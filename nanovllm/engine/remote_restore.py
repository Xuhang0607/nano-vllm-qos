import asyncio
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from threading import Event, Thread

from nanovllm.engine.hierarchical_cache import CacheTier
from nanovllm.engine.radix_cache import BlockKey, RadixPrefixCache
from nanovllm.engine.transfer_coordinator import AsyncKVTransferCoordinator


class RemoteKVPageMissing(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class RemotePageDescriptor:
    object_key: str
    identity_digest: str
    tp_rank: int
    page_index: int

    def __post_init__(self):
        if not self.object_key or not self.identity_digest:
            raise ValueError("remote page keys and identities must not be empty")
        if self.tp_rank < 0 or self.page_index < 0:
            raise ValueError("tp_rank and page_index must be non-negative")


@dataclass(slots=True, frozen=True)
class RemoteRestoreCandidate:
    pages: tuple[RemotePageDescriptor, ...]
    estimated_total_ms: float
    estimated_transfer_bytes: int

    @property
    def num_blocks(self):
        return len(self.pages)


@dataclass(slots=True)
class PendingRemoteRestore:
    seq_id: int
    pages: tuple[RemotePageDescriptor, ...]
    block_ids: tuple[int, ...]
    futures: tuple[Future, ...]

    def __post_init__(self):
        lengths = {len(self.pages), len(self.block_ids), len(self.futures)}
        if len(lengths) != 1 or not self.pages:
            raise ValueError("pending restore vectors must have the same non-zero length")

    @property
    def done(self):
        for future in self.futures:
            if not future.done():
                continue
            if future.cancelled() or future.exception() is not None:
                return True
            if future.result() is None:
                return True
        return all(future.done() for future in self.futures)

    def result(self, timeout=None):
        for descriptor, future in zip(self.pages, self.futures):
            if not future.done():
                continue
            envelope = future.result()
            if envelope is None:
                raise RemoteKVPageMissing(
                    f"remote KV page is missing: {descriptor.object_key}"
                )

        envelopes = []
        for descriptor, future in zip(self.pages, self.futures):
            envelope = future.result(timeout=timeout)
            if envelope is None:
                raise RemoteKVPageMissing(
                    f"remote KV page is missing: {descriptor.object_key}"
                )
            envelopes.append(envelope)
        return tuple(envelopes)

    def cancel_waiters(self):
        for future in self.futures:
            future.cancel()


@dataclass(slots=True)
class PendingRemoteWriteback:
    block_keys: tuple[BlockKey, ...]
    pages: tuple[RemotePageDescriptor, ...]
    write_pages: tuple[RemotePageDescriptor, ...]
    futures: tuple[Future, ...]

    def __post_init__(self):
        if not self.block_keys or len(self.block_keys) != len(self.pages):
            raise ValueError("write-back prefix vectors must have the same non-zero length")
        if not self.futures or len(self.write_pages) != len(self.futures):
            raise ValueError("write-back pages and futures must have the same non-zero length")

    @property
    def done(self):
        for future in self.futures:
            if not future.done():
                continue
            if future.cancelled() or future.exception() is not None:
                return True
        return all(future.done() for future in self.futures)

    def result(self, timeout=None):
        # Surface a completed failure before waiting on unrelated slow pages.
        for future in self.futures:
            if future.done():
                future.result()
        for future in self.futures:
            future.result(timeout=timeout)


def completed_kv_page_span(
    num_cached_tokens: int,
    num_scheduled_tokens: int,
    block_size: int,
) -> tuple[int, int]:
    """Return the half-open range of pages completed by the latest model run."""
    if num_cached_tokens < 0 or num_scheduled_tokens < 0 or block_size <= 0:
        raise ValueError("token counts must be non-negative and block_size positive")
    start = num_cached_tokens // block_size
    end = (num_cached_tokens + num_scheduled_tokens) // block_size
    return start, end


class RemotePrefixCatalog:
    """Map page-aligned token prefixes to remote object descriptors."""

    def __init__(self):
        self.index = RadixPrefixCache()
        self.descriptors: dict[int, RemotePageDescriptor] = {}
        self._next_handle = 0
        self._identity = None

    def register(
        self,
        block_keys: tuple[BlockKey, ...] | list[BlockKey],
        descriptors: tuple[RemotePageDescriptor, ...]
        | list[RemotePageDescriptor],
    ):
        block_keys = tuple(block_keys)
        descriptors = tuple(descriptors)
        if len(block_keys) != len(descriptors):
            raise ValueError("block keys and remote descriptors must have the same length")
        if not block_keys:
            return
        for index, descriptor in enumerate(descriptors):
            if descriptor.page_index != index:
                raise ValueError("remote descriptor page indexes must start at zero")
        identities = {(page.identity_digest, page.tp_rank) for page in descriptors}
        if len(identities) != 1:
            raise ValueError("a remote prefix must belong to one model identity and TP rank")
        identity = next(iter(identities))
        if self._identity is not None and identity != self._identity:
            raise ValueError("remote catalog identity does not match its existing entries")
        self._identity = identity

        handles = tuple(range(self._next_handle, self._next_handle + len(block_keys)))
        self._next_handle += len(handles)
        for handle, descriptor in zip(handles, descriptors):
            self.descriptors[handle] = descriptor
        self.index.insert(block_keys, handles)

        live_handles = set(self.index.block_locations)
        for handle in handles:
            if handle not in live_handles:
                self.descriptors.pop(handle, None)

    def match(self, block_keys) -> tuple[RemotePageDescriptor, ...]:
        handles = self.index.match(tuple(block_keys))
        return tuple(self.descriptors[handle] for handle in handles)

    def validate(self):
        self.index.validate()
        assert set(self.descriptors) == set(self.index.block_locations)


class RemoteKVRestoreService:
    """Run deduplicated backend I/O without moving CUDA work off the main thread."""

    def __init__(self, backend, catalog=None, planner=None):
        self.catalog = catalog or RemotePrefixCatalog()
        self.planner = planner
        self._backend = backend
        self._loop = asyncio.new_event_loop()
        self._ready = Event()
        self._thread = Thread(
            target=self._run_loop,
            name="nanovllm-kv-restore",
            daemon=True,
        )
        self._closed = False
        self._inflight_writes: dict[str, Future] = {}
        self._writeback_metrics = {
            "writeback_submitted": 0,
            "writeback_completed": 0,
            "writeback_failed": 0,
            "writeback_export_failed": 0,
            "writeback_pages": 0,
            "writeback_coalesced_pages": 0,
        }
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("remote KV restore event loop did not start")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._coordinator = AsyncKVTransferCoordinator(self._backend)
        self._ready.set()
        self._loop.run_forever()

    def _submit(self, coroutine):
        if self._closed:
            coroutine.close()
            raise RuntimeError("remote KV restore service is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def publish_prefix(self, block_keys, descriptors, envelopes, timeout=None):
        block_keys = tuple(block_keys)
        descriptors = tuple(descriptors)
        envelopes = tuple(envelopes)
        if len({len(block_keys), len(descriptors), len(envelopes)}) != 1:
            raise ValueError("published remote prefix vectors must have the same length")
        if not descriptors:
            raise ValueError("published remote prefix must contain at least one page")
        futures = [
            self._submit(self._coordinator.put(descriptor.object_key, envelope))
            for descriptor, envelope in zip(descriptors, envelopes)
        ]
        for future in futures:
            future.result(timeout=timeout)
        self.catalog.register(block_keys, descriptors)

    def register_existing_prefix(self, block_keys, descriptors):
        self.catalog.register(block_keys, descriptors)

    def pages_requiring_write(self, block_keys, descriptors) -> tuple[int, ...]:
        block_keys = tuple(block_keys)
        descriptors = tuple(descriptors)
        if not block_keys or len(block_keys) != len(descriptors):
            raise ValueError("write-back prefix vectors must have the same non-zero length")

        published_pages = self.catalog.match(block_keys)
        required = []
        for index, descriptor in enumerate(descriptors):
            if index < len(published_pages):
                continue
            future = self._inflight_writes.get(descriptor.object_key)
            if future is not None and (
                future.cancelled()
                or (future.done() and future.exception() is not None)
            ):
                self._inflight_writes.pop(descriptor.object_key, None)
                future = None
            if future is None:
                required.append(index)
        return tuple(required)

    def submit_writeback(
        self,
        block_keys,
        descriptors,
        envelopes_by_index,
    ) -> PendingRemoteWriteback | None:
        block_keys = tuple(block_keys)
        descriptors = tuple(descriptors)
        if not block_keys or len(block_keys) != len(descriptors):
            raise ValueError("write-back prefix vectors must have the same non-zero length")

        published_pages = self.catalog.match(block_keys)
        write_plan = []
        for index, descriptor in enumerate(descriptors):
            if index < len(published_pages):
                continue
            future = self._inflight_writes.get(descriptor.object_key)
            if future is None and index not in envelopes_by_index:
                raise ValueError(f"missing exported KV page {index} for write-back")
            write_plan.append((index, descriptor, future))

        if not write_plan:
            return None

        write_pages = []
        futures = []
        for index, descriptor, future in write_plan:
            if future is None:
                future = self._submit(
                    self._coordinator.put(
                        descriptor.object_key,
                        envelopes_by_index[index],
                    )
                )
                self._inflight_writes[descriptor.object_key] = future
                self._writeback_metrics["writeback_pages"] += 1
            else:
                self._writeback_metrics["writeback_coalesced_pages"] += 1
            write_pages.append(descriptor)
            futures.append(future)

        self._writeback_metrics["writeback_submitted"] += 1
        return PendingRemoteWriteback(
            block_keys=block_keys,
            pages=descriptors,
            write_pages=tuple(write_pages),
            futures=tuple(futures),
        )

    def complete_writeback(self, pending: PendingRemoteWriteback, timeout=None):
        pending.result(timeout=timeout)
        self.catalog.register(pending.block_keys, pending.pages)
        for page, future in zip(pending.write_pages, pending.futures):
            if self._inflight_writes.get(page.object_key) is future:
                self._inflight_writes.pop(page.object_key, None)
        self._writeback_metrics["writeback_completed"] += 1

    def fail_writeback(self, pending: PendingRemoteWriteback):
        for page, future in zip(pending.write_pages, pending.futures):
            if self._inflight_writes.get(page.object_key) is future:
                self._inflight_writes.pop(page.object_key, None)
        self._writeback_metrics["writeback_failed"] += 1

    def record_writeback_export_failure(self):
        self._writeback_metrics["writeback_export_failed"] += 1

    def plan_restore(
        self,
        seq,
        local_cached_blocks: int,
        prefill_ms_per_token: float,
    ) -> RemoteRestoreCandidate | None:
        cacheable_blocks = max(0, seq.num_blocks - 1)
        block_keys = tuple(tuple(seq.block(i)) for i in range(cacheable_blocks))
        pages = self.catalog.match(block_keys)
        if len(pages) <= local_cached_blocks:
            return None

        if self.planner is None:
            return RemoteRestoreCandidate(pages, 0.0, 0)

        plan = self.planner.plan(
            total_tokens=seq.num_tokens,
            local_cached_tokens=local_cached_blocks * seq.block_size,
            cached_tokens_by_tier={
                CacheTier.MOONCAKE: len(pages) * seq.block_size,
            },
            prefill_ms_per_token=prefill_ms_per_token,
        )
        if plan.action != "restore" or plan.source_tier != CacheTier.MOONCAKE:
            return None
        source_blocks = plan.source_cached_tokens // seq.block_size
        return RemoteRestoreCandidate(
            pages=pages[:source_blocks],
            estimated_total_ms=plan.total_ms,
            estimated_transfer_bytes=plan.transfer_bytes,
        )

    def submit_restore(
        self,
        seq_id: int,
        pages: tuple[RemotePageDescriptor, ...],
        block_ids: tuple[int, ...],
    ) -> PendingRemoteRestore:
        if len(pages) != len(block_ids) or not pages:
            raise ValueError("restore pages and block ids must have the same non-zero length")
        futures = tuple(
            self._submit(self._coordinator.get(page.object_key)) for page in pages
        )
        return PendingRemoteRestore(seq_id, pages, block_ids, futures)

    @staticmethod
    def wait_for_any(pending_restores, timeout=None):
        futures = {
            future
            for pending in pending_restores
            for future in pending.futures
            if not future.done()
        }
        if not futures:
            return
        wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)

    def metrics(self):
        return self._coordinator.metrics() | self._writeback_metrics.copy()

    def close(self):
        if self._closed:
            return
        self._closed = True
        future = asyncio.run_coroutine_threadsafe(
            self._coordinator.close(),
            self._loop,
        )
        future.result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("remote KV restore event loop did not stop")
        self._inflight_writes.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
