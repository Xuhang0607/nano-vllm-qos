import asyncio
from dataclasses import asdict, dataclass
from enum import Enum

from nanovllm.engine.storage_backend import KVStorageBackend


class KVTransferState(str, Enum):
    ABSENT = "absent"
    FETCHING = "fetching"
    RESIDENT = "resident"
    WRITING = "writing"
    EVICTING = "evicting"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class KVTransferSnapshot:
    key: str
    state: KVTransferState
    generation: int
    has_payload: bool
    waiters: int
    error: str | None


@dataclass(slots=True)
class KVTransferMetrics:
    backend_gets: int = 0
    coalesced_gets: int = 0
    backend_puts: int = 0
    backend_removes: int = 0
    cancelled_waiters: int = 0
    failures: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class _Entry:
    state: KVTransferState = KVTransferState.ABSENT
    generation: int = 0
    payload: bytes | None = None
    task: asyncio.Task | None = None
    io_tail: asyncio.Task | None = None
    waiters: int = 0
    error: str | None = None


class AsyncKVTransferCoordinator:
    """Deduplicate KV I/O and serialize conflicting operations per object key."""

    def __init__(self, backend: KVStorageBackend):
        self.backend = backend
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()
        self._metrics = KVTransferMetrics()
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("KV transfer coordinator is closed")

    async def _wait_for_previous(self, previous: asyncio.Task | None):
        if previous is None:
            return
        await asyncio.shield(asyncio.gather(previous, return_exceptions=True))

    async def _finish_success(
        self,
        key: str,
        generation: int,
        state: KVTransferState,
        payload: bytes | None,
    ) -> bool:
        async with self._lock:
            entry = self._entries[key]
            if entry.generation != generation:
                return False
            entry.state = state
            entry.payload = payload
            entry.task = None
            entry.error = None
            return True

    async def _finish_failure(self, key: str, generation: int, exc: Exception):
        async with self._lock:
            entry = self._entries[key]
            if entry.generation == generation:
                entry.state = KVTransferState.FAILED
                entry.payload = None
                entry.task = None
                entry.error = f"{type(exc).__name__}: {exc}"
            self._metrics.failures += 1

    async def _fetch(
        self,
        key: str,
        generation: int,
        previous: asyncio.Task | None,
    ) -> bytes | None:
        await self._wait_for_previous(previous)
        try:
            payload = await asyncio.to_thread(self.backend.get, key)
            payload = bytes(payload) if payload is not None else None
        except Exception as exc:
            await self._finish_failure(key, generation, exc)
            raise

        state = (
            KVTransferState.RESIDENT
            if payload is not None
            else KVTransferState.ABSENT
        )
        current = await self._finish_success(key, generation, state, payload)
        return payload if current else None

    async def _put(
        self,
        key: str,
        payload: bytes,
        generation: int,
        previous: asyncio.Task | None,
    ) -> bytes | None:
        await self._wait_for_previous(previous)
        try:
            await asyncio.to_thread(self.backend.put, key, payload)
        except Exception as exc:
            await self._finish_failure(key, generation, exc)
            raise

        current = await self._finish_success(
            key,
            generation,
            KVTransferState.RESIDENT,
            payload,
        )
        return payload if current else None

    async def _evict(
        self,
        key: str,
        generation: int,
        previous: asyncio.Task | None,
    ) -> None:
        await self._wait_for_previous(previous)
        try:
            await asyncio.to_thread(self.backend.remove, key)
        except Exception as exc:
            await self._finish_failure(key, generation, exc)
            raise

        await self._finish_success(
            key,
            generation,
            KVTransferState.ABSENT,
            None,
        )

    async def get(self, key: str) -> bytes | None:
        self._ensure_open()
        if not key:
            raise ValueError("key must not be empty")

        while True:
            joined = False
            retry_after_wait = False
            async with self._lock:
                entry = self._entries.setdefault(key, _Entry())
                if entry.state == KVTransferState.RESIDENT:
                    return entry.payload

                if entry.state in (
                    KVTransferState.FETCHING,
                    KVTransferState.WRITING,
                    KVTransferState.EVICTING,
                ):
                    if entry.task is None:
                        raise RuntimeError(
                            f"{entry.state.value} entry is missing its I/O task"
                        )
                    task = entry.task
                    entry.waiters += 1
                    joined = True
                    retry_after_wait = entry.state == KVTransferState.EVICTING
                    self._metrics.coalesced_gets += 1
                else:
                    entry.generation += 1
                    entry.state = KVTransferState.FETCHING
                    entry.payload = None
                    entry.error = None
                    previous = entry.io_tail
                    task = asyncio.create_task(
                        self._fetch(key, entry.generation, previous)
                    )
                    entry.task = task
                    entry.io_tail = task
                    self._metrics.backend_gets += 1

            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                self._metrics.cancelled_waiters += 1
                raise
            finally:
                if joined:
                    async with self._lock:
                        entry = self._entries[key]
                        entry.waiters = max(0, entry.waiters - 1)

            if not retry_after_wait:
                return result

    async def put(self, key: str, payload: bytes) -> None:
        self._ensure_open()
        if not key:
            raise ValueError("key must not be empty")
        payload = bytes(payload)

        async with self._lock:
            entry = self._entries.setdefault(key, _Entry())
            entry.generation += 1
            entry.state = KVTransferState.WRITING
            entry.payload = None
            entry.error = None
            previous = entry.io_tail
            task = asyncio.create_task(
                self._put(key, payload, entry.generation, previous)
            )
            entry.task = task
            entry.io_tail = task
            self._metrics.backend_puts += 1

        await asyncio.shield(task)

    async def evict(self, key: str) -> None:
        self._ensure_open()
        if not key:
            raise ValueError("key must not be empty")

        async with self._lock:
            entry = self._entries.setdefault(key, _Entry())
            entry.generation += 1
            entry.state = KVTransferState.EVICTING
            entry.payload = None
            entry.error = None
            previous = entry.io_tail
            task = asyncio.create_task(
                self._evict(key, entry.generation, previous)
            )
            entry.task = task
            entry.io_tail = task
            self._metrics.backend_removes += 1

        await asyncio.shield(task)

    async def snapshot(self, key: str) -> KVTransferSnapshot:
        async with self._lock:
            entry = self._entries.get(key, _Entry())
            return KVTransferSnapshot(
                key=key,
                state=entry.state,
                generation=entry.generation,
                has_payload=entry.payload is not None,
                waiters=entry.waiters,
                error=entry.error,
            )

    def metrics(self):
        return self._metrics.to_dict()

    async def wait_idle(self):
        while True:
            async with self._lock:
                pending = {
                    entry.io_tail
                    for entry in self._entries.values()
                    if entry.io_tail is not None and not entry.io_tail.done()
                }
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self.wait_idle()
        await asyncio.to_thread(self.backend.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()
