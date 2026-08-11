import asyncio
import threading

import pytest

from nanovllm.engine.storage_backend import InMemoryKVStore
from nanovllm.engine.transfer_coordinator import (
    AsyncKVTransferCoordinator,
    KVTransferState,
)


class BlockingGetStore(InMemoryKVStore):
    def __init__(self, objects=None):
        super().__init__()
        self.objects.update(objects or {})
        self.get_started = threading.Event()
        self.allow_get = threading.Event()
        self.get_count = 0
        self.remove_count = 0

    def get(self, key):
        self.get_count += 1
        self.get_started.set()
        if not self.allow_get.wait(timeout=2):
            raise TimeoutError("test did not release the blocked get")
        return super().get(key)

    def remove(self, key):
        self.remove_count += 1
        super().remove(key)


class FailOnceStore(InMemoryKVStore):
    def __init__(self):
        super().__init__()
        self.objects["page"] = b"kv"
        self.get_count = 0

    def get(self, key):
        self.get_count += 1
        if self.get_count == 1:
            raise OSError("temporary remote failure")
        return super().get(key)


def test_concurrent_gets_share_one_backend_fetch():
    async def scenario():
        backend = BlockingGetStore({"page": b"kv-data"})
        coordinator = AsyncKVTransferCoordinator(backend)

        first = asyncio.create_task(coordinator.get("page"))
        assert await asyncio.to_thread(backend.get_started.wait, 1)
        second = asyncio.create_task(coordinator.get("page"))
        await asyncio.sleep(0)
        backend.allow_get.set()

        assert await asyncio.gather(first, second) == [b"kv-data", b"kv-data"]
        assert backend.get_count == 1
        assert coordinator.metrics()["backend_gets"] == 1
        assert coordinator.metrics()["backend_get_bytes"] == len(b"kv-data")
        assert coordinator.metrics()["coalesced_gets"] == 1
        assert (await coordinator.snapshot("page")).state == KVTransferState.RESIDENT

    asyncio.run(scenario())


def test_waiter_cancellation_does_not_cancel_shared_fetch():
    async def scenario():
        backend = BlockingGetStore({"page": b"kv-data"})
        coordinator = AsyncKVTransferCoordinator(backend)

        cancelled = asyncio.create_task(coordinator.get("page"))
        assert await asyncio.to_thread(backend.get_started.wait, 1)
        survivor = asyncio.create_task(coordinator.get("page"))
        await asyncio.sleep(0)

        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        backend.allow_get.set()
        assert await survivor == b"kv-data"
        assert backend.get_count == 1
        assert coordinator.metrics()["cancelled_waiters"] == 1
        assert (await coordinator.snapshot("page")).state == KVTransferState.RESIDENT

    asyncio.run(scenario())


def test_failed_fetch_is_observable_and_retryable():
    async def scenario():
        backend = FailOnceStore()
        coordinator = AsyncKVTransferCoordinator(backend)

        with pytest.raises(OSError, match="temporary remote failure"):
            await coordinator.get("page")

        failed = await coordinator.snapshot("page")
        assert failed.state == KVTransferState.FAILED
        assert "temporary remote failure" in failed.error

        assert await coordinator.get("page") == b"kv"
        assert backend.get_count == 2
        assert coordinator.metrics()["failures"] == 1
        assert (await coordinator.snapshot("page")).state == KVTransferState.RESIDENT

    asyncio.run(scenario())


def test_evict_invalidates_an_inflight_fetch_and_preserves_io_order():
    async def scenario():
        backend = BlockingGetStore({"page": b"stale-kv"})
        coordinator = AsyncKVTransferCoordinator(backend)

        fetch = asyncio.create_task(coordinator.get("page"))
        assert await asyncio.to_thread(backend.get_started.wait, 1)
        eviction = asyncio.create_task(coordinator.evict("page"))
        await asyncio.sleep(0)
        backend.allow_get.set()

        assert await fetch is None
        await eviction
        assert backend.remove_count == 1
        assert not backend.exists("page")
        snapshot = await coordinator.snapshot("page")
        assert snapshot.state == KVTransferState.ABSENT
        assert not snapshot.has_payload

    asyncio.run(scenario())


def test_get_joins_an_inflight_write_and_close_rejects_new_work():
    async def scenario():
        backend = InMemoryKVStore()
        coordinator = AsyncKVTransferCoordinator(backend)

        write = asyncio.create_task(coordinator.put("page", b"fresh-kv"))
        fetched = asyncio.create_task(coordinator.get("page"))
        await write
        assert await fetched == b"fresh-kv"
        assert coordinator.metrics()["backend_gets"] == 0
        assert coordinator.metrics()["backend_put_bytes"] == len(b"fresh-kv")

        await coordinator.close()
        with pytest.raises(RuntimeError, match="closed"):
            await coordinator.get("page")

    asyncio.run(scenario())
