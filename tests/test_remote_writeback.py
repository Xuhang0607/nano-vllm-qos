import threading

import pytest

from nanovllm.engine.remote_restore import (
    RemoteKVRestoreService,
    RemotePageDescriptor,
    completed_kv_page_span,
)
from nanovllm.engine.storage_backend import InMemoryKVStore


class BlockingPutStore(InMemoryKVStore):
    def __init__(self):
        super().__init__()
        self.put_started = threading.Event()
        self.allow_put = threading.Event()
        self.put_count = 0
        self._count_lock = threading.Lock()

    def put(self, key, payload):
        with self._count_lock:
            self.put_count += 1
            self.put_started.set()
        if not self.allow_put.wait(timeout=2):
            raise TimeoutError("test did not release the remote write")
        super().put(key, payload)


class FailingPutStore(InMemoryKVStore):
    def put(self, key, payload):
        if key == "page-1":
            raise OSError("remote write failed")
        super().put(key, payload)


def make_prefix():
    keys = ((1, 2), (3, 4))
    descriptors = (
        RemotePageDescriptor("page-0", "model-a", 0, 0),
        RemotePageDescriptor("page-1", "model-a", 0, 1),
    )
    envelopes = {0: b"envelope-0", 1: b"envelope-1"}
    return keys, descriptors, envelopes


def test_completed_kv_page_span_only_advances_at_page_boundaries():
    assert completed_kv_page_span(0, 1, 2) == (0, 0)
    assert completed_kv_page_span(1, 1, 2) == (0, 1)
    assert completed_kv_page_span(4, 1, 2) == (2, 2)
    assert completed_kv_page_span(4, 2, 2) == (2, 3)


def test_successful_writeback_publishes_catalog_after_backend_puts():
    keys, descriptors, envelopes = make_prefix()
    backend = InMemoryKVStore()
    service = RemoteKVRestoreService(backend)
    try:
        assert service.pages_requiring_write(keys, descriptors) == (0, 1)
        pending = service.submit_writeback(keys, descriptors, envelopes)

        service.complete_writeback(pending, timeout=2)

        assert backend.objects == {
            "page-0": b"envelope-0",
            "page-1": b"envelope-1",
        }
        assert service.catalog.match(keys) == descriptors
        assert service.metrics()["writeback_completed"] == 1
        assert service.metrics()["writeback_pages"] == 2
    finally:
        service.close()


def test_failed_writeback_does_not_publish_a_partial_prefix():
    keys, descriptors, envelopes = make_prefix()
    service = RemoteKVRestoreService(FailingPutStore())
    try:
        pending = service.submit_writeback(keys, descriptors, envelopes)

        with pytest.raises(OSError, match="remote write failed"):
            service.complete_writeback(pending, timeout=2)
        service.fail_writeback(pending)

        assert service.catalog.match(keys) == ()
        assert service.pages_requiring_write(keys, descriptors) == (0, 1)
        assert service.metrics()["writeback_failed"] == 1
    finally:
        service.close()


def test_concurrent_writebacks_coalesce_backend_puts():
    keys, descriptors, envelopes = make_prefix()
    backend = BlockingPutStore()
    service = RemoteKVRestoreService(backend)
    try:
        first = service.submit_writeback(keys, descriptors, envelopes)
        assert backend.put_started.wait(timeout=1)
        assert service.pages_requiring_write(keys, descriptors) == ()
        second = service.submit_writeback(keys, descriptors, {})

        backend.allow_put.set()
        service.complete_writeback(first, timeout=2)
        service.complete_writeback(second, timeout=2)

        assert backend.put_count == 2
        assert service.metrics()["backend_puts"] == 2
        assert service.metrics()["writeback_coalesced_pages"] == 2
        assert service.catalog.match(keys) == descriptors
    finally:
        backend.allow_put.set()
        service.close()


def test_longer_prefix_reuses_an_inflight_shorter_prefix_write():
    keys, descriptors, envelopes = make_prefix()
    backend = BlockingPutStore()
    service = RemoteKVRestoreService(backend)
    try:
        first = service.submit_writeback(
            keys[:1],
            descriptors[:1],
            {0: envelopes[0]},
        )
        assert backend.put_started.wait(timeout=1)
        assert service.pages_requiring_write(keys, descriptors) == (1,)
        extended = service.submit_writeback(
            keys,
            descriptors,
            {1: envelopes[1]},
        )

        backend.allow_put.set()
        service.complete_writeback(extended, timeout=2)
        service.complete_writeback(first, timeout=2)

        assert backend.put_count == 2
        assert service.metrics()["writeback_coalesced_pages"] == 1
        assert service.catalog.match(keys) == descriptors
        service.catalog.validate()
    finally:
        backend.allow_put.set()
        service.close()
