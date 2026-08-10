import pytest

from nanovllm.engine.remote_catalog import (
    RemoteCatalogSnapshotCodec,
    RemoteCatalogSnapshotError,
)
from nanovllm.engine.remote_restore import (
    RemoteKVRestoreService,
    RemotePageDescriptor,
    RemotePrefixCatalog,
)
from nanovllm.engine.storage_backend import InMemoryKVStore


class FailingCatalogStore(InMemoryKVStore):
    def put(self, key, payload):
        if key == "catalog":
            raise OSError("catalog storage unavailable")
        super().put(key, payload)


def make_branched_prefixes():
    shared = RemotePageDescriptor("page-shared", "model-a", 0, 0)
    first_keys = ((1, 2), (3, 4))
    first_pages = (
        shared,
        RemotePageDescriptor("page-first", "model-a", 0, 1),
    )
    second_keys = ((1, 2), (5, 6))
    second_pages = (
        shared,
        RemotePageDescriptor("page-second", "model-a", 0, 1),
    )
    return (first_keys, first_pages), (second_keys, second_pages)


def test_catalog_snapshot_round_trip_preserves_branched_radix_paths():
    first, second = make_branched_prefixes()
    catalog = RemotePrefixCatalog()
    catalog.register(*first)
    catalog.register(*second)

    envelope = RemoteCatalogSnapshotCodec.encode(catalog.snapshot())
    snapshot = RemoteCatalogSnapshotCodec.decode(envelope)
    restored = RemotePrefixCatalog()

    assert restored.merge_snapshot(snapshot) == 2
    assert restored.match(first[0]) == first[1]
    assert restored.match(second[0]) == second[1]
    assert RemoteCatalogSnapshotCodec.encode(restored.snapshot()) == envelope
    restored.validate()


def test_catalog_snapshot_rejects_payload_tampering():
    first, _ = make_branched_prefixes()
    catalog = RemotePrefixCatalog()
    catalog.register(*first)
    envelope = RemoteCatalogSnapshotCodec.encode(catalog.snapshot())
    tampered = envelope.replace(b"page-first", b"page-other", 1)

    with pytest.raises(RemoteCatalogSnapshotError, match="checksum"):
        RemoteCatalogSnapshotCodec.decode(tampered)


def test_service_restart_recovers_the_latest_persistent_catalog():
    first, second = make_branched_prefixes()
    backend = InMemoryKVStore()
    service = RemoteKVRestoreService(
        backend,
        catalog_key="catalog",
        catalog_identity=("model-a", 0),
    )
    try:
        service.register_existing_prefix(*first)
        service.register_existing_prefix(*second)
        service.flush_catalog_saves(wait_for_all=True, timeout=2)
        assert service.metrics()["catalog_save_completed"] == 2
    finally:
        service.close()

    restarted = RemoteKVRestoreService(
        backend,
        catalog_key="catalog",
        catalog_identity=("model-a", 0),
    )
    try:
        assert restarted.catalog.match(first[0]) == first[1]
        assert restarted.catalog.match(second[0]) == second[1]
        assert restarted.metrics()["catalog_loaded_prefixes"] == 2
        restarted.catalog.validate()
    finally:
        restarted.close()


def test_completed_writeback_is_discoverable_after_service_restart():
    first, _ = make_branched_prefixes()
    backend = InMemoryKVStore()
    service = RemoteKVRestoreService(
        backend,
        catalog_key="catalog",
        catalog_identity=("model-a", 0),
    )
    try:
        pending = service.submit_writeback(
            first[0],
            first[1],
            {0: b"shared-envelope", 1: b"first-envelope"},
        )
        service.complete_writeback(pending, timeout=2)
        service.flush_catalog_saves(wait_for_all=True, timeout=2)
    finally:
        service.close()

    restarted = RemoteKVRestoreService(
        backend,
        catalog_key="catalog",
        catalog_identity=("model-a", 0),
    )
    try:
        assert restarted.catalog.match(first[0]) == first[1]
        assert backend.get("page-shared") == b"shared-envelope"
        assert backend.get("page-first") == b"first-envelope"
    finally:
        restarted.close()


def test_corrupted_persistent_catalog_falls_back_to_an_empty_catalog():
    backend = InMemoryKVStore()
    backend.put("catalog", b'{"not":"a valid snapshot"}')

    service = RemoteKVRestoreService(
        backend,
        catalog_key="catalog",
        catalog_identity=("model-a", 0),
    )
    try:
        assert service.catalog.match(((1, 2),)) == ()
        assert service.metrics()["catalog_load_failed"] == 1
        assert service.catalog_errors
    finally:
        service.close()


def test_persistent_catalog_from_another_engine_identity_is_rejected():
    first, _ = make_branched_prefixes()
    catalog = RemotePrefixCatalog()
    catalog.register(*first)
    backend = InMemoryKVStore()
    backend.put(
        "catalog",
        RemoteCatalogSnapshotCodec.encode(catalog.snapshot()),
    )

    service = RemoteKVRestoreService(
        backend,
        catalog_key="catalog",
        catalog_identity=("model-b", 0),
    )
    try:
        assert service.catalog.match(first[0]) == ()
        assert service.metrics()["catalog_load_failed"] == 1
        assert "identity" in service.catalog_errors[-1]
    finally:
        service.close()


def test_catalog_save_failure_does_not_hide_the_in_memory_prefix():
    first, _ = make_branched_prefixes()
    backend = FailingCatalogStore()
    service = RemoteKVRestoreService(
        backend,
        catalog_key="catalog",
        catalog_identity=("model-a", 0),
    )
    try:
        service.register_existing_prefix(*first)
        service.flush_catalog_saves(wait_for_all=True, timeout=2)

        assert service.catalog.match(first[0]) == first[1]
        assert service.metrics()["catalog_save_failed"] == 1
        assert service.metrics()["catalog_save_pending"] == 0
    finally:
        service.close()
