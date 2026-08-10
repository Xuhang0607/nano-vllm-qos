import asyncio

import pytest

from nanovllm.engine.kv_page import (
    AsyncKVPageStore,
    KVPageCodec,
    KVPageCompatibilityError,
    KVPageFormatError,
    KVPageLayout,
    KVPageMetadata,
    TorchKVPageIO,
)
from nanovllm.engine.storage_backend import InMemoryKVStore, KVCacheIdentity
from nanovllm.engine.transfer_coordinator import AsyncKVTransferCoordinator


def make_page():
    layout = KVPageLayout(
        num_layers=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype="bf16",
    )
    identity = KVCacheIdentity("Qwen/Qwen3-0.6B", "main", "bf16", 2, 4)
    metadata = KVPageMetadata(identity.digest, tp_rank=1, page_index=3, layout=layout)
    payload = bytes(index % 251 for index in range(layout.nbytes))
    return metadata, payload


def test_layout_normalizes_dtype_and_computes_packed_page_size():
    layout = KVPageLayout(3, 16, 4, 128, "torch.bfloat16")
    assert layout.shape == (2, 3, 16, 4, 128)
    assert layout.dtype == "bfloat16"
    assert layout.nbytes == 2 * 3 * 16 * 4 * 128 * 2


def test_page_envelope_round_trip_and_compatibility_checks():
    metadata, payload = make_page()
    envelope = KVPageCodec.encode(metadata, payload)

    record = KVPageCodec.decode(
        envelope,
        expected_identity_digest=metadata.identity_digest,
        expected_tp_rank=metadata.tp_rank,
        expected_page_index=metadata.page_index,
        expected_layout=metadata.layout,
    )

    assert record.metadata == metadata
    assert record.payload == payload
    assert envelope != payload


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"expected_identity_digest": "another-model"}, "identity mismatch"),
        ({"expected_tp_rank": 0}, "TP rank mismatch"),
        ({"expected_page_index": 4}, "page index mismatch"),
        ({"expected_layout": KVPageLayout(2, 4, 1, 8, "bf16")}, "layout mismatch"),
    ],
)
def test_page_envelope_rejects_incompatible_consumers(expected, message):
    metadata, payload = make_page()
    envelope = KVPageCodec.encode(metadata, payload)
    with pytest.raises(KVPageCompatibilityError, match=message):
        KVPageCodec.decode(envelope, **expected)


def test_page_envelope_detects_corruption_and_truncation():
    metadata, payload = make_page()
    envelope = bytearray(KVPageCodec.encode(metadata, payload))
    envelope[-20] ^= 0xFF

    with pytest.raises(KVPageFormatError, match="checksum"):
        KVPageCodec.decode(envelope)
    with pytest.raises(KVPageFormatError, match="truncated"):
        KVPageCodec.decode(b"short")


def test_page_envelope_validates_payload_size_before_encoding():
    metadata, payload = make_page()
    with pytest.raises(KVPageCompatibilityError, match="expected"):
        KVPageCodec.encode(metadata, payload[:-1])


def test_async_page_store_round_trip_uses_the_transfer_coordinator():
    async def scenario():
        metadata, payload = make_page()
        backend = InMemoryKVStore()
        coordinator = AsyncKVTransferCoordinator(backend)
        store = AsyncKVPageStore(coordinator)

        await store.put("page-key", metadata, payload)
        assert backend.objects["page-key"] != payload

        record = await store.get(
            "page-key",
            expected_identity_digest=metadata.identity_digest,
            expected_tp_rank=1,
            expected_page_index=3,
            expected_layout=metadata.layout,
        )
        assert record.metadata == metadata
        assert record.payload == payload

        await store.evict("page-key")
        assert await store.get("page-key") is None

    asyncio.run(scenario())


def test_torch_page_io_round_trip_handles_noncontiguous_physical_page():
    torch = pytest.importorskip("torch")
    cache = torch.arange(
        2 * 3 * 4 * 5 * 2 * 8,
        dtype=torch.float32,
    ).reshape(2, 3, 4, 5, 2, 8)
    page_io = TorchKVPageIO(cache)
    expected = cache[:, :, 2].clone()

    payload = page_io.export_page(2)
    cache[:, :, 2].zero_()
    page_io.import_page(2, payload)

    torch.testing.assert_close(cache[:, :, 2], expected)
    assert len(payload) == page_io.layout.nbytes
