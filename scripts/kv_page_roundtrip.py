from __future__ import annotations

import argparse

from nanovllm.engine.kv_page import (
    KVPageCodec,
    KVPageMetadata,
    TorchKVPageIO,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a synthetic KV page export/restore round trip"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def main():
    import torch

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    dtype = getattr(torch, args.dtype)
    shape = (2, 3, 4, 8, 2, 16)
    torch.manual_seed(7)
    cache = torch.randn(shape, dtype=dtype, device=args.device)
    page_io = TorchKVPageIO(cache)
    physical_block_id = 2
    expected = cache[:, :, physical_block_id].clone()

    raw_page = page_io.export_page(physical_block_id)
    metadata = KVPageMetadata(
        identity_digest="synthetic-roundtrip-v1",
        tp_rank=0,
        page_index=5,
        layout=page_io.layout,
    )
    envelope = KVPageCodec.encode(metadata, raw_page)

    cache[:, :, physical_block_id].zero_()
    record = KVPageCodec.decode(
        envelope,
        expected_identity_digest=metadata.identity_digest,
        expected_tp_rank=0,
        expected_page_index=5,
        expected_layout=page_io.layout,
    )
    page_io.import_page(physical_block_id, record.payload)
    torch.testing.assert_close(cache[:, :, physical_block_id], expected)

    print("KV page round trip passed")
    print(f"  device={args.device}, dtype={args.dtype}")
    print(f"  layout={page_io.layout.shape}")
    print(f"  raw_page_bytes={len(raw_page)}")
    print(f"  envelope_bytes={len(envelope)}")


if __name__ == "__main__":
    main()
