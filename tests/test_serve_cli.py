import sys

from nanovllm.serve.__main__ import parse_args


def test_mooncake_serving_defaults_use_external_store_segment(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nanovllm.serve",
            "--model",
            "/models/Qwen3-0.6B",
            "--kv-storage-backend",
            "mooncake",
        ],
    )

    args = parse_args()

    assert args.kv_storage_backend == "mooncake"
    assert args.mooncake_global_segment_mib == 0
    assert args.mooncake_local_buffer_mib == 512
    assert args.mooncake_protocol == "tcp"


def test_mooncake_serving_options_can_be_overridden(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nanovllm.serve",
            "--model",
            "/models/Qwen3-0.6B",
            "--kv-storage-backend",
            "mooncake",
            "--mooncake-global-segment-mib",
            "256",
            "--mooncake-local-buffer-mib",
            "768",
            "--remote-kv-bandwidth-gbps",
            "50",
            "--remote-kv-fixed-latency-ms",
            "0.2",
            "--remote-kv-force-restore",
        ],
    )

    args = parse_args()

    assert args.mooncake_global_segment_mib == 256
    assert args.mooncake_local_buffer_mib == 768
    assert args.remote_kv_bandwidth_gbps == 50
    assert args.remote_kv_fixed_latency_ms == 0.2
    assert args.remote_kv_force_restore is True
