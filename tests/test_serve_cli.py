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
    assert args.kv_reclaim_policy == "slo_aware"
    assert args.kv_reclaim_max_keep_ratio == 0.75
    assert args.kv_reclaim_target_free_blocks == 2
    assert args.num_kvcache_blocks is None
    assert args.kv_compression_policy == "none"
    assert args.kv_compression_sink_blocks == 1
    assert args.kv_compression_recent_blocks == 8
    assert args.kv_compression_importance_blocks == 2
    assert args.metrics_summary_mode == "cached"
    assert args.scheduler_trace_path is None
    assert args.max_num_active_seqs is None
    assert args.kv_admission_lookahead == 0


def test_scheduler_trace_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nanovllm.serve", "--scheduler-trace-path", "/tmp/run.jsonl"])
    assert parse_args().scheduler_trace_path == "/tmp/run.jsonl"


def test_active_limit_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nanovllm.serve", "--max-num-active-seqs", "8"])
    assert parse_args().max_num_active_seqs == 8


def test_full_summary_ablation_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nanovllm.serve", "--mock", "--metrics-summary-mode", "full"])
    assert parse_args().metrics_summary_mode == "full"


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
            "--kv-reclaim-policy",
            "recompute",
            "--kv-reclaim-max-keep-ratio",
            "0.5",
            "--kv-reclaim-target-free-blocks",
            "4",
            "--num-kvcache-blocks",
            "24",
            "--kv-compression-policy",
            "query_aware",
            "--kv-compression-sink-blocks",
            "2",
            "--kv-compression-recent-blocks",
            "6",
            "--kv-compression-importance-blocks",
            "3",
        ],
    )

    args = parse_args()

    assert args.mooncake_global_segment_mib == 256
    assert args.mooncake_local_buffer_mib == 768
    assert args.remote_kv_bandwidth_gbps == 50
    assert args.remote_kv_fixed_latency_ms == 0.2
    assert args.remote_kv_force_restore is True
    assert args.kv_reclaim_policy == "recompute"
    assert args.kv_reclaim_max_keep_ratio == 0.5
    assert args.kv_reclaim_target_free_blocks == 4
    assert args.num_kvcache_blocks == 24
    assert args.kv_compression_policy == "query_aware"
    assert args.kv_compression_sink_blocks == 2
    assert args.kv_compression_recent_blocks == 6
    assert args.kv_compression_importance_blocks == 3
