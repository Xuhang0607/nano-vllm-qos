import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve nano-vLLM over an OpenAI-compatible API"
    )
    parser.add_argument("--model", help="Local Hugging Face model directory")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key")
    parser.add_argument(
        "--mock", action="store_true", help="Use the UI development backend"
    )
    parser.add_argument("--scheduling-policy", choices=("fcfs", "pals"), default="pals")
    parser.add_argument("--scheduler-trace-path", help="Exclusive JSONL flight recorder path; diagnostic only")
    parser.add_argument("--metrics-summary-mode", choices=("cached", "full"), default="cached",
                        help="Reuse unchanged completed-request summaries; full is the ablation baseline")
    parser.add_argument(
        "--prefix-cache-backend", choices=("hash", "radix"), default="radix"
    )
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--kv-admission-lookahead", type=int, default=0,
                        help="Soft priority-aware KV growth reserve in decode tokens; 0 disables")
    parser.add_argument("--max-num-active-seqs", type=int,
                        help="Optional KV-resident request admission limit; distinct from step batch size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--num-kvcache-blocks",
        type=int,
        help="Cap the KV cache block count for reproducible pressure tests",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--kv-reclaim-policy",
        choices=("recompute", "slo_aware"),
        default="slo_aware",
    )
    parser.add_argument("--kv-reclaim-min-keep-ratio", type=float, default=0.0)
    parser.add_argument("--kv-reclaim-max-keep-ratio", type=float, default=0.75)
    parser.add_argument("--kv-reclaim-budget-scale-ms", type=float, default=1000.0)
    parser.add_argument("--kv-reclaim-target-free-blocks", type=int, default=2)
    parser.add_argument(
        "--kv-compression-policy",
        choices=("none", "sink_recent", "query_aware"),
        default="none",
    )
    parser.add_argument("--kv-compression-sink-blocks", type=int, default=1)
    parser.add_argument("--kv-compression-recent-blocks", type=int, default=8)
    parser.add_argument("--kv-compression-importance-blocks", type=int, default=2)
    parser.add_argument("--kv-compression-query-tokens", type=int, default=64)
    parser.add_argument(
        "--kv-compression-trigger-free-ratio", type=float, default=0.15
    )
    parser.add_argument(
        "--kv-storage-backend",
        choices=("none", "mooncake"),
        default="none",
    )
    parser.add_argument("--mooncake-master", default="127.0.0.1:50051")
    parser.add_argument("--mooncake-metadata", default="P2PHANDSHAKE")
    parser.add_argument("--mooncake-hostname", default="localhost")
    parser.add_argument(
        "--mooncake-protocol",
        choices=("tcp", "rdma", "efa"),
        default="tcp",
    )
    parser.add_argument("--mooncake-rdma-devices", default="")
    parser.add_argument("--mooncake-global-segment-mib", type=int, default=0)
    parser.add_argument("--mooncake-local-buffer-mib", type=int, default=512)
    parser.add_argument("--remote-kv-bandwidth-gbps", type=float, default=12.5)
    parser.add_argument("--remote-kv-fixed-latency-ms", type=float, default=0.3)
    parser.add_argument(
        "--remote-kv-force-restore",
        action="store_true",
        help="Disable the cost planner and restore every matching remote prefix",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.mock and not args.model:
        raise SystemExit("--model is required unless --mock is used")
    if args.mock and args.kv_storage_backend != "none":
        raise SystemExit("--kv-storage-backend is unavailable with --mock")
    if args.mooncake_global_segment_mib < 0 or args.mooncake_local_buffer_mib < 0:
        raise SystemExit("Mooncake memory sizes must be non-negative")
    if not (
        0
        <= args.kv_reclaim_min_keep_ratio
        <= args.kv_reclaim_max_keep_ratio
        <= 1
    ):
        raise SystemExit("KV reclaim ratios must satisfy 0 <= min <= max <= 1")
    if args.kv_reclaim_budget_scale_ms <= 0:
        raise SystemExit("--kv-reclaim-budget-scale-ms must be positive")
    if args.kv_reclaim_target_free_blocks < 1:
        raise SystemExit("--kv-reclaim-target-free-blocks must be at least 1")
    if args.num_kvcache_blocks is not None and args.num_kvcache_blocks < 1:
        raise SystemExit("--num-kvcache-blocks must be at least 1")
    if args.max_num_active_seqs is not None and args.max_num_active_seqs < 1:
        raise SystemExit("--max-num-active-seqs must be at least 1")
    if args.kv_admission_lookahead < 0:
        raise SystemExit("--kv-admission-lookahead must be non-negative")
    if args.kv_compression_sink_blocks < 1:
        raise SystemExit("--kv-compression-sink-blocks must be at least 1")
    if args.kv_compression_recent_blocks < 1:
        raise SystemExit("--kv-compression-recent-blocks must be at least 1")
    if args.kv_compression_importance_blocks < 1:
        raise SystemExit("--kv-compression-importance-blocks must be at least 1")
    if args.kv_compression_query_tokens < 1:
        raise SystemExit("--kv-compression-query-tokens must be at least 1")
    if not 0 <= args.kv_compression_trigger_free_ratio <= 1:
        raise SystemExit(
            "--kv-compression-trigger-free-ratio must be between 0 and 1"
        )

    import uvicorn

    from nanovllm.serve.app import create_app

    if args.mock:
        from nanovllm.serve.mock_engine import MockEngine

        engine_factory = MockEngine
        served_model = args.served_model_name or "nano-vllm-ui-demo"
        backend_name = "mock backend"
    else:
        model_path = str(Path(args.model).resolve())
        served_model = args.served_model_name or Path(model_path).name

        def engine_factory():
            from nanovllm import LLM

            engine_kwargs = {
                "scheduling_policy": args.scheduling_policy,
                "metrics_summary_mode": args.metrics_summary_mode,
                "scheduler_trace_path": args.scheduler_trace_path,
                "prefix_cache_backend": args.prefix_cache_backend,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "max_num_active_seqs": args.max_num_active_seqs,
                "kv_admission_lookahead": args.kv_admission_lookahead,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "num_kvcache_blocks_override": args.num_kvcache_blocks,
                "enforce_eager": args.enforce_eager,
                "kv_reclaim_policy": args.kv_reclaim_policy,
                "kv_reclaim_min_keep_ratio": args.kv_reclaim_min_keep_ratio,
                "kv_reclaim_max_keep_ratio": args.kv_reclaim_max_keep_ratio,
                "kv_reclaim_budget_scale_ms": args.kv_reclaim_budget_scale_ms,
                "kv_reclaim_target_free_blocks": args.kv_reclaim_target_free_blocks,
                "kv_compression_policy": args.kv_compression_policy,
                "kv_compression_sink_blocks": args.kv_compression_sink_blocks,
                "kv_compression_recent_blocks": args.kv_compression_recent_blocks,
                "kv_compression_importance_blocks": (
                    args.kv_compression_importance_blocks
                ),
                "kv_compression_query_tokens": args.kv_compression_query_tokens,
                "kv_compression_trigger_free_ratio": (
                    args.kv_compression_trigger_free_ratio
                ),
                "remote_kv_bandwidth_gbps": args.remote_kv_bandwidth_gbps,
                "remote_kv_fixed_latency_ms": args.remote_kv_fixed_latency_ms,
                "remote_kv_cost_aware": not args.remote_kv_force_restore,
            }
            backend = None
            if args.kv_storage_backend == "mooncake":
                from nanovllm.engine.storage_backend import (
                    MooncakeKVStore,
                    MooncakeStoreConfig,
                )

                backend = MooncakeKVStore(
                    MooncakeStoreConfig(
                        local_hostname=args.mooncake_hostname,
                        metadata_server=args.mooncake_metadata,
                        global_segment_size=args.mooncake_global_segment_mib
                        * 1024
                        * 1024,
                        local_buffer_size=args.mooncake_local_buffer_mib
                        * 1024
                        * 1024,
                        protocol=args.mooncake_protocol,
                        rdma_devices=args.mooncake_rdma_devices,
                        master_server_addr=args.mooncake_master,
                    )
                )
                engine_kwargs["kv_storage_backend"] = backend
            try:
                return LLM(model_path, **engine_kwargs)
            except Exception:
                if backend is not None:
                    backend.close()
                raise

        backend_name = (
            "CUDA + Mooncake"
            if args.kv_storage_backend == "mooncake"
            else "CUDA"
        )

    app = create_app(
        engine_factory,
        served_model=served_model,
        backend_name=backend_name,
        api_key=args.api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
