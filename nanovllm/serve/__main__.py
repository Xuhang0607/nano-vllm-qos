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
    parser.add_argument(
        "--prefix-cache-backend", choices=("hash", "radix"), default="radix"
    )
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
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
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.mock and not args.model:
        raise SystemExit("--model is required unless --mock is used")
    if args.mock and args.kv_storage_backend != "none":
        raise SystemExit("--kv-storage-backend is unavailable with --mock")
    if args.mooncake_global_segment_mib < 0 or args.mooncake_local_buffer_mib < 0:
        raise SystemExit("Mooncake memory sizes must be non-negative")

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
                "prefix_cache_backend": args.prefix_cache_backend,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "remote_kv_bandwidth_gbps": args.remote_kv_bandwidth_gbps,
                "remote_kv_fixed_latency_ms": args.remote_kv_fixed_latency_ms,
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
