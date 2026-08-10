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
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.mock and not args.model:
        raise SystemExit("--model is required unless --mock is used")

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

            return LLM(
                model_path,
                scheduling_policy=args.scheduling_policy,
                prefix_cache_backend=args.prefix_cache_backend,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
            )

        backend_name = "CUDA"

    app = create_app(
        engine_factory,
        served_model=served_model,
        backend_name=backend_name,
        api_key=args.api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
