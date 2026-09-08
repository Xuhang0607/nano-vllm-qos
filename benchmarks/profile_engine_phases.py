"""Instrumented offline diagnostic; timings are host wall time, not GPU kernel time."""

import argparse
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import json
from pathlib import Path
from time import perf_counter


@contextmanager
def measure_method(target, name, totals, label, range_factory=None):
    original = getattr(target, name)
    had_local = name in vars(target)

    @wraps(original)
    def measured(*args, **kwargs):
        start = perf_counter()
        try:
            with range_factory(label) if range_factory else nullcontext():
                return original(*args, **kwargs)
        finally:
            item = totals.setdefault(label, {"calls": 0, "wall_ms": 0.0})
            item["calls"] += 1
            item["wall_ms"] += (perf_counter() - start) * 1000

    setattr(target, name, measured)
    try:
        yield
    finally:
        if had_local:
            setattr(target, name, original)
        else:
            delattr(target, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/d/models/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(0, 32), default=0)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--cprofile", action="store_true",
                        help="Collect inclusive Python call timings; diagnostic overhead only")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    if args.warmup_rounds < 0:
        parser.error("warmup-rounds must be non-negative")

    from contextlib import ExitStack
    import cProfile
    import pstats
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.qos import RequestQoS

    engine = LLM(args.model, enforce_eager=True, max_model_len=2048,
                 max_num_batched_tokens=512, max_num_seqs=8,
                 num_kvcache_blocks_override=16, scheduling_policy="pals",
                 prefix_cache_backend="radix", kv_reclaim_policy="recompute",
                 kv_admission_lookahead=args.horizon)
    totals = {}
    outputs = []
    python_profiler = cProfile.Profile() if args.cprofile else None
    try:
        engine.generate([[99] * 32], SamplingParams(max_tokens=2, ignore_eos=True),
                        use_tqdm=False)
        for warmup in range(args.warmup_rounds):
            engine.generate(
                [[1000 + warmup * 12 + index] * (257, 513, 769)[index % 3]
                 for index in range(12)],
                SamplingParams(max_tokens=16, temperature=0.1, ignore_eos=True),
                request_qos=[RequestQoS(priority=10 if index % 2 else 0)
                             for index in range(12)], use_tqdm=False,
            )
        torch.manual_seed(20260907)
        for index in range(12):
            engine.add_request([100 + index] * (257, 513, 769)[index % 3],
                               SamplingParams(max_tokens=16, temperature=0.1,
                                              ignore_eos=True),
                               RequestQoS(priority=10 if index % 2 else 0))
        torch.cuda.synchronize()
        with ExitStack() as stack:
            for target, name, label in (
                (engine.scheduler, "schedule", "schedule"),
                (engine.model_runner, "call", "model_call"),
                (engine.scheduler, "postprocess", "postprocess"),
                (engine.model_runner, "prepare_prefill", "prepare_prefill"),
                (engine.model_runner, "prepare_decode", "prepare_decode"),
                (engine.model_runner, "prepare_sample", "prepare_sample"),
                (engine.model_runner, "run_model", "run_model"),
                (engine.model_runner.sampler, "forward", "sample"),
            ):
                stack.enter_context(measure_method(
                    target, name, totals, label,
                    torch.cuda.nvtx.range if args.nvtx else None,
                ))
            if args.cuda_profiler_range:
                torch.cuda.profiler.start()
            try:
                with torch.cuda.nvtx.range("measured_workload") if args.nvtx else nullcontext():
                    start = perf_counter()
                    if python_profiler:
                        python_profiler.enable()
                    try:
                        while not engine.is_finished():
                            completed, _ = engine.step()
                            outputs.extend(completed)
                        torch.cuda.synchronize()
                    finally:
                        if python_profiler:
                            python_profiler.disable()
                    elapsed_ms = (perf_counter() - start) * 1000
            finally:
                if args.cuda_profiler_range:
                    torch.cuda.profiler.stop()
        assert len(outputs) == 12
        assert all(len(tokens) == 16 for _, tokens in outputs)
        assert len(engine.scheduler.block_manager.free_block_ids) == 16
        root = Path(__file__).resolve().parents[1]
        hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((root / "nanovllm").rglob("*.py"))}
        report = {"scope": __doc__, "horizon": args.horizon,
                  "model": args.model, "gpu": torch.cuda.get_device_name(),
                  "kv_pages": 16, "requests": 12, "output_tokens": 192,
                  "arrival": "all submitted before measurement; not online SLO test",
                  "wall_ms": elapsed_ms, "phases": totals, "source_sha256": hashes,
                  "warmup_rounds": args.warmup_rounds, "nvtx": args.nvtx,
                  "cuda_profiler_range": args.cuda_profiler_range,
                  "timing_note": "inclusive nested host timings; do not sum child and parent",
                  "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "request_metrics": [engine.get_request_metrics(seq_id)
                                      for seq_id, _ in outputs]}
        report["cprofile"] = args.cprofile
        if python_profiler:
            report["python_top_cumulative"] = [
                {"file": key[0], "line": key[1], "function": key[2],
                 "primitive_calls": value[0], "calls": value[1],
                 "self_ms": value[2] * 1000, "cumulative_ms": value[3] * 1000}
                for key, value in sorted(pstats.Stats(python_profiler).stats.items(),
                                         key=lambda item: item[1][3], reverse=True)[:50]
            ]
    finally:
        engine.exit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("source_sha256", "request_metrics", "python_top_cumulative")}))


if __name__ == "__main__":
    main()
