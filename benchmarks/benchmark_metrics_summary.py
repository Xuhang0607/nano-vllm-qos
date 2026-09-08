"""CPU-only microbenchmark of unchanged-history Scheduler.metrics polling."""

import argparse
import cProfile
from dataclasses import replace
import json
from pathlib import Path
import platform
import random
from statistics import median
from time import perf_counter

from benchmarks.benchmark_qos_scheduler import make_config
from nanovllm.engine.qos import RequestMetrics
from nanovllm.engine.scheduler import Scheduler


def build_history(count, seed=20260906):
    rng = random.Random(seed)
    base = RequestMetrics(
        seq_id=0, request_class="interactive", priority=1, prompt_tokens=512,
        completion_tokens=64, queue_ms=0, ttft_ms=0, tpot_ms=None, e2e_ms=0,
        preemptions=0, kv_reclaim_events=0, kv_reclaimed_blocks=0, kv_retained_blocks=0,
        kv_invalidated_tokens=0, kv_recomputed_tokens=0, kv_compression_events=0,
        kv_compression_dropped_blocks=0, kv_compression_dropped_tokens=0,
        kv_restore_wait_ms=0, kv_restored_tokens=0, kv_restore_failures=0,
        ttft_slo_ms=1000, tpot_slo_ms=100, e2e_slo_ms=10000,
        ttft_slo_met=None, tpot_slo_met=None, e2e_slo_met=None,
    )
    return [replace(base, seq_id=index, queue_ms=rng.random() * 100,
                    ttft_ms=rng.random() * 1200, e2e_ms=rng.random() * 12000,
                    tpot_ms=None if index % 3 == 0 else rng.random() * 150,
                    ttft_slo_met=bool(index % 2), tpot_slo_met=None if index % 3 == 0 else True,
                    e2e_slo_met=bool(index % 5), preemptions=index % 4)
            for index in range(count)]


def make_scheduler(mode, history):
    config = make_config("pals", "radix")
    config.num_kvcache_blocks = 64
    config.metrics_summary_mode = mode
    scheduler = Scheduler(config)
    scheduler.completed_metrics.extend(history)
    return scheduler


def semantic_metrics(result):
    return {key: value for key, value in result.items() if not key.startswith("metrics_summary_")}


def benchmark(count, polls, batches):
    history = build_history(count)
    schedulers = {mode: make_scheduler(mode, history) for mode in ("full", "cached")}
    assert semantic_metrics(schedulers["full"].metrics()) == semantic_metrics(schedulers["cached"].metrics())
    samples = {mode: [] for mode in schedulers}
    for batch in range(batches):
        for mode in (("full", "cached") if batch % 2 == 0 else ("cached", "full")):
            started = perf_counter()
            for _ in range(polls):
                schedulers[mode].metrics()
            samples[mode].append((perf_counter() - started) * 1e6 / polls)
    return {"history": count, "polls_per_batch": polls, "batches": batches,
            "full_us": median(samples["full"]), "cached_us": median(samples["cached"]),
            "samples_us": samples, "semantic_equality": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--polls", type=int, default=100)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.histories) < 0 or min(args.polls, args.batches) < 1:
        parser.error("invalid history or repetition count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [benchmark(count, args.polls, args.batches) for count in args.histories]
    result = {"gpu_measurement": False, "python": platform.python_version(),
              "scope": "Unchanged synthetic histories; exact real Scheduler.metrics CPU wall time",
              "rows": rows}
    for mode in ("full", "cached"):
        scheduler = make_scheduler(mode, build_history(max(args.histories)))
        scheduler.metrics()
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.polls):
            scheduler.metrics()
        profiler.disable()
        profiler.dump_stats(str(args.output_dir / f"{mode}.prof"))
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
