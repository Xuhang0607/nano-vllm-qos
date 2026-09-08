"""Isolated WSL/Linux validation suites with equal-budget KV controls."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from benchmarks.aggregate_serving_runs import aggregate_runs, compare_groups, render_markdown
from benchmarks.benchmark_kv_quality import build_quality_matrix, run_quality_benchmark
from benchmarks.compare_kv_quality import compare_quality_runs, render_markdown as quality_markdown
from benchmarks.benchmark_serving_gpu import (
    build_workload, print_report, request_json, run_live_benchmark,
    schedule_sustained, write_results,
)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_status(args, state, **details):
    save_json(args.output_dir / "status.json", {
        "state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **details,
    })


def validate_resume(previous, current):
    ignored = {"resume", "git_diff_stat", "git_commit"}
    mismatches = [key for key in set(previous) | set(current)
                  if key not in ignored and previous.get(key) != current.get(key)]
    if mismatches:
        raise ValueError("resume configuration/source mismatch: " + ", ".join(sorted(mismatches)))


def check_port(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


@contextmanager
def server(args, policy, label):
    # The engine still uses port 2333 for NCCL rendezvous. Never stop someone else's worker.
    check_port(args.port)
    check_port(2333)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({
        "PROJECT_DIR": str(root), "VENV_DIR": str(Path(sys.executable).parent.parent),
        "PORT": str(args.port), "MODEL_DIR": args.model_path,
        "ENABLE_MOONCAKE": "0", "SCHEDULING_POLICY": "pals",
        "PREFIX_CACHE_BACKEND": "radix", "KV_RECLAIM_POLICY": "recompute",
        "KV_COMPRESSION_POLICY": policy, "KV_COMPRESSION_SINK_BLOCKS": "1",
        "KV_COMPRESSION_RECENT_BLOCKS": "4" if policy == "sink_recent" else "2",
        "KV_COMPRESSION_IMPORTANCE_BLOCKS": "2", "KV_COMPRESSION_QUERY_TOKENS": "64",
        "KV_COMPRESSION_TRIGGER_FREE_RATIO": "1.0",
        "MAX_NUM_SEQS": str(args.max_num_seqs), "MAX_MODEL_LEN": "8192",
        "NUM_KVCACHE_BLOCKS": str(args.kv_blocks) if args.kv_blocks else "",
        "GPU_MEMORY_UTILIZATION": "0.80",
        "VALIDATION_SERVE_MODULE": "benchmarks.serve_device_context"
        if getattr(args, "suite", None) == "device" else "nanovllm.serve",
        "NANOVLLM_BENCH_DEVICE_MODE": getattr(args, "device_context_mode", ""),
        "METRICS_SUMMARY_MODE": getattr(args, "metrics_summary_mode", "cached"),
        "MAX_NUM_ACTIVE_SEQS": str(args.max_active_seqs) if getattr(args, "max_active_seqs", None) else "",
        "KV_ADMISSION_LOOKAHEAD": str(getattr(args, "kv_admission_lookahead", 0)),
        "SCHEDULER_TRACE_PATH": str(args.output_dir.resolve() / f"{label}.trace.jsonl")
        if getattr(args, "suite", None) == "trace" else "",
    })
    log_path = args.output_dir / f"{label}.server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(["bash", str(root / "scripts/run_nanovllm_wsl.sh")],
                                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited; inspect {log_path}")
                try:
                    metrics = request_json(f"{args.base_url}/v1/metrics", timeout_s=2)
                    if (metrics.get("worker_alive") and metrics.get("kv_compression_policy") == policy
                            and metrics.get("metrics_summary_mode") == env["METRICS_SUMMARY_MODE"]
                            and (getattr(args, "suite", None) != "device"
                                 or metrics.get("device_context_mode") == args.device_context_mode)
                            and metrics.get("kv_admission_lookahead", 0) == int(env["KV_ADMISSION_LOOKAHEAD"])):
                        break
                except (OSError, RuntimeError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"server startup timed out; inspect {log_path}")
            yield
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            # Clean up only descendants in the process group created by this suite.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def quality_suite(args):
    cases = build_quality_matrix(args.lengths, args.quality_repeats, args.seed)
    runs = []
    for policy in ("none", "sink_recent", "query_aware"):
        result_path = args.output_dir / f"quality-{policy}.json"
        if args.resume and result_path.exists():
            runs.append(json.loads(result_path.read_text(encoding="utf-8")))
            print(f"Resume quality: {policy}", flush=True)
            continue
        print(f"Quality: {policy}, {len(cases)} cases, retained budget=5 pages", flush=True)
        write_status(args, "running", phase="quality", policy=policy, cases=len(cases))
        with server(args, policy, f"quality-{policy}"):
            result = run_quality_benchmark(args.base_url, "Qwen3-0.6B", cases,
                                           max_tokens=64, timeout_s=args.timeout)
        save_json(result_path, result)
        runs.append(result)
    comparison = compare_quality_runs(runs, require_equal_budget=True)
    save_json(args.output_dir / "quality-comparison.json", comparison)
    markdown = quality_markdown(comparison)
    (args.output_dir / "quality-comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


def serving_suite(args):
    for rate in args.rates:
        count = max(args.requests, math.ceil(args.duration * rate) + 1)
        workload = build_workload(
            count // 2, count - count // 2, f"validation-{args.seed}",
            shared_prefix_repeats=8, batch_unique_repeats=64,
            interactive_unique_repeats=16,
            batch_output_tokens=args.batch_output_tokens,
            interactive_output_tokens=args.interactive_output_tokens,
        )
        workload = [replace(item, ttft_slo_ms=1000, tpot_slo_ms=100, e2e_slo_ms=10000)
                    if item.request_class == "interactive" else item for item in workload]
        workload = schedule_sustained(workload, rate, args.seed, args.arrival_process)
        groups = {"none": [], "query_aware": []}
        for repeat in range(args.repeats):
            policies = ("none", "query_aware") if repeat % 2 == 0 else ("query_aware", "none")
            for policy in policies:
                label = f"serving-{rate:g}-{policy}-{repeat + 1}"
                result_path = args.output_dir / f"{label}.json"
                if args.resume and result_path.exists():
                    groups[policy].append(json.loads(result_path.read_text(encoding="utf-8")))
                    print(f"Resume serving: {label}", flush=True)
                    continue
                print(f"Serving: {label}, {count} requests", flush=True)
                write_status(args, "running", phase="serving", policy=policy, rate=rate,
                             repeat=repeat + 1, total_repeats=args.repeats, requests=count)
                with server(args, policy, label):
                    result = run_live_benchmark(
                        args.base_url, "Qwen3-0.6B", workload, timeout_s=args.timeout,
                        warmup_prefix="Independent warmup context.",
                        workload_label=f"sustained-{rate:g}", gpu_sampling_interval_ms=200,
                        workload_parameters={"seed": args.seed, "arrival_rate_rps": rate,
                                             "arrival_process": args.arrival_process},
                        max_inflight=args.max_inflight,
                    )
                write_results(result, result_path)
                print_report(result)
                groups[policy].append(result)
        comparison = compare_groups(aggregate_runs(groups["none"], "No-Compression"),
                                    aggregate_runs(groups["query_aware"], "Query-Aware"))
        save_json(args.output_dir / f"serving-{rate:g}-comparison.json", comparison)
        markdown = render_markdown(comparison)
        (args.output_dir / f"serving-{rate:g}-comparison.md").write_text(markdown, encoding="utf-8")


def sustained_probe_workload(args, rate):
    count = max(args.requests, math.ceil(args.duration * rate) + 1)
    workload = build_workload(
        count // 2, count - count // 2, f"validation-{args.seed}",
        shared_prefix_repeats=8, batch_unique_repeats=64, interactive_unique_repeats=16,
        batch_output_tokens=args.batch_output_tokens,
        interactive_output_tokens=args.interactive_output_tokens,
    )
    workload = [replace(item, ttft_slo_ms=1000, tpot_slo_ms=100, e2e_slo_ms=10000)
                if item.request_class == "interactive" else item for item in workload]
    return count, schedule_sustained(workload, rate, args.seed, args.arrival_process)


def metrics_suite(args):
    # Hold compression and scheduling fixed; only change completed-summary reuse.
    for rate in args.rates:
        count, workload = sustained_probe_workload(args, rate)
        groups = {"full": [], "cached": []}
        for repeat in range(args.repeats):
            modes = ("full", "cached") if repeat % 2 == 0 else ("cached", "full")
            if args.suite == "trace":
                modes = (args.metrics_summary_mode,)
            for mode in modes:
                label = f"metrics-{rate:g}-{mode}-{repeat + 1}"
                path = args.output_dir / f"{label}.json"
                if args.resume and path.exists():
                    groups[mode].append(json.loads(path.read_text(encoding="utf-8")))
                    continue
                args.metrics_summary_mode = mode
                print(f"Metrics ablation: {label}, {count} requests", flush=True)
                write_status(args, "running", phase="metrics", mode=mode, rate=rate,
                             repeat=repeat + 1, total_repeats=args.repeats, requests=count)
                with server(args, "none", label):
                    result = run_live_benchmark(
                        args.base_url, "Qwen3-0.6B", workload, timeout_s=args.timeout,
                        warmup_prefix="Independent warmup context.",
                        workload_label=f"metrics-sustained-{rate:g}", gpu_sampling_interval_ms=200,
                        workload_parameters={"seed": args.seed, "arrival_rate_rps": rate,
                                             "arrival_process": args.arrival_process},
                        max_inflight=args.max_inflight,
                    )
                write_results(result, path)
                print_report(result)
                groups[mode].append(result)
        if args.suite == "trace":
            continue
        comparison = compare_groups(aggregate_runs(groups["full"], "Full summary"),
                                    aggregate_runs(groups["cached"], "Cached summary"),
                                    allow_metrics_mode_change=True)
        save_json(args.output_dir / f"metrics-{rate:g}-comparison.json", comparison)
        (args.output_dir / f"metrics-{rate:g}-comparison.md").write_text(
            render_markdown(comparison), encoding="utf-8")


def admission_suite(args):
    candidate = args.kv_admission_lookahead
    for rate in args.rates:
        count, workload = sustained_probe_workload(args, rate)
        groups = {0: [], candidate: []}
        for repeat in range(args.repeats):
            order = (0, candidate) if repeat % 2 == 0 else (candidate, 0)
            for horizon in order:
                label = f"admission-{rate:g}-h{horizon}-{repeat + 1}"
                path = args.output_dir / f"{label}.json"
                if args.resume and path.exists():
                    groups[horizon].append(json.loads(path.read_text(encoding="utf-8")))
                    continue
                args.kv_admission_lookahead = horizon
                print(f"Admission ablation: {label}, {count} requests", flush=True)
                write_status(args, "running", phase="admission", horizon=horizon,
                             rate=rate, repeat=repeat + 1, requests=count)
                with server(args, "none", label):
                    result = run_live_benchmark(
                        args.base_url, "Qwen3-0.6B", workload, timeout_s=args.timeout,
                        warmup_prefix="Independent warmup context.",
                        workload_label=f"admission-sustained-{rate:g}", gpu_sampling_interval_ms=200,
                        workload_parameters={"seed": args.seed, "arrival_rate_rps": rate,
                                             "arrival_process": args.arrival_process},
                        max_inflight=args.max_inflight,
                    )
                write_results(result, path)
                print_report(result)
                groups[horizon].append(result)
        comparison = compare_groups(aggregate_runs(groups[0], "No page reservation"),
                                    aggregate_runs(groups[candidate], f"Priority reserve {candidate}"),
                                    allow_admission_change=True)
        save_json(args.output_dir / f"admission-{rate:g}-comparison.json", comparison)
        (args.output_dir / f"admission-{rate:g}-comparison.md").write_text(
            render_markdown(comparison), encoding="utf-8")
    args.kv_admission_lookahead = candidate


def device_suite(args):
    for rate in args.rates:
        count, workload = sustained_probe_workload(args, rate)
        groups = {"legacy_cpu": [], "scoped": []}
        for repeat in range(args.repeats):
            modes = ("legacy_cpu", "scoped") if repeat % 2 == 0 else ("scoped", "legacy_cpu")
            for mode in modes:
                label = f"device-{rate:g}-{mode}-{repeat + 1}"
                path = args.output_dir / f"{label}.json"
                if args.resume and path.exists():
                    groups[mode].append(json.loads(path.read_text(encoding="utf-8")))
                    continue
                args.device_context_mode = mode
                write_status(args, "running", phase="device", mode=mode, rate=rate,
                             repeat=repeat + 1, requests=count)
                print(f"Device-context ablation: {label}, {count} requests", flush=True)
                with server(args, "none", label):
                    result = run_live_benchmark(
                        args.base_url, "Qwen3-0.6B", workload, timeout_s=args.timeout,
                        warmup_prefix="Independent warmup context.",
                        workload_label=f"device-sustained-{rate:g}", gpu_sampling_interval_ms=200,
                        workload_parameters={"seed": args.seed, "arrival_rate_rps": rate,
                                             "arrival_process": args.arrival_process},
                        max_inflight=args.max_inflight,
                    )
                write_results(result, path)
                print_report(result)
                groups[mode].append(result)
        comparison = compare_groups(aggregate_runs(groups["legacy_cpu"], "CPU device context"),
                                    aggregate_runs(groups["scoped"], "Scoped initialization"),
                                    allow_device_context_change=True)
        save_json(args.output_dir / f"device-{rate:g}-comparison.json", comparison)
        (args.output_dir / f"device-{rate:g}-comparison.md").write_text(
            render_markdown(comparison), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("quality", "serving", "all", "metrics", "trace", "admission", "device"))
    parser.add_argument("--model-path", default="/mnt/d/models/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed trials only with matching source/config")
    parser.add_argument("--metrics-summary-mode", choices=("cached", "full"), default="cached")
    parser.add_argument("--kv-blocks", type=int, default=64, help="0 uses normal automatic capacity")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-active-seqs", type=int)
    parser.add_argument("--kv-admission-lookahead", type=int, default=0)
    parser.add_argument("--quality-repeats", type=int, default=6)
    parser.add_argument("--lengths", type=int, nargs="+", default=[48, 96, 144])
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.5, 1, 2])
    parser.add_argument("--arrival-process", choices=("uniform", "poisson"), default="poisson")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-output-tokens", type=int, default=128)
    parser.add_argument("--interactive-output-tokens", type=int, default=64)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=240)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("run this suite inside WSL/Linux")
    if (args.requests < 2 or args.repeats < 1 or args.kv_blocks < 0 or args.max_num_seqs < 1
            or args.kv_admission_lookahead < 0
            or (args.suite == "admission" and args.kv_admission_lookahead == 0)
            or (args.max_active_seqs is not None and args.max_active_seqs < 1)
            or args.duration <= 0 or args.max_inflight < 1
            or any(not math.isfinite(rate) or rate <= 0 for rate in args.rates)):
        parser.error("invalid count, duration, capacity or arrival rate")
    args.base_url = f"http://127.0.0.1:{args.port}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()) and not args.resume:
        parser.error("output directory must be empty to preserve prior measurements")
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest["git_diff_stat"] = subprocess.check_output(["git", "-c", "core.autocrlf=false", "diff", "--stat"], text=True)
    root = Path(__file__).resolve().parents[1]
    manifest["source_sha256"] = {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for directory in ("nanovllm", "benchmarks")
        for path in sorted((root / directory).rglob("*.py"))
    }
    manifest["source_sha256"].update({
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "scripts").glob("*.sh"))
    })
    manifest_path = args.output_dir / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            parser.error("resume requires an existing manifest")
        validate_resume(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)
    else:
        save_json(manifest_path, manifest)
    try:
        if args.suite in ("quality", "all"):
            quality_suite(args)
        if args.suite in ("serving", "all"):
            serving_suite(args)
        if args.suite in ("metrics", "trace"):
            metrics_suite(args)
        if args.suite == "admission":
            admission_suite(args)
        if args.suite == "device":
            device_suite(args)
    except (Exception, KeyboardInterrupt) as exc:
        write_status(args, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    write_status(args, "complete")


if __name__ == "__main__":
    main()
