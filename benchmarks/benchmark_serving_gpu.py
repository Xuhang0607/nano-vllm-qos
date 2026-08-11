"""Live serving benchmark for the nano-vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

LATENCY_FIELDS = ("queue_ms", "ttft_ms", "tpot_ms", "e2e_ms")
SLO_FIELDS = ("ttft_slo_met", "tpot_slo_met", "e2e_slo_met")
COUNTER_FIELDS = (
    "prefix_cache_lookups",
    "prefix_cache_queried_blocks",
    "prefix_cache_hit_blocks",
    "prefix_cache_evictions",
    "remote_restore_started",
    "remote_restore_completed",
    "remote_restore_failed",
    "kv_restored_tokens",
    "remote_io_backend_gets",
    "remote_io_backend_get_bytes",
    "remote_io_backend_puts",
    "remote_io_backend_put_bytes",
    "remote_io_failures",
)


@dataclass(slots=True, frozen=True)
class RequestSpec:
    request_id: str
    request_class: str
    priority: int
    arrival_ms: float
    messages: tuple[tuple[str, str], ...]
    max_tokens: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    e2e_slo_ms: float

    def payload(self, model: str):
        return {
            "model": model,
            "messages": [
                {"role": role, "content": content} for role, content in self.messages
            ],
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "stream": False,
            "priority": self.priority,
            "request_class": self.request_class,
            "ttft_slo_ms": self.ttft_slo_ms,
            "tpot_slo_ms": self.tpot_slo_ms,
            "e2e_slo_ms": self.e2e_slo_ms,
        }


def percentile(values, ratio: float):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(records, field: str):
    values = [
        record["metrics"][field]
        for record in records
        if record.get("success") and record.get("metrics", {}).get(field) is not None
    ]
    return {
        "p50": median(values) if values else 0.0,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def _request_slo_met(record):
    checks = [
        record.get("metrics", {}).get(field)
        for field in SLO_FIELDS
        if record.get("metrics", {}).get(field) is not None
    ]
    return all(checks) if checks else None


def summarize_records(records, duration_s: float):
    completed = [record for record in records if record.get("success")]
    evaluated = [record for record in completed if _request_slo_met(record) is not None]
    slo_met = sum(_request_slo_met(record) is True for record in evaluated)
    prompt_tokens = sum(
        record.get("usage", {}).get("prompt_tokens", 0) for record in completed
    )
    completion_tokens = sum(
        record.get("usage", {}).get("completion_tokens", 0) for record in completed
    )
    duration_s = max(duration_s, 1e-9)
    return {
        "requests": len(records),
        "completed_requests": len(completed),
        "failed_requests": len(records) - len(completed),
        "duration_s": duration_s,
        "request_throughput_rps": len(completed) / duration_s,
        "output_throughput_tokens_per_s": completion_tokens / duration_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "slo_evaluated_requests": len(evaluated),
        "slo_met_requests": slo_met,
        "slo_attainment": slo_met / len(evaluated) if evaluated else None,
        "slo_goodput_rps": slo_met / duration_s,
        "latency_ms": {
            field: _latency_summary(completed, field) for field in LATENCY_FIELDS
        },
    }


def metric_deltas(before, after):
    deltas = {
        field: max(0, int(after.get(field, 0)) - int(before.get(field, 0)))
        for field in COUNTER_FIELDS
    }
    queried = deltas["prefix_cache_queried_blocks"]
    deltas["prefix_cache_block_hit_rate"] = (
        deltas["prefix_cache_hit_blocks"] / queried if queried else 0.0
    )
    deltas["remote_io_transfer_bytes"] = (
        deltas["remote_io_backend_get_bytes"] + deltas["remote_io_backend_put_bytes"]
    )
    return deltas


def build_shared_prefix(workload_id: str, repeats: int):
    sentence = (
        f"[{workload_id}] nano-vLLM 性能实验固定上下文，"
        "所有请求必须基于相同系统信息独立回答。"
    )
    return "\n".join(f"{index:03d}: {sentence}" for index in range(repeats))


def build_workload(
    batch_requests: int,
    interactive_requests: int,
    workload_id: str,
    shared_prefix_repeats: int = 32,
    interactive_delay_ms: float = 100.0,
    batch_output_tokens: int = 16,
    interactive_output_tokens: int = 8,
):
    shared_prefix = build_shared_prefix(workload_id, shared_prefix_repeats)
    workload = []
    for index in range(batch_requests):
        workload.append(
            RequestSpec(
                request_id=f"batch-{index}",
                request_class="batch",
                priority=0,
                arrival_ms=0.0,
                messages=(
                    ("system", shared_prefix),
                    ("user", f"离线任务 {index}：用一句话概括以上上下文。"),
                ),
                max_tokens=batch_output_tokens,
                ttft_slo_ms=10000.0,
                tpot_slo_ms=500.0,
                e2e_slo_ms=30000.0,
            )
        )
    for index in range(interactive_requests):
        workload.append(
            RequestSpec(
                request_id=f"interactive-{index}",
                request_class="interactive",
                priority=10,
                arrival_ms=interactive_delay_ms * (index + 1),
                messages=(("user", f"交互请求 {index}：只回答数字 {index}。"),),
                max_tokens=interactive_output_tokens,
                ttft_slo_ms=2000.0,
                tpot_slo_ms=300.0,
                e2e_slo_ms=5000.0,
            )
        )
    return sorted(workload, key=lambda item: (item.arrival_ms, item.request_id))


def request_json(url, payload=None, api_key=None, timeout_s=120.0):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", {}).get("message", body)
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def wait_for_idle(base_url, api_key=None, timeout_s=120.0):
    deadline = time.perf_counter() + timeout_s
    stable_polls = 0
    pending_fields = (
        "serving_active_requests",
        "serving_queued_requests",
        "waiting_requests",
        "running_requests",
        "remote_restore_pending",
        "remote_io_writeback_pending",
        "remote_io_catalog_save_pending",
    )
    while time.perf_counter() < deadline:
        metrics = request_json(
            f"{base_url}/v1/metrics", api_key=api_key, timeout_s=timeout_s
        )
        if all(int(metrics.get(field, 0)) == 0 for field in pending_fields):
            stable_polls += 1
            if stable_polls >= 2:
                return metrics
        else:
            stable_polls = 0
        time.sleep(0.1)
    raise TimeoutError("server did not become idle before the benchmark timeout")


def _run_request(base_url, model, spec, start_time, api_key, timeout_s):
    target = start_time + spec.arrival_ms / 1000.0
    time.sleep(max(0.0, target - time.perf_counter()))
    request_started = time.perf_counter()
    try:
        response = request_json(
            f"{base_url}/v1/chat/completions",
            spec.payload(model),
            api_key,
            timeout_s,
        )
        return {
            "request_id": spec.request_id,
            "request_class": spec.request_class,
            "priority": spec.priority,
            "arrival_ms": spec.arrival_ms,
            "success": True,
            "client_e2e_ms": (time.perf_counter() - request_started) * 1000.0,
            "usage": response.get("usage", {}),
            "metrics": response.get("x_nanovllm_metrics", {}),
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
        }
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        return {
            "request_id": spec.request_id,
            "request_class": spec.request_class,
            "priority": spec.priority,
            "arrival_ms": spec.arrival_ms,
            "success": False,
            "client_e2e_ms": (time.perf_counter() - request_started) * 1000.0,
            "error": f"{type(exc).__name__}: {exc}",
            "usage": {},
            "metrics": {},
        }


def run_live_benchmark(
    base_url: str,
    model: str,
    workload,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    warmup_prefix: str | None = None,
    hardware: str | None = None,
):
    base_url = base_url.rstrip("/")
    health = request_json(f"{base_url}/health", api_key=api_key, timeout_s=timeout_s)
    if health.get("status") != "ok":
        raise RuntimeError(f"server is not healthy: {health}")

    if warmup_prefix:
        warmup = RequestSpec(
            request_id="warmup",
            request_class="warmup",
            priority=0,
            arrival_ms=0,
            messages=(("system", warmup_prefix), ("user", "预热，只回答 1。")),
            max_tokens=1,
            ttft_slo_ms=30000,
            tpot_slo_ms=1000,
            e2e_slo_ms=30000,
        )
        warmup_record = _run_request(
            base_url, model, warmup, time.perf_counter(), api_key, timeout_s
        )
        if not warmup_record["success"]:
            raise RuntimeError(f"warmup failed: {warmup_record['error']}")

    metrics_before = wait_for_idle(base_url, api_key, timeout_s)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(workload))) as executor:
        futures = [
            executor.submit(
                _run_request,
                base_url,
                model,
                spec,
                started,
                api_key,
                timeout_s,
            )
            for spec in workload
        ]
        records = [future.result() for future in as_completed(futures)]
    duration_s = time.perf_counter() - started
    records.sort(key=lambda item: item["request_id"])
    metrics_after = wait_for_idle(base_url, api_key, timeout_s)
    classes = sorted({record["request_class"] for record in records})
    return {
        "metadata": {
            "benchmark": "live nano-vLLM serving benchmark",
            "gpu_measurement": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": base_url,
            "model": model,
            "backend": health.get("backend"),
            "hardware": hardware,
            "policy": metrics_before.get("policy"),
            "prefix_cache_backend": metrics_before.get("prefix_cache_backend"),
            "max_model_len": metrics_before.get("max_model_len"),
            "requested_max_model_len": metrics_before.get("requested_max_model_len"),
            "max_num_seqs": metrics_before.get("max_num_seqs"),
            "max_num_batched_tokens": metrics_before.get("max_num_batched_tokens"),
            "kvcache_block_size": metrics_before.get("kvcache_block_size"),
            "num_kvcache_blocks": metrics_before.get("num_kvcache_blocks"),
            "kv_cache_capacity_tokens": metrics_before.get("kv_cache_capacity_tokens"),
            "requests": len(workload),
        },
        "overall": summarize_records(records, duration_s),
        "by_class": {
            name: summarize_records(
                [record for record in records if record["request_class"] == name],
                duration_s,
            )
            for name in classes
        },
        "metric_deltas": metric_deltas(metrics_before, metrics_after),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "requests": records,
    }


def write_results(results, output_json: Path, output_csv: Path | None = None):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_csv = output_csv or output_json.with_suffix(".requests.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id",
        "request_class",
        "priority",
        "arrival_ms",
        "success",
        "client_e2e_ms",
        "prompt_tokens",
        "completion_tokens",
        *LATENCY_FIELDS,
        *SLO_FIELDS,
        "preemptions",
        "kv_restored_tokens",
        "error",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in results["requests"]:
            metrics = record.get("metrics", {})
            usage = record.get("usage", {})
            writer.writerow(
                {
                    **{field: record.get(field) for field in fieldnames},
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    **{field: metrics.get(field) for field in LATENCY_FIELDS},
                    **{field: metrics.get(field) for field in SLO_FIELDS},
                    "preemptions": metrics.get("preemptions"),
                    "kv_restored_tokens": metrics.get("kv_restored_tokens"),
                }
            )
    return output_csv


def print_report(results):
    metadata = results["metadata"]
    print("nano-vLLM live serving benchmark")
    print(
        f"backend={metadata['backend']} model={metadata['model']} "
        f"policy={metadata['policy']} prefix={metadata['prefix_cache_backend']}"
    )
    print(
        f"{'Class':<14} {'Req':>4} {'TTFT p50':>10} {'TTFT p95':>10} "
        f"{'TTFT p99':>10} {'E2E p95':>10} {'SLO':>8}"
    )
    for name, summary in results["by_class"].items():
        ttft = summary["latency_ms"]["ttft_ms"]
        e2e = summary["latency_ms"]["e2e_ms"]
        attainment = summary["slo_attainment"]
        print(
            f"{name:<14} {summary['completed_requests']:>4} "
            f"{ttft['p50']:>10.2f} {ttft['p95']:>10.2f} {ttft['p99']:>10.2f} "
            f"{e2e['p95']:>10.2f} "
            f"{('-' if attainment is None else f'{attainment:.1%}'):>8}"
        )
    overall = results["overall"]
    cache = results["metric_deltas"]
    print(f"throughput: {overall['request_throughput_rps']:.2f} req/s")
    print(f"SLO goodput: {overall['slo_goodput_rps']:.2f} req/s")
    print(
        "prefix cache: "
        f"{cache['prefix_cache_hit_blocks']}/{cache['prefix_cache_queried_blocks']} "
        f"blocks ({cache['prefix_cache_block_hit_rate']:.1%})"
    )
    print(
        "Mooncake transfer: "
        f"{cache['remote_io_transfer_bytes'] / 1024**2:.2f} MiB, "
        f"failures={cache['remote_io_failures']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--api-key")
    parser.add_argument("--hardware")
    parser.add_argument("--workload-id", default="gpu-ablation-v1")
    parser.add_argument("--batch-requests", type=int, default=4)
    parser.add_argument("--interactive-requests", type=int, default=4)
    parser.add_argument("--shared-prefix-repeats", type=int, default=32)
    parser.add_argument("--interactive-delay-ms", type=float, default=100.0)
    parser.add_argument("--batch-output-tokens", type=int, default=16)
    parser.add_argument("--interactive-output-tokens", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.batch_requests < 0 or args.interactive_requests < 0:
        parser.error("request counts must not be negative")
    if args.batch_requests + args.interactive_requests == 0:
        parser.error("at least one request is required")
    if args.shared_prefix_repeats <= 0:
        parser.error("shared-prefix-repeats must be positive")

    workload = build_workload(
        args.batch_requests,
        args.interactive_requests,
        args.workload_id,
        args.shared_prefix_repeats,
        args.interactive_delay_ms,
        args.batch_output_tokens,
        args.interactive_output_tokens,
    )
    shared_prefix = build_shared_prefix(args.workload_id, args.shared_prefix_repeats)
    results = run_live_benchmark(
        args.base_url,
        args.model,
        workload,
        args.api_key,
        args.timeout_s,
        None if args.no_warmup else shared_prefix,
        args.hardware,
    )
    print_report(results)
    output_csv = write_results(results, args.output_json, args.output_csv)
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {output_csv}")


if __name__ == "__main__":
    main()
