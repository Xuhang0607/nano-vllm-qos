"""Aggregate repeated live serving runs and compare two experiment groups."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
CONFIG_FIELDS = (
    "device_context_mode",
    "kv_admission_lookahead",
    "max_num_active_seqs",
    "scheduler_trace_enabled",
    "metrics_summary_mode",
    "client_max_inflight",
    "model",
    "backend",
    "hardware",
    "policy",
    "prefix_cache_backend",
    "kv_reclaim_policy",
    "kv_compression_policy",
    "kv_compression_sink_blocks",
    "kv_compression_recent_blocks",
    "kv_compression_importance_blocks",
    "kv_compression_query_tokens",
    "kv_compression_trigger_free_ratio",
    "num_kvcache_blocks",
    "num_kvcache_blocks_override",
    "max_num_seqs",
    "max_model_len",
    "remote_kv_cost_aware",
    "workload_fingerprint",
    "workload_label",
    "interactive_delay_ms",
    "interactive_arrival_rate_rps",
    "warmup_enabled",
    "gpu_sampling_interval_ms",
    "requests",
    "workload_parameters",
)
CROSS_GROUP_FIELDS = (
    "client_max_inflight",
    "model",
    "hardware",
    "max_num_seqs",
    "max_model_len",
    "num_kvcache_blocks",
    "num_kvcache_blocks_override",
    "workload_fingerprint",
    "workload_label",
    "interactive_delay_ms",
    "interactive_arrival_rate_rps",
    "warmup_enabled",
    "gpu_sampling_interval_ms",
    "requests",
    "workload_parameters",
)


def summarize_values(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot summarize an empty sample")
    sample_mean = mean(values)
    sample_stdev = stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
        half_width = critical * sample_stdev / math.sqrt(len(values))
    else:
        half_width = None
    return {
        "samples": len(values),
        "mean": sample_mean,
        "stdev": sample_stdev,
        "min": min(values),
        "max": max(values),
        "ci95_low": sample_mean - half_width if half_width is not None else None,
        "ci95_high": sample_mean + half_width if half_width is not None else None,
        "ci95_half_width": half_width,
    }


def _run_metrics(run):
    request_metrics = [item.get("metrics", {}) for item in run.get("requests", [])]
    metrics = {
        "observability.summary_calls": run["metric_deltas"].get("metrics_summary_calls", 0),
        "observability.summary_refreshes": run["metric_deltas"].get("metrics_summary_refreshes", 0),
        "observability.summary_compute_ms": run["metric_deltas"].get("metrics_summary_compute_ms", 0),
        "overall.failed_requests": run["overall"].get("failed_requests", 0),
        "overall.request_throughput_rps": run["overall"]["request_throughput_rps"],
        "overall.output_throughput_tokens_per_s": run["overall"][
            "output_throughput_tokens_per_s"
        ],
        "overall.slo_goodput_rps": run["overall"]["slo_goodput_rps"],
        "cache.prefix_cache_block_hit_rate": run["metric_deltas"][
            "prefix_cache_block_hit_rate"
        ],
        "remote.transfer_bytes": run["metric_deltas"]["remote_io_transfer_bytes"],
        "remote.kv_restored_tokens": run["metric_deltas"]["kv_restored_tokens"],
        "remote.restore_started": run["metric_deltas"]["remote_restore_started"],
        "remote.restore_completed": run["metric_deltas"][
            "remote_restore_completed"
        ],
        "remote.backend_get_bytes": run["metric_deltas"][
            "remote_io_backend_get_bytes"
        ],
        "remote.backend_put_bytes": run["metric_deltas"][
            "remote_io_backend_put_bytes"
        ],
        "kv.preemptions": sum(item.get("preemptions", 0) for item in request_metrics),
        "kv.reclaim_events": sum(
            item.get("kv_reclaim_events", 0) for item in request_metrics
        ),
        "kv.reclaimed_blocks": sum(
            item.get("kv_reclaimed_blocks", 0) for item in request_metrics
        ),
        "kv.retained_blocks": sum(
            item.get("kv_retained_blocks", 0) for item in request_metrics
        ),
        "kv.invalidated_tokens": sum(
            item.get("kv_invalidated_tokens", 0) for item in request_metrics
        ),
        "kv.recomputed_tokens": sum(
            item.get("kv_recomputed_tokens", 0) for item in request_metrics
        ),
        "kv.forced_fallbacks": run["metric_deltas"].get(
            "kv_reclaim_forced_fallbacks", 0
        ),
        "kv.compression_events": sum(
            item.get("kv_compression_events", 0) for item in request_metrics
        ),
        "kv.compression_dropped_blocks": sum(
            item.get("kv_compression_dropped_blocks", 0)
            for item in request_metrics
        ),
        "kv.compression_dropped_tokens": sum(
            item.get("kv_compression_dropped_tokens", 0)
            for item in request_metrics
        ),
    }
    for class_name, summary in run["by_class"].items():
        metrics[f"{class_name}.failed_requests"] = summary.get("failed_requests", 0)
        if summary.get("offered_slo_attainment") is not None:
            metrics[f"{class_name}.offered_slo_attainment"] = summary["offered_slo_attainment"]
        for field, values in summary.get("client_latency_ms", {}).items():
            if values["samples"]:
                metrics[f"{class_name}.{field}_p95"] = values["p95"]
        for latency in ("ttft_ms", "tpot_ms", "e2e_ms"):
            for percentile in ("p50", "p95", "p99"):
                metrics[f"{class_name}.{latency}_{percentile}"] = summary["latency_ms"][
                    latency
                ][percentile]
        if summary["slo_attainment"] is not None:
            metrics[f"{class_name}.slo_attainment"] = summary["slo_attainment"]
    gpu = run.get("gpu", {})
    if gpu.get("available"):
        for field in (
            "utilization_gpu_percent",
            "memory_used_mib",
            "power_watts",
        ):
            summary = gpu.get(field)
            if summary:
                for statistic in ("mean", "p95", "max"):
                    metrics[f"gpu.{field}.{statistic}"] = summary[statistic]
        if gpu.get("peak_memory_fraction") is not None:
            metrics["gpu.peak_memory_fraction"] = gpu["peak_memory_fraction"]
    return metrics


def _validate_group(runs):
    if not runs:
        raise ValueError("at least one run is required")
    expected = runs[0]["metadata"]
    for index, run in enumerate(runs[1:], start=2):
        actual = run["metadata"]
        mismatches = [
            field for field in CONFIG_FIELDS if expected.get(field) != actual.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"run {index} does not match group configuration: "
                + ", ".join(mismatches)
            )


def aggregate_runs(runs, name: str):
    _validate_group(runs)
    extracted = [_run_metrics(run) for run in runs]
    metric_names = set(extracted[0])
    for run_metrics in extracted[1:]:
        metric_names &= set(run_metrics)
    return {
        "name": name,
        "runs": len(runs),
        "evidence_warnings": sorted({warning for run in runs for warning in run.get("evidence_warnings", [])}),
        "configuration": {
            field: runs[0]["metadata"].get(field) for field in CONFIG_FIELDS
        },
        "metrics": {
            field: summarize_values([item[field] for item in extracted])
            for field in sorted(metric_names)
        },
    }


def compare_groups(baseline, candidate, *, allow_metrics_mode_change=False,
                   allow_admission_change=False, allow_device_context_change=False):
    mismatches = [
        field
        for field in CROSS_GROUP_FIELDS
        if baseline["configuration"].get(field)
        != candidate["configuration"].get(field)
    ]
    if (not allow_device_context_change
            and baseline["configuration"].get("device_context_mode")
            != candidate["configuration"].get("device_context_mode")):
        mismatches.append("device_context_mode")
    if (baseline["configuration"].get("scheduler_trace_enabled")
            != candidate["configuration"].get("scheduler_trace_enabled")):
        mismatches.append("scheduler_trace_enabled")
    if (not allow_admission_change
            and baseline["configuration"].get("kv_admission_lookahead")
            != candidate["configuration"].get("kv_admission_lookahead")):
        mismatches.append("kv_admission_lookahead")
    if (not allow_metrics_mode_change
            and baseline["configuration"].get("metrics_summary_mode")
            != candidate["configuration"].get("metrics_summary_mode")):
        mismatches.append("metrics_summary_mode")
    if mismatches:
        raise ValueError(
            "comparison groups use different workloads: " + ", ".join(mismatches)
        )
    common = sorted(set(baseline["metrics"]) & set(candidate["metrics"]))
    metrics = {}
    for field in common:
        before = baseline["metrics"][field]["mean"]
        after = candidate["metrics"][field]["mean"]
        metrics[field] = {
            "baseline_mean": before,
            "candidate_mean": after,
            "change_percent": (
                (after - before) / before * 100.0 if before != 0 else None
            ),
        }
    return {"baseline": baseline, "candidate": candidate, "metrics": metrics}


def _format_ci(summary):
    if summary["ci95_half_width"] is None:
        return f"{summary['mean']:.3f}"
    return f"{summary['mean']:.3f} +/- {summary['ci95_half_width']:.3f}"


def render_markdown(comparison):
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    lines = [
        "# Repeated Live Serving Ablation",
        "",
        f"Baseline: **{baseline['name']}** ({baseline['runs']} runs)",
        "",
        f"Candidate: **{candidate['name']}** ({candidate['runs']} runs)",
        "",
        "Values are mean +/- 95% Student-t confidence-interval half-width.",
        "",
        "| Metric | Baseline | Candidate | Mean change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field, values in comparison["metrics"].items():
        before = baseline["metrics"][field]
        after = candidate["metrics"][field]
        change = values["change_percent"]
        change_text = "n/a" if change is None else f"{change:+.1f}%"
        lines.append(
            f"| {field} | {_format_ci(before)} | {_format_ci(after)} | {change_text} |"
        )
    lines.append("")
    warnings = sorted(set(baseline.get("evidence_warnings", []))
                      | set(candidate.get("evidence_warnings", [])))
    if min(baseline["runs"], candidate["runs"]) < 3:
        warnings.append("Fewer than three repetitions: preliminary result only.")
    if warnings:
        lines.extend(["## Evidence Limits", ""] + [f"- {warning}" for warning in warnings] + [""])
    return "\n".join(lines)


def _load(paths):
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    baseline = aggregate_runs(_load(args.baseline), args.baseline_name)
    candidate = aggregate_runs(_load(args.candidate), args.candidate_name)
    comparison = compare_groups(baseline, candidate)
    markdown = render_markdown(comparison)
    print(markdown)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
