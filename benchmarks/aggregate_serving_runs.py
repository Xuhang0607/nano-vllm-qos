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
    "model",
    "backend",
    "hardware",
    "policy",
    "prefix_cache_backend",
    "max_num_seqs",
    "max_model_len",
    "remote_kv_cost_aware",
    "workload_fingerprint",
    "warmup_enabled",
    "requests",
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
    metrics = {
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
    }
    for class_name, summary in run["by_class"].items():
        for latency in ("ttft_ms", "tpot_ms", "e2e_ms"):
            for percentile in ("p50", "p95", "p99"):
                metrics[f"{class_name}.{latency}_{percentile}"] = summary["latency_ms"][
                    latency
                ][percentile]
        if summary["slo_attainment"] is not None:
            metrics[f"{class_name}.slo_attainment"] = summary["slo_attainment"]
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
        "configuration": {
            field: runs[0]["metadata"].get(field) for field in CONFIG_FIELDS
        },
        "metrics": {
            field: summarize_values([item[field] for item in extracted])
            for field in sorted(metric_names)
        },
    }


def compare_groups(baseline, candidate):
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
