"""Compare two live serving benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def percent_change(baseline: float, candidate: float):
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def compare_runs(baseline, candidate):
    classes = sorted(set(baseline["by_class"]) & set(candidate["by_class"]))
    class_metrics = {}
    for name in classes:
        before = baseline["by_class"][name]
        after = candidate["by_class"][name]
        class_metrics[name] = {
            "ttft_ms_p95": {
                "baseline": before["latency_ms"]["ttft_ms"]["p95"],
                "candidate": after["latency_ms"]["ttft_ms"]["p95"],
            },
            "e2e_ms_p95": {
                "baseline": before["latency_ms"]["e2e_ms"]["p95"],
                "candidate": after["latency_ms"]["e2e_ms"]["p95"],
            },
            "slo_attainment": {
                "baseline": before["slo_attainment"],
                "candidate": after["slo_attainment"],
            },
        }
        for values in class_metrics[name].values():
            values["change_percent"] = percent_change(
                values["baseline"], values["candidate"]
            )

    overall_metrics = {}
    for field in (
        "request_throughput_rps",
        "output_throughput_tokens_per_s",
        "slo_goodput_rps",
    ):
        before = baseline["overall"][field]
        after = candidate["overall"][field]
        overall_metrics[field] = {
            "baseline": before,
            "candidate": after,
            "change_percent": percent_change(before, after),
        }
    return {
        "baseline": baseline["metadata"],
        "candidate": candidate["metadata"],
        "overall": overall_metrics,
        "by_class": class_metrics,
    }


def _change(value):
    return "n/a" if value is None else f"{value:+.1f}%"


def render_markdown(comparison):
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    lines = [
        "# Live Serving Ablation",
        "",
        "| Configuration | Policy | Prefix cache | Backend | Max sequences |",
        "| --- | --- | --- | --- | ---: |",
        (
            f"| Baseline | {baseline.get('policy')} | "
            f"{baseline.get('prefix_cache_backend')} | {baseline.get('backend')} | "
            f"{baseline.get('max_num_seqs')} |"
        ),
        (
            f"| Candidate | {candidate.get('policy')} | "
            f"{candidate.get('prefix_cache_backend')} | {candidate.get('backend')} | "
            f"{candidate.get('max_num_seqs')} |"
        ),
        "",
        "| Scope | Metric | Baseline | Candidate | Change |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for field, values in comparison["overall"].items():
        lines.append(
            f"| overall | {field} | {values['baseline']:.3f} | "
            f"{values['candidate']:.3f} | {_change(values['change_percent'])} |"
        )
    for name, metrics in comparison["by_class"].items():
        for field, values in metrics.items():
            lines.append(
                f"| {name} | {field} | {values['baseline']:.3f} | "
                f"{values['candidate']:.3f} | {_change(values['change_percent'])} |"
            )
    lines.extend(
        [
            "",
            "> Positive latency changes are regressions; positive throughput, goodput,",
            "> and SLO-attainment changes are improvements.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparison = compare_runs(baseline, candidate)
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
