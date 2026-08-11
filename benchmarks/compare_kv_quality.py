"""Compare live KV compression quality benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def summarize_run(run):
    cases = run["cases"]
    return {
        "policy": run["metadata"]["kv_compression_policy"],
        "accuracy": run["summary"]["accuracy"],
        "correct": run["summary"]["correct"],
        "cases": run["summary"]["cases"],
        "compression_drop_ratio": run["summary"]["compression_drop_ratio"],
        "ttft_ms_mean": mean(item["metrics"]["ttft_ms"] for item in cases),
        "e2e_ms_mean": mean(item["metrics"]["e2e_ms"] for item in cases),
    }


def compare_quality_runs(runs):
    if not runs:
        raise ValueError("at least one quality run is required")
    expected_cases = [item["case_id"] for item in runs[0]["cases"]]
    expected_max_tokens = runs[0]["metadata"].get("max_tokens")
    for run in runs[1:]:
        if [item["case_id"] for item in run["cases"]] != expected_cases:
            raise ValueError("quality runs do not contain the same cases")
        if run["metadata"].get("max_tokens") != expected_max_tokens:
            raise ValueError("quality runs use different max token budgets")
    return {
        "benchmark": "KV compression quality comparison",
        "case_ids": expected_cases,
        "policies": [summarize_run(run) for run in runs],
    }


def render_markdown(comparison):
    lines = [
        "# KV Compression Quality Comparison",
        "",
        "Live Qwen3-0.6B Needle-in-a-Haystack evaluation.",
        "",
        "| Policy | Accuracy | KV token drop | Mean TTFT | Mean E2E |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in comparison["policies"]:
        lines.append(
            f"| {item['policy']} | {item['correct']}/{item['cases']} "
            f"({item['accuracy']:.1%}) | {item['compression_drop_ratio']:.1%} | "
            f"{item['ttft_ms_mean']:.1f} ms | {item['e2e_ms_mean']:.1f} ms |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    comparison = compare_quality_runs(runs)
    markdown = render_markdown(comparison)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
