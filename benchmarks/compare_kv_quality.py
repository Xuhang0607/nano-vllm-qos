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


def compare_quality_runs(runs, require_equal_budget=False):
    if not runs:
        raise ValueError("at least one quality run is required")
    expected_cases = [item["case_id"] for item in runs[0]["cases"]]
    expected_max_tokens = runs[0]["metadata"].get("max_tokens")
    for run in runs[1:]:
        if [item["case_id"] for item in run["cases"]] != expected_cases:
            raise ValueError("quality runs do not contain the same cases")
        if run["metadata"].get("max_tokens") != expected_max_tokens:
            raise ValueError("quality runs use different max token budgets")
        for field in ("model", "case_fingerprint", "temperature", "kvcache_block_size",
                      "num_kvcache_blocks", "kv_compression_trigger_free_ratio"):
            if run["metadata"].get(field) != runs[0]["metadata"].get(field):
                raise ValueError(f"quality runs differ in {field}")
    compressed = [run for run in runs if run["metadata"]["kv_compression_policy"] != "none"]
    budgets = [run["metadata"].get("retained_page_budget") for run in compressed]
    equal_budget = bool(budgets) and None not in budgets and len(set(budgets)) == 1
    if require_equal_budget and (not equal_budget or not runs[0]["metadata"].get("case_fingerprint")):
        raise ValueError("equal retained page budgets and case fingerprints are required")
    baseline = next((run for run in runs if run["metadata"]["kv_compression_policy"] == "none"), None)
    paired = {}
    if baseline and all("correct" in item for run in runs for item in run["cases"]):
        for run in compressed:
            pairs = list(zip(baseline["cases"], run["cases"]))
            paired[run["metadata"]["kv_compression_policy"]] = {
                "baseline_correct": sum(item["correct"] for item in baseline["cases"]),
                "lost_correct": sum(a["correct"] and not b["correct"] for a, b in pairs),
                "gained_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
            }
    return {
        "benchmark": "KV compression quality comparison",
        "case_ids": expected_cases,
        "policies": [summarize_run(run) for run in runs],
        "equal_retained_page_budget": equal_budget,
        "retained_page_budget": budgets[0] if equal_budget else None,
        "by_task": {run["metadata"]["kv_compression_policy"]: run.get("by_task", {}) for run in runs},
        "paired_vs_uncompressed": paired,
    }


def render_markdown(comparison):
    lines = [
        "# KV Compression Quality Comparison",
        "",
        "Live synthetic retrieval evaluation; not a general model-quality benchmark.",
        "",
        f"Equal retained page budget: {comparison['equal_retained_page_budget']} "
        f"(pages: {comparison['retained_page_budget']}).",
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
    lines.extend(["", "## Task Breakdown", "", "| Policy | Task | Correct / Cases |",
                  "| --- | --- | ---: |"])
    for policy, tasks in comparison.get("by_task", {}).items():
        for task, summary in tasks.items():
            lines.append(f"| {policy} | {task} | {summary['correct']}/{summary['cases']} |")
    if comparison.get("paired_vs_uncompressed"):
        lines.extend(["", "## Paired Against Uncompressed", "",
                      "| Policy | Baseline correct | Lost correct | Gained correct |",
                      "| --- | ---: | ---: | ---: |"])
        for policy, counts in comparison["paired_vs_uncompressed"].items():
            lines.append(f"| {policy} | {counts['baseline_correct']} | "
                         f"{counts['lost_correct']} | {counts['gained_correct']} |")
        lines.extend(["", "Sampling is stochastic; paired changes are descriptive, not proof of causality."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--require-equal-budget", action="store_true")
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    comparison = compare_quality_runs(runs, args.require_equal_budget)
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
