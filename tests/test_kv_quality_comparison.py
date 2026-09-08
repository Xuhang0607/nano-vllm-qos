import pytest

from benchmarks.compare_kv_quality import compare_quality_runs, render_markdown


def make_run(policy, correct, drop_ratio):
    return {
        "metadata": {"kv_compression_policy": policy, "max_tokens": 128},
        "summary": {
            "accuracy": correct / 2,
            "correct": correct,
            "cases": 2,
            "compression_drop_ratio": drop_ratio,
        },
        "cases": [
            {"case_id": "a", "metrics": {"ttft_ms": 10, "e2e_ms": 30}},
            {"case_id": "b", "metrics": {"ttft_ms": 20, "e2e_ms": 50}},
        ],
    }


def test_quality_comparison_summarizes_accuracy_compression_and_latency():
    comparison = compare_quality_runs(
        [make_run("none", 2, 0), make_run("query_aware", 2, 0.5)]
    )

    candidate = comparison["policies"][1]
    assert candidate["accuracy"] == 1
    assert candidate["compression_drop_ratio"] == 0.5
    assert candidate["ttft_ms_mean"] == 15
    assert candidate["e2e_ms_mean"] == 40
    assert "query_aware" in render_markdown(comparison)


def test_quality_comparison_rejects_different_case_sets():
    first = make_run("none", 2, 0)
    second = make_run("query_aware", 2, 0.5)
    second["cases"][1]["case_id"] = "different"

    with pytest.raises(ValueError, match="same cases"):
        compare_quality_runs([first, second])

    second = make_run("query_aware", 2, 0.5)
    second["metadata"]["max_tokens"] = 256
    with pytest.raises(ValueError, match="token budgets"):
        compare_quality_runs([first, second])


def test_strict_comparison_requires_same_budget_and_prompt_fingerprint():
    runs = [make_run("none", 2, 0), make_run("sink_recent", 1, 0.5),
            make_run("query_aware", 2, 0.5)]
    for run in runs:
        run["metadata"].update(case_fingerprint="same-prompts", retained_page_budget=5)
    assert compare_quality_runs(runs, True)["equal_retained_page_budget"]
    runs[-1]["metadata"]["retained_page_budget"] = 7
    with pytest.raises(ValueError, match="equal retained"):
        compare_quality_runs(runs, True)
    runs[-1]["metadata"]["retained_page_budget"] = 5
    runs[-1]["metadata"]["case_fingerprint"] = "changed-prompts"
    with pytest.raises(ValueError, match="case_fingerprint"):
        compare_quality_runs(runs, True)


def test_paired_quality_distinguishes_lost_and_gained_correct_answers():
    before, after = make_run("none", 1, 0), make_run("query_aware", 1, 0.5)
    for item, correct in zip(before["cases"], [True, False]):
        item["correct"] = correct
    for item, correct in zip(after["cases"], [False, True]):
        item["correct"] = correct
    paired = compare_quality_runs([before, after])["paired_vs_uncompressed"]["query_aware"]
    assert paired == {"baseline_correct": 1, "lost_correct": 1, "gained_correct": 1}
