import pytest

from benchmarks.compare_serving_runs import compare_runs, render_markdown


def make_run(policy, interactive_e2e_p95, goodput):
    summary = {
        "latency_ms": {
            "ttft_ms": {"p95": 100.0},
            "e2e_ms": {"p95": interactive_e2e_p95},
        },
        "slo_attainment": goodput,
    }
    return {
        "metadata": {
            "policy": policy,
            "prefix_cache_backend": "radix",
            "backend": "CUDA",
            "max_num_seqs": 1,
        },
        "overall": {
            "request_throughput_rps": 2.0,
            "output_throughput_tokens_per_s": 20.0,
            "slo_goodput_rps": goodput,
        },
        "by_class": {"interactive": summary},
    }


def test_compare_runs_reports_candidate_changes_and_markdown():
    baseline = make_run("fcfs", 1000.0, 0.5)
    candidate = make_run("pals", 100.0, 1.0)

    comparison = compare_runs(baseline, candidate)

    assert comparison["overall"]["slo_goodput_rps"]["change_percent"] == 100
    assert comparison["by_class"]["interactive"]["e2e_ms_p95"][
        "change_percent"
    ] == pytest.approx(-90)
    markdown = render_markdown(comparison)
    assert "| Baseline | fcfs | radix | CUDA | 1 |" in markdown
    assert "| Candidate | pals | radix | CUDA | 1 |" in markdown
    assert "-90.0%" in markdown
