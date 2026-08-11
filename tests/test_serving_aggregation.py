import pytest

from benchmarks.aggregate_serving_runs import (
    aggregate_runs,
    compare_groups,
    render_markdown,
    summarize_values,
)


def make_run(policy="pals", throughput=2.0, fingerprint="same"):
    latency = {
        name: {"p50": throughput, "p95": throughput * 2, "p99": throughput * 3}
        for name in ("ttft_ms", "tpot_ms", "e2e_ms")
    }
    return {
        "metadata": {
            "model": "Qwen3-0.6B",
            "backend": "CUDA",
            "hardware": "test-gpu",
            "policy": policy,
            "prefix_cache_backend": "radix",
            "max_num_seqs": 1,
            "max_model_len": 40960,
            "remote_kv_cost_aware": True,
            "workload_fingerprint": fingerprint,
            "warmup_enabled": True,
            "requests": 2,
        },
        "overall": {
            "request_throughput_rps": throughput,
            "output_throughput_tokens_per_s": throughput * 10,
            "slo_goodput_rps": throughput / 2,
        },
        "by_class": {
            "interactive": {
                "latency_ms": latency,
                "slo_attainment": 1.0,
            }
        },
        "metric_deltas": {
            "prefix_cache_block_hit_rate": 0.5,
            "remote_io_transfer_bytes": 0,
            "kv_restored_tokens": 0,
            "remote_restore_started": 0,
            "remote_restore_completed": 0,
            "remote_io_backend_get_bytes": 0,
            "remote_io_backend_put_bytes": 0,
        },
    }


def test_summarize_values_uses_student_t_interval():
    summary = summarize_values([1, 2, 3])

    assert summary["mean"] == 2
    assert summary["stdev"] == 1
    assert summary["ci95_half_width"] == pytest.approx(4.303 / 3**0.5)
    assert summary["min"] == 1
    assert summary["max"] == 3


def test_aggregate_runs_validates_configuration_and_aggregates_metrics():
    group = aggregate_runs([make_run(throughput=2), make_run(throughput=4)], "pals")

    assert group["runs"] == 2
    assert group["metrics"]["overall.request_throughput_rps"]["mean"] == 3
    assert group["metrics"]["interactive.e2e_ms_p95"]["mean"] == 6

    with pytest.raises(ValueError, match="workload_fingerprint"):
        aggregate_runs([make_run(), make_run(fingerprint="other")], "invalid")


def test_compare_groups_and_render_markdown():
    baseline = aggregate_runs([make_run(throughput=2)], "hash")
    candidate = aggregate_runs([make_run(throughput=3)], "radix")

    comparison = compare_groups(baseline, candidate)

    assert (
        comparison["metrics"]["overall.request_throughput_rps"]["change_percent"] == 50
    )
    markdown = render_markdown(comparison)
    assert "Baseline: **hash** (1 runs)" in markdown
    assert "+50.0%" in markdown
