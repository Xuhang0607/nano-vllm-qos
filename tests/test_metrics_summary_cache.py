from dataclasses import replace

import pytest

from benchmarks.benchmark_metrics_summary import build_history, make_scheduler, semantic_metrics
from benchmarks.benchmark_serving_gpu import metric_deltas
from nanovllm.engine.qos import summarize_metrics
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


@pytest.mark.parametrize("count", [0, 1, 23, 1000])
def test_cached_summary_matches_full_and_reuses_unchanged_history(count):
    history = build_history(count)
    cached = make_scheduler("cached", history)
    full = make_scheduler("full", history)
    for _ in range(4):
        assert semantic_metrics(cached.metrics()) == semantic_metrics(full.metrics())
    assert cached._summary_refreshes == 1
    assert full._summary_refreshes == 4


def test_summary_refreshes_after_completion_and_does_not_leak_mutations():
    scheduler = make_scheduler("cached", build_history(3))
    first = scheduler.metrics()
    first["ttft_ms_p95"] = -1
    assert scheduler.metrics()["ttft_ms_p95"] >= 0
    scheduler.completed_metrics.append(replace(build_history(1)[0], seq_id=4, ttft_ms=99999))
    result = scheduler.metrics()
    assert result["completed_requests"] == 4
    assert result["ttft_ms_p95"] == summarize_metrics(scheduler.completed_metrics)["ttft_ms_p95"]
    assert result["metrics_summary_refreshes"] == 2


def test_live_counters_refresh_without_recomputing_history():
    scheduler = make_scheduler("cached", build_history(3))
    scheduler.metrics()
    seq = Sequence([1, 2], SamplingParams(max_tokens=1))
    scheduler.add(seq)
    scheduler.kv_compression_events += 7
    result = scheduler.metrics()
    assert result["waiting_requests"] == 1
    assert result["kv_compression_events"] == 7
    assert result["metrics_summary_refreshes"] == 1
    scheduler.cancel(seq.seq_id)
    result = scheduler.metrics()
    assert result["waiting_requests"] == 0
    assert result["cancelled_requests"] == 1
    assert result["metrics_summary_refreshes"] == 1


def test_full_mode_can_be_selected_for_ablation():
    with pytest.raises(ValueError, match="metrics_summary_mode"):
        make_scheduler("invalid", [])


def test_summary_duration_delta_preserves_fractional_milliseconds():
    result = metric_deltas({"metrics_summary_compute_ms": 0.2},
                           {"metrics_summary_compute_ms": 0.35})
    assert result["metrics_summary_compute_ms"] == pytest.approx(0.15)
