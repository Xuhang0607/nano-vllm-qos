import csv
import json

import pytest

from benchmarks.benchmark_serving_gpu import (
    build_workload,
    describe_workload,
    metric_deltas,
    parse_nvidia_smi_sample,
    summarize_gpu_samples,
    summarize_records,
    workload_fingerprint,
    write_results,
)


def make_record(request_id, request_class, ttft_ms, slo_met):
    return {
        "request_id": request_id,
        "request_class": request_class,
        "priority": 10 if request_class == "interactive" else 0,
        "arrival_ms": 0,
        "success": True,
        "client_e2e_ms": ttft_ms + 10,
        "usage": {"prompt_tokens": 100, "completion_tokens": 4},
        "metrics": {
            "queue_ms": 1,
            "ttft_ms": ttft_ms,
            "tpot_ms": 2,
            "e2e_ms": ttft_ms + 10,
            "ttft_slo_met": slo_met,
            "tpot_slo_met": True,
            "e2e_slo_met": slo_met,
            "preemptions": 0,
            "kv_restored_tokens": 0,
        },
    }


def test_build_workload_has_fixed_classes_priorities_and_arrivals():
    workload = build_workload(2, 2, "test", shared_prefix_repeats=2)

    assert [item.request_id for item in workload] == [
        "batch-0",
        "batch-1",
        "interactive-0",
        "interactive-1",
    ]
    assert [item.priority for item in workload] == [0, 0, 10, 10]
    assert [item.arrival_ms for item in workload] == [0, 0, 100, 200]
    assert workload[0].messages[0][1] == workload[1].messages[0][1]
    assert workload_fingerprint(workload) == workload_fingerprint(
        build_workload(2, 2, "test", shared_prefix_repeats=2)
    )
    assert workload_fingerprint(workload) != workload_fingerprint(
        build_workload(2, 2, "different", shared_prefix_repeats=2)
    )
    description = describe_workload(workload, "medium")
    assert description["label"] == "medium"
    assert description["request_classes"] == {"batch": 2, "interactive": 2}
    assert description["interactive_delay_ms"] == 100
    assert description["interactive_arrival_rate_rps"] == 10


def test_build_workload_can_add_request_private_pressure_context():
    workload = build_workload(
        2,
        0,
        "pressure",
        shared_prefix_repeats=2,
        batch_unique_repeats=3,
    )

    assert workload[0].messages[0][1] == workload[1].messages[0][1]
    assert "private-0-0002" in workload[0].messages[1][1]
    assert "private-1-0002" in workload[1].messages[1][1]
    assert workload[0].messages[1][1] != workload[1].messages[1][1]

    interactive = build_workload(
        0,
        2,
        "pressure",
        shared_prefix_repeats=2,
        interactive_unique_repeats=3,
    )
    assert "interactive-0-0002" in interactive[0].messages[0][1]
    assert "interactive-1-0002" in interactive[1].messages[0][1]


def test_gpu_sample_parser_and_summary_handle_optional_power():
    first = parse_nvidia_smi_sample(
        "0, NVIDIA GeForce RTX 4060 Laptop GPU, 75, 7000, 8188, 88.5"
    )
    second = parse_nvidia_smi_sample(
        "0, NVIDIA GeForce RTX 4060 Laptop GPU, 95, 7200, 8188, [N/A]"
    )

    summary = summarize_gpu_samples([first, second])

    assert summary["available"] is True
    assert summary["sample_count"] == 2
    assert summary["utilization_gpu_percent"]["mean"] == 85
    assert summary["memory_used_mib"]["max"] == 7200
    assert summary["power_watts"]["mean"] == 88.5
    assert summary["peak_memory_fraction"] == pytest.approx(7200 / 8188)

    with pytest.raises(ValueError, match="column count"):
        parse_nvidia_smi_sample("0, incomplete")


def test_summary_reports_percentiles_throughput_and_slo_goodput():
    records = [
        make_record("a", "interactive", 10, True),
        make_record("b", "interactive", 30, False),
        {
            "request_id": "failed",
            "request_class": "interactive",
            "success": False,
            "usage": {},
            "metrics": {},
        },
    ]

    summary = summarize_records(records, duration_s=2)

    assert summary["completed_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["request_throughput_rps"] == 1
    assert summary["output_throughput_tokens_per_s"] == 4
    assert summary["slo_attainment"] == 0.5
    assert summary["slo_goodput_rps"] == 0.5
    assert summary["latency_ms"]["ttft_ms"]["p50"] == 20
    assert summary["latency_ms"]["ttft_ms"]["p95"] == pytest.approx(29)
    assert summary["latency_ms"]["ttft_ms"]["p99"] == pytest.approx(29.8)


def test_metric_deltas_isolate_cache_and_remote_io_for_one_run():
    before = {
        "prefix_cache_queried_blocks": 10,
        "prefix_cache_hit_blocks": 4,
        "remote_io_backend_get_bytes": 100,
        "remote_io_backend_put_bytes": 200,
        "kv_reclaim_events": 2,
        "kv_reclaim_freed_blocks": 3,
    }
    after = {
        "prefix_cache_queried_blocks": 30,
        "prefix_cache_hit_blocks": 14,
        "remote_io_backend_get_bytes": 500,
        "remote_io_backend_put_bytes": 800,
        "kv_reclaim_events": 7,
        "kv_reclaim_freed_blocks": 11,
    }

    delta = metric_deltas(before, after)

    assert delta["prefix_cache_queried_blocks"] == 20
    assert delta["prefix_cache_hit_blocks"] == 10
    assert delta["prefix_cache_block_hit_rate"] == 0.5
    assert delta["remote_io_backend_get_bytes"] == 400
    assert delta["remote_io_backend_put_bytes"] == 600
    assert delta["remote_io_transfer_bytes"] == 1000
    assert delta["kv_reclaim_events"] == 5
    assert delta["kv_reclaim_freed_blocks"] == 8


def test_write_results_creates_json_and_flat_request_csv(tmp_path):
    record = make_record("a", "interactive", 10, True)
    results = {"metadata": {}, "requests": [record]}
    json_path = tmp_path / "result.json"

    csv_path = write_results(results, json_path)

    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["requests"][0]["request_id"]
        == "a"
    )
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["request_id"] == "a"
    assert rows[0]["ttft_ms"] == "10"
