import pytest

from benchmarks.analyze_load_progress import analyze_run


def record(arrival, duration, success=True):
    return {"arrival_ms": arrival, "scheduled_e2e_ms": duration, "success": success,
            "metrics": {"queue_ms": 10} if success else None}


def test_counts_last_boundary_once_and_keeps_failed_requests():
    run = {"metadata": {}, "requests": [record(0, 1000), record(60000, 90000, False),
                                          record(120000, 1000)]}
    report = analyze_run(run)
    assert [item["cohort_requests"] for item in report["windows"]] == [1, 2]
    assert [item["client_outstanding"] for item in report["windows"]] == [1, 2]
    assert report["failures"] == 1
    assert report["drain_s"] == 30


def test_missing_terminal_time_is_not_zero_backlog():
    report = analyze_run({"metadata": {}, "requests": [record(0, None, False), record(1000, 0)]})
    assert report["windows"][0]["client_outstanding"] is None
    assert report["drain_s"] is None


@pytest.mark.parametrize("window", [0, -1, float("nan")])
def test_invalid_window(window):
    with pytest.raises(ValueError):
        analyze_run({}, window)
