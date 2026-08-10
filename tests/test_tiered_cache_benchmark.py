from benchmarks.benchmark_tiered_cache import run_benchmark


def test_cost_aware_planner_beats_static_cache_strategies():
    results = run_benchmark(120)["strategies"]
    adaptive = results["cost_aware"]

    assert adaptive["estimated_total_ms"] <= results["local_recompute"][
        "estimated_total_ms"
    ]
    assert adaptive["estimated_total_ms"] <= results["always_restore"][
        "estimated_total_ms"
    ]
    assert 0 < adaptive["restore_requests"] < adaptive["requests"]
    assert adaptive["recomputed_tokens"] < results["local_recompute"][
        "recomputed_tokens"
    ]
