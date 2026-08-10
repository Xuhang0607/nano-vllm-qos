from benchmarks.benchmark_qos_scheduler import build_workload, run_policy


def test_pals_improves_interactive_slo_under_mixed_load():
    workload = build_workload(batch_requests=12, interactive_requests=12)
    fcfs = run_policy("fcfs", workload)
    pals = run_policy("pals", workload)

    assert (
        pals["by_class"]["interactive"]["ttft_ms_p95"]
        < fcfs["by_class"]["interactive"]["ttft_ms_p95"]
    )
    assert pals["service_gain"]["ratio"] > fcfs["service_gain"]["ratio"]
    assert pals["by_class"]["interactive"]["ttft_slo_attainment"] == 1.0
    assert pals["by_class"]["interactive"]["e2e_slo_attainment"] == 1.0
    assert pals["by_class"]["batch"]["e2e_slo_attainment"] == 1.0
