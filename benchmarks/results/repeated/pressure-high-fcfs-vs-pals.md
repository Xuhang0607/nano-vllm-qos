# Repeated Live Serving Ablation

Baseline: **FCFS** (3 runs)

Candidate: **PALS** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.e2e_ms_p50 | 6311.929 +/- 1555.698 | 10307.323 +/- 298.191 | +63.3% |
| batch.e2e_ms_p95 | 8992.560 +/- 1926.021 | 10350.060 +/- 299.968 | +15.1% |
| batch.e2e_ms_p99 | 9231.753 +/- 1953.196 | 10353.885 +/- 300.163 | +12.2% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 98.957 +/- 25.183 | 119.690 +/- 1.869 | +21.0% |
| batch.tpot_ms_p95 | 140.743 +/- 30.676 | 155.894 +/- 4.187 | +10.8% |
| batch.tpot_ms_p99 | 144.456 +/- 31.104 | 160.593 +/- 5.044 | +11.2% |
| batch.ttft_ms_p50 | 79.398 +/- 36.824 | 2766.882 +/- 415.776 | +3384.8% |
| batch.ttft_ms_p95 | 126.272 +/- 51.884 | 3309.704 +/- 376.709 | +2521.1% |
| batch.ttft_ms_p99 | 131.144 +/- 52.829 | 3358.159 +/- 373.273 | +2460.7% |
| cache.prefix_cache_block_hit_rate | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| gpu.memory_used_mib.max | 6430.000 +/- 62.108 | 6418.333 +/- 7.590 | -0.2% |
| gpu.memory_used_mib.mean | 6411.920 +/- 38.778 | 6403.056 +/- 10.044 | -0.1% |
| gpu.memory_used_mib.p95 | 6428.000 +/- 54.825 | 6418.150 +/- 7.012 | -0.2% |
| gpu.peak_memory_fraction | 0.785 +/- 0.008 | 0.784 +/- 0.001 | -0.2% |
| gpu.utilization_gpu_percent.max | 48.000 +/- 17.915 | 57.667 +/- 53.070 | +20.1% |
| gpu.utilization_gpu_percent.mean | 34.078 +/- 2.460 | 34.403 +/- 3.004 | +1.0% |
| gpu.utilization_gpu_percent.p95 | 42.533 +/- 2.008 | 40.667 +/- 1.434 | -4.4% |
| interactive.e2e_ms_p50 | 9528.785 +/- 2096.290 | 863.848 +/- 172.568 | -90.9% |
| interactive.e2e_ms_p95 | 9695.237 +/- 2133.082 | 1177.784 +/- 284.003 | -87.9% |
| interactive.e2e_ms_p99 | 9713.777 +/- 2128.495 | 1217.662 +/- 303.335 | -87.5% |
| interactive.slo_attainment | 0.000 +/- 0.000 | 1.000 +/- 0.000 | n/a |
| interactive.tpot_ms_p50 | 3139.841 +/- 669.849 | 33.400 +/- 9.517 | -98.9% |
| interactive.tpot_ms_p95 | 3207.104 +/- 664.307 | 38.791 +/- 9.456 | -98.8% |
| interactive.tpot_ms_p99 | 3214.322 +/- 660.611 | 39.772 +/- 9.250 | -98.8% |
| interactive.ttft_ms_p50 | 109.134 +/- 99.891 | 767.923 +/- 151.778 | +603.7% |
| interactive.ttft_ms_p95 | 146.581 +/- 79.752 | 1073.848 +/- 231.844 | +632.6% |
| interactive.ttft_ms_p99 | 150.621 +/- 83.258 | 1112.098 +/- 250.693 | +638.3% |
| overall.output_throughput_tokens_per_s | 28.498 +/- 5.753 | 27.713 +/- 0.734 | -2.8% |
| overall.request_throughput_rps | 1.187 +/- 0.240 | 1.155 +/- 0.031 | -2.8% |
| overall.slo_goodput_rps | 0.396 +/- 0.080 | 1.155 +/- 0.031 | +191.7% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
