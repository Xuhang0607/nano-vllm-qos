# Repeated Live Serving Ablation

Baseline: **FCFS** (3 runs)

Candidate: **PALS** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.e2e_ms_p50 | 6300.949 +/- 630.153 | 10291.071 +/- 506.503 | +63.3% |
| batch.e2e_ms_p95 | 9058.057 +/- 642.792 | 10338.846 +/- 515.791 | +14.1% |
| batch.e2e_ms_p99 | 9305.085 +/- 637.087 | 10343.004 +/- 517.050 | +11.2% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 98.445 +/- 9.706 | 128.083 +/- 4.608 | +30.1% |
| batch.tpot_ms_p95 | 141.442 +/- 10.000 | 157.402 +/- 7.834 | +11.3% |
| batch.tpot_ms_p99 | 145.296 +/- 9.929 | 160.874 +/- 8.053 | +10.7% |
| batch.ttft_ms_p50 | 98.900 +/- 23.714 | 2221.832 +/- 231.053 | +2146.5% |
| batch.ttft_ms_p95 | 147.206 +/- 29.264 | 3000.038 +/- 405.413 | +1938.0% |
| batch.ttft_ms_p99 | 151.450 +/- 29.420 | 3065.248 +/- 414.977 | +1923.9% |
| cache.prefix_cache_block_hit_rate | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| gpu.memory_used_mib.max | 6352.333 +/- 67.642 | 6362.333 +/- 84.516 | +0.2% |
| gpu.memory_used_mib.mean | 6343.107 +/- 40.969 | 6345.784 +/- 69.489 | +0.0% |
| gpu.memory_used_mib.p95 | 6350.667 +/- 60.667 | 6361.667 +/- 84.480 | +0.2% |
| gpu.peak_memory_fraction | 0.776 +/- 0.008 | 0.777 +/- 0.010 | +0.2% |
| gpu.utilization_gpu_percent.max | 48.333 +/- 23.083 | 49.000 +/- 18.756 | +1.4% |
| gpu.utilization_gpu_percent.mean | 34.597 +/- 2.244 | 34.025 +/- 3.252 | -1.7% |
| gpu.utilization_gpu_percent.p95 | 42.000 +/- 4.303 | 41.167 +/- 3.993 | -2.0% |
| interactive.e2e_ms_p50 | 7562.697 +/- 621.849 | 142.845 +/- 7.133 | -98.1% |
| interactive.e2e_ms_p95 | 8545.422 +/- 544.964 | 297.944 +/- 105.518 | -96.5% |
| interactive.e2e_ms_p99 | 8562.426 +/- 564.625 | 315.154 +/- 113.431 | -96.3% |
| interactive.slo_attainment | 0.000 +/- 0.000 | 1.000 +/- 0.000 | n/a |
| interactive.tpot_ms_p50 | 2508.252 +/- 204.933 | 32.262 +/- 2.313 | -98.7% |
| interactive.tpot_ms_p95 | 2832.326 +/- 164.148 | 35.980 +/- 4.955 | -98.7% |
| interactive.tpot_ms_p99 | 2836.190 +/- 164.422 | 36.684 +/- 6.992 | -98.7% |
| interactive.ttft_ms_p50 | 36.641 +/- 8.723 | 43.648 +/- 15.339 | +19.1% |
| interactive.ttft_ms_p95 | 52.818 +/- 54.732 | 196.088 +/- 91.530 | +271.3% |
| interactive.ttft_ms_p99 | 57.700 +/- 74.141 | 214.848 +/- 106.402 | +272.4% |
| overall.output_throughput_tokens_per_s | 28.204 +/- 1.649 | 27.806 +/- 1.379 | -1.4% |
| overall.request_throughput_rps | 1.175 +/- 0.069 | 1.159 +/- 0.057 | -1.4% |
| overall.slo_goodput_rps | 0.392 +/- 0.023 | 1.159 +/- 0.057 | +195.8% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
