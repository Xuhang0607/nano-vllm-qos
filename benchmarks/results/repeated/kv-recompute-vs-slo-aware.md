# Repeated Live Serving Ablation

Baseline: **Recompute** (3 runs)

Candidate: **SLO-Aware-KV** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.e2e_ms_p50 | 9590.125 +/- 1441.029 | 10341.439 +/- 1367.771 | +7.8% |
| batch.e2e_ms_p95 | 16531.677 +/- 960.009 | 17852.989 +/- 1336.970 | +8.0% |
| batch.e2e_ms_p99 | 17506.635 +/- 881.571 | 18906.744 +/- 1480.490 | +8.0% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 35.657 +/- 5.229 | 38.691 +/- 4.821 | +8.5% |
| batch.tpot_ms_p95 | 63.202 +/- 5.914 | 67.536 +/- 8.627 | +6.9% |
| batch.tpot_ms_p99 | 67.022 +/- 6.647 | 71.521 +/- 9.990 | +6.7% |
| batch.ttft_ms_p50 | 654.602 +/- 872.496 | 679.703 +/- 1375.474 | +3.8% |
| batch.ttft_ms_p95 | 657.059 +/- 871.129 | 681.842 +/- 1376.181 | +3.8% |
| batch.ttft_ms_p99 | 657.304 +/- 870.959 | 682.036 +/- 1376.241 | +3.8% |
| cache.prefix_cache_block_hit_rate | 0.332 +/- 0.565 | 0.463 +/- 0.281 | +39.5% |
| gpu.memory_used_mib.max | 3334.667 +/- 50.202 | 3334.333 +/- 51.055 | -0.0% |
| gpu.memory_used_mib.mean | 3308.075 +/- 46.637 | 3309.978 +/- 84.594 | +0.1% |
| gpu.memory_used_mib.p95 | 3334.333 +/- 48.767 | 3333.117 +/- 50.874 | -0.0% |
| gpu.peak_memory_fraction | 0.407 +/- 0.006 | 0.407 +/- 0.006 | -0.0% |
| gpu.utilization_gpu_percent.max | 84.667 +/- 36.201 | 99.000 +/- 4.303 | +16.9% |
| gpu.utilization_gpu_percent.mean | 31.823 +/- 1.558 | 32.149 +/- 5.123 | +1.0% |
| gpu.utilization_gpu_percent.p95 | 42.067 +/- 5.645 | 40.933 +/- 1.293 | -2.7% |
| interactive.e2e_ms_p50 | 707.564 +/- 99.664 | 896.213 +/- 135.333 | +26.7% |
| interactive.e2e_ms_p95 | 1008.114 +/- 490.524 | 1166.024 +/- 731.573 | +15.7% |
| interactive.e2e_ms_p99 | 1041.215 +/- 565.628 | 1188.785 +/- 783.990 | +14.2% |
| interactive.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| interactive.tpot_ms_p50 | 35.336 +/- 5.093 | 38.113 +/- 11.697 | +7.9% |
| interactive.tpot_ms_p95 | 37.300 +/- 2.464 | 42.678 +/- 5.349 | +14.4% |
| interactive.tpot_ms_p99 | 37.560 +/- 2.521 | 43.250 +/- 4.352 | +15.1% |
| interactive.ttft_ms_p50 | 458.253 +/- 135.779 | 629.508 +/- 105.810 | +37.4% |
| interactive.ttft_ms_p95 | 769.139 +/- 447.133 | 898.768 +/- 654.418 | +16.9% |
| interactive.ttft_ms_p99 | 803.851 +/- 510.710 | 921.507 +/- 706.305 | +14.6% |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 1706.333 +/- 1834.512 | 1109.000 +/- 1467.323 | -35.0% |
| kv.preemptions | 1.333 +/- 1.434 | 1.333 +/- 1.434 | +0.0% |
| kv.reclaim_events | 1.333 +/- 1.434 | 1.333 +/- 1.434 | +0.0% |
| kv.reclaimed_blocks | 5.333 +/- 5.737 | 4.333 +/- 5.737 | -18.8% |
| kv.recomputed_tokens | 1194.333 +/- 732.944 | 1023.667 +/- 1100.134 | -14.3% |
| kv.retained_blocks | 0.000 +/- 0.000 | 2.333 +/- 1.434 | n/a |
| overall.output_throughput_tokens_per_s | 58.911 +/- 3.923 | 55.020 +/- 4.176 | -6.6% |
| overall.request_throughput_rps | 0.449 +/- 0.025 | 0.417 +/- 0.032 | -7.3% |
| overall.slo_goodput_rps | 0.449 +/- 0.025 | 0.417 +/- 0.032 | -7.3% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
