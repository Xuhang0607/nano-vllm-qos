# Repeated Live Serving Ablation

Baseline: **CPU device context** (3 runs)

Candidate: **Scoped initialization** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 6833.800 +/- 1500.307 | 5440.613 +/- 685.346 | -20.4% |
| batch.dispatch_lag_ms_p95 | 1.365 +/- 0.253 | 1.398 +/- 0.216 | +2.5% |
| batch.e2e_ms_p50 | 5723.939 +/- 514.310 | 4684.856 +/- 119.296 | -18.2% |
| batch.e2e_ms_p95 | 6779.137 +/- 1405.136 | 5408.181 +/- 678.481 | -20.2% |
| batch.e2e_ms_p99 | 7080.872 +/- 1486.440 | 5628.186 +/- 1176.686 | -20.5% |
| batch.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| batch.offered_slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.scheduled_e2e_ms_p95 | 6834.935 +/- 1500.488 | 5441.670 +/- 685.765 | -20.4% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 37.816 +/- 3.936 | 31.778 +/- 2.442 | -16.0% |
| batch.tpot_ms_p95 | 42.936 +/- 9.360 | 34.549 +/- 3.976 | -19.5% |
| batch.tpot_ms_p99 | 45.121 +/- 12.598 | 35.647 +/- 7.136 | -21.0% |
| batch.ttft_ms_p50 | 1065.140 +/- 7.180 | 808.726 +/- 111.540 | -24.1% |
| batch.ttft_ms_p95 | 1624.588 +/- 258.644 | 1195.759 +/- 209.765 | -26.4% |
| batch.ttft_ms_p99 | 1794.304 +/- 576.939 | 1316.103 +/- 327.916 | -26.7% |
| cache.prefix_cache_block_hit_rate | 0.167 +/- 0.000 | 0.167 +/- 0.000 | +0.0% |
| gpu.memory_used_mib.max | 5028.333 +/- 494.392 | 4963.333 +/- 386.315 | -1.3% |
| gpu.memory_used_mib.mean | 4910.165 +/- 401.680 | 4853.722 +/- 177.489 | -1.1% |
| gpu.memory_used_mib.p95 | 4956.617 +/- 449.509 | 4926.933 +/- 319.370 | -0.6% |
| gpu.peak_memory_fraction | 0.614 +/- 0.060 | 0.606 +/- 0.047 | -1.3% |
| gpu.utilization_gpu_percent.max | 96.000 +/- 17.212 | 96.000 +/- 9.937 | +0.0% |
| gpu.utilization_gpu_percent.mean | 30.574 +/- 1.417 | 32.270 +/- 0.846 | +5.5% |
| gpu.utilization_gpu_percent.p95 | 50.667 +/- 2.869 | 50.000 +/- 0.000 | -1.3% |
| interactive.client_e2e_ms_p95 | 2762.972 +/- 699.824 | 2263.846 +/- 256.910 | -18.1% |
| interactive.dispatch_lag_ms_p95 | 1.330 +/- 0.050 | 1.568 +/- 0.995 | +17.9% |
| interactive.e2e_ms_p50 | 2379.506 +/- 215.048 | 1956.290 +/- 115.744 | -17.8% |
| interactive.e2e_ms_p95 | 2724.321 +/- 681.242 | 2225.069 +/- 258.628 | -18.3% |
| interactive.e2e_ms_p99 | 2822.341 +/- 800.783 | 2327.805 +/- 392.648 | -17.5% |
| interactive.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| interactive.offered_slo_attainment | 0.998 +/- 0.009 | 1.000 +/- 0.000 | +0.2% |
| interactive.scheduled_e2e_ms_p95 | 2764.179 +/- 699.872 | 2265.020 +/- 257.120 | -18.1% |
| interactive.slo_attainment | 0.998 +/- 0.009 | 1.000 +/- 0.000 | +0.2% |
| interactive.tpot_ms_p50 | 36.923 +/- 3.290 | 30.470 +/- 1.918 | -17.5% |
| interactive.tpot_ms_p95 | 42.363 +/- 10.856 | 34.624 +/- 4.267 | -18.3% |
| interactive.tpot_ms_p99 | 43.743 +/- 12.089 | 36.104 +/- 5.589 | -17.5% |
| interactive.ttft_ms_p50 | 41.881 +/- 6.286 | 33.632 +/- 2.679 | -19.7% |
| interactive.ttft_ms_p95 | 104.828 +/- 4.623 | 81.012 +/- 19.036 | -22.7% |
| interactive.ttft_ms_p99 | 126.123 +/- 38.643 | 98.712 +/- 9.759 | -21.7% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.preemptions | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.reclaim_events | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.reclaimed_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.recomputed_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| observability.summary_calls | 8169.000 +/- 813.927 | 9894.000 +/- 417.066 | +21.1% |
| observability.summary_compute_ms | 53.545 +/- 2.208 | 58.065 +/- 1.774 | +8.4% |
| observability.summary_refreshes | 261.000 +/- 4.303 | 283.667 +/- 14.975 | +8.7% |
| overall.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| overall.output_throughput_tokens_per_s | 94.243 +/- 0.228 | 94.469 +/- 0.082 | +0.2% |
| overall.request_throughput_rps | 0.983 +/- 0.002 | 0.985 +/- 0.001 | +0.2% |
| overall.slo_goodput_rps | 0.982 +/- 0.007 | 0.985 +/- 0.001 | +0.4% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |

## Evidence Limits

- batch: only 150 completions; tail percentiles are exploratory.
- interactive: only 151 completions; tail percentiles are exploratory.
