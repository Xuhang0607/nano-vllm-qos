# Repeated Live Serving Ablation

Baseline: **Full summary** (3 runs)

Candidate: **Cached summary** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 9173.684 +/- 1787.428 | 13518.381 +/- 21937.797 | +47.4% |
| batch.dispatch_lag_ms_p95 | 0.821 +/- 0.155 | 0.824 +/- 0.017 | +0.4% |
| batch.e2e_ms_p50 | 6968.636 +/- 1072.211 | 8294.700 +/- 8193.977 | +19.0% |
| batch.e2e_ms_p95 | 9131.135 +/- 1794.437 | 13469.536 +/- 21930.848 | +47.5% |
| batch.e2e_ms_p99 | 9870.494 +/- 1802.324 | 16088.798 +/- 30673.048 | +63.0% |
| batch.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| batch.offered_slo_attainment | 1.000 +/- 0.000 | 0.980 +/- 0.086 | -2.0% |
| batch.scheduled_e2e_ms_p95 | 9174.305 +/- 1787.517 | 13519.006 +/- 21937.896 | +47.4% |
| batch.slo_attainment | 1.000 +/- 0.000 | 0.980 +/- 0.086 | -2.0% |
| batch.tpot_ms_p50 | 45.986 +/- 8.445 | 50.205 +/- 39.591 | +9.2% |
| batch.tpot_ms_p95 | 65.957 +/- 12.968 | 79.079 +/- 83.616 | +19.9% |
| batch.tpot_ms_p99 | 72.264 +/- 10.993 | 105.935 +/- 173.860 | +46.6% |
| batch.ttft_ms_p50 | 720.442 +/- 34.576 | 1163.465 +/- 2027.261 | +61.5% |
| batch.ttft_ms_p95 | 1980.330 +/- 178.156 | 4538.059 +/- 11380.851 | +129.2% |
| batch.ttft_ms_p99 | 2225.301 +/- 74.189 | 5246.933 +/- 13359.159 | +135.8% |
| cache.prefix_cache_block_hit_rate | 0.167 +/- 0.002 | 0.171 +/- 0.020 | +2.9% |
| gpu.memory_used_mib.max | 5041.000 +/- 158.122 | 5173.333 +/- 601.735 | +2.6% |
| gpu.memory_used_mib.mean | 4981.324 +/- 138.156 | 5014.142 +/- 226.551 | +0.7% |
| gpu.memory_used_mib.p95 | 5035.667 +/- 158.031 | 5110.667 +/- 456.945 | +1.5% |
| gpu.peak_memory_fraction | 0.616 +/- 0.019 | 0.632 +/- 0.073 | +2.6% |
| gpu.utilization_gpu_percent.max | 100.000 +/- 0.000 | 100.000 +/- 0.000 | +0.0% |
| gpu.utilization_gpu_percent.mean | 39.287 +/- 0.574 | 39.086 +/- 3.267 | -0.5% |
| gpu.utilization_gpu_percent.p95 | 65.467 +/- 1.696 | 63.000 +/- 2.484 | -3.8% |
| interactive.client_e2e_ms_p95 | 2788.597 +/- 243.450 | 3056.119 +/- 1530.405 | +9.6% |
| interactive.dispatch_lag_ms_p95 | 0.795 +/- 0.098 | 0.810 +/- 0.019 | +1.9% |
| interactive.e2e_ms_p50 | 2468.739 +/- 173.035 | 2497.120 +/- 541.468 | +1.1% |
| interactive.e2e_ms_p95 | 2743.223 +/- 255.130 | 3014.531 +/- 1497.379 | +9.9% |
| interactive.e2e_ms_p99 | 2833.701 +/- 199.853 | 3249.265 +/- 2134.124 | +14.7% |
| interactive.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| interactive.offered_slo_attainment | 0.997 +/- 0.014 | 0.988 +/- 0.052 | -0.9% |
| interactive.scheduled_e2e_ms_p95 | 2789.226 +/- 243.437 | 3057.536 +/- 1528.710 | +9.6% |
| interactive.slo_attainment | 0.997 +/- 0.014 | 0.988 +/- 0.052 | -0.9% |
| interactive.tpot_ms_p50 | 37.418 +/- 2.275 | 37.475 +/- 6.121 | +0.2% |
| interactive.tpot_ms_p95 | 40.755 +/- 2.935 | 42.570 +/- 11.581 | +4.5% |
| interactive.tpot_ms_p99 | 41.818 +/- 3.466 | 45.250 +/- 16.795 | +8.2% |
| interactive.ttft_ms_p50 | 107.965 +/- 23.781 | 102.305 +/- 58.875 | -5.2% |
| interactive.ttft_ms_p95 | 211.588 +/- 58.081 | 382.398 +/- 838.336 | +80.7% |
| interactive.ttft_ms_p99 | 263.919 +/- 40.100 | 681.908 +/- 1930.228 | +158.4% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 0.000 +/- 0.000 | 6223.333 +/- 26779.003 | n/a |
| kv.preemptions | 0.000 +/- 0.000 | 4.333 +/- 18.646 | n/a |
| kv.reclaim_events | 0.000 +/- 0.000 | 4.333 +/- 18.646 | n/a |
| kv.reclaimed_blocks | 0.000 +/- 0.000 | 21.667 +/- 93.232 | n/a |
| kv.recomputed_tokens | 0.000 +/- 0.000 | 5114.000 +/- 22005.542 | n/a |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| observability.summary_calls | 8160.667 +/- 418.914 | 8260.000 +/- 735.813 | +1.2% |
| observability.summary_compute_ms | 2801.820 +/- 104.974 | 207.459 +/- 23.542 | -92.6% |
| observability.summary_refreshes | 8160.667 +/- 418.914 | 564.000 +/- 16.291 | -93.1% |
| overall.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| overall.output_throughput_tokens_per_s | 188.891 +/- 0.458 | 187.132 +/- 8.738 | -0.9% |
| overall.request_throughput_rps | 1.969 +/- 0.005 | 1.950 +/- 0.091 | -0.9% |
| overall.slo_goodput_rps | 1.965 +/- 0.017 | 1.920 +/- 0.223 | -2.3% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |

## Evidence Limits

- batch: 300 completions; P99 still has limited tail samples.
- interactive: 301 completions; P99 still has limited tail samples.
