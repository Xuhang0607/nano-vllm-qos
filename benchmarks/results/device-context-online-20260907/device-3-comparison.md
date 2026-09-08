# Repeated Live Serving Ablation

Baseline: **CPU device context** (3 runs)

Candidate: **Scoped initialization** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 34298.475 +/- 4688.599 | 17925.885 +/- 5229.001 | -47.7% |
| batch.dispatch_lag_ms_p95 | 1.129 +/- 1.347 | 9.356 +/- 36.812 | +728.8% |
| batch.e2e_ms_p50 | 27924.003 +/- 5353.983 | 15329.535 +/- 5768.332 | -45.1% |
| batch.e2e_ms_p95 | 34261.226 +/- 4673.247 | 17864.308 +/- 5252.273 | -47.9% |
| batch.e2e_ms_p99 | 44869.223 +/- 14012.129 | 25220.380 +/- 21003.893 | -43.8% |
| batch.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| batch.offered_slo_attainment | 0.267 +/- 0.208 | 0.819 +/- 0.364 | +206.9% |
| batch.scheduled_e2e_ms_p95 | 34299.464 +/- 4690.072 | 17926.412 +/- 5229.548 | -47.7% |
| batch.slo_attainment | 0.267 +/- 0.208 | 0.819 +/- 0.364 | +206.9% |
| batch.tpot_ms_p50 | 81.941 +/- 7.353 | 62.151 +/- 10.234 | -24.2% |
| batch.tpot_ms_p95 | 114.041 +/- 41.189 | 83.022 +/- 24.414 | -27.2% |
| batch.tpot_ms_p99 | 231.715 +/- 20.115 | 150.318 +/- 154.690 | -35.1% |
| batch.ttft_ms_p50 | 17287.621 +/- 5408.880 | 6609.477 +/- 4250.636 | -61.8% |
| batch.ttft_ms_p95 | 26932.425 +/- 5699.459 | 11721.977 +/- 5228.632 | -56.5% |
| batch.ttft_ms_p99 | 28418.976 +/- 4831.091 | 12025.421 +/- 5289.549 | -57.7% |
| cache.prefix_cache_block_hit_rate | 0.188 +/- 0.005 | 0.190 +/- 0.002 | +1.2% |
| gpu.memory_used_mib.max | 5175.667 +/- 224.751 | 5160.667 +/- 79.822 | -0.3% |
| gpu.memory_used_mib.mean | 5100.466 +/- 52.250 | 5122.026 +/- 76.105 | +0.4% |
| gpu.memory_used_mib.p95 | 5152.333 +/- 141.723 | 5152.667 +/- 84.771 | +0.0% |
| gpu.peak_memory_fraction | 0.632 +/- 0.027 | 0.630 +/- 0.010 | -0.3% |
| gpu.utilization_gpu_percent.max | 94.333 +/- 24.384 | 91.333 +/- 37.293 | -3.2% |
| gpu.utilization_gpu_percent.mean | 34.923 +/- 4.863 | 41.513 +/- 2.717 | +18.9% |
| gpu.utilization_gpu_percent.p95 | 52.000 +/- 4.303 | 58.900 +/- 3.873 | +13.3% |
| interactive.client_e2e_ms_p95 | 13512.193 +/- 6763.357 | 2919.857 +/- 896.299 | -78.4% |
| interactive.dispatch_lag_ms_p95 | 1.004 +/- 0.412 | 2.497 +/- 6.827 | +148.6% |
| interactive.e2e_ms_p50 | 3998.515 +/- 4432.312 | 2235.632 +/- 192.710 | -44.1% |
| interactive.e2e_ms_p95 | 13328.359 +/- 6404.404 | 2819.912 +/- 1014.225 | -78.8% |
| interactive.e2e_ms_p99 | 16729.022 +/- 6567.881 | 3913.026 +/- 3515.404 | -76.6% |
| interactive.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| interactive.offered_slo_attainment | 0.568 +/- 0.428 | 0.982 +/- 0.057 | +72.9% |
| interactive.scheduled_e2e_ms_p95 | 13513.313 +/- 6764.695 | 2925.598 +/- 884.006 | -78.4% |
| interactive.slo_attainment | 0.568 +/- 0.428 | 0.982 +/- 0.057 | +72.9% |
| interactive.tpot_ms_p50 | 41.825 +/- 3.972 | 33.470 +/- 3.448 | -20.0% |
| interactive.tpot_ms_p95 | 63.010 +/- 63.930 | 38.773 +/- 6.014 | -38.5% |
| interactive.tpot_ms_p99 | 66.087 +/- 61.676 | 41.692 +/- 5.990 | -36.9% |
| interactive.ttft_ms_p50 | 670.093 +/- 1548.637 | 110.953 +/- 11.812 | -83.4% |
| interactive.ttft_ms_p95 | 10749.610 +/- 6342.760 | 405.402 +/- 529.799 | -96.2% |
| interactive.ttft_ms_p99 | 13997.614 +/- 7005.161 | 1424.514 +/- 3301.569 | -89.8% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 3662.333 +/- 1940.219 | 1840.000 +/- 3959.137 | -49.8% |
| kv.preemptions | 2.667 +/- 1.434 | 1.333 +/- 2.869 | -50.0% |
| kv.reclaim_events | 2.667 +/- 1.434 | 1.333 +/- 2.869 | -50.0% |
| kv.reclaimed_blocks | 13.333 +/- 7.172 | 6.667 +/- 14.343 | -50.0% |
| kv.recomputed_tokens | 2979.667 +/- 1573.766 | 1498.667 +/- 3224.845 | -49.7% |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| observability.summary_calls | 2340.000 +/- 11.385 | 2381.667 +/- 155.590 | +1.8% |
| observability.summary_compute_ms | 22.049 +/- 2.764 | 23.881 +/- 3.747 | +8.3% |
| observability.summary_refreshes | 165.000 +/- 19.875 | 177.000 +/- 0.000 | +7.3% |
| overall.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| overall.output_throughput_tokens_per_s | 183.486 +/- 7.465 | 222.282 +/- 24.202 | +21.1% |
| overall.request_throughput_rps | 1.915 +/- 0.078 | 2.320 +/- 0.253 | +21.1% |
| overall.slo_goodput_rps | 0.802 +/- 0.630 | 2.094 +/- 0.693 | +161.0% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |

## Evidence Limits

- Measurement lasted less than 300 seconds; not a sustained-load result.
- batch: only 90 completions; tail percentiles are exploratory.
- interactive: only 91 completions; tail percentiles are exploratory.
