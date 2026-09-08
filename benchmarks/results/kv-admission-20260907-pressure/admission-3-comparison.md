# Repeated Live Serving Ablation

Baseline: **No page reservation** (3 runs)

Candidate: **Priority reserve 32** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 63480.296 +/- 16690.052 | 69927.129 +/- 2109.162 | +10.2% |
| batch.dispatch_lag_ms_p95 | 0.874 +/- 0.277 | 0.927 +/- 0.086 | +6.1% |
| batch.e2e_ms_p50 | 45394.707 +/- 15792.331 | 50079.075 +/- 5189.915 | +10.3% |
| batch.e2e_ms_p95 | 63440.880 +/- 16694.723 | 69836.914 +/- 2040.069 | +10.1% |
| batch.e2e_ms_p99 | 75221.384 +/- 25152.333 | 81185.500 +/- 13458.467 | +7.9% |
| batch.failed_requests | 0.000 +/- 0.000 | 0.333 +/- 1.434 | n/a |
| batch.offered_slo_attainment | 0.159 +/- 0.107 | 0.143 +/- 0.021 | -10.5% |
| batch.scheduled_e2e_ms_p95 | 63480.788 +/- 16690.352 | 69927.682 +/- 2108.911 | +10.2% |
| batch.slo_attainment | 0.159 +/- 0.107 | 0.143 +/- 0.021 | -10.3% |
| batch.tpot_ms_p50 | 86.279 +/- 4.679 | 88.440 +/- 6.333 | +2.5% |
| batch.tpot_ms_p95 | 241.519 +/- 57.219 | 230.135 +/- 59.566 | -4.7% |
| batch.tpot_ms_p99 | 281.782 +/- 71.531 | 284.341 +/- 23.883 | +0.9% |
| batch.ttft_ms_p50 | 29120.721 +/- 16556.675 | 34474.106 +/- 7033.461 | +18.4% |
| batch.ttft_ms_p95 | 56129.509 +/- 15623.326 | 62370.561 +/- 2435.032 | +11.1% |
| batch.ttft_ms_p99 | 57374.697 +/- 16218.773 | 64127.948 +/- 3145.145 | +11.8% |
| cache.prefix_cache_block_hit_rate | 0.186 +/- 0.005 | 0.192 +/- 0.004 | +3.6% |
| gpu.memory_used_mib.max | 5155.000 +/- 423.673 | 5178.000 +/- 201.277 | +0.4% |
| gpu.memory_used_mib.mean | 4952.814 +/- 290.205 | 4964.642 +/- 225.262 | +0.2% |
| gpu.memory_used_mib.p95 | 5100.000 +/- 495.207 | 5105.517 +/- 99.162 | +0.1% |
| gpu.peak_memory_fraction | 0.630 +/- 0.052 | 0.632 +/- 0.025 | +0.4% |
| gpu.utilization_gpu_percent.max | 100.000 +/- 0.000 | 100.000 +/- 0.000 | +0.0% |
| gpu.utilization_gpu_percent.mean | 36.624 +/- 1.141 | 36.349 +/- 1.135 | -0.8% |
| gpu.utilization_gpu_percent.p95 | 56.750 +/- 1.708 | 55.667 +/- 1.434 | -1.9% |
| interactive.client_e2e_ms_p95 | 42360.841 +/- 13359.266 | 44727.183 +/- 8231.208 | +5.6% |
| interactive.dispatch_lag_ms_p95 | 0.845 +/- 0.118 | 0.852 +/- 0.192 | +0.8% |
| interactive.e2e_ms_p50 | 15948.942 +/- 12733.151 | 16739.604 +/- 8983.945 | +5.0% |
| interactive.e2e_ms_p95 | 42161.729 +/- 14028.956 | 44347.989 +/- 7561.054 | +5.2% |
| interactive.e2e_ms_p99 | 46455.072 +/- 15469.375 | 48516.074 +/- 4365.793 | +4.4% |
| interactive.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| interactive.offered_slo_attainment | 0.238 +/- 0.145 | 0.297 +/- 0.104 | +24.8% |
| interactive.scheduled_e2e_ms_p95 | 42361.399 +/- 13359.446 | 44727.791 +/- 8231.042 | +5.6% |
| interactive.slo_attainment | 0.238 +/- 0.145 | 0.297 +/- 0.104 | +24.8% |
| interactive.tpot_ms_p50 | 40.721 +/- 5.020 | 43.721 +/- 0.157 | +7.4% |
| interactive.tpot_ms_p95 | 50.123 +/- 15.309 | 55.174 +/- 6.194 | +10.1% |
| interactive.tpot_ms_p99 | 55.092 +/- 15.381 | 71.159 +/- 15.635 | +29.2% |
| interactive.ttft_ms_p50 | 13185.084 +/- 11230.168 | 13550.372 +/- 9311.924 | +2.8% |
| interactive.ttft_ms_p95 | 39543.570 +/- 14056.935 | 41723.115 +/- 7707.451 | +5.5% |
| interactive.ttft_ms_p99 | 44038.457 +/- 15338.950 | 45975.391 +/- 5079.001 | +4.4% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 24134.000 +/- 11142.195 | 17419.000 +/- 4997.908 | -27.8% |
| kv.preemptions | 17.333 +/- 7.986 | 12.667 +/- 3.795 | -26.9% |
| kv.reclaim_events | 17.333 +/- 7.986 | 12.667 +/- 3.795 | -26.9% |
| kv.reclaimed_blocks | 86.667 +/- 39.930 | 63.333 +/- 18.974 | -26.9% |
| kv.recomputed_tokens | 19696.667 +/- 9098.149 | 14176.333 +/- 4026.426 | -28.0% |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| observability.summary_calls | 4622.667 +/- 39.620 | 4662.333 +/- 129.146 | +0.9% |
| observability.summary_compute_ms | 67.987 +/- 1.302 | 67.850 +/- 7.953 | -0.2% |
| observability.summary_refreshes | 297.000 +/- 28.651 | 298.333 +/- 29.639 | +0.4% |
| overall.failed_requests | 0.000 +/- 0.000 | 0.333 +/- 1.434 | n/a |
| overall.output_throughput_tokens_per_s | 187.865 +/- 19.485 | 181.682 +/- 2.611 | -3.3% |
| overall.request_throughput_rps | 1.959 +/- 0.203 | 1.895 +/- 0.026 | -3.3% |
| overall.slo_goodput_rps | 0.390 +/- 0.261 | 0.417 +/- 0.121 | +6.9% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |

## Evidence Limits

- Measurement lasted less than 300 seconds; not a sustained-load result.
- batch: only 179 completions; tail percentiles are exploratory.
- batch: only 180 completions; tail percentiles are exploratory.
- interactive: only 181 completions; tail percentiles are exploratory.
