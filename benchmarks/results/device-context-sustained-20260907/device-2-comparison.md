# Repeated Live Serving Ablation

Baseline: **CPU device context** (3 runs)

Candidate: **Scoped initialization** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 9637.144 +/- 4185.118 | 5769.800 +/- 995.696 | -40.1% |
| batch.dispatch_lag_ms_p95 | 3.813 +/- 7.163 | 1.240 +/- 1.731 | -67.5% |
| batch.e2e_ms_p50 | 6245.457 +/- 1014.894 | 4832.801 +/- 629.800 | -22.6% |
| batch.e2e_ms_p95 | 9601.773 +/- 4169.316 | 5732.517 +/- 998.741 | -40.3% |
| batch.e2e_ms_p99 | 10796.672 +/- 6066.571 | 6053.451 +/- 1462.348 | -43.9% |
| batch.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| batch.offered_slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.scheduled_e2e_ms_p95 | 9637.837 +/- 4184.799 | 5771.134 +/- 997.598 | -40.1% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 40.851 +/- 8.048 | 31.474 +/- 3.621 | -23.0% |
| batch.tpot_ms_p95 | 65.709 +/- 21.812 | 37.660 +/- 9.683 | -42.7% |
| batch.tpot_ms_p99 | 75.762 +/- 30.584 | 42.508 +/- 12.211 | -43.9% |
| batch.ttft_ms_p50 | 711.270 +/- 81.332 | 623.218 +/- 26.980 | -12.4% |
| batch.ttft_ms_p95 | 1982.404 +/- 407.221 | 1571.444 +/- 215.172 | -20.7% |
| batch.ttft_ms_p99 | 2529.961 +/- 1501.912 | 1718.094 +/- 282.851 | -32.1% |
| cache.prefix_cache_block_hit_rate | 0.168 +/- 0.002 | 0.167 +/- 0.000 | -0.4% |
| gpu.memory_used_mib.max | 5112.000 +/- 415.501 | 5064.667 +/- 87.530 | -0.9% |
| gpu.memory_used_mib.mean | 4984.588 +/- 164.694 | 4951.803 +/- 24.399 | -0.7% |
| gpu.memory_used_mib.p95 | 5045.667 +/- 295.650 | 5021.333 +/- 88.267 | -0.5% |
| gpu.peak_memory_fraction | 0.624 +/- 0.051 | 0.619 +/- 0.011 | -0.9% |
| gpu.utilization_gpu_percent.max | 100.000 +/- 0.000 | 96.333 +/- 8.725 | -3.7% |
| gpu.utilization_gpu_percent.mean | 35.547 +/- 1.842 | 38.210 +/- 2.426 | +7.5% |
| gpu.utilization_gpu_percent.p95 | 63.000 +/- 4.303 | 61.783 +/- 0.932 | -1.9% |
| interactive.client_e2e_ms_p95 | 2788.340 +/- 296.429 | 2269.033 +/- 421.376 | -18.6% |
| interactive.dispatch_lag_ms_p95 | 2.370 +/- 4.846 | 1.145 +/- 1.083 | -51.7% |
| interactive.e2e_ms_p50 | 2370.999 +/- 199.583 | 1961.563 +/- 191.958 | -17.3% |
| interactive.e2e_ms_p95 | 2748.719 +/- 277.311 | 2223.549 +/- 420.862 | -19.1% |
| interactive.e2e_ms_p99 | 2864.392 +/- 373.355 | 2450.991 +/- 619.057 | -14.4% |
| interactive.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| interactive.offered_slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| interactive.scheduled_e2e_ms_p95 | 2794.725 +/- 313.434 | 2270.629 +/- 424.090 | -18.8% |
| interactive.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| interactive.tpot_ms_p50 | 36.134 +/- 2.640 | 30.318 +/- 3.201 | -16.1% |
| interactive.tpot_ms_p95 | 41.338 +/- 4.019 | 34.037 +/- 6.091 | -17.7% |
| interactive.tpot_ms_p99 | 42.581 +/- 4.638 | 36.680 +/- 6.611 | -13.9% |
| interactive.ttft_ms_p50 | 81.787 +/- 44.817 | 50.604 +/- 31.530 | -38.1% |
| interactive.ttft_ms_p95 | 194.028 +/- 47.859 | 125.485 +/- 16.780 | -35.3% |
| interactive.ttft_ms_p99 | 243.365 +/- 71.528 | 169.819 +/- 81.371 | -30.2% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 936.000 +/- 4027.608 | 0.000 +/- 0.000 | -100.0% |
| kv.preemptions | 0.667 +/- 2.869 | 0.000 +/- 0.000 | -100.0% |
| kv.reclaim_events | 0.667 +/- 2.869 | 0.000 +/- 0.000 | -100.0% |
| kv.reclaimed_blocks | 3.333 +/- 14.343 | 0.000 +/- 0.000 | -100.0% |
| kv.recomputed_tokens | 765.333 +/- 3293.229 | 0.000 +/- 0.000 | -100.0% |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| observability.summary_calls | 8460.667 +/- 611.230 | 9983.333 +/- 1040.823 | +18.0% |
| observability.summary_compute_ms | 206.240 +/- 28.131 | 192.192 +/- 32.039 | -6.8% |
| observability.summary_refreshes | 561.000 +/- 28.651 | 504.333 +/- 37.622 | -10.1% |
| overall.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| overall.output_throughput_tokens_per_s | 188.428 +/- 0.560 | 188.978 +/- 0.367 | +0.3% |
| overall.request_throughput_rps | 1.964 +/- 0.006 | 1.970 +/- 0.004 | +0.3% |
| overall.slo_goodput_rps | 1.964 +/- 0.006 | 1.970 +/- 0.004 | +0.3% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |

## Evidence Limits

- batch: 300 completions; P99 still has limited tail samples.
- interactive: 301 completions; P99 still has limited tail samples.
