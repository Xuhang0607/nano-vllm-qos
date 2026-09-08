# Repeated Live Serving Ablation

Baseline: **No-Compression** (3 runs)

Candidate: **Query-Aware** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 41099.288 +/- 83961.566 | 33817.993 +/- 29141.273 | -17.7% |
| batch.dispatch_lag_ms_p95 | 0.979 +/- 0.372 | 1.041 +/- 0.340 | +6.4% |
| batch.e2e_ms_p50 | 25169.930 +/- 54034.460 | 22900.742 +/- 18769.891 | -9.0% |
| batch.e2e_ms_p95 | 40955.803 +/- 84130.081 | 33757.618 +/- 29137.300 | -17.6% |
| batch.e2e_ms_p99 | 49093.125 +/- 84907.022 | 36777.547 +/- 38125.610 | -25.1% |
| batch.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| batch.offered_slo_attainment | 0.698 +/- 1.265 | 0.631 +/- 0.867 | -9.6% |
| batch.scheduled_e2e_ms_p95 | 41100.039 +/- 83961.459 | 33818.724 +/- 29141.210 | -17.7% |
| batch.slo_attainment | 0.698 +/- 1.265 | 0.631 +/- 0.867 | -9.6% |
| batch.tpot_ms_p50 | 86.400 +/- 35.479 | 109.279 +/- 24.410 | +26.5% |
| batch.tpot_ms_p95 | 146.449 +/- 134.976 | 137.979 +/- 38.315 | -5.8% |
| batch.tpot_ms_p99 | 205.907 +/- 103.237 | 156.557 +/- 85.809 | -24.0% |
| batch.ttft_ms_p50 | 13599.791 +/- 49412.469 | 8034.064 +/- 14818.537 | -40.9% |
| batch.ttft_ms_p95 | 27885.347 +/- 84571.846 | 19047.659 +/- 29699.433 | -31.7% |
| batch.ttft_ms_p99 | 29085.366 +/- 86154.665 | 20103.746 +/- 31070.635 | -30.9% |
| cache.prefix_cache_block_hit_rate | 0.182 +/- 0.014 | 0.183 +/- 0.011 | +0.6% |
| gpu.memory_used_mib.max | 5697.333 +/- 537.791 | 5547.333 +/- 397.139 | -2.6% |
| gpu.memory_used_mib.mean | 5492.836 +/- 114.846 | 5415.015 +/- 220.913 | -1.4% |
| gpu.memory_used_mib.p95 | 5581.000 +/- 212.944 | 5476.200 +/- 323.378 | -1.9% |
| gpu.peak_memory_fraction | 0.696 +/- 0.066 | 0.677 +/- 0.049 | -2.6% |
| gpu.utilization_gpu_percent.max | 100.000 +/- 0.000 | 100.000 +/- 0.000 | +0.0% |
| gpu.utilization_gpu_percent.mean | 63.126 +/- 5.288 | 61.686 +/- 1.733 | -2.3% |
| gpu.utilization_gpu_percent.p95 | 82.000 +/- 2.484 | 78.667 +/- 1.434 | -4.1% |
| interactive.client_e2e_ms_p95 | 22417.607 +/- 82154.011 | 12684.888 +/- 26310.629 | -43.4% |
| interactive.dispatch_lag_ms_p95 | 0.994 +/- 0.305 | 1.037 +/- 0.257 | +4.3% |
| interactive.e2e_ms_p50 | 10665.626 +/- 33707.208 | 4274.151 +/- 5770.310 | -59.9% |
| interactive.e2e_ms_p95 | 22355.997 +/- 82161.499 | 12586.801 +/- 26306.291 | -43.7% |
| interactive.e2e_ms_p99 | 23618.335 +/- 85934.081 | 13472.813 +/- 28194.380 | -43.0% |
| interactive.failed_requests | 0.333 +/- 1.434 | 0.000 +/- 0.000 | -100.0% |
| interactive.offered_slo_attainment | 0.693 +/- 1.249 | 0.663 +/- 0.817 | -4.3% |
| interactive.scheduled_e2e_ms_p95 | 22418.292 +/- 82154.030 | 12685.570 +/- 26310.534 | -43.4% |
| interactive.slo_attainment | 0.693 +/- 1.248 | 0.663 +/- 0.817 | -4.3% |
| interactive.tpot_ms_p50 | 44.598 +/- 13.516 | 44.332 +/- 6.324 | -0.6% |
| interactive.tpot_ms_p95 | 52.908 +/- 28.632 | 57.270 +/- 24.527 | +8.2% |
| interactive.tpot_ms_p99 | 62.691 +/- 63.072 | 66.583 +/- 38.235 | +6.2% |
| interactive.ttft_ms_p50 | 7696.840 +/- 32341.375 | 1379.763 +/- 4951.776 | -82.1% |
| interactive.ttft_ms_p95 | 19526.432 +/- 81841.538 | 9547.052 +/- 25896.401 | -51.1% |
| interactive.ttft_ms_p99 | 20770.684 +/- 85735.743 | 10536.428 +/- 28217.878 | -49.3% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 502.667 +/- 5.737 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 128682.667 +/- 1468.757 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 502.667 +/- 5.737 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 27817.333 +/- 23471.475 | 11645.667 +/- 38589.294 | -58.1% |
| kv.preemptions | 19.667 +/- 17.449 | 8.333 +/- 27.702 | -57.6% |
| kv.reclaim_events | 19.667 +/- 17.449 | 8.333 +/- 27.702 | -57.6% |
| kv.reclaimed_blocks | 98.333 +/- 87.247 | 39.000 +/- 134.865 | -60.3% |
| kv.recomputed_tokens | 22356.000 +/- 19223.948 | 9427.000 +/- 31136.780 | -57.8% |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| overall.failed_requests | 0.333 +/- 1.434 | 0.000 +/- 0.000 | -100.0% |
| overall.output_throughput_tokens_per_s | 167.880 +/- 68.261 | 175.883 +/- 15.190 | +4.8% |
| overall.request_throughput_rps | 1.749 +/- 0.713 | 1.833 +/- 0.158 | +4.8% |
| overall.slo_goodput_rps | 1.314 +/- 2.486 | 1.200 +/- 1.638 | -8.6% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |

## Evidence Limits

- batch: 300 completions; P99 still has limited tail samples.
- interactive: 300 completions; P99 still has limited tail samples.
- interactive: 301 completions; P99 still has limited tail samples.
