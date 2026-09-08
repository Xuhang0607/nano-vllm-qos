# Repeated Live Serving Ablation

Baseline: **No-Compression** (1 runs)

Candidate: **Query-Aware** (1 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 15397.000 | 13317.096 | -13.5% |
| batch.dispatch_lag_ms_p95 | 0.837 | 0.897 | +7.2% |
| batch.e2e_ms_p50 | 7289.549 | 7195.202 | -1.3% |
| batch.e2e_ms_p95 | 15043.207 | 13278.534 | -11.7% |
| batch.e2e_ms_p99 | 22932.133 | 13582.439 | -40.8% |
| batch.failed_requests | 0.000 | 0.000 | n/a |
| batch.offered_slo_attainment | 1.000 | 1.000 | +0.0% |
| batch.scheduled_e2e_ms_p95 | 15397.675 | 13317.477 | -13.5% |
| batch.slo_attainment | 1.000 | 1.000 | +0.0% |
| batch.tpot_ms_p50 | 49.691 | 48.775 | -1.8% |
| batch.tpot_ms_p95 | 96.187 | 93.234 | -3.1% |
| batch.tpot_ms_p99 | 140.553 | 99.095 | -29.5% |
| batch.ttft_ms_p50 | 1067.286 | 1025.425 | -3.9% |
| batch.ttft_ms_p95 | 5510.132 | 2137.185 | -61.2% |
| batch.ttft_ms_p99 | 6875.414 | 3135.434 | -54.4% |
| cache.prefix_cache_block_hit_rate | 0.172 | 0.169 | -1.8% |
| gpu.memory_used_mib.max | 4631.000 | 4779.000 | +3.2% |
| gpu.memory_used_mib.mean | 4596.862 | 4622.847 | +0.6% |
| gpu.memory_used_mib.p95 | 4625.000 | 4750.000 | +2.7% |
| gpu.peak_memory_fraction | 0.566 | 0.584 | +3.2% |
| gpu.utilization_gpu_percent.max | 100.000 | 100.000 | +0.0% |
| gpu.utilization_gpu_percent.mean | 38.161 | 36.028 | -5.6% |
| gpu.utilization_gpu_percent.p95 | 63.000 | 60.000 | -4.8% |
| interactive.client_e2e_ms_p95 | 3751.155 | 2926.257 | -22.0% |
| interactive.dispatch_lag_ms_p95 | 0.957 | 0.946 | -1.2% |
| interactive.e2e_ms_p50 | 2413.132 | 2512.765 | +4.1% |
| interactive.e2e_ms_p95 | 3323.537 | 2893.763 | -12.9% |
| interactive.e2e_ms_p99 | 3827.004 | 3201.244 | -16.4% |
| interactive.failed_requests | 0.000 | 0.000 | n/a |
| interactive.offered_slo_attainment | 0.997 | 1.000 | +0.3% |
| interactive.scheduled_e2e_ms_p95 | 3751.799 | 2926.867 | -22.0% |
| interactive.slo_attainment | 0.997 | 1.000 | +0.3% |
| interactive.tpot_ms_p50 | 36.950 | 37.934 | +2.7% |
| interactive.tpot_ms_p95 | 46.998 | 43.959 | -6.5% |
| interactive.tpot_ms_p99 | 54.734 | 49.005 | -10.5% |
| interactive.ttft_ms_p50 | 81.328 | 107.648 | +32.4% |
| interactive.ttft_ms_p95 | 357.318 | 224.335 | -37.2% |
| interactive.ttft_ms_p99 | 969.850 | 256.093 | -73.6% |
| kv.compression_dropped_blocks | 0.000 | 500.000 | n/a |
| kv.compression_dropped_tokens | 0.000 | 128000.000 | n/a |
| kv.compression_events | 0.000 | 500.000 | n/a |
| kv.forced_fallbacks | 0.000 | 0.000 | n/a |
| kv.invalidated_tokens | 5812.000 | 0.000 | -100.0% |
| kv.preemptions | 4.000 | 0.000 | -100.0% |
| kv.reclaim_events | 4.000 | 0.000 | -100.0% |
| kv.reclaimed_blocks | 20.000 | 0.000 | -100.0% |
| kv.recomputed_tokens | 4788.000 | 0.000 | -100.0% |
| kv.retained_blocks | 0.000 | 0.000 | n/a |
| overall.failed_requests | 0.000 | 0.000 | n/a |
| overall.output_throughput_tokens_per_s | 183.293 | 188.195 | +2.7% |
| overall.request_throughput_rps | 1.910 | 1.961 | +2.7% |
| overall.slo_goodput_rps | 1.907 | 1.961 | +2.8% |
| remote.backend_get_bytes | 0.000 | 0.000 | n/a |
| remote.backend_put_bytes | 0.000 | 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 | 0.000 | n/a |
| remote.restore_completed | 0.000 | 0.000 | n/a |
| remote.restore_started | 0.000 | 0.000 | n/a |
| remote.transfer_bytes | 0.000 | 0.000 | n/a |

## Evidence Limits

- batch: 300 completions; P99 still has limited tail samples.
- interactive: 301 completions; P99 still has limited tail samples.
- Fewer than three repetitions: preliminary result only.
