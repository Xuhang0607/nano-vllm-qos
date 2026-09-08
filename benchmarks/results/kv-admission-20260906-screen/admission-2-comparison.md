# Repeated Live Serving Ablation

Baseline: **No page reservation** (1 runs)

Candidate: **Priority reserve 32** (1 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 10217.874 | 7973.907 | -22.0% |
| batch.dispatch_lag_ms_p95 | 1.049 | 1.111 | +6.0% |
| batch.e2e_ms_p50 | 6729.406 | 6571.559 | -2.3% |
| batch.e2e_ms_p95 | 10097.263 | 7942.765 | -21.3% |
| batch.e2e_ms_p99 | 10639.045 | 8274.987 | -22.2% |
| batch.failed_requests | 0.000 | 0.000 | n/a |
| batch.offered_slo_attainment | 1.000 | 1.000 | +0.0% |
| batch.scheduled_e2e_ms_p95 | 10218.579 | 7974.385 | -22.0% |
| batch.slo_attainment | 1.000 | 1.000 | +0.0% |
| batch.tpot_ms_p50 | 44.788 | 41.921 | -6.4% |
| batch.tpot_ms_p95 | 70.557 | 57.006 | -19.2% |
| batch.tpot_ms_p99 | 77.438 | 60.147 | -22.3% |
| batch.ttft_ms_p50 | 930.195 | 696.392 | -25.1% |
| batch.ttft_ms_p95 | 1995.905 | 1975.825 | -1.0% |
| batch.ttft_ms_p99 | 2493.091 | 2105.635 | -15.5% |
| cache.prefix_cache_block_hit_rate | 0.168 | 0.167 | -0.5% |
| gpu.memory_used_mib.max | 5028.000 | 4770.000 | -5.1% |
| gpu.memory_used_mib.mean | 4879.699 | 4734.410 | -3.0% |
| gpu.memory_used_mib.p95 | 4990.000 | 4754.000 | -4.7% |
| gpu.peak_memory_fraction | 0.614 | 0.583 | -5.1% |
| gpu.utilization_gpu_percent.max | 100.000 | 100.000 | +0.0% |
| gpu.utilization_gpu_percent.mean | 36.864 | 36.633 | -0.6% |
| gpu.utilization_gpu_percent.p95 | 64.000 | 66.000 | +3.1% |
| interactive.client_e2e_ms_p95 | 2861.510 | 2661.208 | -7.0% |
| interactive.dispatch_lag_ms_p95 | 1.117 | 1.041 | -6.8% |
| interactive.e2e_ms_p50 | 2411.005 | 2412.131 | +0.0% |
| interactive.e2e_ms_p95 | 2841.263 | 2615.313 | -8.0% |
| interactive.e2e_ms_p99 | 3167.798 | 2723.681 | -14.0% |
| interactive.failed_requests | 0.000 | 0.000 | n/a |
| interactive.offered_slo_attainment | 0.990 | 1.000 | +1.0% |
| interactive.scheduled_e2e_ms_p95 | 2862.261 | 2662.078 | -7.0% |
| interactive.slo_attainment | 0.990 | 1.000 | +1.0% |
| interactive.tpot_ms_p50 | 36.603 | 36.680 | +0.2% |
| interactive.tpot_ms_p95 | 41.753 | 39.325 | -5.8% |
| interactive.tpot_ms_p99 | 43.987 | 42.194 | -4.1% |
| interactive.ttft_ms_p50 | 93.575 | 103.520 | +10.6% |
| interactive.ttft_ms_p95 | 220.953 | 187.485 | -15.1% |
| interactive.ttft_ms_p99 | 377.382 | 219.506 | -41.8% |
| kv.compression_dropped_blocks | 0.000 | 0.000 | n/a |
| kv.compression_dropped_tokens | 0.000 | 0.000 | n/a |
| kv.compression_events | 0.000 | 0.000 | n/a |
| kv.forced_fallbacks | 0.000 | 0.000 | n/a |
| kv.invalidated_tokens | 0.000 | 0.000 | n/a |
| kv.preemptions | 0.000 | 0.000 | n/a |
| kv.reclaim_events | 0.000 | 0.000 | n/a |
| kv.reclaimed_blocks | 0.000 | 0.000 | n/a |
| kv.recomputed_tokens | 0.000 | 0.000 | n/a |
| kv.retained_blocks | 0.000 | 0.000 | n/a |
| observability.summary_calls | 8176.000 | 8266.000 | +1.1% |
| observability.summary_compute_ms | 224.351 | 219.047 | -2.4% |
| observability.summary_refreshes | 569.000 | 561.000 | -1.4% |
| overall.failed_requests | 0.000 | 0.000 | n/a |
| overall.output_throughput_tokens_per_s | 189.041 | 189.159 | +0.1% |
| overall.request_throughput_rps | 1.970 | 1.971 | +0.1% |
| overall.slo_goodput_rps | 1.960 | 1.971 | +0.6% |
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
