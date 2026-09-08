# Repeated Live Serving Ablation

Baseline: **CPU device context** (3 runs)

Candidate: **Scoped initialization** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 120861.569 +/- 94786.503 | 44972.627 +/- 54228.953 | -62.8% |
| batch.dispatch_lag_ms_p95 | 0.803 +/- 0.230 | 0.666 +/- 0.260 | -17.1% |
| batch.e2e_ms_p50 | 65617.731 +/- 49562.106 | 25371.717 +/- 40474.647 | -61.3% |
| batch.e2e_ms_p95 | 120351.277 +/- 93270.125 | 43774.147 +/- 56180.144 | -63.6% |
| batch.e2e_ms_p99 | 131629.754 +/- 93533.406 | 52151.246 +/- 62887.737 | -60.4% |
| batch.failed_requests | 0.333 +/- 1.434 | 0.000 +/- 0.000 | -100.0% |
| batch.offered_slo_attainment | 0.116 +/- 0.116 | 0.461 +/- 0.734 | +296.2% |
| batch.scheduled_e2e_ms_p95 | 120862.028 +/- 94786.843 | 44973.091 +/- 54228.951 | -62.8% |
| batch.slo_attainment | 0.116 +/- 0.116 | 0.461 +/- 0.734 | +296.0% |
| batch.tpot_ms_p50 | 78.187 +/- 17.149 | 63.525 +/- 16.988 | -18.8% |
| batch.tpot_ms_p95 | 246.811 +/- 69.935 | 176.064 +/- 50.216 | -28.7% |
| batch.tpot_ms_p99 | 284.323 +/- 80.836 | 199.546 +/- 49.843 | -29.8% |
| batch.ttft_ms_p50 | 52934.545 +/- 46256.245 | 16258.913 +/- 37620.075 | -69.3% |
| batch.ttft_ms_p95 | 112507.505 +/- 91241.839 | 32575.704 +/- 63621.030 | -71.0% |
| batch.ttft_ms_p99 | 114309.424 +/- 90928.882 | 33901.140 +/- 65465.908 | -70.3% |
| cache.prefix_cache_block_hit_rate | 0.186 +/- 0.005 | 0.187 +/- 0.004 | +0.3% |
| gpu.memory_used_mib.max | 5293.000 +/- 233.528 | 5146.333 +/- 294.164 | -2.8% |
| gpu.memory_used_mib.mean | 5114.055 +/- 163.884 | 5066.231 +/- 146.867 | -0.9% |
| gpu.memory_used_mib.p95 | 5271.667 +/- 160.722 | 5118.000 +/- 195.979 | -2.9% |
| gpu.peak_memory_fraction | 0.646 +/- 0.029 | 0.629 +/- 0.036 | -2.8% |
| gpu.utilization_gpu_percent.max | 100.000 +/- 0.000 | 100.000 +/- 0.000 | +0.0% |
| gpu.utilization_gpu_percent.mean | 35.785 +/- 2.692 | 42.520 +/- 4.565 | +18.8% |
| gpu.utilization_gpu_percent.p95 | 62.350 +/- 6.195 | 63.000 +/- 0.000 | +1.0% |
| interactive.client_e2e_ms_p95 | 102379.845 +/- 92699.515 | 24459.026 +/- 62659.463 | -76.1% |
| interactive.dispatch_lag_ms_p95 | 0.806 +/- 0.214 | 0.684 +/- 0.224 | -15.1% |
| interactive.e2e_ms_p50 | 36283.181 +/- 37120.991 | 8542.652 +/- 28083.197 | -76.5% |
| interactive.e2e_ms_p95 | 102108.821 +/- 92131.877 | 24167.164 +/- 62156.288 | -76.3% |
| interactive.e2e_ms_p99 | 105472.409 +/- 93133.476 | 26074.022 +/- 64550.849 | -75.3% |
| interactive.failed_requests | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| interactive.offered_slo_attainment | 0.152 +/- 0.118 | 0.526 +/- 0.802 | +245.6% |
| interactive.scheduled_e2e_ms_p95 | 102380.211 +/- 92699.368 | 24459.514 +/- 62659.568 | -76.1% |
| interactive.slo_attainment | 0.152 +/- 0.118 | 0.526 +/- 0.802 | +245.6% |
| interactive.tpot_ms_p50 | 37.656 +/- 7.904 | 30.484 +/- 6.455 | -19.0% |
| interactive.tpot_ms_p95 | 43.737 +/- 15.405 | 33.475 +/- 8.217 | -23.5% |
| interactive.tpot_ms_p99 | 53.342 +/- 50.063 | 36.933 +/- 9.484 | -30.8% |
| interactive.ttft_ms_p50 | 33711.095 +/- 36312.644 | 6544.966 +/- 27463.516 | -80.6% |
| interactive.ttft_ms_p95 | 99877.409 +/- 91955.428 | 22254.644 +/- 61980.087 | -77.7% |
| interactive.ttft_ms_p99 | 103224.492 +/- 92761.458 | 24155.305 +/- 64635.846 | -76.6% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 79103.333 +/- 15114.498 | 74352.667 +/- 17174.851 | -6.0% |
| kv.preemptions | 56.000 +/- 10.829 | 52.333 +/- 14.557 | -6.5% |
| kv.reclaim_events | 56.000 +/- 10.829 | 52.333 +/- 14.557 | -6.5% |
| kv.reclaimed_blocks | 280.000 +/- 54.145 | 260.667 +/- 68.549 | -6.9% |
| kv.recomputed_tokens | 64426.000 +/- 13792.450 | 60699.333 +/- 13599.383 | -5.8% |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| observability.summary_calls | 11443.667 +/- 163.369 | 11381.333 +/- 142.505 | -0.5% |
| observability.summary_compute_ms | 336.108 +/- 51.275 | 404.909 +/- 98.432 | +20.5% |
| observability.summary_refreshes | 673.667 +/- 22.405 | 791.667 +/- 188.946 | +17.5% |
| overall.failed_requests | 0.333 +/- 1.434 | 0.000 +/- 0.000 | -100.0% |
| overall.output_throughput_tokens_per_s | 204.793 +/- 48.349 | 256.600 +/- 48.332 | +25.3% |
| overall.request_throughput_rps | 2.134 +/- 0.503 | 2.674 +/- 0.504 | +25.3% |
| overall.slo_goodput_rps | 0.292 +/- 0.300 | 1.360 +/- 2.225 | +365.9% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |

## Evidence Limits

- batch: 449 completions; P99 still has limited tail samples.
- batch: 450 completions; P99 still has limited tail samples.
- interactive: 451 completions; P99 still has limited tail samples.
