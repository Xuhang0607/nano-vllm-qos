# Repeated Live Serving Ablation

Baseline: **No-Compression** (3 runs)

Candidate: **Query-Aware** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.client_e2e_ms_p95 | 253581.265 +/- 86560.786 | 259243.411 +/- 109276.156 | +2.2% |
| batch.dispatch_lag_ms_p95 | 1.015 +/- 0.080 | 1.126 +/- 0.322 | +11.0% |
| batch.e2e_ms_p50 | 136536.789 +/- 17318.417 | 140060.576 +/- 88675.595 | +2.6% |
| batch.e2e_ms_p95 | 252624.875 +/- 84514.759 | 258946.147 +/- 108837.738 | +2.5% |
| batch.e2e_ms_p99 | 258905.607 +/- 87876.987 | 270710.971 +/- 111758.830 | +4.6% |
| batch.failed_requests | 18.000 +/- 71.097 | 43.333 +/- 95.875 | +140.7% |
| batch.offered_slo_attainment | 0.057 +/- 0.021 | 0.053 +/- 0.061 | -7.8% |
| batch.scheduled_e2e_ms_p95 | 253582.095 +/- 86560.616 | 259244.138 +/- 109276.330 | +2.2% |
| batch.slo_attainment | 0.059 +/- 0.018 | 0.057 +/- 0.056 | -3.5% |
| batch.tpot_ms_p50 | 104.310 +/- 24.443 | 128.053 +/- 39.309 | +22.8% |
| batch.tpot_ms_p95 | 332.915 +/- 99.217 | 357.781 +/- 117.548 | +7.5% |
| batch.tpot_ms_p99 | 400.587 +/- 103.628 | 415.275 +/- 142.091 | +3.7% |
| batch.ttft_ms_p50 | 115775.541 +/- 3416.924 | 117570.716 +/- 79040.949 | +1.6% |
| batch.ttft_ms_p95 | 241231.067 +/- 77467.711 | 243353.310 +/- 102216.267 | +0.9% |
| batch.ttft_ms_p99 | 248258.043 +/- 84496.575 | 255023.410 +/- 118746.400 | +2.7% |
| cache.prefix_cache_block_hit_rate | 0.187 +/- 0.003 | 0.186 +/- 0.003 | -0.8% |
| gpu.memory_used_mib.max | 5203.000 +/- 843.392 | 5472.000 +/- 1177.506 | +5.2% |
| gpu.memory_used_mib.mean | 5035.005 +/- 974.161 | 5161.194 +/- 984.602 | +2.5% |
| gpu.memory_used_mib.p95 | 5155.000 +/- 877.689 | 5391.583 +/- 1134.919 | +4.6% |
| gpu.peak_memory_fraction | 0.635 +/- 0.103 | 0.668 +/- 0.144 | +5.2% |
| gpu.utilization_gpu_percent.max | 100.000 +/- 0.000 | 100.000 +/- 0.000 | +0.0% |
| gpu.utilization_gpu_percent.mean | 42.219 +/- 28.978 | 42.483 +/- 26.202 | +0.6% |
| gpu.utilization_gpu_percent.p95 | 65.217 +/- 28.928 | 65.500 +/- 21.765 | +0.4% |
| interactive.client_e2e_ms_p95 | 235916.487 +/- 103132.137 | 254419.277 +/- 140870.909 | +7.8% |
| interactive.dispatch_lag_ms_p95 | 1.018 +/- 0.077 | 1.115 +/- 0.335 | +9.5% |
| interactive.e2e_ms_p50 | 120131.783 +/- 46084.153 | 129205.891 +/- 116905.755 | +7.6% |
| interactive.e2e_ms_p95 | 234769.518 +/- 104305.355 | 250943.836 +/- 133449.494 | +6.9% |
| interactive.e2e_ms_p99 | 240228.887 +/- 103842.777 | 258907.998 +/- 146453.924 | +7.8% |
| interactive.failed_requests | 1.333 +/- 3.795 | 8.667 +/- 33.083 | +550.0% |
| interactive.offered_slo_attainment | 0.073 +/- 0.019 | 0.060 +/- 0.065 | -18.2% |
| interactive.scheduled_e2e_ms_p95 | 235917.318 +/- 103132.218 | 254419.962 +/- 140870.981 | +7.8% |
| interactive.slo_attainment | 0.073 +/- 0.020 | 0.061 +/- 0.066 | -16.7% |
| interactive.tpot_ms_p50 | 49.936 +/- 7.979 | 53.142 +/- 16.465 | +6.4% |
| interactive.tpot_ms_p95 | 63.060 +/- 14.481 | 66.632 +/- 17.205 | +5.7% |
| interactive.tpot_ms_p99 | 67.181 +/- 14.000 | 80.962 +/- 8.682 | +20.5% |
| interactive.ttft_ms_p50 | 116351.729 +/- 44893.576 | 125449.693 +/- 115589.990 | +7.8% |
| interactive.ttft_ms_p95 | 232101.135 +/- 103588.965 | 247789.659 +/- 132317.045 | +6.8% |
| interactive.ttft_ms_p99 | 237510.013 +/- 103355.094 | 255788.754 +/- 145303.903 | +7.7% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 724.000 +/- 173.050 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 185344.000 +/- 44300.750 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 724.000 +/- 173.050 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 68168.667 +/- 21659.567 | 63087.000 +/- 17491.221 | -7.5% |
| kv.preemptions | 48.333 +/- 14.975 | 45.000 +/- 12.909 | -6.9% |
| kv.reclaim_events | 48.333 +/- 14.975 | 45.000 +/- 12.909 | -6.9% |
| kv.reclaimed_blocks | 241.667 +/- 74.874 | 224.333 +/- 61.676 | -7.2% |
| kv.recomputed_tokens | 55795.333 +/- 17826.009 | 51567.000 +/- 14186.907 | -7.6% |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| overall.failed_requests | 19.333 +/- 70.676 | 52.000 +/- 122.415 | +169.0% |
| overall.output_throughput_tokens_per_s | 149.599 +/- 43.585 | 138.094 +/- 66.665 | -7.7% |
| overall.request_throughput_rps | 1.568 +/- 0.416 | 1.457 +/- 0.657 | -7.0% |
| overall.slo_goodput_rps | 0.104 +/- 0.045 | 0.089 +/- 0.121 | -14.9% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |

## Evidence Limits

- batch: 376 completions; P99 still has limited tail samples.
- batch: 394 completions; P99 still has limited tail samples.
- batch: 399 completions; P99 still has limited tail samples.
- batch: 447 completions; P99 still has limited tail samples.
- batch: 450 completions; P99 still has limited tail samples.
- interactive: 427 completions; P99 still has limited tail samples.
- interactive: 448 completions; P99 still has limited tail samples.
- interactive: 449 completions; P99 still has limited tail samples.
- interactive: 450 completions; P99 still has limited tail samples.
- interactive: 451 completions; P99 still has limited tail samples.
