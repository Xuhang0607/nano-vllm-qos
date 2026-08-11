# Repeated Live Serving Ablation

Baseline: **No-Compression** (3 runs)

Candidate: **Query-Aware-KV** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.e2e_ms_p50 | 8284.533 +/- 481.456 | 8645.866 +/- 1218.493 | +4.4% |
| batch.e2e_ms_p95 | 14660.024 +/- 1010.154 | 9170.939 +/- 87.152 | -37.4% |
| batch.e2e_ms_p99 | 15559.971 +/- 1147.078 | 9244.948 +/- 75.672 | -40.6% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 30.925 +/- 1.362 | 32.277 +/- 4.941 | +4.4% |
| batch.tpot_ms_p95 | 56.995 +/- 5.349 | 34.829 +/- 2.616 | -38.9% |
| batch.tpot_ms_p99 | 60.675 +/- 5.947 | 35.189 +/- 2.301 | -42.0% |
| batch.ttft_ms_p50 | 397.675 +/- 132.110 | 414.312 +/- 79.810 | +4.2% |
| batch.ttft_ms_p95 | 399.887 +/- 133.344 | 416.708 +/- 78.894 | +4.2% |
| batch.ttft_ms_p99 | 400.102 +/- 133.498 | 416.933 +/- 78.796 | +4.2% |
| cache.prefix_cache_block_hit_rate | 0.201 +/- 0.000 | 0.231 +/- 0.000 | +15.0% |
| gpu.memory_used_mib.max | 3332.000 +/- 23.958 | 3333.333 +/- 79.434 | +0.0% |
| gpu.memory_used_mib.mean | 3292.096 +/- 116.893 | 3327.660 +/- 61.178 | +1.1% |
| gpu.memory_used_mib.p95 | 3331.200 +/- 20.565 | 3333.333 +/- 79.434 | +0.1% |
| gpu.peak_memory_fraction | 0.407 +/- 0.003 | 0.407 +/- 0.010 | +0.0% |
| gpu.utilization_gpu_percent.max | 86.000 +/- 30.223 | 85.333 +/- 40.161 | -0.8% |
| gpu.utilization_gpu_percent.mean | 32.209 +/- 1.767 | 30.236 +/- 4.293 | -6.1% |
| gpu.utilization_gpu_percent.p95 | 42.333 +/- 1.434 | 40.950 +/- 2.052 | -3.3% |
| interactive.e2e_ms_p50 | 659.276 +/- 45.127 | 666.814 +/- 107.070 | +1.1% |
| interactive.e2e_ms_p95 | 816.249 +/- 104.608 | 744.216 +/- 108.920 | -8.8% |
| interactive.e2e_ms_p99 | 825.683 +/- 114.163 | 744.262 +/- 108.924 | -9.9% |
| interactive.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| interactive.tpot_ms_p50 | 34.056 +/- 1.491 | 32.925 +/- 3.975 | -3.3% |
| interactive.tpot_ms_p95 | 35.008 +/- 1.611 | 37.398 +/- 7.640 | +6.8% |
| interactive.tpot_ms_p99 | 35.098 +/- 1.599 | 37.703 +/- 7.883 | +7.4% |
| interactive.ttft_ms_p50 | 417.240 +/- 45.283 | 430.187 +/- 83.585 | +3.1% |
| interactive.ttft_ms_p95 | 608.623 +/- 104.182 | 529.961 +/- 101.586 | -12.9% |
| interactive.ttft_ms_p99 | 622.969 +/- 114.196 | 530.006 +/- 101.589 | -14.9% |
| kv.compression_dropped_blocks | 0.000 +/- 0.000 | 4.000 +/- 0.000 | n/a |
| kv.compression_dropped_tokens | 0.000 +/- 0.000 | 1024.000 +/- 0.000 | n/a |
| kv.compression_events | 0.000 +/- 0.000 | 4.000 +/- 0.000 | n/a |
| kv.forced_fallbacks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| kv.invalidated_tokens | 1280.000 +/- 0.000 | 1280.000 +/- 0.000 | +0.0% |
| kv.preemptions | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| kv.reclaim_events | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| kv.reclaimed_blocks | 4.000 +/- 0.000 | 4.000 +/- 0.000 | +0.0% |
| kv.recomputed_tokens | 1024.000 +/- 0.000 | 1024.000 +/- 0.000 | +0.0% |
| kv.retained_blocks | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| overall.output_throughput_tokens_per_s | 66.228 +/- 3.720 | 112.731 +/- 2.937 | +70.2% |
| overall.request_throughput_rps | 0.507 +/- 0.038 | 0.862 +/- 0.011 | +70.1% |
| overall.slo_goodput_rps | 0.507 +/- 0.038 | 0.862 +/- 0.011 | +70.1% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
