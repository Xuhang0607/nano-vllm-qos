# Repeated Live Serving Ablation

Baseline: **FCFS** (3 runs)

Candidate: **PALS** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.e2e_ms_p50 | 6076.962 +/- 246.290 | 10124.783 +/- 1200.856 | +66.6% |
| batch.e2e_ms_p95 | 8699.821 +/- 319.622 | 10169.859 +/- 1203.411 | +16.9% |
| batch.e2e_ms_p99 | 8934.783 +/- 318.803 | 10174.032 +/- 1203.843 | +13.9% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 95.556 +/- 4.319 | 117.579 +/- 17.639 | +23.0% |
| batch.tpot_ms_p95 | 136.570 +/- 4.964 | 153.108 +/- 18.572 | +12.1% |
| batch.tpot_ms_p99 | 140.234 +/- 4.932 | 157.745 +/- 18.672 | +12.5% |
| batch.ttft_ms_p50 | 62.865 +/- 17.348 | 2717.301 +/- 207.972 | +4222.4% |
| batch.ttft_ms_p95 | 96.283 +/- 6.195 | 3202.077 +/- 266.236 | +3225.7% |
| batch.ttft_ms_p99 | 100.126 +/- 8.129 | 3244.460 +/- 270.382 | +3140.4% |
| cache.prefix_cache_block_hit_rate | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| gpu.memory_used_mib.max | 6409.000 +/- 88.290 | 6394.000 +/- 25.937 | -0.2% |
| gpu.memory_used_mib.mean | 6391.536 +/- 31.641 | 6391.094 +/- 26.712 | -0.0% |
| gpu.memory_used_mib.p95 | 6405.233 +/- 72.100 | 6393.667 +/- 24.510 | -0.2% |
| gpu.peak_memory_fraction | 0.783 +/- 0.011 | 0.781 +/- 0.003 | -0.2% |
| gpu.utilization_gpu_percent.max | 43.000 +/- 2.484 | 47.667 +/- 19.926 | +10.9% |
| gpu.utilization_gpu_percent.mean | 32.819 +/- 3.661 | 30.814 +/- 2.271 | -6.1% |
| gpu.utilization_gpu_percent.p95 | 41.383 +/- 0.827 | 40.217 +/- 0.932 | -2.8% |
| interactive.e2e_ms_p50 | 8335.787 +/- 286.887 | 627.027 +/- 42.524 | -92.5% |
| interactive.e2e_ms_p95 | 8504.967 +/- 298.118 | 746.343 +/- 68.753 | -91.2% |
| interactive.e2e_ms_p99 | 8521.783 +/- 304.569 | 756.799 +/- 92.124 | -91.1% |
| interactive.slo_attainment | 0.000 +/- 0.000 | 1.000 +/- 0.000 | n/a |
| interactive.tpot_ms_p50 | 2756.087 +/- 95.274 | 30.274 +/- 3.807 | -98.9% |
| interactive.tpot_ms_p95 | 2798.259 +/- 98.787 | 34.358 +/- 8.229 | -98.8% |
| interactive.tpot_ms_p99 | 2801.146 +/- 104.694 | 35.376 +/- 9.346 | -98.7% |
| interactive.ttft_ms_p50 | 53.661 +/- 3.772 | 533.806 +/- 42.960 | +894.8% |
| interactive.ttft_ms_p95 | 145.773 +/- 59.280 | 653.584 +/- 55.351 | +348.4% |
| interactive.ttft_ms_p99 | 154.317 +/- 72.691 | 663.024 +/- 75.874 | +329.7% |
| overall.output_throughput_tokens_per_s | 29.265 +/- 0.885 | 28.244 +/- 3.394 | -3.5% |
| overall.request_throughput_rps | 1.219 +/- 0.037 | 1.177 +/- 0.141 | -3.5% |
| overall.slo_goodput_rps | 0.406 +/- 0.012 | 1.177 +/- 0.141 | +189.5% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
