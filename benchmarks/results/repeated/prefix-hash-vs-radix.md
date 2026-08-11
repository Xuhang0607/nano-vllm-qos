# Repeated Live Serving Ablation

Baseline: **Hash** (3 runs)

Candidate: **Radix** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.e2e_ms_p50 | 563.697 +/- 22.603 | 612.071 +/- 100.205 | +8.6% |
| batch.e2e_ms_p95 | 568.404 +/- 23.400 | 614.809 +/- 98.495 | +8.2% |
| batch.e2e_ms_p99 | 568.780 +/- 23.680 | 615.055 +/- 98.373 | +8.1% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 44.511 +/- 7.988 | 39.421 +/- 9.074 | -11.4% |
| batch.tpot_ms_p95 | 44.511 +/- 7.988 | 45.314 +/- 6.128 | +1.8% |
| batch.tpot_ms_p99 | 44.511 +/- 7.988 | 45.314 +/- 6.128 | +1.8% |
| batch.ttft_ms_p50 | 252.123 +/- 39.795 | 336.127 +/- 131.248 | +33.3% |
| batch.ttft_ms_p95 | 256.830 +/- 36.502 | 338.864 +/- 129.420 | +31.9% |
| batch.ttft_ms_p99 | 257.206 +/- 35.961 | 339.110 +/- 129.316 | +31.8% |
| cache.prefix_cache_block_hit_rate | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| overall.output_throughput_tokens_per_s | 79.453 +/- 7.060 | 59.509 +/- 35.803 | -25.1% |
| overall.request_throughput_rps | 9.932 +/- 0.882 | 7.439 +/- 4.475 | -25.1% |
| overall.slo_goodput_rps | 9.932 +/- 0.882 | 7.439 +/- 4.475 | -25.1% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
