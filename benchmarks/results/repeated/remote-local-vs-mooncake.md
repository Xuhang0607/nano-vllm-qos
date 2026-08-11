# Repeated Live Serving Ablation

Baseline: **Local** (3 runs)

Candidate: **Mooncake** (3 runs)

Values are mean +/- 95% Student-t confidence-interval half-width.

| Metric | Baseline | Candidate | Mean change |
| --- | ---: | ---: | ---: |
| batch.e2e_ms_p50 | 1999.753 +/- 291.177 | 3007.438 +/- 711.927 | +50.4% |
| batch.e2e_ms_p95 | 1999.753 +/- 291.177 | 3007.438 +/- 711.927 | +50.4% |
| batch.e2e_ms_p99 | 1999.753 +/- 291.177 | 3007.438 +/- 711.927 | +50.4% |
| batch.slo_attainment | 1.000 +/- 0.000 | 1.000 +/- 0.000 | +0.0% |
| batch.tpot_ms_p50 | 307.916 +/- 72.537 | 326.794 +/- 169.586 | +6.1% |
| batch.tpot_ms_p95 | 307.916 +/- 72.537 | 326.794 +/- 169.586 | +6.1% |
| batch.tpot_ms_p99 | 307.916 +/- 72.537 | 326.794 +/- 169.586 | +6.1% |
| batch.ttft_ms_p50 | 1076.003 +/- 75.359 | 2027.055 +/- 284.708 | +88.4% |
| batch.ttft_ms_p95 | 1076.003 +/- 75.359 | 2027.055 +/- 284.708 | +88.4% |
| batch.ttft_ms_p99 | 1076.003 +/- 75.359 | 2027.055 +/- 284.708 | +88.4% |
| cache.prefix_cache_block_hit_rate | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| overall.output_throughput_tokens_per_s | 1.976 +/- 0.291 | 1.320 +/- 0.288 | -33.2% |
| overall.request_throughput_rps | 0.494 +/- 0.073 | 0.330 +/- 0.072 | -33.2% |
| overall.slo_goodput_rps | 0.494 +/- 0.073 | 0.330 +/- 0.072 | -33.2% |
| remote.backend_get_bytes | 0.000 +/- 0.000 | 117441336.000 +/- 0.000 | n/a |
| remote.backend_put_bytes | 0.000 +/- 0.000 | 0.000 +/- 0.000 | n/a |
| remote.kv_restored_tokens | 0.000 +/- 0.000 | 1024.000 +/- 0.000 | n/a |
| remote.restore_completed | 0.000 +/- 0.000 | 1.000 +/- 0.000 | n/a |
| remote.restore_started | 0.000 +/- 0.000 | 1.000 +/- 0.000 | n/a |
| remote.transfer_bytes | 0.000 +/- 0.000 | 117441336.000 +/- 0.000 | n/a |
