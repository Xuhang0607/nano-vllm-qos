# Repeated page-admission pressure test

Six trials: H=0/H=32, three repetitions each, alternating order. Same frozen
Python source, seed 20260907, uniform 3 requests/s, 120-second arrival window,
361 requests/trial, 64 KV pages, PALS/Radix, eager Qwen3-0.6B on RTX 4060 8GB.
No trace, compression, Mooncake or static resident limit. This is a short
pressure test, not a long-duration stability or peak-capacity claim.

2,166 requests offered, 2,165 succeeded. Candidate trial 3 contains one
connection timeout (`batch-120`, Errno 110). It was retained, not replaced.
The server log has no corresponding exception stack; the cause is unproven.

| Three-run mean | H=0 | H=32 |
| --- | ---: | ---: |
| Reclaims | 17.33 | 12.67 |
| Recomputed tokens | 19696.67 | 14176.33 |
| Interactive TTFT P95 (s) | 39.54 | 41.72 |
| Interactive E2E P95 (s) | 42.16 | 44.35 |
| Batch E2E P95 (s) | 63.44 | 69.84 |
| Output tokens/s | 187.86 | 181.68 |
| SLO goodput (requests/s) | 0.390 | 0.417 |

Lower recomputation did not translate into better latency/throughput. The
goodput mean increased 6.9%, but the 95% Student-t half-widths are about
0.26 and 0.12 req/s, with direction reversal between repetitions. This is
not evidence of a stable optimization. Keep the candidate disabled by default.
Candidate trials recorded 58,369 deferred checks, not that many requests.

See `admission-3-comparison.md` for all means and intervals, and JSON/CSV
files for all request results. Percentiles with about 180 class samples per
trial are exploratory, especially P99.

The scheduler's partial-prefill/self-preemption progress fixes were made
AFTER these runs. The manifest describes the benchmark version, not those
later fixes. A separate GPU smoke validates execution of the latest version;
the full six-trial benchmark was not rerun after the progress fixes.
