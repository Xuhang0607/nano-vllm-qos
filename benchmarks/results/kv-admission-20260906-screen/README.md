# Page-reserve screening

One run per strategy, same frozen Python source, Qwen3-0.6B, RTX 4060 8GB,
64 KV pages, PALS/Radix, eager mode, no compression, Mooncake or scheduler
trace. Uniform 2 requests/s, seed 20260906, 300-second arrival window.

Both runs completed 601/601 requests and generated 57,664 output tokens.
The workload fingerprint is `4f001b067b528ab1` in both results.

| Metric | H=0 | H=32 |
| --- | ---: | ---: |
| Deferred admission checks | 0 | 3 |
| Reclaim events | 0 | 0 |
| Interactive TTFT P95 (ms) | 220.95 | 187.48 |
| Interactive E2E P95 (ms) | 2841.26 | 2615.31 |
| Batch E2E P95 (ms) | 10097.26 | 7942.77 |
| Output tokens/s | 189.04 | 189.16 |

Only three candidate checks deferred admission. This single screening pair
does not establish that page reservation caused the latency differences.
No reduction in reclamation was demonstrated because neither run reclaimed
pages. Do not promote these numbers as a stable speedup or capacity result.
The follow-up pressure suite uses a higher offered rate and repeated trials.
