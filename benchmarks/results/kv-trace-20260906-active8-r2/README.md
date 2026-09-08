# Resident-cap diagnostic, 2026-09-06

Same workload as `../kv-trace-20260906-r2`, with `--max-active-seqs 8`.
601/601 requests succeeded; 57,664 output tokens; zero reclaim events and
zero recomputed tokens. The trace contains only its header because no
configured incident triggered capture.

This candidate is NOT a demonstrated optimization. Interactive TTFT P95
was 4,850.22 ms versus 304.72 ms in the earlier diagnostic. Interactive
E2E P95 was 7,657.08 ms versus 3,183.51 ms. SLO goodput was 1.213 req/s
versus 1.943 req/s. The optional cap remains disabled by default.

Both runs are single, instrumented diagnostics with separate source manifests,
not a repeated uninstrumented speedup experiment. The earlier baseline also
hit the trace byte limit; instrumentation overhead is not identical.
Do not cherry-pick zero preemptions while omitting queueing/latency costs.
