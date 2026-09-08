# Completed Repeated Serving Validation

Twelve isolated GPU trials completed: two arrival rates (2 and 3 req/s), two
compression policies (`none` and `query_aware`), three repetitions each.
All trials use PALS and Radix. This is not an upstream nano-vLLM comparison.

- Model: local Qwen3-0.6B BF16; GPU: RTX 4060 8GB; WSL/Linux.
- Arrivals: fixed-interval, shuffled class mix, seed 20260906.
- Offered: 9,012; completed: 8,797; failed: 215 timeouts.
- Client-cap rejections: zero; client concurrency cap: 512.
- The 2 req/s mean output-throughput observation is +4.8%, versus -7.7% at 3 req/s.
  Between-run variability is large. These results do not establish stable acceleration.
- The non-streaming HTTP benchmark uses engine-side TTFT/TPOT and separately
  measured client E2E. Failed requests are included in offered-SLO denominators.

Each JSON has request-level records and configuration; CSV files provide tabular
records; server logs and the source/configuration manifest are retained. No failed
trial was replaced with a successful rerun. The status file records suite completion,
not a claim that all requests succeeded.

See [the full report](../../../docs/final_validation_zh.md),
[2 req/s aggregation](serving-2-comparison.md), and
[3 req/s aggregation](serving-3-comparison.md).
