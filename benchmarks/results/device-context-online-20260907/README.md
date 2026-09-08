# CPU device-context online ablation

See [protocol and interpretation](../../../docs/device_context_online_zh.md).

- Same runtime source, legacy CPU device dispatch enabled vs scoped initialization.
- RTX 4060, Qwen3-0.6B, eager, PALS/Radix, 64 KV pages; compression/Mooncake/reservation disabled.
- Three alternating runs per mode, 181 requests/run, uniform 3 requests/s for 60 seconds.
- 1,086 successful requests, zero failed; 17,344 output tokens/run; identical workload fingerprints.
- Mean output throughput: 183.49 to 222.28 tokens/s (+21.1%).
- Mean per-run interactive E2E P95: 13.33 to 2.82 seconds (-78.8%).
- Raw request records, server logs, metadata, source hashes and confidence intervals are retained.

This is a short single-load ablation, not a sustained-capacity or upstream speedup claim.
Candidate completion rate is still below the offered rate. Do not promise a permanent SLO attainment rate.
