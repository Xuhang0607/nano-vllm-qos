# Metrics Summary Ablation

Six GPU trials completed, three per summary mode. All 3,606 requests succeeded.
The models, workloads, scheduling policy, KV capacity, and compression settings
were held fixed. Only completed-history summary reuse was intentionally changed.

The cached mode reduced mean historical-summary compute time by 92.6%, but did
not improve mean end-to-end throughput. Its third trial experienced 13 KV
preemptions and higher latency; it is retained in the aggregate, not excluded.
This suite does not establish a sustained capacity increase or fix for earlier
overload timeouts. All measured throughput includes the arrival window and drain.

- [Full analysis and reproduction](../../../docs/metrics_summary_optimization_zh.md)
- [Three-run comparison](metrics-2-comparison.md)
- Per-trial JSON, request CSV, logs, source manifest and final status are retained.
