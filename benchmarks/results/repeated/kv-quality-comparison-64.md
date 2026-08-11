# KV Compression Quality Comparison

Live Qwen3-0.6B Needle-in-a-Haystack evaluation.

| Policy | Accuracy | KV token drop | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: |
| none | 9/9 (100.0%) | 0.0% | 202.4 ms | 776.2 ms |
| sink_recent | 0/9 (0.0%) | 67.6% | 205.4 ms | 630.2 ms |
| query_aware | 9/9 (100.0%) | 40.5% | 201.3 ms | 801.5 ms |
