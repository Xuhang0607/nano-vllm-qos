# KV incident capture, 2026-09-06

Diagnostic run, not an uninstrumented performance benchmark. Qwen3-0.6B,
RTX 4060 8GB, 64 KV pages, step batch limit 8, PALS/Radix, no compression,
no Mooncake. Fixed seed 20260906, uniform 2 requests/s, 601 offered and
601 successful requests, no failed requests.

Whole-run counters: 5 reclaim events, 25 freed pages, 7,228 invalidated tokens.
The bounded trace contains 3 append-pressure events before hitting its byte
limit. Use the JSON benchmark for totals; the trace is only incident windows.

The first incident follows admission at step 4170: the running queue grows
from 14 to 15 while free pages fall from 2 to 0. At step 4176 request 303
needs its seventh page at sequence length 1,537, causing request 310 to be
preempted. A step batch limit of 8 did not cap KV-resident requests at 8.

`manifest.json` records the source before adding the optional admission cap.
`incident-summary.json` is derived from `metrics-2-cached-1.trace.jsonl`.
Do not infer that all of the earlier run's 13 preemptions had the same cause.
