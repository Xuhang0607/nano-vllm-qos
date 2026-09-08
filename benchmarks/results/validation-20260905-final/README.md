# Quality Results and an Interrupted Serving Diagnostic

The three quality trials completed on the independent seed 20260906. Their
equal-budget comparison is valid for the unchanged KV selection algorithm.

The first serving baseline hit the load generator's 128-request limit and
rejected 126 of 601 arrivals. Client response times were also much larger than
engine times. Investigation identified blocking per-request event waits in the
shared executor. The serving campaign was stopped intentionally.

Serving now uses asynchronous queue notifications, and non-streaming requests
decode only at completion. The repeated campaign in
`../validation-20260905-repeated/` uses the same fixed configuration for both
compression policies, including a 512-request client cap. Do not mix the
interrupted serving trial into that comparison or attribute all differences to
one isolated change.
