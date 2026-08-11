# Live Serving Ablation

| Configuration | Policy | Prefix cache | Backend | Max sequences |
| --- | --- | --- | --- | ---: |
| Baseline | fcfs | radix | CUDA | 1 |
| Candidate | pals | radix | CUDA | 1 |

| Scope | Metric | Baseline | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| overall | request_throughput_rps | 0.783 | 0.780 | -0.4% |
| overall | output_throughput_tokens_per_s | 26.627 | 26.518 | -0.4% |
| overall | slo_goodput_rps | 0.392 | 0.780 | +99.2% |
| batch | ttft_ms_p95 | 143.852 | 2731.302 | +1798.7% |
| batch | e2e_ms_p95 | 8678.685 | 9375.084 | +8.0% |
| batch | slo_attainment | 1.000 | 1.000 | +0.0% |
| interactive | ttft_ms_p95 | 76.517 | 504.785 | +559.7% |
| interactive | e2e_ms_p95 | 8963.876 | 591.079 | -93.4% |
| interactive | slo_attainment | 0.000 | 1.000 | n/a |

> Positive latency changes are regressions; positive throughput, goodput,
> and SLO-attainment changes are improvements.
