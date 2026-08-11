# PALS Multi-Arrival-Rate GPU Pressure Test

Values are mean +/- 95% Student-t confidence-interval half-width.

| Load | Rate | Policy | Interactive E2E p95 | Interactive SLO | SLO Goodput | GPU util mean | Peak GPU memory |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| low | 2.0 req/s | FCFS | 8545.42 +/- 544.96 ms | 0.00 +/- 0.00% | 0.39 +/- 0.02 req/s | 34.60 +/- 2.24% | 6352.33 +/- 67.64 MiB |
| low | 2.0 req/s | PALS | 297.94 +/- 105.52 ms | 100.00 +/- 0.00% | 1.16 +/- 0.06 req/s | 34.03 +/- 3.25% | 6362.33 +/- 84.52 MiB |
| medium | 5.0 req/s | FCFS | 8504.97 +/- 298.12 ms | 0.00 +/- 0.00% | 0.41 +/- 0.01 req/s | 32.82 +/- 3.66% | 6409.00 +/- 88.29 MiB |
| medium | 5.0 req/s | PALS | 746.34 +/- 68.75 ms | 100.00 +/- 0.00% | 1.18 +/- 0.14 req/s | 30.81 +/- 2.27% | 6394.00 +/- 25.94 MiB |
| high | 20.0 req/s | FCFS | 9695.24 +/- 2133.08 ms | 0.00 +/- 0.00% | 0.40 +/- 0.08 req/s | 34.08 +/- 2.46% | 6430.00 +/- 62.11 MiB |
| high | 20.0 req/s | PALS | 1177.78 +/- 284.00 ms | 100.00 +/- 0.00% | 1.15 +/- 0.03 req/s | 34.40 +/- 3.00% | 6418.33 +/- 7.59 MiB |

The chart reports engine-side latency and goodput. Browser rendering time is excluded.
