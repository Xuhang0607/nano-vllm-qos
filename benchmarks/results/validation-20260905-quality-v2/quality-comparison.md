# KV Compression Quality Comparison

Live synthetic retrieval evaluation; not a general model-quality benchmark.

Equal retained page budget: True (pages: 5).

| Policy | Accuracy | KV token drop | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: |
| none | 162/216 (75.0%) | 0.0% | 155.6 ms | 528.7 ms |
| sink_recent | 58/216 (26.9%) | 58.3% | 142.5 ms | 547.6 ms |
| query_aware | 152/216 (70.4%) | 58.3% | 140.2 ms | 483.9 ms |

## Task Breakdown

| Policy | Task | Correct / Cases |
| --- | --- | ---: |
| none | distractors | 41/54 |
| none | literal | 53/54 |
| none | paraphrase | 16/54 |
| none | two_hop | 52/54 |
| sink_recent | distractors | 16/54 |
| sink_recent | literal | 18/54 |
| sink_recent | paraphrase | 7/54 |
| sink_recent | two_hop | 17/54 |
| query_aware | distractors | 41/54 |
| query_aware | literal | 52/54 |
| query_aware | paraphrase | 7/54 |
| query_aware | two_hop | 52/54 |

## Paired Against Uncompressed

| Policy | Baseline correct | Lost correct | Gained correct |
| --- | ---: | ---: | ---: |
| sink_recent | 162 | 105 | 1 |
| query_aware | 162 | 14 | 4 |

Sampling is stochastic; paired changes are descriptive, not proof of causality.
