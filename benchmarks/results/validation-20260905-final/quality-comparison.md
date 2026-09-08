# KV Compression Quality Comparison

Live synthetic retrieval evaluation; not a general model-quality benchmark.

Equal retained page budget: True (pages: 5).

| Policy | Accuracy | KV token drop | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: |
| none | 158/216 (73.1%) | 0.0% | 235.9 ms | 804.2 ms |
| sink_recent | 59/216 (27.3%) | 58.3% | 220.0 ms | 828.9 ms |
| query_aware | 153/216 (70.8%) | 58.3% | 220.2 ms | 722.0 ms |

## Task Breakdown

| Policy | Task | Correct / Cases |
| --- | --- | ---: |
| none | distractors | 40/54 |
| none | literal | 54/54 |
| none | paraphrase | 15/54 |
| none | two_hop | 49/54 |
| sink_recent | distractors | 17/54 |
| sink_recent | literal | 18/54 |
| sink_recent | paraphrase | 8/54 |
| sink_recent | two_hop | 16/54 |
| query_aware | distractors | 40/54 |
| query_aware | literal | 54/54 |
| query_aware | paraphrase | 7/54 |
| query_aware | two_hop | 52/54 |

## Paired Against Uncompressed

| Policy | Baseline correct | Lost correct | Gained correct |
| --- | ---: | ---: | ---: |
| sink_recent | 158 | 102 | 3 |
| query_aware | 158 | 13 | 8 |

Sampling is stochastic; paired changes are descriptive, not proof of causality.
