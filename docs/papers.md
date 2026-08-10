# Paper Map

This extension combines a small, testable subset of ideas from recent LLM
serving research. Paper-reported improvements are not results of this project.

| Work | Venue / version | Idea used in this project | Not implemented here |
| --- | --- | --- | --- |
| [Cascade](https://arxiv.org/abs/2608.06557) | arXiv v1, 2026-08-06 | Per-request latency budget; least-budget-first scheduling; largest-budget preemption | Hierarchical KV-cache placement and production-trace evaluation |
| [ProServe](https://arxiv.org/abs/2512.12928) | arXiv v2, 2026-06-12 | Client priority and weighted service gain | Multi-instance routing, asynchronous KV offload, SlideBatching |
| [SOLA](https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html) | MLSys 2025 | State-aware, iteration-level prefill/decode decisions | SOLA's complete optimization formulation |
| [QoServe / Niyama](https://www.microsoft.com/en-us/research/publication/niyama-breaking-the-silos-of-llm-inference-serving/) | ASPLOS 2026 | Fine-grained QoS classes and fairness under shared serving | Dynamic chunk sizing and selective relegation |
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | OSDI 2024 | Uses the chunked-prefill foundation already present upstream | Pipeline-parallel stall-free batching |
| [FastServe](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang) | NSDI 2026 | Motivation for iteration-level preemption | Distributed skip-join MLFQ design |
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104) | NeurIPS 2024 | Explicit radix-tree prefix reuse and cache-aware scheduling | Full SGLang runtime and kernel stack |
| [Mooncake](https://arxiv.org/abs/2407.00079) | FAST 2025 Best Paper / arXiv v4 | Tiered KV catalog, Store adapter and transfer-aware planning | Real GPU tensor restore, PD clusters and RDMA validation |

## Project-specific synthesis

PALS uses one explainable score to combine phase-specific SLO headroom,
prefix-cache-aware remaining work, client priority, and anti-starvation aging.
The hierarchical extension adds a page-aligned radix index and chooses remote
restore or recomputation from predicted total cost. FCFS, hash prefix caching,
and static remote-restore policies remain available as compatibility and
ablation baselines.
