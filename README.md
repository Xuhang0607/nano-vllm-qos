<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

This fork adds **PALS**, a priority-aware latency-budget scheduler, a
page-aligned **RadixAttention prefix index**, and an experimental hierarchical
KV cache planner with an optional Mooncake Store adapter. The original FCFS and
hash-prefix policies remain available as baselines.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* **QoS scheduling** - Per-request TTFT, TPOT, E2E SLOs and client priorities
* **Radix KV cache** - Longest-prefix matching, canonical concurrent inserts and LRU leaf eviction
* **Tiered cache planning** - Transfer-vs-recompute decisions across GPU, CPU and Mooncake

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## QoS-Aware Scheduling

Each request can declare a client priority and TTFT/TPOT/E2E service-level
objectives. PALS compares the predicted latency headroom of waiting prefill
requests and running decode requests at every scheduling step.

```python
from nanovllm import LLM, RequestQoS, SamplingParams

llm = LLM(
    "/YOUR/MODEL/PATH",
    scheduling_policy="pals",
    prefix_cache_backend="radix",
    enforce_eager=True,
)
outputs = llm.generate(
    ["Summarize this incident report."],
    SamplingParams(temperature=0.6, max_tokens=128),
    request_qos=RequestQoS(
        priority=4,
        ttft_slo_ms=100.0,
        tpot_slo_ms=20.0,
        e2e_slo_ms=1000.0,
        request_class="interactive",
    ),
)

print(outputs[0]["metrics"])
print(llm.get_scheduler_metrics())
```

Run the deterministic control-plane comparison without model weights:

```bash
python -m pip install -r requirements-control.txt
python -m pytest -q
python -m benchmarks.benchmark_qos_scheduler \
  --output-json benchmarks/results/qos_simulation.json
python -m benchmarks.benchmark_tiered_cache \
  --output-json benchmarks/results/tiered_cache_simulation.json
```

The shared-prefix workload reduces simulated interactive TTFT p95 from 221.08 ms
under FCFS to 10.61 ms under PALS while all batch E2E SLOs remain satisfied.
The PALS makespan is 21.8% higher in this workload. In the tiered-cache
simulation, cost-aware restore averages 95.96 ms versus 107.20 ms for local-only
recompute and 168.93 ms for always restoring remote KV. These are simulator results,
not GPU throughput measurements. See [the Chinese design note](docs/qos_scheduler_zh.md)
and [the hierarchical Radix/Mooncake note](docs/hierarchical_radix_mooncake_zh.md)
for the algorithms, experiment interpretation, and current implementation boundary.

## Optional Mooncake Smoke Test

The adapter follows Mooncake Store's byte-object and batch APIs. On Ubuntu,
install the optional dependency, start `mooncake_master`, and run a TCP
round-trip before attempting GPU KV integration:

```bash
python -m pip install -e ".[mooncake]"
mooncake_master
python -m scripts.mooncake_smoke --protocol tcp
```

The adapter and its error handling are covered with a fake store in CPU CI.
The real Mooncake process and GPU tensor data path are not validated on Windows.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
