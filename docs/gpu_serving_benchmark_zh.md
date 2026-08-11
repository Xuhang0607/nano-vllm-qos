# 真实 GPU Serving Benchmark 与 Ablation

## 1. 为什么需要这套实验

控制面模拟适合验证 Scheduler 决策和成本模型，但不能回答真实 GPU 上的延迟与吞吐。
本实验直接调用 nano-vLLM 的 OpenAI 兼容 API，并使用引擎内部时间戳统计指标，避免把
浏览器渲染、网络抖动或客户端线程调度误算成模型执行时间。

每轮实验同时保存：

- TTFT、TPOT、E2E 和 Queue Latency 的 p50/p95/p99；
- 请求吞吐、输出 Token 吞吐、SLO 达成率和 SLO Goodput；
- Prefix Cache 查询块、命中块、淘汰块和本轮命中率；
- Mooncake GET/PUT 次数、传输字节、恢复 Token 和失败数；
- 每个请求的类别、优先级、到达时间、Token 数、抢占次数和 SLO 结果。

## 2. 已验证环境

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GiB
- 模型：Qwen3-0.6B，BF16
- 系统：Windows 11 + WSL2 Ubuntu-22.04
- CUDA 主路径：Triton + FlashAttention
- KV Page：256 Token
- 调度实验：4 个 Batch 请求、4 个 Interactive 请求
- Batch 输出上限：64 Token
- Interactive 输出上限：4 Token
- Interactive 到达间隔：100 ms
- Prefix：32 段固定文本，实验前执行一次 Warmup

调度实验使用 `max_num_seqs=1`，目的是让不同策略对同一 GPU 执行槽的选择可观察。
这不是正常聊天服务的推荐配置；正常服务保持默认 `64` 以启用 Continuous Batching。

## 3. 运行 FCFS 基线

先保持 Mooncake 关闭，避免远端恢复成为额外变量：

```bash
cd /mnt/d/nano-vllm-qos
SCHEDULING_POLICY=fcfs \
PREFIX_CACHE_BACKEND=radix \
MAX_NUM_SEQS=1 \
ENABLE_MOONCAKE=0 \
bash scripts/run_nanovllm_wsl.sh
```

在另一个终端运行：

```bash
python -m benchmarks.benchmark_serving_gpu \
  --hardware "NVIDIA GeForce RTX 4060 Laptop GPU 8GB" \
  --workload-id rtx4060-scheduler-v1 \
  --batch-requests 4 \
  --interactive-requests 4 \
  --shared-prefix-repeats 32 \
  --batch-output-tokens 64 \
  --interactive-output-tokens 4 \
  --output-json benchmarks/results/serving_gpu_fcfs_radix_local_scheduler.json
```

## 4. 运行 PALS 候选策略

停止 FCFS Worker，只改变调度策略：

```bash
SCHEDULING_POLICY=pals \
PREFIX_CACHE_BACKEND=radix \
MAX_NUM_SEQS=1 \
ENABLE_MOONCAKE=0 \
bash scripts/run_nanovllm_wsl.sh
```

再次运行完全相同的 Benchmark 命令，仅修改输出文件名：

```bash
python -m benchmarks.benchmark_serving_gpu \
  --hardware "NVIDIA GeForce RTX 4060 Laptop GPU 8GB" \
  --workload-id rtx4060-scheduler-v1 \
  --batch-requests 4 \
  --interactive-requests 4 \
  --shared-prefix-repeats 32 \
  --batch-output-tokens 64 \
  --interactive-output-tokens 4 \
  --output-json benchmarks/results/serving_gpu_pals_radix_local_scheduler.json
```

`workload-id` 必须保持一致，否则两组 Prompt 不同，不能构成控制变量实验。

## 5. 自动生成对比报告

```bash
python -m benchmarks.compare_serving_runs \
  --baseline benchmarks/results/serving_gpu_fcfs_radix_local_scheduler.json \
  --candidate benchmarks/results/serving_gpu_pals_radix_local_scheduler.json \
  --output-json benchmarks/results/serving_gpu_scheduler_comparison.json \
  --output-markdown benchmarks/results/serving_gpu_scheduler_comparison.md
```

当前小样本结果：

| 指标 | FCFS | PALS | 变化 |
| --- | ---: | ---: | ---: |
| 请求吞吐 | 0.783 req/s | 0.780 req/s | -0.4% |
| SLO Goodput | 0.392 req/s | 0.780 req/s | +99.2% |
| Interactive TTFT p95 | 76.52 ms | 504.78 ms | +559.7% |
| Interactive E2E p95 | 8963.88 ms | 591.08 ms | -93.4% |
| Batch TTFT p95 | 143.85 ms | 2731.30 ms | +1798.7% |
| Batch E2E p95 | 8678.68 ms | 9375.08 ms | +8.0% |

FCFS 会较早完成各请求 Prefill，因此 Interactive TTFT 较低；但在单执行槽下，后续
Decode 长时间被 Batch 请求占用，Interactive E2E SLO 全部失败。PALS 重新分配 Decode
机会，使 Interactive E2E 显著下降并达到 SLO，代价是 Batch 和首 Token 延迟增加。
这说明优化目标不是让每个指标同时变小，而是在几乎不损失吞吐的情况下提高关键请求
的 SLO Goodput。

## 6. Prefix Cache 与 Mooncake Ablation

后续完整矩阵应逐项改变一个变量：

| 组别 | Policy | Prefix | Remote KV | 目的 |
| --- | --- | --- | --- | --- |
| A | FCFS | Radix | 关闭 | 调度基线 |
| B | PALS | Radix | 关闭 | 调度 Ablation |
| C | PALS | Hash | 关闭 | Hash/Radix 对照 |
| D | PALS | Radix | Mooncake | Remote KV 对照 |

测量远端恢复时使用 `--no-warmup`，先在 D 组写回固定 Prefix，只重启 GPU Worker，再用
相同 `workload-id` 发起请求。否则本地 Warmup 会先建立 GPU Cache，掩盖 Mooncake Restore。

## 7. 如何形成可信简历结论

当前结果证明 Benchmark 链路和调度趋势可复现，但请求数仍少。正式报告应满足：

1. 每组至少重复 5 到 10 次，报告均值、标准差或置信区间；
2. 在低、中、高三个请求到达率下重复；
3. 固定模型、GPU、输出长度、Prompt 分布、Warmup 和缓存初始状态；
4. 同时报告收益与代价，不只选择改善的指标；
5. 将模拟结果、单机 TCP 结果和未来 RDMA/多机结果明确分开。

在完成多轮实验前，简历可写“构建了可复现的 GPU Ablation 框架，并在单机小样本中
观察到 PALS 提升交互请求 SLO Goodput”，不要把当前百分比描述成普遍性能保证。
