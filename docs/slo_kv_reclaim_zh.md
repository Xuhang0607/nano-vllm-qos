# SLO 感知 KV Cache 回收设计

## 1. 问题

原始 nano-vLLM 在显存 KV Block 不足时会抢占一个运行请求，并释放它拥有的全部
KV Block。请求恢复后，如果这些 Block 已被其他请求覆盖，就必须重新 Prefill 已经计算过的
Token。

PALS 已经能够根据 TTFT、TPOT、E2E SLO、请求优先级和等待老化选择抢占对象，但原来的
释放动作只有一种：全部释放。当前里程碑进一步回答：

> 抢占请求后应该释放多少 KV，才能回收足够显存，同时减少紧急请求的重复计算？

## 2. 当前实现

`SLOAwareKVReclaimPolicy` 将 PALS 的剩余延迟预算转换为 `[0, 1]` 的紧迫度：

```text
budget <= 0: urgency = 1
budget > 0:  urgency = 1 / (1 + budget / budget_scale)

keep_ratio = min_keep_ratio
             + (max_keep_ratio - min_keep_ratio) * urgency
```

策略只保留已经计算完成、连续且按 Page 对齐的 KV 前缀。其余后缀 Block 被释放，恢复时由
`BlockManager.resume()` 先重新查询 Radix Cache，接回请求等待期间可复用的后缀，再补齐
物理 Block 并从最终命中边界继续 Prefill。因此当前实现不改变 Position ID，也不会近似
删除参与 Attention 的历史 Token，生成语义保持不变。

缓存压力同样参与决策。每次抢占至少释放足够 Block，使空闲 Block 达到
`kv_reclaim_target_free_blocks`。如果多个 Waiting 请求保留前缀后造成恢复阻塞，Scheduler
会从最不紧急的 Waiting 请求开始执行完整回收，保证系统能够继续运行。

## 3. 配置

```bash
python -m nanovllm.serve \
  --model /path/to/Qwen3-0.6B \
  --scheduling-policy pals \
  --kv-reclaim-policy slo_aware \
  --kv-reclaim-min-keep-ratio 0.0 \
  --kv-reclaim-max-keep-ratio 0.75 \
  --kv-reclaim-budget-scale-ms 1000 \
  --kv-reclaim-target-free-blocks 2
```

`--kv-reclaim-policy recompute` 保留原来全部释放的行为，作为实验基线。

## 4. 可观测指标

- `kv_reclaim_events`：KV 回收决策次数；
- `kv_reclaim_requested_blocks`：策略计划释放的 Block 数；
- `kv_reclaim_freed_blocks`：考虑共享引用后实际释放的 Block 数；
- `kv_reclaim_retained_blocks`：显式保留的完整前缀 Block 数；
- `kv_reclaim_invalidated_tokens`：因回收而失效的 KV Token 数；
- `kv_recomputed_tokens`：恢复阶段实际重新 Prefill 的 Token 数；
- `kv_reclaim_forced_fallbacks`：为保证进度而执行完整回收的次数。

## 5. 当前验证结果

运行：

```bash
python -m benchmarks.benchmark_kv_reclaim \
  --output-json benchmarks/results/kv_reclaim_simulation.json
```

确定性控制面实验包含两个部分：

1. 小容量 Scheduler 压力实验中，`slo_aware` 将失效 Token 从 12 降至 8；由于 Hash
   Prefix Cache 恰好复用了部分已释放 Block，两种策略最终都重算 8 Token，模拟 Makespan
   均为 20.3 ms。
2. 加入中间 Block 覆盖的隔离实验后，`recompute` 前缀存活 0 Token，需要重算 12 Token；
   `slo_aware` 显式保留 4 Token，需要重算 8 Token。

真实 GPU 对照使用显式小容量 KV Pool 构造压力：

```bash
REPEATS=3 bash scripts/run_gpu_ablation_wsl.sh kv-reclaim
```

RTX 4060 + Qwen3-0.6B 三轮均值显示：失效 Token 降低 35.0%，实际重计算 Token 降低
14.3%；但请求吞吐降低 7.3%，Interactive E2E p95 增加 15.7%。这说明当前策略减少了
重计算，却没有转化为端到端加速。恢复调度、短 Prefill 和小样本波动都可能抵消收益，后续
需要 Kernel 级分解和更长上下文验证。原始聚合报告为
`benchmarks/results/repeated/kv-recompute-vs-slo-aware.md`。

## 6. 与近似 KV 压缩的边界

本文件描述的是正确性优先的“连续前缀保留 + 后缀重算”，不是有损 KV 压缩。项目已经在
独立路径实现 Query-Aware 近似压缩，包括逻辑/物理长度分离、Sink/Query/Recent 选页、
抢占回退和 9 样本 Needle 质量探针。两条路径不能混为一谈：

- `slo_aware`：完整历史最终都会参与 Attention，语义无损，但需要重计算后缀；
- `query_aware`：删除部分物理历史页以减少 Attention 成本，需要任务级质量验证。

详见 [Query-Aware 近似 KV 压缩](query_aware_kv_compression_zh.md)。
