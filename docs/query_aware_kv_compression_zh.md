# Query-Aware 近似 KV 压缩设计与评测

## 1. 要解决的问题

nano-vLLM 原始路径会让每个仍在运行的序列保留完整 KV Cache。长上下文或高并发使 GPU
Page 接近耗尽时，引擎只能抢占请求并丢弃全部 KV，之后重新 Prefill。固定的
Sink + Recent Window 虽然能快速释放显存，但容易删除回答所需的中间证据。

本项目增加 `query_aware` 策略：在显存压力出现时，用最后一段用户 Query 近似表示当前
信息需求，从历史 KV Page 中保留更相关的页。目标不是宣称通用无损，而是在明确的质量
预算下减少物理 Attention 上下文和 Page 占用。

## 2. 与上游 nano-vLLM 的区别

上游序列只有一个上下文长度，逻辑 Token 位置、Block Table 和参与 Attention 的 KV 长度
一一对应。本项目增加了两套状态：

- `num_cached_tokens`：逻辑上下文长度，用于 RoPE 位置和生成语义；
- `num_physical_cached_tokens`：当前仍保留的物理 KV 长度，用于 FlashAttention
  `context_lens`；
- `kv_block_logical_indices`：每个物理 Block 对应的原始逻辑页号。

因此压缩后 Token 仍位于原来的位置，Attention 只读取被选中的物理页。这个解耦是该功能
最关键的执行层改动，而不只是 Scheduler 中增加一个开关。

## 3. 页面选择算法

一个可压缩序列的完整页被划分为三组：

1. **Sink**：保留开头 `sink_blocks` 页，维持开头全局信息；
2. **Query-Aware History**：取最后 `query_tokens` 个 Token 作为 Query，对历史候选页按
   Token 重合打分；在候选历史页中出现越少的 Query Token 权重越高，降低常见 Token
   对打分的干扰；
3. **Recent**：保留最后 `recent_blocks` 页，保留局部连续性。

历史页按得分、最近位置和稳定页号排序，最多选择 `importance_blocks` 页。三组去重后按
原逻辑顺序重排 Block Table，多余物理页归还空闲池。

压缩由 `trigger_free_ratio` 控制：只有空闲 Page 比例低于阈值时才触发。一个序列可在
Decode 期间周期性继续压缩，但不会删除未填满的尾页。

## 4. 正确性边界与失败回退

`query_aware` 是近似 Attention，不能把它写成无损 KV Cache。工程上增加了以下边界：

- 被压缩序列停止 Prefix Hash/Radix 发布，避免其他请求复用不完整前缀；
- 被压缩序列停止 Mooncake Write-Back，避免远端持久化近似 KV；
- 若被压缩序列之后遭到抢占，放弃不完整 KV 并从 Token 序列全量重计算；
- 统计压缩事件、释放页和丢弃 Token，便于把收益与近似程度一起报告；
- `none` 和 `sink_recent` 均作为可插拔基线保留。

这一设计优先保证状态不会被误当成完整 KV 使用，但不保证压缩后的模型输出与完整上下文
一致。业务使用需要结合任务级准确率或人工质量指标设定开关。

## 5. 代码落点

| 文件 | 作用 |
| --- | --- |
| `nanovllm/config.py` | 压缩策略、保留页数、Query 长度与压力阈值配置 |
| `nanovllm/engine/sequence.py` | 逻辑/物理长度、逻辑页索引与压缩统计 |
| `nanovllm/engine/block_manager.py` | Sink/Recent 与 Query-Aware 页面选择、Block Table 压缩 |
| `nanovllm/engine/scheduler.py` | 压力触发、候选序列选择、统计与抢占回退 |
| `nanovllm/engine/model_runner.py` | RoPE 逻辑位置和 FlashAttention 物理长度解耦 |
| `nanovllm/engine/llm_engine.py` | 禁止压缩前缀的远端写回 |
| `benchmarks/benchmark_kv_quality.py` | 真实模型 Needle-in-a-Haystack 质量探针 |
| `benchmarks/benchmark_serving_gpu.py` | 实机吞吐、尾延迟和压缩计数采集 |

## 6. 复现实验

性能消融会自动重启服务，固定模型、Block 容量和请求轨迹，只改变压缩策略：

```bash
cd /mnt/d/nano-vllm-qos
REPEATS=3 bash scripts/run_gpu_ablation_wsl.sh kv-compression
```

质量探针固定早/中/晚三个证据位置，每个位置重复三次：

```bash
bash scripts/run_gpu_ablation_wsl.sh kv-quality
```

单独启动 Query-Aware 服务：

```bash
NUM_KVCACHE_BLOCKS=18 \
KV_COMPRESSION_POLICY=query_aware \
KV_COMPRESSION_SINK_BLOCKS=1 \
KV_COMPRESSION_RECENT_BLOCKS=2 \
KV_COMPRESSION_IMPORTANCE_BLOCKS=2 \
KV_COMPRESSION_QUERY_TOKENS=64 \
KV_COMPRESSION_TRIGGER_FREE_RATIO=1.0 \
ENABLE_MOONCAKE=0 bash scripts/run_nanovllm_wsl.sh
```

`NUM_KVCACHE_BLOCKS=18` 和 `TRIGGER_FREE_RATIO=1.0` 是为了在 8GB 显卡上构造可控压力，
不是线上推荐默认值。

## 7. RTX 4060 实机结果

环境为 Qwen3-0.6B BF16、RTX 4060 Laptop GPU 8GB、WSL2。固定压力负载各重复 3 轮，
报告 Student-t 95% 置信区间的原始数据位于
`benchmarks/results/repeated/kv-compression-none-vs-query.md`。

| 指标 | 不压缩 | Query-Aware | 均值变化 |
| --- | ---: | ---: | ---: |
| 请求吞吐 | 0.507 req/s | 0.862 req/s | +70.1% |
| 输出吞吐 | 66.228 token/s | 112.731 token/s | +70.2% |
| Batch E2E p95 | 14660.0 ms | 9170.9 ms | -37.4% |
| Interactive E2E p95 | 816.2 ms | 744.2 ms | -8.8% |
| Interactive TTFT p95 | 608.6 ms | 530.0 ms | -12.9% |
| SLO 达成率 | 100% | 100% | 不变 |

每轮候选策略发生 4 次压缩，共释放 4 页、丢弃 1024 个物理 KV Token。该负载中两组的
抢占和重计算次数相同，因此吞吐差异主要来自压缩后的 Attention 上下文，而不是减少抢占。

## 8. 质量结果和可信表述

在关闭思考、64 Token 输出预算并只校验最终答案区的 9 个 Needle-in-a-Haystack 样本中：

| 策略 | 准确率 | 累计 KV Token 丢弃比例 | 平均 TTFT | 平均 E2E |
| --- | ---: | ---: | ---: | ---: |
| none | 9/9 | 0.0% | 202.4 ms | 776.2 ms |
| sink_recent | 0/9 | 67.6% | 205.4 ms | 630.2 ms |
| query_aware | 9/9 | 40.5% | 201.3 ms | 801.5 ms |

可以说：**在 9 样本定向 Needle 测试中保持 100% 命中，并在固定压力负载中提升 70.1%
请求吞吐。** 不能说：**压缩 40.5% KV 且模型质量无损。** 后者需要 LongBench、真实对话、
多模型和更大样本验证。

## 9. 面试可讨论的设计取舍

- 为什么保留逻辑位置，而不是压缩后重新编号 RoPE？
- 为什么固定 Sink + Recent 在中间证据任务上失败？
- Token ID 重合为什么只是低成本代理，下一步如何换成 Attention Score 或 Hidden-State
  相似度？
- 为什么压缩序列被抢占后选择全量重计算，而不是恢复剩余页？
- 70.1% 吞吐提升是否来自更少抢占？如何用计数器证明不是？
- 如何把单请求质量探针扩展为 LongBench 和质量约束下的 Pareto 曲线？

这些问题对应实际源码、失败实验和边界条件，比只描述“接入了 KV Cache 压缩”更适合面试深挖。
