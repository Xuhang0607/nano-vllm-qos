# nano-vLLM PALS 调度器设计说明

## 1. 项目目标

原版 nano-vLLM 已经实现 continuous batching、chunked prefill、prefix cache、
抢占、CUDA Graph 和 Tensor Parallel。这个扩展没有重复实现这些功能，而是研究一个
在线推理服务问题：**当离线长任务和在线短对话共享同一张 GPU 时，如何优先保障重要
请求的 TTFT/E2E SLO，同时避免低优先级请求永久饥饿。**

PALS（Priority-Aware Latency-budget Scheduling）是一项论文启发的工程原型，
不是对某一篇论文的完整复现，也不宣称已经构成新的学术成果。

## 2. 请求与指标

调用者可以通过 `RequestQoS` 声明：

- `priority`：非负整数，数值越大表示业务优先级越高；
- `ttft_slo_ms`：Time To First Token 的目标；
- `tpot_slo_ms`：Time Per Output Token 的目标；
- `e2e_slo_ms`：从到达到生成结束的目标；
- `request_class`：用于按 interactive、batch 等类别聚合指标。

每个请求记录 arrival、first scheduled、first token、finish 时间以及抢占次数，输出：

- queue latency、TTFT、TPOT、E2E latency；
- p50/p95 聚合值；
- TTFT/TPOT/E2E SLO 达成率；
- 抢占次数和 EWMA 执行时间估计。

## 3. 核心公式

对请求 `i`，调度器计算调整后的剩余延迟预算：

```text
raw_budget_i = target_slo_i - elapsed_i - predicted_remaining_i

budget_i = raw_budget_i
           - priority_i * priority_boost
           - waiting_steps_i * aging_credit
```

预算越小，请求越紧急。优先级 credit 让高价值请求更早执行；aging credit 随等待
轮数增长，避免 best-effort 请求永久饥饿。

在第一个 token 生成前，存在 TTFT SLO 时使用 TTFT 作为目标，并只估计剩余
prefill；第一个 token 生成后，调度器同时检查下一 token 的 TPOT deadline 和
E2E SLO，并采用更紧急的预算。prefill 的剩余
工作量使用 `num_tokens - num_cached_tokens`，所以 prefix cache 命中会直接增加请求
的延迟余量。

## 4. 调度流程

每一轮调度包含四个决策：

1. 找到 waiting 队列中预算最小的 prefill 请求；
2. 找到 running 队列中预算最小的 decode 请求；
3. 比较二者预算，选择本轮执行 prefill 或 decode；
4. KV cache 不足时，优先抢占预算最大的运行请求。

FCFS 仍然是默认策略，因此原有调用行为不变。设置
`scheduling_policy="pals"` 才启用新策略。

## 5. 在线执行时间估计

模型执行后，调度器根据本轮 token 数和耗时计算 `ms/token` 样本，并用 EWMA
分别更新 prefill、decode 估计：

```text
estimate = alpha * observed + (1 - alpha) * estimate
```

分开估计两个阶段，是因为 prefill 通常是 compute-bound，而 decode 更容易受到
memory bandwidth 和 batch size 影响。当前版本使用简单的 per-token 模型，便于解释；
后续可以升级成按 batch size、context length 分桶的预测器。

## 6. 与论文的关系

- **Cascade（2026）**：借鉴 latency budget/headroom，以及用同一预算选择紧急请求和
  抢占对象；当前已实现分层目录与成本规划，但真实 KV 数据迁移仍待 Linux 验证。
- **ProServe（2025/2026）**：借鉴多优先级服务和 service gain 指标；本项目加入
  priority credit，但未实现多实例路由和异步 KV offload。
- **SOLA（MLSys 2025）**：借鉴按请求状态、系统状态进行细粒度调度；本项目在每一轮
  显式比较 prefill 与 decode 的预算。
- **QoServe/Niyama（ASPLOS 2026）**：借鉴共享基础设施上的细粒度 QoS 和公平性；
  本项目用 aging 处理饥饿，复用 nano-vLLM 已有的 chunked prefill。
- **Sarathi-Serve（OSDI 2024）**：其 chunked prefill 思想已存在于上游 nano-vLLM，
  本项目在此基础上决定哪一个请求获得 chunk。

论文链接集中在 [papers.md](papers.md)。

## 7. 可复现实验

```bash
python -m pytest -q
python -m benchmarks.benchmark_qos_scheduler \
  --output-json benchmarks/results/qos_simulation.json
```

默认 workload 包含 12 个长 prompt 的 batch 请求和 12 个持续到达的 interactive
请求。它调用真实的 `Scheduler` 与 `BlockManager`，只把 GPU 执行替换为确定性的
耗时模型。

| 策略 | 类别 | TTFT p95 | TPOT p95 | E2E p95 | TTFT/TPOT/E2E SLO |
| --- | --- | ---: | ---: | ---: | ---: |
| FCFS | interactive | 221.08 ms | 27.50 ms | 395.02 ms | 25.0% / 0.0% / 16.7% |
| PALS | interactive | 10.61 ms | 1.65 ms | 22.16 ms | 100% / 100% / 100% |
| FCFS | batch | 230.84 ms | 11.86 ms | 401.32 ms | - / - / 100% |
| PALS | batch | 397.72 ms | 9.98 ms | 542.62 ms | - / - / 100% |

PALS 将 interactive TTFT p95 降低 95.2%，但模拟 makespan 从 445.42 ms 增加到
542.62 ms（+21.8%）。这展示的是延迟隔离和总体吞吐之间的取舍。以上全部是
**控制面模拟结果，不是 GPU 性能数据**。

## 8. 当前限制与下一步

1. 当前预测器假设剩余输出长度等于 `max_tokens`，实际系统可接入长度预测模型；
2. 尚未针对 batch size、context length 和 prefix-cache 命中率校准执行时间；
3. 已实现 Radix 元数据、成本规划和 Mooncake adapter，但尚未恢复真实 GPU KV tensor；
4. Windows 环境完成了控制面测试，Triton/FlashAttention 数据面需在 Linux + NVIDIA
   GPU 上运行真实 trace，报告真实 TTFT、TPOT、throughput 和 goodput；
5. 公平性目前使用 aging 保底，可以进一步加入 Jain fairness index 和 overload
   admission control。

## 9. RadixAttention 与 Mooncake 扩展

Mooncake 完整系统涉及远端 KV Cache、RDMA/多级存储和全局调度，不能在单进程
nano-vLLM 中仅靠增加一个依赖完成。当前已经完成：

- page-aligned Radix Tree、最长前缀匹配和 LRU 叶子驱逐；
- GPU/CPU/Mooncake 分层目录；
- transfer-vs-recompute 成本规划；
- Mooncake Store 单页/批量 byte API adapter；
- TCP/RDMA smoke test 脚本。

下一阶段的数据面工作包括：

1. 在 ModelRunner 中打包、写回和恢复真实 GPU KV tensor；
2. 将选中的 restore/recompute 时间加入 PALS 的 `predicted_remaining`；
3. 实现异步 prefetch timeout、失败回退和请求取消；
4. 过载时基于“最优可行路径仍无法满足 SLO”执行可解释的早拒绝；
5. 用 Mooncake 官方 trace 回放，并在 Linux 节点验证真实 Store/RDMA 数据面。

实验应至少比较：local-only FCFS、local-only PALS、固定 remote restore、
transfer-vs-recompute PALS。报告 SLO goodput、TTFT/TPOT、传输字节、remote hit rate、
recompute tokens 和 rejection rate。这样能回答“何时值得传 KV、何时重算、网络拥塞
如何改变调度决策”，比单纯调用 Mooncake API 更能体现系统思考。

详细设计和实验见 [hierarchical_radix_mooncake_zh.md](hierarchical_radix_mooncake_zh.md)。

面试中应把它描述为“基于最新调度论文思想完成的可运行二次开发与实验框架”，而不是
“复现了 Cascade/ProServe”或“提出了已被证明的新算法”。
