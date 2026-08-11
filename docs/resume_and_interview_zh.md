# nano-vLLM-QoS 简历与面试指南

## 1. 项目名称

**nano-vLLM-QoS：面向 SLO 的大模型推理调度与分层 KV Cache 系统**

技术栈：Python、PyTorch、CUDA、Triton、FlashAttention、FastAPI、Radix Tree、Paged KV
Cache、Mooncake、WSL2、Qwen3-0.6B。

## 2. 推荐的简历描述

下面这版适合放在一页中文技术简历中：

> **nano-vLLM-QoS：大模型推理引擎二次开发**  |  个人项目
>
> - 基于 nano-vLLM 二次开发 SLO 感知调度器 PALS，引入请求优先级、TTFT/TPOT/E2E
>   延迟预算、老化和执行时间 EWMA 估计，并接入 OpenAI 兼容流式对话服务。
> - 在 RTX 4060 8GB + Qwen3-0.6B 上构建 2/5/20 req/s 多到达率 GPU 压力实验；相比
>   FCFS，PALS 将 Interactive E2E p95 降低 87.9% 到 96.5%，SLO 达成率由 0% 提升至
>   100%，总吞吐变化控制在 3.5% 内。
> - 实现 Page-Aligned Radix Prefix Cache、Paged KV Cache 与 Mooncake Remote KV
>   Write-Back/Restore；定位固定 Catalog Key 使用 Insert 导致跨 Worker 读取旧快照的问题，
>   改为 Upsert 后连续 3 轮稳定恢复 1024 Token、读取约 112 MiB KV 数据。
> - 构建可复现 Benchmark 框架，使用工作负载 SHA-256 指纹、逐请求 CSV、GPU 利用率/显存
>   采样与 Student-t 95% 置信区间，自动生成 FCFS/PALS、Hash/Radix、Local/Mooncake
>   Ablation 报告和性能曲线。

## 3. 为什么这不是简单复现

复现上游项目通常只能说明“模型跑起来了”。这个项目新增了可以定位到源码、测试和实机
数据的系统能力：

1. 修改 Scheduler 决策目标，从到达顺序扩展为 Priority、Aging 和 SLO Slack 联合排序；
2. 保留 FCFS 和 Hash Cache 作为基线，能够进行控制变量实验；
3. 新增真实 KV Page 序列化、远端写回、跨 Worker Catalog 重建与恢复状态机；
4. 新增 OpenAI API、对话前端、请求取消、会话删除和运行指标；
5. 不只展示成功案例，还报告 PALS 的 Batch 延迟代价和 Mooncake TCP Restore 更慢的结果。

## 4. 一分钟项目介绍

> 我基于 nano-vLLM 做了一个面向在线推理 QoS 的二次开发项目。原始 FCFS 调度在混合
> Batch 和 Interactive 请求时，会让交互请求长时间等待 Decode，TTFT 可能不高，但 E2E
> 尾延迟会超过 SLO。我实现了 PALS 调度器，根据优先级、等待老化、SLO 剩余时间和执行
> 时间估计选择请求，并保留 FCFS 作为基线。
>
> 我在 RTX 4060 上做了三档到达率、每组 3 轮的实机实验。PALS 将交互 E2E p95 降低
> 87.9% 到 96.5%，SLO 从 0% 提升到 100%，总吞吐变化不超过 3.5%，而 GPU 利用率基本
> 一致。项目还实现了 Radix Prefix Cache 和 Mooncake Remote KV，并修复了 Catalog
> Insert/Upsert 语义导致跨 Worker 恢复失败的问题。

## 5. 面试官问“最难的问题是什么”

### 难点一：为什么 TTFT 很低，交互体验仍然很差

FCFS 实验中 Interactive TTFT 只有几十毫秒，但 E2E p95 超过 8 秒。原因是请求很快完成
Prefill 并输出首 Token，后续 Decode 却被长 Batch 请求持续占用。因此只优化 TTFT 会得出
错误结论。

修改方式：

- 为每个请求记录 Queue、TTFT、TPOT、E2E 和三类 SLO；
- 调度器每步重新计算优先级，而不是只在请求入队时排序；
- 使用 Aging 防止 Batch 永久饥饿；
- 同时报告 Interactive 收益和 Batch 延迟代价。

### 难点二：Mooncake 已经 PUT 数据，为什么重启后没有 Restore

最初指标显示约 112 MiB PUT，但 `remote_restore_started=0`。先通过强制 Restore 开关排除
Cost-Aware Planner，随后发现新 Worker 加载的 Catalog 仍是旧版本。

根因是 KV Page 与 Catalog 的对象语义不同：KV Page 使用内容寻址 Key，是不可变对象；
Catalog 使用固定 Key，是可变元数据。旧实现对两者都调用 Insert 风格的 `put`，导致
Catalog 无法覆盖。

修改方式：

- 在存储协议中区分 `put` 和 `upsert`；
- KV Page 保持普通 `put`，Catalog 使用 Mooncake `upsert_batch`；
- 增加模拟 Insert-Only Backend 的回归测试；
- 重启 Worker 后验证 GET 字节、Restore 次数和 Restored Token 三类指标。

### 难点三：如何证明优化来自调度而不是实验波动

- 固定模型、GPU、Prefix、输出长度、请求到达序列和 `max_num_seqs`；
- 为完整工作负载生成 SHA-256 指纹，聚合器拒绝混合不同指纹；
- 每组重启 Worker，避免本地 KV Cache 污染；
- 每组重复 3 轮，报告 Sample Standard Deviation 和 Student-t 95% CI；
- 同时采集 GPU 利用率和显存，确认两种策略使用的硬件资源相近。

## 6. 指标应该怎样解释

- **TTFT**：请求到第一个输出 Token 的时间，主要反映排队与 Prefill。
- **TPOT**：首 Token 之后平均每个 Token 的时间，主要反映 Decode 调度。
- **E2E**：完整请求结束时间，最接近用户真实等待时间。
- **SLO Goodput**：单位时间内满足所有 SLO 的请求数，比单纯吞吐更适合 QoS 调度。
- **Prefix Block Hit Rate**：命中的完整 KV Page 数除以查询 Page 数。

## 7. 不能写进简历的夸张说法

不要写“Mooncake 将推理提速 88%”。当前 WSL2 单机 TCP 实验中，强制 Remote Restore
实际上比本地 Recompute 更慢，Cost-Aware Planner 选择 Recompute 才是正确结果。

不要写“实现了生产级 vLLM”。当前自动 Remote Restore 限定 TP=1，Catalog 主要验证
单写者重启场景，RDMA、多机和多实例一致性仍是后续工作。

不要只写“使用了 RadixAttention、PagedAttention、Mooncake”。面试官更关心这些组件
解决了什么问题、修改了哪些执行路径、如何验证以及有什么边界。

## 8. 证据入口

- 压力图：`assets/pals-pressure.svg`
- 多到达率聚合报告：`benchmarks/results/repeated/pals-pressure-summary.md`
- 实机实验方法：`docs/gpu_serving_benchmark_zh.md`
- 调度器：`nanovllm/engine/scheduler.py`、`nanovllm/engine/qos.py`
- Remote KV：`nanovllm/engine/remote_restore.py`、`nanovllm/engine/storage_backend.py`
- 一键实验：`scripts/run_gpu_ablation_wsl.sh`
