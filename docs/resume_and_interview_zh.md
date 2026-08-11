# nano-vLLM-QoS 简历与面试指南

## 1. 项目名称

**nano-vLLM-QoS：面向 SLO 与显存压力的大模型推理引擎优化**

技术栈：Python、PyTorch、CUDA、Triton、FlashAttention、FastAPI、Radix Tree、Paged KV
Cache、Query-Aware KV Compression、Mooncake、WSL2、Qwen3-0.6B。

## 2. 推荐的简历描述

下面这版适合放在一页中文技术简历中：

> **nano-vLLM-QoS：大模型推理调度与 KV Cache 优化**  |  个人项目
>
> - 基于 nano-vLLM 改造 Scheduler 与 KV 执行路径，实现 Priority/Aging/SLO Slack 联合调度、
>   Page-Aligned Radix Prefix Cache 及 OpenAI 兼容流式服务。
> - 设计 Query-Aware KV 压缩：解耦 RoPE 逻辑位置与 FlashAttention 物理上下文，按
>   Attention Sink、Query 相关历史页和 Recent Window 动态回收 GPU Page；固定压力负载下
>   请求吞吐提升 70.1%，Batch E2E p95 降低 37.4%。
> - 在 Qwen3-0.6B + RTX 4060 8GB 上构建 9 样本 Needle-in-a-Haystack 质量探针；压缩策略
>   累计丢弃 40.5% KV Token 时命中 9/9，固定 Sink+Recent 基线命中 0/9，并实现抢占后的
>   全量重计算安全回退。
> - 建立重启隔离的 GPU Ablation 框架，采集 TTFT/TPOT/E2E、SLO Goodput、KV 重计算量、
>   GPU 利用率和 Student-t 95% CI；另完成 Mooncake KV Write-Back/Restore 与持久化 Catalog。

## 3. 为什么这不是简单复现

复现上游项目通常只能说明“模型跑起来了”。这个项目新增了可以定位到源码、测试和实机
数据的系统能力：

1. 修改 Scheduler 决策目标，从到达顺序扩展为 Priority、Aging 和 SLO Slack 联合排序；
2. 修改 Sequence、BlockManager、Scheduler 和 ModelRunner，支持逻辑/物理 KV 长度解耦和
   Query-Aware 压缩，而不是只在 API 层增加配置；
3. 新增真实 KV Page 序列化、远端写回、跨 Worker Catalog 重建与恢复状态机；
4. 保留 FCFS、Hash、全量重计算和不压缩策略作为控制变量基线；
5. 不只展示正结果，也报告无损部分回收吞吐下降、Mooncake TCP Restore 慢于重计算等反例。

## 4. 一分钟项目介绍

> 我基于 nano-vLLM 做了一个面向在线 QoS 和显存压力的二次开发项目。调度侧实现 PALS，
> 根据优先级、等待老化、SLO 剩余时间和执行时间估计选择请求；KV 侧实现 Query-Aware
> 压缩，在保留 RoPE 原始位置的同时缩短 FlashAttention 实际读取的物理上下文。
>
> 在 RTX 4060 上的三轮固定压力实验中，Query-Aware 将请求吞吐提升 70.1%、Batch E2E
> p95 降低 37.4%；9 样本 Needle 测试命中 9/9，而固定 Sink+Recent 为 0/9。我也保留了
> 失败结果：无损部分回收虽然少重算 14.3% Token，但吞吐下降 7.3%，说明减少重计算不一定
> 等价于端到端更快。项目还实现了 Radix Prefix Cache、Mooncake Remote KV 和可复现实验框架。

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

### 难点四：压缩 KV 后为什么位置和 Attention 长度不能共用一个变量

最初 nano-vLLM 默认“逻辑 Token 数 = 物理 KV Token 数”。删除中间页后，如果直接缩短
位置编号，RoPE 会把后续 Token 当成前移，模型语义发生额外变化；如果仍把逻辑长度传给
FlashAttention，又会读取不存在的 Block。

修改方式：

- Sequence 同时保存逻辑缓存长度、物理缓存长度和物理页对应的逻辑页号；
- RoPE `positions` 使用逻辑位置，FlashAttention `context_lens` 使用物理长度；
- BlockManager 只压缩完整页，并保持选中页按原逻辑顺序排列；
- 压缩后禁用 Prefix 发布和 Remote Write-Back，抢占时回退全量重计算。

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

不要写“KV 压缩 40.5% 且质量无损”。当前结论只来自 9 个定向 Needle 样本，应写成
“在该 9 样本探针中命中 9/9”，并说明仍需 LongBench、多模型和真实业务验证。

## 8. 证据入口

- 压力图：`assets/pals-pressure.svg`
- 多到达率聚合报告：`benchmarks/results/repeated/pals-pressure-summary.md`
- KV 压缩聚合报告：`benchmarks/results/repeated/kv-compression-none-vs-query.md`
- KV 质量对照：`benchmarks/results/repeated/kv-quality-comparison-64.md`
- Query-Aware 设计：`docs/query_aware_kv_compression_zh.md`
- 实机实验方法：`docs/gpu_serving_benchmark_zh.md`
- 调度器：`nanovllm/engine/scheduler.py`、`nanovllm/engine/qos.py`
- Remote KV：`nanovllm/engine/remote_restore.py`、`nanovllm/engine/storage_backend.py`
- 一键实验：`scripts/run_gpu_ablation_wsl.sh`
