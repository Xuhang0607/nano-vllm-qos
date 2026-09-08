# nano-vLLM-QoS 简历与面试指南

> 评测范围更正：下文 70.1%/70.2% 吞吐和 37.4% 尾延迟数据来自每轮仅 8 个请求的
> 探索性实验；9/9 质量数据来自同模板检索，且两种压缩策略保留预算不同。
> 不建议直接把这些数字作为通用性能结论写入简历。新增公平预算与持续负载评测见
> [扩大评测协议](validation_protocol_zh.md)，简历量化结果应以完成后的扩大实验为准。

最新证据见[独立样本与重复实机验证](final_validation_zh.md)。同预算质量对照已扩大到
两组独立样本、每策略 432 题。历史单轮 +2.7% 吞吐同样不作为稳定提升写入简历。

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
>   Attention Sink、Query 相关历史页和 Recent Window 动态回收 GPU Page；零匹配时按
>   近期优先补足预算，实现压缩后的前缀隔离与抢占重计算回退。
> - 在 Qwen3-0.6B + RTX 4060 8GB 上完成两组独立样本、共 432 题合成检索对照；相同
>   5 页保留预算下，准确率由固定窗口的 27.1% 提高至 70.6%，均累计丢弃 58.3% KV Token；
>   相比不压缩的 74.1% 低约 3.5 个百分点。
> - 将服务层阻塞线程池等待改为跨线程异步事件通知，补齐取消与超时处理，非流式请求仅在
>   结束时解码一次；完成两档负载、各策略三轮的 9012 请求实机对照，当前 WSL 213 项回归测试通过。

使用范围：9012 为发出请求数，其中成功 8797、超时 215，并非全部通过。2 req/s 的平均
吞吐观察值 +4.8%，3 req/s 为 -7.7%，轮间波动较大，因此本版不写稳定提速结论。

最新可补充的工程点：定位每步全量扫描、排序历史请求指标的问题，改为按完成记录变化
失效的精确摘要缓存，保持实时队列和 KV 计数更新。在另一个 3606 请求、各模式三轮的
实机对照中，历史汇总累计耗时下降 92.6%，但整体吞吐未提升。该数字不能写成推理加速。
见[指标汇总优化报告](metrics_summary_optimization_zh.md)。

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
> 在 RTX 4060 上，我将原先 9 题的质量探针扩成两组独立样本、共 432 题，并统一两种策略
> 的保留页预算。Query-Aware 答对 305 题，固定窗口答对 117 题，不压缩答对 320 题。
> 因此我能说明选页策略在这组测试中有价值，也能明确说明它存在质量损失。我也保留了
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

### 难点五：已经完成的结果为何迟迟回不到客户端

旧服务为每个请求执行 `asyncio.to_thread(queue.get)`。大量尚未完成的请求可能占满默认
执行器的线程，使其他已就绪请求的取结果任务也排不到线程。这是服务层等待机制的问题，
不能只看 GPU 利用率判断。

改为线程安全队列加 `loop.call_soon_threadsafe` 通知：先注册等待者再检查队列，避免
丢唤醒；取消等待时移除 Future，但不取走下一条事件。非流式请求不再逐 Token 解码累计
文本。回归测试占满默认执行器，同时挂起 128 个异步等待者，确认最后就绪的请求仍能返回。
这证明等待路径不再依赖线程池容量，不等于证明引擎支持 128 路 GPU 并发。

## 6. 指标应该怎样解释

- **TTFT**：请求到第一个输出 Token 的时间，主要反映排队与 Prefill。
- **TPOT**：首 Token 之后平均每个 Token 的时间，主要反映 Decode 调度。
- **E2E**：完整请求结束时间，最接近用户真实等待时间。
- **SLO Goodput**：单位时间内满足所有 SLO 的请求数，比单纯吞吐更适合 QoS 调度。
- **Prefix Block Hit Rate**：命中的完整 KV Page 数除以查询 Page 数。

本轮非流式压测中的 TTFT/TPOT 来自引擎埋点，不是客户端 SSE 首包或逐 Token 网络延迟。
客户端 E2E 单独测量。失败请求必须计入 offered SLO 的分母；不能仅对成功请求报达标率。

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

- 最新质量和持续压测：[最终验证报告](final_validation_zh.md)
- 质量对照：`benchmarks/results/validation-20260905-quality-v2/`、`benchmarks/results/validation-20260905-final/`
- 重复性能对照：`benchmarks/results/validation-20260905-repeated/`
- 以下为历史探索性结果，不能与新版配置混合聚合。

- 压力图：`assets/pals-pressure.svg`
- 多到达率聚合报告：`benchmarks/results/repeated/pals-pressure-summary.md`
- KV 压缩聚合报告：`benchmarks/results/repeated/kv-compression-none-vs-query.md`
- KV 质量对照：`benchmarks/results/repeated/kv-quality-comparison-64.md`
- Query-Aware 设计：`docs/query_aware_kv_compression_zh.md`
- 实机实验方法：`docs/gpu_serving_benchmark_zh.md`
- 调度器：`nanovllm/engine/scheduler.py`、`nanovllm/engine/qos.py`
- Remote KV：`nanovllm/engine/remote_restore.py`、`nanovllm/engine/storage_backend.py`
- 一键实验：`scripts/run_gpu_ablation_wsl.sh`
