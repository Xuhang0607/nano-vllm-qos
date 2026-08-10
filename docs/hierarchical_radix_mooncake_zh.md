# 分层 RadixAttention 与 Mooncake 设计

## 1. 为什么不直接复制 HiCache

原版 nano-vLLM 的 prefix cache 使用“前缀哈希 -> 物理 block”映射。它能复用完整
page，但没有显式的前缀树、分层位置、叶子驱逐和远端传输成本模型。

SGLang HiCache 已经实现成熟的 HiRadixTree 与 Mooncake L3。如果只把同样功能搬到
nano-vLLM，项目仍然主要是复现。本项目的差异化目标是：

> 将请求 SLO/优先级、Radix prefix locality、KV 传输成本和重计算成本放进同一套
> 可解释的调度决策。

## 2. 当前已经实现

### Page-aligned Radix Tree

`RadixPrefixCache` 的一个符号对应一个完整 KV page，而不是一个 token。树的边可以
压缩多个连续 page，并支持：

- 最长连续前缀匹配；
- 分叉时自动拆分压缩边；
- 较短前缀命中压缩边中部；
- 物理 block 覆盖时，从失效位置删除所有后缀；
- 删除分支后的节点重压缩；
- LRU 叶子尾 page 驱逐；
- 命中率、节点数、缓存 page 数和驱逐次数统计。

`prefix_cache_backend="hash"` 保留上游行为，设置为 `radix` 才启用新后端。

### 并发重复前缀

同一轮 batch 中，两个请求可能同时 miss 相同前缀，并分别计算出内容相同但物理地址
不同的 KV。第二次插入不能报错，也不能让树同时保留两套前缀。当前实现保留第一份
canonical block，并把第二个请求的新后缀分支连接到 canonical 路径。多余 block 在
请求释放后回到 free list。

### 分层目录与成本规划

`HierarchicalRadixCache` 为 GPU、CPU 和 Mooncake 分别维护 radix 索引，所以 L1
物理 block 被覆盖后，L3 的远端副本仍然可命中。

KV 大小按以下公式计算：

```text
kv_bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
```

对于一个远端前缀，成本规划器同时计算：

```text
recompute_only = remaining_from_local * prefill_ms_per_token

restore_then_recompute = fixed_latency
                       + transfer_bytes / effective_bandwidth
                       + remaining_from_remote * prefill_ms_per_token
```

它在 GPU、CPU、Mooncake 和本地重算候选中选择预计总时间最小的方案。网络拥塞通过
`congestion_multiplier` 进入有效传输时间。

### Mooncake Store Adapter

`MooncakeKVStore` 对接官方 `MooncakeDistributedStore`，覆盖：

- `put/get/is_exist/remove`；
- `upsert_batch/get_batch/batch_is_exist/batch_remove`；
- TCP、RDMA、EFA 配置；
- 模型指纹、TP rank、page 序号和累计前缀哈希组成的隔离对象键；
- 延迟导入和错误码检查，不安装 Mooncake 时不影响本地 nano-vLLM。

## 3. 控制面实验

```bash
python -m benchmarks.benchmark_tiered_cache \
  --output-json benchmarks/results/tiered_cache_simulation.json
```

默认实验使用 120 个不同长度、缓存命中率、prefill 成本和网络拥塞的请求：

| 策略 | 平均预计时间 | 远端恢复请求 | 传输量 | 重算 token |
| --- | ---: | ---: | ---: | ---: |
| local recompute | 107.20 ms | 0 | 0 GiB | 201,600 |
| always restore | 168.93 ms | 120 | 13.20 GiB | 77,986 |
| cost aware | 95.96 ms | 70 | 3.46 GiB | 169,208 |

成本感知策略比 local-only 预计时间降低 10.5%，比盲目 restore 降低 43.2%。它没有
追求最高 remote hit rate，而是在拥塞时接受更多重算。这是控制面模型结果，不是
Mooncake、RDMA 或 GPU 性能数据。

## 4. 当前数据面状态

当前已经完成：

1. 稳定、带版本和 Checksum 的 KV Page Envelope；
2. ModelRunner Rank-Local Tensor Page 打包与恢复；
3. Remote Prefix Cost Model 与 `WAITING_FOR_KV` 调度状态；
4. 后台 GET、重复请求合并、取消隔离和失败回退；
5. 主线程 GPU Import、BlockManager 原子注册与 Scheduler 唤醒；
6. 新完成 Page 的主线程导出、后台 PUT、写入合并与 Remote Catalog 原子发布。

当前仍不能声称“完成 nano-vLLM + Mooncake 分布式推理”。仍需：

1. 多实例 Remote Prefix Catalog 的 CAS/事务一致性；
2. TP>1 Shard 恢复与跨 Rank 同步；
3. 独立 CUDA Stream/Event 和注册内存零拷贝路径；
4. 真实 Mooncake 进程端到端性能验证。

官方 Mooncake 建议先在 Ubuntu 使用 TCP 完成正确性验证，再在具备 RDMA/GPUDirect
条件的节点测试零拷贝路径。仓库提供 `python -m scripts.mooncake_smoke` 做第一步。

## 5. 下一项创新：SLO-aware Cache Value

普通 LRU 只看最近访问时间。下一阶段可以为 radix 叶子计算：

```text
cache_value = reuse_probability
            * saved_recompute_ms
            * request_priority_weight
            - restore_cost
            - storage_cost
```

显存不足时驱逐 cache value 最低的叶子；过载时只为延迟预算紧张、高价值的请求保留
或加载远端 KV。实验必须与 LRU、LFU 和 size-aware greedy 做消融，避免只展示单一
策略结果。
