# KV Page 数据面与稳定序列化格式

## 1. 从控制面走向真实数据

前一阶段的 `HierarchicalRadixCache`、`KVTransferPlanner` 和
`AsyncKVTransferCoordinator` 解决了“去哪里找”和“多个请求如何共享传输”的问题，但它们处理的仍是元数据或普通字节对象。要恢复真实 KV Cache，还需要回答：

1. nano-vLLM 中一个物理 KV Block 对应 Tensor 的哪一部分？
2. 如何把非连续 GPU Tensor 打包成 Mooncake 能存储的稳定字节对象？
3. 如何保证错误模型、错误 TP Rank 或损坏数据不会写入 KV Cache？
4. 如何把字节对象恢复回新的物理 Block，而不依赖旧 Block ID？

本阶段新增 `KVPageCodec`、`TorchKVPageIO` 和 `AsyncKVPageStore`，完成单个物理 Page 的真实导出、校验、存储与恢复原语。

## 2. nano-vLLM 的 KV Cache 布局

`ModelRunner.allocate_kv_cache()` 分配：

```text
kv_cache.shape = [
    2,                    # K / V
    num_layers,
    num_physical_blocks,
    block_size,
    num_kv_heads_per_rank,
    head_dim,
]
```

一个物理 Block `block_id` 的完整 Page 是：

```python
page = kv_cache[:, :, block_id]
```

其逻辑 Shape 为：

```text
[2, num_layers, block_size, num_kv_heads_per_rank, head_dim]
```

这里有一个容易忽略的细节：`physical_block` 位于 Tensor 第 3 维。固定该维后，不同 Layer
之间仍隔着其他 Physical Block 的存储区域，因此这个 Page 通常不是连续内存。不能直接把
`data_ptr()` 后面的 `page.nbytes` 字节上传，否则会混入其他 Block，或者漏掉后续 Layer。

正确做法是先把所有 K/V、所有 Layer 的指定 Block Copy 到连续 Staging Buffer，再转为字节对象。

## 3. 稳定 Page Envelope

Mooncake Store 只存储 Key/Bytes，不理解 Torch Tensor。`KVPageCodec` 定义以下版本化格式：

```text
+----------------------+----------------------+-------------------+------------------+
| Fixed Prefix         | Canonical JSON       | Raw Tensor Bytes  | BLAKE2b-128      |
| magic/version/length | Metadata             | C-order packed    | Checksum         |
+----------------------+----------------------+-------------------+------------------+
```

固定 Prefix 使用 Little-Endian 编码，包含：

| 字段 | 作用 |
| --- | --- |
| `magic` | 识别 Nano-vLLM KV Page 对象 |
| `version` | 支持未来格式升级与拒绝未知版本 |
| `header_size` | 定位 Metadata 边界 |
| `payload_size` | 校验完整对象长度并限制异常大对象 |

Metadata 包含：

| 字段 | 作用 |
| --- | --- |
| `identity_digest` | 隔离模型、Revision、DType、TP Size 与 Page Size |
| `tp_rank` | 防止把其他 Tensor Parallel Rank 的 KV 写入当前 Rank |
| `page_index` | 标识该 Prefix 中的逻辑 Page 位置，不绑定本地 Physical Block ID |
| `layout` | Layers、Block Size、KV Heads、Head Dim 与 DType |

Checksum 覆盖 Prefix、Metadata 和 Raw Payload。Decode 顺序为：

```text
长度上限 -> Magic/Version -> Envelope 总长度 -> Checksum
-> Metadata Schema -> Layout/Payload Size -> Consumer Compatibility
```

在所有检查通过前，不会把 Payload 写入 KV Tensor。

## 4. 为什么逻辑 Page Index 与物理 Block ID 分离

Remote Cache 保存的是“某个 Token Prefix 的第几个 Page”，而不是某台 GPU 当前分配到的
Physical Block ID。请求再次命中时，`BlockManager` 可能分配完全不同的空闲 Block。

```text
第一次请求: logical page 3 -> physical block 87
缓存恢复后: logical page 3 -> physical block 12
```

因此：

- Remote Key 和 Envelope 使用 `page_index`。
- `TorchKVPageIO.import_page()` 接受本次新分配的 `block_id`。
- Radix Prefix Metadata 负责把 Token Prefix 与逻辑 Page 顺序关联。
- BlockManager 继续拥有 Physical Block 的生命周期。

这保持了逻辑前缀、远端对象与本地物理显存三者的解耦。

## 5. Torch Tensor 导出与恢复

### Export

```text
non-contiguous CUDA page
-> contiguous CPU pinned staging buffer
-> wait for current CUDA stream
-> uint8 view
-> raw bytes
-> KVPageCodec envelope
```

### Import

```text
verified envelope
-> raw bytes
-> CPU pinned staging buffer
-> copy to selected CUDA physical block
-> wait for current CUDA stream
```

通过把原始 DType 的 Staging Tensor 重新解释为 `uint8`，代码可以完整保留 BF16 位模式，
不需要先把 BF16 转成 NumPy 不一定支持的数值类型，也不会发生精度转换。

当前 API 在返回前同步 CUDA Stream，语义简单且便于验证。后续要实现传输与计算重叠时，应改为返回 CUDA Event，并由 Scheduler 根据 Event 唤醒请求，而不是直接删除同步点。

## 6. 与 ModelRunner 的接入

`ModelRunner` 分配 KV Cache 后创建 `TorchKVPageIO`，并提供两个 Rank-Local 方法：

```python
envelope = model_runner.export_kv_page(
    block_id,
    identity_digest,
    logical_page_index,
)

restored_bytes = model_runner.import_kv_page(
    new_block_id,
    envelope,
    identity_digest,
    logical_page_index,
)
```

导入时会自动校验当前 `ModelRunner.rank` 和实际 KV Tensor Layout。TP=1 下已经由
Scheduler 自动触发，完整流程见
[Remote KV Restore 与 Scheduler 唤醒闭环](remote_restore_scheduler_zh.md)。

## 7. 分层存储桥接

`AsyncKVPageStore` 把 Codec 与上一阶段的 `AsyncKVTransferCoordinator` 组合：

```text
KV Page Metadata + Raw Bytes
-> Encode Envelope
-> Coordinator.put
-> InMemory / Mooncake Backend

Coordinator.get
-> Decode + Compatibility Check
-> KV Page Record
```

这样可以同时复用 Duplicate Fetch Coalescing、Cancellation Isolation、Failure Retry 与
Fetch/Evict I/O Ordering。

## 8. 验证结果

控制面测试覆盖：

- BF16 Layout 与字节数计算。
- Envelope 编解码往返。
- Identity、TP Rank、Page Index 和 Layout 不匹配拒绝。
- Payload 损坏与截断检测。
- Async Store/Coordinator/Backend 端到端往返。
- Torch CPU 非连续 Physical Page 往返（安装 Torch 时执行）。

本机额外完成真实 GPU 验证：

```text
GPU: NVIDIA GeForce RTX 4060
DType: BF16
Synthetic KV Cache: [2, 3, 4, 8, 2, 16]
Physical Page Layout: [2, 3, 8, 2, 16]
Raw Page: 3072 bytes
Result: export -> zero original -> import -> bitwise equality passed
```

可在符合项目 Python/PyTorch 版本要求的环境运行：

```bash
python -m scripts.kv_page_roundtrip --device cuda --dtype bfloat16
```

该测试验证数据正确性，不代表传输性能 Benchmark。

## 9. 当前边界与下一步

已经完成：

1. Page-Aligned 真实 Tensor 打包与恢复。
2. 版本化 Envelope、Checksum 与消费端兼容性校验。
3. Pinned CPU Staging Buffer 的 GPU/CPU 搬运原语。
4. Async Coordinator 与存储 Backend 的 Page 对象桥接。
5. `ModelRunner` Rank-Local 导出/导入入口。
6. TP=1 下的 `WAITING_FOR_KV`、后台 Fetch、主线程 Import、BlockManager 原子注册与唤醒。
7. TP=1 下新完成 Page 的安全点导出、后台写入与 Catalog 原子发布。

仍需完成：

1. 多实例 Remote Prefix Catalog 的 CAS/事务一致性。
2. Tensor Parallel 下每个 Rank 独立读写自己的 Shard，并进行跨 Rank 完成同步。
3. 使用独立 CUDA Stream 与 Event 实现传输/计算重叠。
4. 接入真实 Mooncake 进程并测量 TTFT、带宽与 Crossover Point。
