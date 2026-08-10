# Remote Prefix Catalog 持久化与重启恢复

## 1. 为什么只保存 KV Page 还不够

自动 Write-Back 已经能把 KV Page 写入远端存储，但恢复路径首先需要回答：

```text
这段 Token Prefix 对应哪些 Remote Object Key？
```

这个映射原本只在当前进程的 `RemotePrefixCatalog` 中。进程退出后，Page Bytes 仍可能存在，
Radix 索引却消失，新服务无法发现这些对象。因此需要同时持久化 Catalog Metadata。

## 2. 不能持久化哪些内容

本地 Radix Tree 中包含两类仅对当前进程有意义的数据：

- 内部 Handle；
- 指向 RadixNode 的位置关系。

本地 `BlockManager` 还持有 Physical GPU Block ID。这些数字在新进程中都会重新分配，不能
写入持久化格式。快照只保存可跨进程解释的逻辑数据：

```text
Token Block Path
-> Remote Object Key
-> Model Identity Digest
-> TP Rank
-> Logical Page Index
```

恢复时重新调用 `RemotePrefixCatalog.register()`，由当前进程生成新的 Handle，并复用原有
Canonical Insert 规则。

## 3. 快照格式

`remote_catalog.py` 定义版本 1 的 JSON Envelope：

```text
schema
version
checksum
payload:
  identity_digest
  tp_rank
  prefixes:
    block_keys
    page descriptors
```

Payload 使用字段排序和紧凑分隔符生成 Canonical JSON，再计算 128-bit BLAKE2b Checksum。
加载时会检查：

1. Schema 与 Version；
2. Checksum；
3. Token ID 是否可表示为 uint32；
4. 所有 Prefix 是否使用相同 Page Size；
5. Page Index 是否从 0 连续增长；
6. 每个 Page 的模型身份与 TP Rank 是否和快照一致；
7. 快照身份是否和当前 Engine 的 `KVCacheIdentity` 一致。

## 4. 为什么保存叶路径而不是内部树节点

快照遍历 Remote Radix 的叶节点，并保存从 Root 到 Leaf 的完整逻辑路径。多个叶路径可以在
重建时重新形成共享前缀和分叉结构。这种格式容易校验，也不依赖 RadixNode 的内部实现。

代价是共享前缀会在多个叶记录中重复，且每次更新需要写完整快照。它适合当前原型和单写者
重启验证，但大规模 Catalog 更适合增量日志、紧凑 Trie 编码或数据库索引。

## 5. 对象 Key 与身份隔离

Catalog Key 由 `make_kv_catalog_key()` 生成：

```text
nanovllm-kv/{identity_digest}/tp-{rank}/catalog-v1
```

`identity_digest` 已包含模型 ID、Revision、DType、TP Size、Page Size 和 Layout Version。
因此不同模型、精度、并行配置或 Page Size 不会误用同一个 Catalog。

## 6. 保存时序

Catalog 绝不能先于 Page 变得可见：

```text
all KV Page PUTs succeed
-> register prefix in memory
-> encode complete catalog snapshot
-> enqueue background catalog PUT
```

同一 Catalog Key 的 PUT 通过 `AsyncKVTransferCoordinator` 的 I/O Tail 串行执行，所以旧快照
不会在新快照之后覆盖 Backend。每个 Engine Step 会回收已完成 Future，正常退出时等待所有
Catalog Save 完成。

如果进程在 Page PUT 成功后、Catalog Save 完成前崩溃，最多产生暂时无法发现的 Orphan
Page，不会发布指向缺失 Page 的无效 Catalog。

## 7. 启动恢复与失败降级

配置 `kv_storage_backend` 后，Catalog 持久化默认开启：

```python
LLM(
    model_path,
    kv_storage_backend=backend,
    remote_kv_persist_catalog=True,
    remote_kv_catalog_timeout_s=10.0,
)
```

新服务启动时读取身份隔离的 Catalog Object。验证成功后，把所有叶路径合并进新的
`RemotePrefixCatalog`。以下情况会退化为空 Catalog，而不会阻止本地推理：

- Catalog Object 不存在；
- Backend GET 失败或超时；
- JSON、Version 或 Checksum 无效；
- 模型身份或 TP Rank 不匹配。

后续本地 Prefill 仍可重新产生 Page，并通过自动 Write-Back 覆盖有效快照。

## 8. 可观测指标

新增 Remote I/O 指标：

- `catalog_loads`；
- `catalog_load_failed`；
- `catalog_loaded_prefixes`；
- `catalog_save_submitted`；
- `catalog_save_completed`；
- `catalog_save_failed`；
- `catalog_save_pending`。

最近的加载和保存错误保存在有上限的 `catalog_errors` 队列中，避免长期运行时无限增长。

## 9. 当前一致性边界

当前协议提供的是 **单写者下的跨进程重启恢复**。两个服务实例如果同时读取旧快照、分别
加入不同 Prefix 并覆盖同一个 Catalog Key，仍可能发生 Last-Writer-Wins 丢更新。

要支持多实例强一致，需要 Backend 至少提供一种能力：

- Compare-And-Swap Version；
- Transaction；
- Append-Only Log 和可枚举对象；
- 独立的强一致 Metadata Service。

在 Mooncake 实际 API 和部署方案确认前，项目不会把单对象 PUT 描述为多副本一致性协议。

## 10. 测试覆盖

`tests/test_remote_catalog.py` 覆盖：

1. 分叉 Radix 快照往返和确定性编码；
2. Payload 被修改后的 Checksum 拒绝；
3. 连续快照按顺序保存，重启后恢复最新完整分支；
4. 自动 Write-Back 产生的 Prefix 在新服务中可发现；
5. 损坏快照回退为空 Catalog；
6. 其他模型身份的快照被拒绝；
7. Catalog Save 失败不隐藏当前进程内已经可用的 Prefix。
