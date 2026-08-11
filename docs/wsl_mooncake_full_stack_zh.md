# WSL2 下运行 Qwen3、PALS、Radix Cache 与 Mooncake

本文面向第一次接触 AI Infra 的读者，目标是在一台 Windows + NVIDIA GPU
电脑上跑通完整 nano-vLLM 主路径，而不是 Windows Transformers 兼容后端。

## 1. 最终运行的是什么

完整链路由三个 Linux 进程组成：

```text
浏览器 / OpenAI API
        |
        v
nano-vLLM GPU Worker (Qwen3-0.6B)
  |- Continuous Batching
  |- PALS 调度
  |- Paged KV Cache
  |- Radix Prefix Cache
  `- KV Page Export / Import
        |
        | TCP GET / PUT
        v
Mooncake Master ------ Mooncake Store Service
  元数据和分配决策         常驻 2 GiB 内存段，保存 KV Page 和 Catalog
```

三个进程不能随意合并。`mooncake_master` 只管理元数据，不保存 KV Page
字节；推理 Worker 的生命周期又比较短。如果只让 Worker 注册全局内存段，Worker
退出后该内存段会被卸载，重启后的 Worker 就找不到旧对象。常驻 Store Service
提供独立于 Worker 生命周期的存储段，才能验证真正的跨进程恢复。

## 2. 已验证环境

- Windows 11 + WSL2 `Ubuntu-22.04`
- NVIDIA GeForce RTX 4060 Laptop GPU，8 GiB 显存
- Python 3.10.12
- CUDA Toolkit 12.9
- PyTorch 2.8.0 + CUDA 12.9
- Triton 3.4.0
- FlashAttention 2.8.3
- Mooncake Transfer Engine 0.3.12.post1
- 模型：本地 `Qwen3-0.6B`，BF16

WSL 的 `C:\Users\<用户名>\.wslconfig` 可使用：

```ini
[wsl2]
networkingMode=mirrored
autoProxy=true
dnsTunneling=true
firewall=true
memory=10GB
processors=8
swap=8GB
```

修改后在 PowerShell 执行 `wsl --shutdown`，再重新进入 WSL。

## 3. Python 环境

以下命令都在 WSL 中执行：

```bash
python3 -m venv /home/xuhang/.venvs/nanovllm-qos
source /home/xuhang/.venvs/nanovllm-qos/bin/activate
cd /mnt/d/nano-vllm-qos

python -m pip install -e ".[serve,mooncake]"
```

FlashAttention 必须与当前 PyTorch、CUDA 和 Python ABI 匹配。安装后可先检查：

```bash
python -c "import torch, triton, flash_attn; print(torch.__version__, triton.__version__, flash_attn.__version__)"
```

## 4. 启动三个进程

打开三个 WSL 终端，均进入仓库目录。

终端一启动 Mooncake Master：

```bash
cd /mnt/d/nano-vllm-qos
bash scripts/run_mooncake_wsl.sh master
```

终端二启动常驻 Store Service：

```bash
cd /mnt/d/nano-vllm-qos
bash scripts/run_mooncake_wsl.sh store
```

终端三启动 Qwen3 推理服务：

```bash
cd /mnt/d/nano-vllm-qos
ENABLE_MOONCAKE=1 bash scripts/run_nanovllm_wsl.sh
```

默认端口如下：

| 服务 | 端口 | 用途 |
| --- | --- | --- |
| nano-vLLM | `8020` | 对话前端、OpenAI API、指标 |
| Mooncake Master | `50051` | Store 元数据 RPC |
| Master metrics | `9003` | Mooncake Master 指标 |
| Store REST | `9004` | Store Service 管理接口 |

浏览器打开 `http://127.0.0.1:8020/` 即可使用对话前端。

## 5. 启动参数与功能的关系

`run_nanovllm_wsl.sh` 默认传入：

```text
--scheduling-policy pals          启用优先级与 SLO Slack 调度
--prefix-cache-backend radix      启用页对齐 Radix 前缀索引
--max-num-seqs 64                 允许多个请求进入 Continuous Batching
--max-model-len 40960             请求输入与输出共享的模型原生窗口
--gpu-memory-utilization 0.90     为完整 40960 窗口分配足够 KV 块
--kv-storage-backend mooncake     启用 Remote KV 写回与恢复
--mooncake-global-segment-mib 0   Worker 不承担持久存储角色
--mooncake-local-buffer-mib 512   KV Page 并发传输的本地 staging 空间
```

Paged KV Cache 和 Continuous Batching 是 nano-vLLM 引擎本身的执行机制，不需要
额外开关。PALS、Radix 和 Mooncake 则分别由上面的参数选择。

引擎会在启动后计算 `num_kvcache_blocks * kvcache_block_size`，并把有效上下文限制为
模型窗口与物理 KV 容量中的较小值。当前 RTX 4060 8GB 在 0.90 下分配 163 个 256-Token
块，即 41728 Token，因此可以覆盖 Qwen3-0.6B 的 40960 Token 原生窗口。

需要演示优先级排队时，可以临时把并发槽位限制为 1：

```bash
MAX_NUM_SEQS=1 ENABLE_MOONCAKE=1 bash scripts/run_nanovllm_wsl.sh
```

此时较晚到达的紧急请求可以越过已经等待的普通请求。正常聊天时不要限制为 1，
直接使用默认值 64，才能保留 Continuous Batching 的并发能力。也可以设置
`SCHEDULING_POLICY=fcfs` 运行相同负载，作为不考虑优先级的对照组。

为什么本地 Buffer 是 512 MiB：Qwen3-0.6B 在当前布局下每个 KV Page Envelope
约 28 MiB。2174 Token 的请求会产生 8 个可写回页面，并发 PUT 需要约 224 MiB。
128 MiB 会在第五页开始出现 `Failed to allocate buffer`，512 MiB 能覆盖 4096
上下文下的本项目实验。

## 6. 如何确认不是“只启动了”

先查看健康状态：

```bash
curl http://127.0.0.1:8020/health
curl http://127.0.0.1:8020/v1/metrics
```

健康状态中的 Backend 应为 `CUDA + Mooncake`，指标应包含：

```text
policy = pals
prefix_cache_backend = radix
remote_io_writeback_completed
remote_restore_completed
kv_restored_tokens
```

本机端到端实验过程是：

1. 发送一个 2174 Token 的长请求，生成 8 个完整 KV Page。
2. 后台向 Mooncake PUT 8 个 Page，再保存 1 个 Catalog 对象。
3. 优雅关闭并重新启动 nano-vLLM Worker，Master 与 Store Service 保持运行。
4. 新 Worker 启动时本地 Radix Cache 为 0，但加载到 1 条远端 Catalog 前缀。
5. 再次发送相同请求，远端 GET 8 个 Page，并导入新的 GPU Block。

实测功能指标：

```text
remote_restore_started   = 1
remote_restore_completed = 1
remote_restore_failed    = 0
kv_restored_tokens       = 2048
radix_cached_blocks      = 8
remote_io_failures       = 0
```

这证明 Remote KV 数据面已经闭环。当前 WSL 单机 TCP 恢复约等待 1.5 秒，不能据此
宣称性能提升；它验证的是正确性和系统集成。后续要用独立机器、RDMA、固定负载与
TTFT/吞吐基线来判断性能收益。

## 7. 常用环境变量

```bash
PORT=8021 ENABLE_MOONCAKE=1 bash scripts/run_nanovllm_wsl.sh

MOONCAKE_STORE_SEGMENT_BYTES=3221225472 \
  bash scripts/run_mooncake_wsl.sh store

MOONCAKE_PROTOCOL=rdma MOONCAKE_RDMA_DEVICES=mlx5_0 \
  bash scripts/run_mooncake_wsl.sh store
```

本机没有 RDMA 网卡，因此已验证的是 TCP。切换 RDMA 前需要确认 Linux 能看到
RDMA 设备，并让 Store 与 Worker 使用一致的协议和设备名。

## 8. 排错顺序

1. `nvidia-smi`：WSL 是否能看到 GPU。
2. `curl /health`：模型服务是否启动。
3. `curl /v1/metrics`：Backend、PALS、Radix 是否生效。
4. 查看 Master 是否仍运行：没有 Master 就无法发现远端对象。
5. 查看 Store Service 是否仍运行：只有 Master 而没有存储段，数据不会跨 Worker 重启保留。
6. 如果出现 Buffer 分配失败，提高 `MOONCAKE_LOCAL_BUFFER_MIB`。
7. 如果 Catalog 能加载但没有 Restore，检查成本模型是否选择了重新 Prefill；可调整
   `REMOTE_KV_BANDWIDTH_GBPS` 与 `REMOTE_KV_FIXED_LATENCY_MS` 做受控实验。

## 9. 当前边界

- 已接通 TP=1 的真实 GPU KV Page 写回与恢复。
- 已验证 Qwen3-0.6B、BF16、CUDA、FlashAttention、TCP Mooncake。
- 当前 Catalog 快照采用单写者模型。
- 尚未实现 Tensor Parallel 多 Rank 恢复。
- 尚未实现 CUDA Stream 上的传输与计算重叠。
- 尚未完成 RDMA、多机和统一负载下的性能对比。
