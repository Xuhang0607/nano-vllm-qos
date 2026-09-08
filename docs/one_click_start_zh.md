# 一键启动 nano-vLLM QoS 服务

项目根目录提供了 Windows 一键启动脚本：

```bat
start_nanovllm_qos.cmd
```

真正的参数配置在：

```text
start_nanovllm_qos.ps1
```

双击这个文件，或者在 PowerShell / CMD 中执行：

```bat
D:\nano-vllm-qos\start_nanovllm_qos.cmd
```

启动脚本会自动打开浏览器。启动成功后访问：

```text
http://127.0.0.1:8020/
```

## 默认启动内容

脚本默认启动的是 WSL2 里的完整服务链路：

- 模型：`/mnt/d/models/Qwen3-0.6B`
- 端口：`8020`
- 调度策略：`pals`
- Prefix Cache：`radix`
- Mooncake Master：自动检查并启动
- Mooncake Store Service：自动检查并启动
- Mooncake Remote KV：开启
- KV 回收：`slo_aware`
- KV 压缩：默认关闭，日常聊天更稳定
- 最大上下文：`40960`

## 常用修改

如果模型目录不同，修改 `start_nanovllm_qos.ps1` 顶部：

```powershell
$ModelDir = "/mnt/d/models/Qwen3-0.6B"
```

如果端口被占用，修改：

```powershell
$Port = "8020"
```

如果只是日常聊天，建议保持：

```powershell
$KvCompressionPolicy = "none"
```

如果要演示 Query-Aware KV 压缩，把它改成：

```powershell
$KvCompressionPolicy = "query_aware"
```

如果要制造显存压力，观察 KV 回收或压缩效果，可以修改：

```powershell
$NumKvCacheBlocks = ""
```

改成：

```powershell
$NumKvCacheBlocks = "18"
```

## 停止服务

保持启动窗口打开。需要停止 nano-vLLM 时，在启动窗口按：

```text
Ctrl+C
```

然后确认终止即可。

Mooncake Master 和 Store 使用后台进程启动。如果要完全停止它们，可以关闭 WSL，或者在 WSL 中手动结束对应进程。

## 如果端口被占用

如果看到类似 `address already in use`，说明已有服务占用了端口。最简单的处理方式是：

1. 关闭之前打开的启动窗口。
2. 或者把 `PORT=8020` 改成其他端口，比如 `8021`。
3. 重新运行 `start_nanovllm_qos.cmd`。

## 面试时可以这样解释

这个脚本把本地推理服务的关键运行参数收敛到一个入口里，方便复现实验环境。默认配置启动 Qwen3-0.6B、PALS 调度、Radix Prefix Cache、Mooncake Remote KV 和 SLO-Aware KV 回收；需要做压缩实验时，只要把 `KV_COMPRESSION_POLICY` 从 `none` 切换成 `query_aware`，就可以观察前端指标面板里的 KV 压缩事件、丢弃 token 数和延迟变化。
