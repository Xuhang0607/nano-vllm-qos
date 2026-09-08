# Validated environment snapshot / 已验证环境

This records the existing test environment, not a portable or fully hashed dependency lock.
No packages or drivers were upgraded for the benchmark.

本文件记录实测环境，不是可跨平台直接安装的完整依赖锁文件。本轮没有升级依赖或驱动。

| Component | Observed version |
| --- | --- |
| Python | 3.10.12, GCC 11.4.0 |
| OS | WSL2 Linux x86_64, kernel 6.18.33.2-microsoft-standard-WSL2, glibc 2.35 |
| GPU | NVIDIA GeForce RTX 4060, 8188 MiB |
| CUDA Toolkit | 12.9, nvcc V12.9.86 |
| PyTorch | 2.8.0+cu129 |
| Triton | 3.4.0 |
| Transformers | 4.57.6 |
| Tokenizers | 0.22.2 |
| FlashAttention | 2.8.3, local wheel listed below |
| Mooncake Transfer Engine | 0.3.12.post1, local wheel listed below; disabled in device-context ablations |
| FastAPI | 0.141.1 |
| Starlette | 1.6.0 |
| Uvicorn | 0.52.1 |
| NumPy | 2.2.6 |
| safetensors | 0.8.0 |
| xxhash | 3.8.1 |
| pytest | 9.1.1 |
| Ruff | 0.16.2 |

## Local wheel identity / 本地 wheel 标识

The installed package metadata records these artifacts. Paths under `/tmp` are not download URLs
and are not suitable as portable requirements entries.

安装元数据记录了以下文件；`/tmp` 是本地路径，不应直接复制成跨机器依赖配置。

```text
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
SHA256: 75c51f34bb93c5a4438d6b767547ca6c22a9bfc6b29b5646dda98e943cd75d99

mooncake_transfer_engine-0.3.12.post1-cp310-cp310-manylinux_2_28_x86_64.whl
SHA256: 3ae0dedd85b57edf87eec36cad3f36e9f7fdddb72f5afb92e6efb7c4e240a45d
```

The Python ABI, PyTorch/CUDA build and C++ ABI matter for FlashAttention. A matching version
number alone is not sufficient. A fresh-machine installation of this snapshot has not been tested.

FlashAttention 需要同时匹配 Python ABI、PyTorch/CUDA 构建和 C++ ABI，不能只匹配版本号。
本轮没有在全新机器/全新虚拟环境中重装验证，因此不承诺本记录等同于一键环境恢复。

## Source identity / 源码版本

`pip freeze` identifies an editable nano-vLLM checkout, but the checkout contains uncommitted
changes. The commit printed by pip is not the exact tested source. Use the experiment's
`manifest.json` source hashes for runtime, benchmark and launcher identity.

当前 nano-vLLM 是 editable 安装，工作区包含未提交修改；pip 展示的 commit 不能代表完整实测源码。
每次正式实验的 `manifest.json` 记录运行时、评测工具与 Shell 启动脚本的 SHA256，应以它核对。
正式发布还需要固定 Git 提交或保留对应源码包，不能仅凭旧 commit 声称可复现。

## Entry points / 入口

- Daily local service: `start_nanovllm_qos.cmd`; see [启动说明](one_click_start_zh.md).
- Device-context ablation: `python -m benchmarks.run_validation device`; see [在线实验](device_context_online_zh.md).
- Regression tests: `python -m pytest tests -q` from the repository root in the validated WSL environment.
- Profiler integrity: see [执行阶段诊断](engine_phase_profile_zh.md). The installed Nsight build did not collect GPU kernel events.

Do not replace the model path with the README placeholder literally. The measured model is
the local Qwen3-0.6B directory; model weights are not included in the source repository.

模型路径应替换为真实目录，不要原样输入占位符。当前测量使用本地 Qwen3-0.6B，源码仓库不包含模型权重。
