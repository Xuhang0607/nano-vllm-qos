# 推理阶段耗时诊断

## 目的与边界

页预留实验减少了重算，但没有带来稳定的服务收益。本轮先增加独立诊断脚本，
诊断脚本不修改默认推理路径。脚本在模型加载和预热之后，测量 Scheduler.schedule、
ModelRunner.call 与 Scheduler.postprocess 的主机墙钟时间，记录源码 SHA256。
方法包装在异常时也会恢复；运行结束检查请求完成、输出长度和全部 KV 页归还。

这不是 Nsight GPU 时间线，也不是纯 Kernel 计时。模型调用包括输入准备、
模型执行、采样和输出转为主机列表；不能将其耗时比例直接解释为 GPU 利用率。
不在每个阶段插入 CUDA 同步，仅在整个测量区间两端同步。

## 复现

在配置好的 WSL Python 环境、项目根目录中运行，输出路径必须尚不存在：

```bash
python -m benchmarks.profile_engine_phases --horizon 0 --output benchmarks/results/phase-profile-h0.json
python -m benchmarks.profile_engine_phases --horizon 32 --output benchmarks/results/phase-profile-h32.json
```

使用 Qwen3-0.6B、RTX 4060、16 页 KV、每步 512 Token、PALS/Radix、eager 执行。
12 个请求在计时前全部提交，输入长度为 257/513/769 Token，输出各 16 Token，
交替分配优先级 0 和 10。输入是合成 Token 序列，不是质量测试；没有在线到达、
HTTP、Mooncake 和客户端等待，因此不能替代之前的服务压测。
初版仅预热短请求；后续版本默认额外预热一轮同结构的 12 请求负载，Token 值不同，
不复用测量请求的前缀。可用 `--warmup-rounds` 修改，不保证所有形状充分预热。
预热也会影响 PALS 的执行时间估计，不能把新旧预热方案直接混合比较。
单轮带测量开销的数据不可作为加速结论。

## 首轮观察

2026-09-07，H=0：12 请求全部完成，192 输出 Token，106 次模型调用，全部页归还。
整体 4431.81 ms，模型调用 4416.37 ms，调度 9.16 ms，后处理 2.16 ms。
结果保存为 `benchmarks/results/phase-profile-20260907-h0.json`。

H=32 同样完成 12 请求、192 输出 Token并归还全部页：整体 4004.69 ms，
模型调用 3987.80 ms，调度 11.70 ms，后处理 1.96 ms，模型调用 76 次。
结果保存为 `benchmarks/results/phase-profile-20260907-h32.json`。
两个实验都是新建引擎运行，未进行多轮交替重复；采样调用分组也可能不同，
不能把单轮时间差归结为稳定加速或输出质量一致。两组源码哈希随 JSON 留存。
新增方法包装测试覆盖异常恢复及返回值保留，本轮完整回归为 194 项通过。

在这个样例中，调度并非主要墙钟耗时来源；但不能据此排除更长等待队列中的扫描成本。
下一步应在真实在线负载上捕获 CPU/CUDA 时间线，拆分模型调用中的输入准备、
Kernel 执行、采样和同步，并比较排队时间，而不是继续仅以回收次数衡量策略优劣。

## Nsight 采集与完整性检查

后续加入 `--nvtx`、`--cuda-profiler-range` 和 `--cprofile`，仅在测量区间开启采集。
新增 prepare_prefill、prepare_decode、prepare_sample、run_model 和 sample 子阶段。
这些阶段嵌套于 model_call 内，不能把父子耗时相加；NVTX 汇总的百分比也不是 GPU 占用率。

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output=benchmarks/results/capture \
  python -m benchmarks.profile_engine_phases --nvtx --cuda-profiler-range \
  --output benchmarks/results/capture.json
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
  --format csv --output benchmarks/results/capture benchmarks/results/capture.nsys-rep
python -m benchmarks.audit_nsys_capture benchmarks/results/capture.sqlite \
  --output benchmarks/results/capture-audit.json --require-kernels
python -m benchmarks.profile_engine_phases --cprofile --output benchmarks/results/python-profile.json
```

本机 Nsight Systems 2025.1.3 提示驱动 13.2 不受支持，回退到 12.9 采集库。
实际导出的 `phase-nsys-20260907-h0.sqlite` 有 40,090 条 CUDA 主机 API 和 533 条 NVTX 事件，
没有 Kernel 或 GPU Memcpy 事件。因此 GPU 耗时不可用，不能计算 GPU 忙碌比例或 Kernel 占比。
这是本轮采集限制，不等于 GPU 没有工作；未升级驱动或覆盖用户工具链。
审计脚本保留诊断消息，并在指定 `--require-kernels` 时写出报告后以状态码 2 退出。
原始大体积捕获保存在本地，JSON/CSV 可提交，原始 `.nsys-rep` 与 `.sqlite` 默认忽略。

主机 NVTX 显示 run_model 约 3138.59 ms，整个区间 3310.40 ms。
这只说明主机调用链的归属，不是 94.8% 时间在执行 GPU Kernel。
Python 调用剖析使用另一独立进程：`phase-python-20260907-h0.json`。
其中 `torch/utils/_device.py::__torch_function__` 调用 198,310 次，self 时间 112.92 ms，
inclusive 时间 767.99 ms；后者含子调用，不能全算作可消除的开销。

## 从诊断到修复

ModelRunner 原先使用 `torch.set_default_device("cuda")` 初始化模型，随后调用
`torch.set_default_device("cpu")`。后者仍保留全局 DeviceContext，影响后续 Torch API 调用。
当前改为只在初始化、预热、KV 分配和图捕获期间使用 `with torch.device("cuda")`，
退出时自动恢复调用者上下文；默认 dtype 用 finally 恢复，包括初始化失败路径。
输入准备中所有 pinned-memory 张量均显式指定 `device="cpu"`，不再依赖全局默认设备。
不修改模型权重、Attention 数学、采样规则或默认调度策略。

首次在同进程反复切换 CPU 设备上下文的 logits 测试触发 Torch 编译重编译上限，
随后精确比较失败，不能把该测试当作通过。验证工具改为独立进程输出固定输入的 Prefill/Decode
logits，且使用固定 Token 作为 Decode 输入，避免采样和累计编译状态干扰。
独立进程对照两阶段均逐元素一致，最大绝对误差为 0；见 `device-context-20260907-logits.json`。
这只覆盖指定输入，不是完整模型质量评测。

```bash
python -m benchmarks.verify_device_context --legacy --output benchmarks/results/legacy.pt
python -m benchmarks.verify_device_context --output benchmarks/results/scoped.pt
python -m benchmarks.verify_device_context --compare benchmarks/results/legacy.pt benchmarks/results/scoped.pt \
  --output benchmarks/results/logits-comparison.json
```

当前完整回归 203 项通过，Ruff F 类检查与 git diff 空白检查通过。
新增测试覆盖初始化成功/失败、调用者已有设备上下文、图捕获失败时状态恢复，
以及 Prefill/Decode/采样准备显式选择 pinned CPU 内存。

## 最终版本离线对照

同一诊断脚本、H=0、额外预热一轮、无 NVTX/cProfile，改动前三轮与最终代码三轮分别为：

| 轮次 | 改动前墙钟 ms | 最终版本墙钟 ms | 改动前模型调用数 | 最终版本模型调用数 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2955.97 | 2567.67 | 96 | 95 |
| 2 | 3239.98 | 2134.03 | 76 | 76 |
| 3 | 3227.55 | 2432.06 | 92 | 76 |
| 均值 | 3141.17 | 2377.92 | - | - |

数据来自 `device-context-20260907-before-{1,2,3}.json` 和
`device-context-20260907-final-{1,2,3}.json`。每轮均完成 12 请求、192 输出 Token，全部 KV 页归还。
均值下降 24.3%，但只适用于该带阶段计时的小规模离线样例。
六轮按先基线、后修改的顺序运行，没有交错随机化；PALS 受实际执行时间估计影响，批次划分也不同。
不能把这当作固定调度计划下的纯 Python 开销收益，更不能直接写成在线吞吐提升或 SLO 改善。

中间仅修改设备作用域的两次运行 `device-context-20260907-after-{1,2}.json` 保留，
但不混入最终版本均值。首次 shell 循环的输出文件名展开失败，第二次调用因目标文件已存在退出；
首次成功结果重命名为 before-1，后续显式指定路径运行，没有覆盖或筛选性能数据。

## 收尾边界

本轮落地的是初始化作用域和输入设备的修复，以及可复现的诊断工具，不是新增模型算法。
优先级页预留与静态驻留上限仍默认关闭，未用本轮离线数据推翻此前在线压测的负面结论。
现有 432 题检索质量结果属于此前版本的实验记录，本轮没有重跑该完整质量集或在线持续负载。
仍需兼容的 Nsight/驱动组合才能补齐 GPU Kernel 时间线；本轮没有升级或修改用户工具链。

最终代码另外完成独立进程 logits 复核：`device-context-20260907-logits-final.json`，
Prefill 和 Decode 均逐元素一致，最大绝对误差为 0。
`verify_device_context --cuda-graph` 也完成 Prefill/Decode 并归还全部页，
输出存于本地 `device-context-20260907-logits-graph.pt`；这项只作图捕获/回放冒烟，
没有把 Graph 输出与 eager 做完整数值或性能比较。
六轮对照的诊断脚本哈希一致，运行时源码仅 `nanovllm/engine/model_runner.py` 不同；
最终数据的源码哈希与收尾时文件匹配。所有本轮 GPU 进程已正常退出，2333 端口已释放。
未提交 Git commit 或推送，也未修改在线服务的调度/缓存默认配置。
