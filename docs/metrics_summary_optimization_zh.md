# 指标汇总热路径优化

## 问题与范围

在线服务在每个引擎 step 后调用 `get_scheduler_metrics()`。旧实现每次都遍历已完成
请求的全部历史，求和并排序计算中位数与 P95。即使一个 Decode step 没有完成新请求，
也会重复同样的历史计算。成本随累计历史增长，不只与当前 GPU batch 大小有关。

本次优化只处理这部分重复计算，不改变调度、Attention、KV 压缩或生成参数。
它不等于修复了所有持续负载超时问题，也不解决历史记录本身的长期内存增长。

## 实现

- 默认 `metrics_summary_mode=cached`；保留 `full` 模式复现原来的每次重算行为。
- 已完成的 `RequestMetrics` 是不可变对象，生产路径只向历史列表追加。
  以完成记录数量判断摘要是否失效；首读和新增完成记录后的首读重新计算。
- 复用的是历史汇总，不是完整指标响应。队列长度、取消计数、KV 页池、压缩和远端
  I/O 指标每次重新读取，避免以性能优化为由返回过期状态。
- 返回摘要副本，避免调用者写入或实时字段覆盖污染后续结果。
- 分位数仍用原有精确算法，未改成近似直方图；请求完成时仍可能有历史扫描成本。
- 记录调用次数、重算次数和实际汇总耗时；耗时差值保留浮点毫秒。

## CPU 测量

Intel Core i5-14400F 主机，Python 3.10.12 / WSL。使用真实 `Scheduler.metrics()` 和合成不可变请求历史，
预热后每批读取 100 次、共 5 批，两模式交替测量；下表为每次读取耗时的批均值中位数。
这是历史不变时的 CPU 微基准，不是模型推理吞吐，也不包含新增完成记录时的重算成本。

| 历史记录数 | full，微秒 | cached，微秒 |
| ---: | ---: | ---: |
| 100 | 73.17 | 4.12 |
| 1000 | 939.37 | 3.22 |
| 10000 | 12020.95 | 3.56 |

所有场景的业务指标逐字段一致。独立 cProfile 诊断中，10000 条历史、100 次 full
读取触发 100 次汇总和 1000 次排序；该诊断开启 profiler，耗时不与上表混用。

- [隔离测量结果](../benchmarks/results/metrics-summary-20260906-cpu-isolated/summary.json)
- 同目录保留 `full.prof` 与 `cached.prof`。
- `metrics-summary-20260906-cpu/` 为早期检查，可能与测试进程部分重叠，不作为主结果。

## 真实服务对照

实验已完成，结果目录为 `benchmarks/results/metrics-summary-20260906-gpu/`。
Qwen3-0.6B BF16、RTX 4060 8GB、WSL；PALS + Radix、压缩关闭、64 页 KV 池、每页
256 Token、最多 8 个序列、eager 执行。两组唯一有意改变的配置为历史指标汇总模式。

固定间隔 2 req/s、每轮 601 请求、到达持续 300 秒，各模式三轮并交替先后顺序。
记录失败请求、客户端 E2E、引擎 TTFT/TPOT、输出 Token、SLO Goodput 和汇总计数。
每轮重启服务，预热后开始统计；保留源码摘要、日志和逐请求 CSV。
旧压缩实验与本轮配置不同，不混合聚合。

共 6 轮、3606 个请求，全部成功，无客户端并发上限拒绝。每轮汇总开销如下：

| 模式 | 轮次 | 指标调用 | 历史重算 | 汇总耗时，毫秒 | 输出 Token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 1 | 8044 | 8044 | 2846.31 | 188.76 |
| full | 2 | 8084 | 8084 | 2762.23 | 188.81 |
| full | 3 | 8354 | 8354 | 2796.91 | 189.10 |
| cached | 1 | 8494 | 557 | 198.02 | 189.15 |
| cached | 2 | 8359 | 565 | 207.39 | 189.17 |
| cached | 3 | 7927 | 570 | 216.97 | 183.07 |

按三轮均值，重算次数下降 93.1%，汇总累计耗时由 2801.82 ms 降至 207.46 ms，
下降 92.6%。一次 step 可完成多个请求，所以 cached 重算次数可以少于完成请求数。
这里的毫秒统计仅覆盖历史 `summarize_metrics`，不等于全部服务层 CPU 时间。

端到端结果并非全面改善。输出吞吐均值从 188.89 降到 187.13 Token/s（-0.9%）；
Batch 客户端 E2E p95 的轮均值从 9.17 升到 13.52 秒，交互类从 2.79 升到 3.06 秒。
第三轮 cached 发生 13 次 KV 抢占，前五轮均没有；该轮完整保留，未删除或用重跑替代。
吞吐均值的 Student-t 95% CI 为 full 188.89 +/- 0.46、cached 187.13 +/- 8.74 Token/s，
轮间波动明显，不能把均值差当作稳定性能结论。

该负载到达率固定为 2 req/s，前五轮接近到达速率限制；它不是最大容量测试。没有证据
证明本改动解决了上一轮 3 req/s 的超时。抢占变化究竟来自时序、调度反馈还是运行环境，
仍需带 step 时间线的定位，不能仅凭 GPU 利用率或一轮异常归因。

[完整三轮对照与置信区间](../benchmarks/results/metrics-summary-20260906-gpu/metrics-2-comparison.md)

```bash
cd /mnt/d/nano-vllm-qos
source /home/xuhang/.venvs/nanovllm-qos/bin/activate
python -m benchmarks.run_validation metrics \
  --seed 20260906 --requests 600 --duration 300 --rates 2 --repeats 3 \
  --arrival-process uniform --max-inflight 512 \
  --output-dir benchmarks/results/my-metrics-ablation

# 单独运行 CPU 测量，不与测试或 GPU 压测同时执行。
python -m benchmarks.benchmark_metrics_summary \
  --histories 100 1000 10000 --polls 100 --batches 5 \
  --output-dir benchmarks/results/my-metrics-cpu
```

## 回归覆盖

当前 152 项测试通过。新增检查包括空历史、单条与大量历史、追加完成记录后的失效、
返回字典修改隔离、实时队列/取消/压缩计数、full 模式对照，以及耗时统计精度。
普通消融聚合器拒绝悄悄混入不同汇总模式；专用 metrics 对照显式允许该配置差异。

## 使用

正常启动默认使用缓存模式，不需要用户调整聊天参数。定位问题时可以通过
`--metrics-summary-mode full` 或 WSL 启动环境变量 `METRICS_SUMMARY_MODE=full` 切回基线。
本轮证实的是历史统计的降耗，未证实整体吞吐收益或高负载超时修复。

## 完成范围

源码清单中的 45 个 Python 文件在实机实验中保持一致，实验进程已正常退出并释放
8031 和 2333 端口。没有改变 KV 压缩、模型计算和 PALS 评分逻辑。
下一步应定位抢占与模型执行的时间线；对长期运行的历史内存上界仍需另行设计。
