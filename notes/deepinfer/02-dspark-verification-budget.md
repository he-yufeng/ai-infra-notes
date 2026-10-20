# DSpark 的预算在什么时候决定：从接受率读到异步流水线

2026-10-20 · DeepInfer 源码导读 02

> 想观察预算怎样变成实际验证宽度，可以跑我做的 [Core DeepInfer](https://github.com/he-yufeng/core-deepinfer)：用 NumPy 看固定总量下的重分配、草稿写入和回滚。它采用随机参数、草稿自概率与合成成本，不冒充本文的训练置信度或硬件实现。

草稿模型一次生成五个 Token，目标模型就必须验五个吗？DeepInfer 的答案是否定的，但光知道“按置信度少验几个”，还不足以把算法接进一个异步推理引擎。

总预算用哪一轮的置信度算，逐请求长度在哪个设备上改，图尺寸如何影响估时，采样计时里又包含哪些工作，这些条件才决定了这项优化实际在做什么。

本文固定于 [DeepInfer `8b161497`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b)，跟随 `MTPVarlen` 与 worker 调用链。没有运行模型或硬件 benchmark。下文数字算例是解释，不是线上测量。

## 草稿已经产生，验证仍然有成本

投机解码里，后面的位置只有在前面的接受链走通后才有用。若每一步条件接受率都是 0.8，前五个全部接受的概率是 `0.8⁵ = 0.32768`。

实际调度不是给每个请求套上这个常数，而是使用 Confidence Head 的估计。[`scheduler.hpp:1239至1250`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/scheduler.hpp#L1239-L1250) 将逐位置 confidence 做前缀累乘，存为 `accept_rates`。

把这些值逐个加入，才能估计再增加一个验证位置带来多少预期输出。直接把独立位置的 confidence 当成互不相关的收益，会忽略前缀接受这个条件。

## 先给所有请求一个主 Token 位置

[`MTPVarlen::schedule`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/mtp_varlen.hpp#L150-L266) 的搜索从每请求一个主 Token 位置开始，再逐个加入按累计接受率降序排列的草稿位置。

对每个候选总预算，比较：

```cpp
const double time_ms = baseline_ms + slope_ms * graph_size + sampler_ms;
const double cur_throughput = cur_expected / time_ms;
if (cur_throughput > best_throughput) {
    best_throughput = cur_throughput;
    best_bsz = cur_bsz;
}
```

分子是预期有效 Token 数。分母里，`baseline_ms` 来自启动性能曲线，`slope_ms` 补运行时负载带来的偏差，`sampler_ms` 补采样阶段耗时。

因此，置信度高不等于一定多验。增加一个位置后，分子在涨，分母也可能跨过一个图尺寸台阶。

## graph size 上取整，不等于预算上取整

假设捕获的图尺寸只有 128、256。候选预算为 129 时，要按能容纳它的 256 图估时；但保存的 `best_bsz` 仍是 129，而不是给请求分配 256 个有效位置。

这是一个假想尺寸例子，说明代码里两个量的区别。实际图尺寸取决于运行配置。

graph padding 会影响耗时，但它不是多验证了一批真实草稿。调参时若只看执行张量的 padded size，不看分给请求的实际长度，就容易把这两类工作混为一谈。

## sampler 不只是从分布里抽一个 Token

[`model_runner.py:337至365`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/model_runner.py#L337-L365) 的 `device_sample` 计时包住整个 `model.sample`。

decode 路径继续经 [`sampler.py`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/sampler.py#L225-L251) 调到 DFlash，其中包含下一批草稿的计算或图重放。所以不能把 `sampler_ms` 理解成只测最后的随机抽样。

漏掉这部分会低估一轮的成本，也可能改变最佳验证预算。是否减少目标模型验证，并不能只用主模型 forward 的时间来算。

## 启动曲线没有被逐点重写

硬件曲线不包含所有运行时变化，例如上下文长度带来的负载。实现用 forward 的归一化残差修正它：

```text
(forward_ms − baseline_ms) / graph_size
```

[`ForwardRuntimeEstimator`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/mtp_varlen.hpp#L44-L131) 保留最近 5000 步，按图尺寸分组。一个组至少有 10 个样本才参与，先取各组中位数，再对组中位数取中位数。各图等权，避免某个常用图仅因为样本多就完全主导估计。

采样阶段另用 9 样本窗口，至少 5 样本才返回估计。优先取相同请求数对应的值，样本不足再退到全局值。forward 斜率被限制为非负，启动曲线作为估时下界。

这不是把启动曲线上每一个点反复覆盖成刚刚测到的数，也不是看到一轮变慢就立刻把全部预算砍掉。

## 第 N 轮调度拿到的，不一定是第 N−1 轮结果

预算算法里最值得读的注释，解释了为什么同时维护 `accept_rates`、`prev_accept_rates` 和两个计数器。

- `schedule_cnt == process_cnt`：上一轮结果已处理，`accept_rates` 对应 N−1，`prev_accept_rates` 对应 N−2。
- `schedule_cnt == process_cnt + 1`：最新一轮还在途，CPU 的 `accept_rates` 本身就只到 N−2。

总预算始终用 N−2 信息算。逐请求长度则使用当前可得的较新信息：非 ZOS 可以直接在 CPU 使用 N−1；ZOS 路径再由 GPU 修正。

源码还有一个约束：后面位置的 confidence 不应反过来影响前面草稿是否验证。讨论这项优化时，不能简单把“CPU 已经等到了最新概率”当成无代价前提。

## GPU 长度修正在主 forward 之前

看 [`worker.py:113至234`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py#L113-L234)，一轮并不是严格的“调度、forward、处理结果、结束”四段串行：

1. driver 为本轮准备长度修正，必要时在 GPU 上重分配。
2. 启动本轮主 forward。
3. 处理上一轮已完成的输出。
4. 本轮 sample 前，等待长度 D2H 并更新 CPU 侧采样元数据。
5. 执行本轮 sample；ZOS decode 将本轮收尾留到下一轮。

修正需要有效的上轮状态、两批请求有交集，以及变长验证开启。新请求没有可 remap 的旧 confidence，不能凭空套上上一批某个请求的值。

CUDA TileLang 实现分两个 kernel 阶段：先 remap 并计算前缀累乘概率，再用并行排名计数选预算内的位置。GPU 改的是各请求分到多少，总预算仍沿用 CPU 的决定。

这也解释了为什么不能把它概括成“每轮先在 CPU 排一遍，再在 GPU 排一遍”。非 ZOS 可以直接在 CPU 分配；没有重叠请求时，也没有这次修正。

## 哪些条件没满足，它就不会做预算搜索

`DEEPINFER_USE_NEXTN_DECODE` 默认 5，`DEEPINFER_ENABLE_MTP_VARLEN` 却需要另外开启。默认草稿数与变长验证开关不是同一个配置。

正常初始化中，启动曲线由 CUDA Graph 路径提供。曲线为空时，`schedule` 不搜索预算，直接为每个已调度请求分配一个主 Token 位置，再加现有的全部草稿位置。

全部验证也不是全部接受。目标模型的验证仍会决定最后能留下哪些 Token。

另外，GPU 修正本身不要求“开启 CUDA Graph”这个条件。曲线的生成、运行计时回灌、异步长度修正是不同分支，不能因为它们都与变长验证有关，就合并成一个开关判断。

## 约 6% 的收益该归给什么

[官方报告](https://zhuanlan.zhihu.com/p/2095560295930508953) 比较的是线上变长验证与固定验证五草稿，双方已经都使用投机解码。约 6% 是在此基础上的额外收益，不是投机解码相对于逐 Token 生成的总加速。

独立评估时，我会先确认开关、曲线和长度分配真的生效，再记录实际预算、padded graph size、接受长度与阶段耗时。否则，测试可能只证明“用了五草稿”，而没有测到想比较的那项优化。

## 源码入口

- [预算与运行时估时](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/mtp_varlen.hpp)
- [变长验证环境变量](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/environ.py#L128-L132)
- [CUDA 分配实现](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py)
- [上一篇：YoCo 的可见域与裁行](01-yoco-bounded-replay.md)
- [下一篇：recipe 怎样保留 Token 身份](03-recipe-token-identity.md)

代码短引文来自 Apache-2.0 许可的 DeepInfer。这里的源码解释不等于已复现其线上收益，也不代表官方部署建议。
