# YoCo 的少算发生在哪里：从 SWA 可见域读到层边界裁剪

2026-10-20 · DeepInfer 源码导读 01

> 想先把状态变化跑出来，可以看我做的配套参考 [Core DeepInfer](https://github.com/he-yufeng/core-deepinfer)。它用随机参数的 NumPy 模型观察前缀恢复、有限回放与 KV 回滚，不需要 GPU；不是 DeepSeek 权重推理。下面仍按原版源码分析。

DeepSeek 的部署报告提到，V4.1 在 SWA 状态缺失后只做有限窗口回放，而不沿着多层依赖精确重建。这件事有两层问题：模型是否能接受近似状态，以及引擎怎样真的少执行那些计算。

这篇只跟后一个问题。读 `mask` 和 `compact` 两条路径，能看到一个容易混淆的区别：**让一个 query 少看一些位置，不等于让后面的层少处理一些 query。**

阅读固定在 [DeepInfer `8b161497`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b)。本文没有运行 GPU/NPU 模型。行数算例按源码公式计算，已有等价验证的断言与执行脚本也只作阅读，不当成已跑过的结果。

## 先把精确恢复和截断实现分开

每层只看附近 128 个 Token，不代表多层网络的精确依赖范围也只有 128。当前层的输入来自上一层，上一层又依赖更早的输入。官方报告用 20 层 Decoder 举例，精确恢复需要覆盖最后 2560 个 Token。

有限回放只处理最近 128 个，并把回放时的注意力也限制在这段里。它恢复的是近似状态。长期 Block KV 仍然参与计算，因此这不是把整个模型的上下文长度缩成 128。

V4.1 的 Causal Encoder-Decoder 结构也很关键：Decoder Block KV 可以由 Encoder 末层 hidden state 投影得到，Decoder 再恢复自己的 SWA。报告中近乎减半的 Prefill 计算来自这些条件共同作用，不能只拿走一个窗口参数就期待同样收益。

下面讨论的 `mask/compact`，是在给定截断语义之后，选择怎样实现它。先验证两种实现能否对齐，再评估截断与精确恢复的差异，是两个不同的实验。

## 可见域同时受三个下界约束

入口在 [`prepare_tensor_input_v4.cpp:149至167`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp#L149-L167)。一个 query 的可见左边界这样计算：

```cpp
int32_t abs_lo = std::clamp(
    std::max({pos - window_size + 1, rect_lo, valid_start}),
    0,
    pos + 1);
```

这里不是只有一个滑窗：

- `pos - window_size + 1` 是正常滑窗下界。
- `rect_lo = data_len - keep_tokens` 是本次保留的尾部区间。
- `valid_start` 是状态有效位置的下界。恢复时一部分旧状态可能被清零，不能再读这些槽位。

三者取最大值后再 clamp。对完全位于保留区间之前的 query，边界可以变成 `pos + 1`，也就是没有可见 key。

后续元数据把 key 分成两段：chunk 之前已有的 SWA 状态，以及当前 chunk 内新产生的状态。两个区间都必须服从同一条绝对位置边界。只改当前 chunk 的 mask，而保留对旧 ring 的读取，语义就变了。

## compact 裁的是行，不只是可见位置

同一文件的 [`fill_yoco_truncation_metadata`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp#L203-L289) 决定去掉 chunk 开头多少行：

```cpp
int32_t clip_prefix = compact
    ? std::clamp(data_len - keep_tokens - q_start, 0, q_len - 1)
    : 0;
```

`mask` 的 `clip_prefix` 为 0，所有 query 行仍然存在。`compact` 才会根据保留区间裁行。

这里上界是 `q_len - 1`，不是 `q_len`。一个 chunk 即使完全落在保留区间之前，也留下最后一行占位。

考虑一个单请求、没有 CP 重排的算例：输入总长 1024，两个 chunk 各长 512，保留长度为 140。逻辑保留区间是 `[884, 1024)`。

| chunk | q_start | compact 去掉的前缀行 | compact 留下的行 | mask 留下的行 |
| --- | ---: | ---: | ---: | ---: |
| `[0, 512)` | 0 | 511 | 1 | 512 |
| `[512, 1024)` | 512 | 372 | 140 | 512 |

这只是 consumer 布局的行数推导，不能读成整个模型总共只算了 141 行。前面的 Encoder 和仍需要完整来源的计算并没有一起消失。

占位行也不能污染真实状态。代码只有在 `pos >= data_len - keep_tokens`，并且处于 chunk 最后的 `swa_block_size` 个位置时，才分配 ring 写入槽位，否则写入映射是 `-1`。

所以，还要再区分“保留了多少 query 行”与“向 ring 写入了多少状态”。

## 元数据变小以后，张量也得跟着换

模型层循环在首个 consumer 层切换 metadata；[`decoder_layer.py:134至152`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/decoder_layer.py#L134-L152) 接着处理张量：

```python
compact_inputs, full_inputs = compact_yoco_boundary(
    hidden_states, residual, input_metadata, cross_layer_runtime,
    hc_enabled=self.hc_enabled,
)
cross_layer_runtime.consumer_full_hidden = self.attn_norm_forward(*full_inputs)[1]
hidden_states, residual = compact_inputs
```

这里留下了两套输入。

后续 consumer 的 query 路径处理 compact 后的 hidden states 和 residual。需要 full-source 的路径则保留完整输入对应的状态。把两条路径合成“Decoder 之前把旧数据全丢掉”，就解释不了 `consumer_full_hidden` 为什么仍然存在。

`compact_yoco_boundary` 还会裁剪共享的候选索引，并清空派生的 attention 缓存。原因很直接：query 行号变了，基于旧布局算出的派生结构不能继续用。

## CP 路径不能直接拿 full 索引去 gather

单卡时，边界裁剪可以按 `yoco_consumer_query_indices` 直接 gather。CP 大于 1 时，原张量已经按 full 序列做过 mirror/zigzag 分布，不能再把一个全局 full 下标当成本 rank 的本地下标。

[`yoco_boundary_compact`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/yoco.py#L55-L76) 实际做了三步：

1. 各 rank 选出自己持有、且裁剪后仍需要的贡献行。
2. all-gather 汇总贡献。
3. 按 compact 后的新 CP 布局，为本 rank 重新选择行。

[`get_consumer_view`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/input_metadata/yoco.py#L94-L180) 同时重建 positions、query 边界与相关索引。原 full metadata 与视图仍共享一部分 tensor，这些共享字段不能随便原地修改。

读这段时，值得检查的不只是最终 shape。绝对位置、每请求边界、原布局与新布局之间的映射，都得一起对上。相同 shape 并不能证明行属于同一个请求或同一个位置。

## 128、134、140 不是互相矛盾的配置

| 数字 | 在讨论什么 |
| --- | --- |
| 128 | 生产报告中的 SWA 窗口与有限回放长度 |
| 134 | 报告配置下，128 窗口加 6 个回退位置的 inference ring |
| 140 | 当前 README 给 compact 模式推荐的保留长度 |

实现并没有无条件硬编码“只算 128”。截断默认 `off`，保留长度默认 0；启用时要显式配置，C++ 检查 `keep_tokens >= window_size + max_backtracking_tokens`。

这与投机解码有关。草稿被拒绝后可能需要回退，ring 得覆盖回退后的完整窗口。减小保留长度不能只看当前一步是否能算出结果，还要看回退后的状态是否仍然有效。

## 现有验证回答了什么

仓库 [`examples/yoco/README.md`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/examples/yoco/README.md) 包括元数据比较与固定轨迹的 logprob 比较，也可以覆盖 CP 配置。

它适合回答：相同截断范围下，compact 的布局转换是否比 mask 多引入了差异。

它不适合单独回答：有限回放是否与精确恢复等价，或者某种长工具任务是否会因此多一次重试。后一类问题需要独立的模型质量与任务评估。

这也是我读这条路径后最想继续追的地方：布局正确性可以用严格对照验证，近似恢复是否划算则必须把质量与节省的工作一起量。两类结论各自有依据，不应该共用一句“等价测试通过”。

## 继续阅读

- [模型层循环](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/deepseek.py#L190-L207)
- [状态 ring 尺寸](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/cache_engine.py#L88-L110)
- [DeepSeek 官方部署报告](https://zhuanlan.zhihu.com/p/2095560295930508953)
- 下一篇：[DSpark 的总预算与逐请求分配](02-dspark-verification-budget.md)

代码短引文来自 Apache-2.0 许可的 DeepInfer，完整实现及归属以链接版本为准。本文分析不代表 DeepSeek 官方结论。
