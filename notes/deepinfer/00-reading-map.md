# DeepInfer / recipe：首轮源码阅读记录

2026-10-20。对照 DeepSeek 的[部署报告](https://zhuanlan.zhihu.com/p/2095560295930508953)，先追了有限窗口回放、变长验证和协议编解码三条调用链。这里记录能从当前代码确认的条件，后续再分别展开。

固定版本：

- DeepInfer：[`8b161497`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b)
- deepseek-recipe：[`935817c6`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)

本轮是源码阅读，没有运行模型或集群 benchmark。文中提到的测试均为阅读已有断言，不代表已经在本地跑过。

## YoCo：mask 和 compact 省的不是同一份计算

入口从 `deepinfer/model_executor/models/deepseek.py` 的层循环看起。在首个 YoCo consumer 层，`mask` 换入收窄 SWA 可见域的 metadata；`compact` 则在切 metadata 后，由 `decoder_layer.py` 的边界处理 gather 出保留的 hidden states 和 residual。

两种模式可以具有相同的 SWA 可见范围，但 `mask` 保留全部 query 行，`compact` 才让后续 consumer 层少处理这些行。完整输入中仍被 full-source 路径需要的状态另行保留，并非简单地把所有前缀数据删掉。

保留范围在 C++ 中按 `data_len - keep_tokens` 计算，可见下界还要与滑窗下界和有效状态下界取交集。截断默认关闭；启用时，代码检查 `keep_tokens >= window_size + max_backtracking_tokens`。README 推荐 compact 模式保留 140 个 Token，不能把它直接替换为生产报告中描述的 128-token 回放窗口。

阅读位置：

- [截断元数据与约束](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp#L149-L359)
- [模型层循环的 metadata 切换](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/deepseek.py#L190-L207)
- [hidden / residual 的 compact 边界](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/decoder_layer.py#L134-L152)
- [等价验证的范围](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/examples/yoco/README.md)

最后一项比较 mask 与 compact，不是证明有限回放与完整精确重建等价。

## DSpark：五个草稿不等于默认变长验证

`DEEPINFER_USE_NEXTN_DECODE` 默认是 5，但 `DEEPINFER_ENABLE_MTP_VARLEN` 需要另行开启。正常初始化路径中，启用 CUDA Graph 与变长验证才会提供对应的性能曲线；没有曲线时，调度器退回验证现有全部草稿。

模型返回的逐位置 confidence 在 `process_MTPInfo` 里被累乘成前缀接受率。`MTPVarlen::schedule` 比较预期有效 Token 数与预计耗时的比值，选择总预算。运行时修正的是前向耗时残差与 sampler 估计，不是直接重写启动曲线上的每个点。

CPU 选择总预算使用较早的置信度。异步 decode 留下前一步状态后，对于两批有交集的请求，GPU 才进一步修正逐请求长度；不开异步调度时，CPU 可以直接分配。这里的条件不能省略成“所有请求都先 CPU 排，再 GPU 重排”。

阅读位置：

- [MTP 环境变量](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/environ.py#L128-L132)
- [confidence 的前缀累乘](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/scheduler.hpp#L1239-L1250)
- [预算搜索与退化路径](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/mtp_varlen.hpp#L150-L266)
- [GPU 修正的调用条件](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py#L182-L219)

## Recipe：区分控制标记，也保留失败状态

输入端的 `special_ranges` 在构建 Prompt 时记录。`PromptTokenizer` 另外准备一个清空 added vocabulary 的 tokenizer 副本，用它批量编码普通文本区域，再按原顺序插回显式标记的 ID。外部普通文本不能只因为写出了同样的标签，就获得结构身份。

图片占位符、assistant 历史里指定的 grounding 标记，以及调用方显式构造的特殊区间，另有保留规则。不能把这个机制理解成“一律剥掉所有 special token”。

输出端也要分 Token-ID 与纯文本两条路径。前者带有标记身份，后者只能按拼写匹配。普通 function 工具名先被缓冲，读到结束引号才发出；参数随后转换为增量 JSON 片段。转换不意味着修复：测试明确保留被截断的半截参数以及 `Length` 结束原因。

阅读位置：

- [Tokenizer 副本与区间编码](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/tokenizer.rs#L46-L116)
- [Assistant grounding 标记](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/v4/mod.rs#L20-L33)
- [工具名缓冲与参数片段](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/processor.rs#L333-L360)
- [截断工具调用的测试](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/processor.rs#L940-L969)

## 后续

接下来几天继续完整梳理两个仓库，把这些调用链、配置差异和测试展开成系列文章。完整集群的 KV 存储、跨实例路由和部署性能，仍需与官方报告区分：当前开源仓库没有交付整套生产服务，不能把报告数据当成源码阅读已经验证的结果。
