# AI Infra Notes

[English](README.md) · 中文

我在阅读和维护 AI 基础设施项目时写下的技术分析与源码导读，主要关注推理系统、KV Cache、调度与通信、模型服务协议，以及 Agent 运行时。

每篇尽量沿一条具体调用链讲清楚：设计解决什么问题，工作在哪一步省掉，依赖哪些条件，又该怎样验证。内容不局限于某个框架或厂商。

## DeepInfer / deepseek-recipe 源码系列

DeepSeek 于 2026 年 10 月 20 日公开 DeepInfer 推理核心与 deepseek-recipe 编解码组件。目前五篇从实际调用链出发，既看计算与状态，也看批量失败和工具历史的协议边界，不把生产报告中的整套基础设施当成已经随引擎交付。

| 文章 | 这篇具体看什么 |
| --- | --- |
| [01 · YoCo：少算发生在哪里](notes/deepinfer/01-yoco-bounded-replay.md) | 可见域与裁行、chunk 占位、ring 写入、完整来源与 compact query、CP 布局变换 |
| [02 · DSpark：预算在什么时候决定](notes/deepinfer/02-dspark-verification-budget.md) | 累计接受率、图尺寸与实际预算、包含起草的采样计时、N−2/N−1 时序与 GPU 修正 |
| [03 · recipe：为什么不先变成字符串](notes/deepinfer/03-recipe-token-identity.md) | 显式特殊区间、真实 tokenizer 对比、Token-ID/Text 两条路径、跨 chunk 标记与半截工具参数 |
| [04 · 批次排空以后，才轮到返回结果](notes/deepinfer/04-offline-generation-failures.md) | 失败终态、最后一次 flush、池化通知的生命周期，以及未入队输入造成的短列表 |
| [05 · 工具从声明删掉，名字为什么还不能复用](notes/deepinfer/05-history-tool-identity.md) | 历史工具身份、namespace 别名、先保留后分配，以及原生回归的验证边界 |
| [首轮源码阅读地图](notes/deepinfer/00-reading-map.md) | 首轮入口、当时验证范围与后续补录 |

固定阅读版本：DeepInfer [`8b161497`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b)，recipe [`935817c6`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)。DeepInfer 随后的 `774ce244` 只调整部署文档，本文仍使用原代码锚点。

想先动手跑推理状态，可以看我另做的 [Core DeepInfer](https://github.com/he-yufeng/core-deepinfer)：用随机未训练参数的 NumPy CPU 模型观察前缀恢复、验证预算与 KV 回滚，[PyPI 0.1.0](https://pypi.org/project/core-deepinfer/0.1.0/) 已可安装。它不加载 DeepSeek 权重、不实现 recipe 编解码；本仓原版源码分析与小参考的验证范围分别保留。

第三篇附有可运行的 [tokenizer 对比示例](notes/examples/token_identity.py)。使用本地 V4.1 tokenizer JSON，不下载权重、不调用 API：

```bash
python -m pip install 'tokenizers>=0.23,<0.24'
python notes/examples/token_identity.py /path/to/deepseek-recipe/static/tokenizers/v41/tokenizer.json
```

后续技术分析和源码导读会持续更新在这里，也会加入其他 AI Infra 项目。觉得这些分析有帮助，欢迎点个 Star，后面继续一起读代码。

## 验证范围

文章标出源码版本，方便对照。官方性能披露、源码推导、本地实验分别说明，不互相替代。

前三篇保留最初阅读时的范围。第四、五篇另行说明2026-10-20留下的验证：DeepInfer真实Python入口加替身24项回归，以及Rust1.92下recipe的85个原生测试；后者不含OpenCV/Python完整workspace。修复分支仍开放，不等同上游已合并。本轮未执行的重跑不计成新验证，也没有DeepInfer原生scheduler、GPU/NPU模型或集群benchmark结论。

短代码引用遵循对应项目许可，原始实现与归属通过固定源码链接保留。欢迎在 issue 中附上版本、代码位置或复现条件指出错误。

作者：[何宇峰 / Yufeng He](https://github.com/he-yufeng)。
