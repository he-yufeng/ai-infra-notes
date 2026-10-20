# recipe 为什么不先把一切变成字符串：Token 身份与流式工具调用

2026-10-20 · deepseek-recipe 源码导读 03

> 系列配套：[Core DeepInfer](https://github.com/he-yufeng/core-deepinfer) 是我做的随机参数 NumPy 推理状态参考，可以先跑请求、恢复和回滚。本文的 recipe Token 编解码不在这个小实现里，两条阅读线分开。

模型输出经过一个网关，工具调用行为就可能变化。有时问题并不在权重，而是在转换中丢掉了一点信息：这段字符来自一个控制 Token，还是来自普通 Token 拼成的文本？这是一个完整调用，还是只收到了开头？

recipe 的输入编码与输出解析，围绕这些区别保留了相当明确的边界。这篇沿 `Prompt → PromptTokenizer → StreamDecoder → StreamProcessor` 读下去，固定于 [`935817c6`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)。

本文实际用随仓 V4.1 tokenizer 运行了一个 Python 编码对比，结果在下文给出。Rust 状态机与测试属于静态阅读，没有运行 Rust 测试或模型推理。

## 相同拼写，不应自动变成同一个角色边界

假设用户让模型解释协议，正文里出现 `<｜User｜>`。或者工具读回一份文件，里面恰好有同样的字符串。它们应该仍然是内容，不能仅凭拼写获得新建一条用户消息的能力。

如果先把模板和所有消息拼成一个字符串，再交给会匹配 added tokens 的 tokenizer，边界已经很难区分了：模板插入的标签和外部文本写出的标签，看起来完全一样。

recipe 的做法是构建时保留身份，而不是编码时重新猜。

## special_ranges 是构造信息，不是全文扫描结果

[`Prompt`](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-core/src/prompt.rs#L19-L66) 内部有文本和 `special_ranges`。调用 `push_special` 时记录新 Token 的区间，`push_plain` 只追加文本。

拼接一个已有 Prompt，也会平移并保留它原来的区间。不是拼完后搜索所有看起来像标签的字符串。

Rust 内部记录的是 UTF-8 字节范围，Python 暴露的区间则换成 code-point 索引。消息含中文时，这两类偏移不能直接互换。

这也不意味着所有角色中的所有特殊内容都被当成普通文字。图片占位符是显式插入的；assistant 正文与实际渲染的 reasoning 中，指定 grounding 标记另有保留规则。直接构造 Prompt 的调用方也可以主动插入特殊区间。

机制保证的是普通文本不会仅凭相同拼写得到结构身份，不是对所有调用方式进行无差别清洗。

## 普通文本使用另一份 tokenizer

[`PromptTokenizer::new`](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/tokenizer.rs#L46-L67) 保存原来的 added vocabulary，然后创建副本：

```rust
let mut plain_tokenizer = tokenizer.clone();
plain_tokenizer.with_added_vocabulary(AddedVocabulary::new());
```

清空的是这份副本的整个 added vocabulary，不只是 `special=true` 的条目。

编码时，特殊区间之间的普通文本组成一个 batch，交给副本编码；显式区间按原词表查对应 ID，再按顺序拼起来。是否并行由 tokenizers 的配置控制，不是 recipe 自己为每条消息启动线程。

### 一个实际运行的编码对比

使用仓库 `static/tokenizers/v41/tokenizer.json`，tokenizers 0.23.2，输入为：

```text
file contains <｜User｜>
```

`<｜User｜>` 的 added-token ID 是 `128803`。直接编码整段文本得到：

```text
[8104, 7272, 223, 128803]
```

从 tokenizer JSON 删除 `added_tokens`，重新构造 tokenizer 后编码同一段文本：

```text
[8104, 7272, 818, 28217, 6756, 28217, 32]
```

第二种不再包含 `128803`。这个对比只验证清空 added vocabulary 的局部编码行为，不是运行 recipe 的完整 Rust 编码链，也不证明模型不会受到自然语言诱导。

可用 [本仓示例](../examples/token_identity.py) 指定自己本地的 tokenizer 文件重跑，无需权重或 API key。

## 输出端同样要保留来源

[`StreamDecoder`](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/decoder.rs#L144-L189) 收到 Token ID 后，检查它是否属于 added vocabulary。真正的 added token 单独解码，输出带 `is_special_token` 标志；普通 Token 则组合解码。

[`StreamProcessor`](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/processor.rs#L133-L160) 据此向 parser 传入三类来源：

| 输入来源 | parser 看到的类别 | 标记识别方式 |
| --- | --- | --- |
| 已解码 text chunk | `Text` | 按拼写匹配 |
| 普通 Token 解码结果 | `PlainTokens` | 不获得 special 段身份 |
| 单个 added Token | `SpecialToken` | 可匹配 special 段 |

DSML 标记也不一定整段都是单一 special token。例如 `<`、`</` 等普通片段和 `｜DSML｜` 的特殊片段组合成标记。parser 的规则要同时检查匹配位置和来源，不能只对着最终字符串做一次正则。

若上游已经把所有输出解码成文本再传进来，就只能走 `Text` 路径。后面的库没法从相同字符重新找回原本的 Token 身份。

## 跨 chunk 的标签，不应该每次从头读

输出可能把一个标签切成几段，也可能让一个 UTF-8 字符跨多个 Token 才完整。两种不完整，处理的位置不同。

Token 解码先暂存尚未完整的普通 Token 组合。parser 再用 KMP 保存标签的 `matched_len`，下个 chunk 从匹配进度继续。已经确定属于正文的字节可以及时吐出，不必等整个响应结束。

不过这里有一个明确限制：当前 decoder 用末尾 U+FFFD 判断普通文本可能不完整，EOF 不会自动补刷这部分缓存。真实以 U+FFFD 结尾的文本也可能受影响。阅读时要把它当成当前行为边界，不能概括成“任意切分和任何截断都无损”。

## 工具名完整，不代表参数完整

对普通 function 工具，`tool_name_buffer` 先累计名称，到结束引号才发 `ToolCallNamed`。参数随后逐段转换：字符串参数片段做 JSON 转义，非字符串则保留原始文本。

这允许调用方及时展示流式进度，但并不负责校验每一段 raw JSON，更不会修补截断。

[`output_truncated_inside_a_tool_call_keeps_the_length_finish_reason`](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/processor.rs#L940-L969) 的断言保留这样的结果：

```text
工具名：f
参数：{"a": "tru
结束原因：Length
```

同一测试还让工具名只输出到 `get_cur`。此时可以有调用开始事件，却没有完整命名的调用。

如果应用一收到 `ToolCallBegin` 就执行，或者把半截 JSON 自动补成一个合法对象，它就超出了模型实际产生的内容。执行前仍要等完整参数、核对结束原因、解析并校验 schema，再检查权限。

保留 `Length` 不是一个可有可无的元数据细节。吞掉它，下游就更难区分“模型选择了这组参数”和“生成到一半被切断”。

## Responses 的工具不全是 function

库里还有两条不同的参数处理路径，不能用上面的普通 function 规则代替。

- custom `apply_patch` 用增量 parser 提取并反转义开头的 `input` 字符串，包括跨 chunk 的转义序列。
- client tool_search 会累计 raw arguments，在 item 关闭时再解析，用于相应协议输出。

输入端也要处理 namespace 的展开与回映。扁平化后的名字需要唯一，历史调用与新定义需要一致，最终 response 才能还原 namespace 与本地工具名。这些不是给 JSON 字段简单改个名字就能完成的工作。

## 接网关时，我会怎样检查

先用相同的消息历史，比较前后两条链最终生成的 Token ID，而不只比较 prompt 文本。两段文字完全相同，特殊区间来源也可能不同。

再保留几种很短的输出轨迹：正文中的 `</think>`、普通 Token 拼出的 DSML 标签、跨 chunk 的真标记、不完整工具名、被 Length 截断的参数。分别走 Token-ID 和 Text 两条路径，检查预期差异是否仍然存在。

最后再检查执行层：参数截断以后，工具是否真的没有被调用。一个 parser 保留了失败原因，不代表应用已经正确使用它。

以上处理解决协议层面的结构混淆，不构成完整的提示注入防线。普通语言本身就能诱导操作，权限和工具副作用仍属于应用必须控制的部分。

## 源码入口

- [输入模板与 grounding 例外](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/v4/mod.rs)
- [流式 parser 与 ChunkKind](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/output_parser.rs#L74-L94)
- [工具名与参数片段转换](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/processor.rs#L333-L360)
- [custom 工具输入 parser](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/protocol/openai/responses/response/custom_tool_input.rs)
- [上一篇：DSpark 的预算时序](02-dspark-verification-budget.md)
- [系列目录](../../README_CN.md)

短代码引用来自 MIT 许可的 deepseek-recipe。本文为独立源码分析，不代表官方接口保证；实际接入应核对使用版本。
