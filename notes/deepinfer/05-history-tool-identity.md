# recipe：工具已经从声明里删掉，名字为什么还不能复用

2026-10-21。原始版本固定为 [`935817c6`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)，修复对照为 [PR #41 的 `32ca266`](https://github.com/he-yufeng/deepseek-recipe/commit/32ca266005b8d63ed312683b29d8fe50875a3b20)。问题先记录在 [Issue #40](https://github.com/deepseek-ai/deepseek-recipe/issues/40)，PR 当前仍开放，本文不把它写成已合并行为。

> 这一篇继续讲 recipe 工具协议；推理执行部分另有我做的 [Core DeepInfer](https://github.com/he-yufeng/core-deepinfer) 可运行参考。它用随机参数 NumPy 模型观察缓存、验证和回滚，不包含本文的 Responses 工具名映射。

第三篇讨论的是 Token 的控制身份。这次沿着 Responses 的工具转换继续往下看，会遇到另一种身份丢失：两个不同的工具，到了模型对话里却变成了同一个名字。

## `call_id` 没撞，工具身份仍然能撞

Responses 可以把工具表示为 `(namespace, local_name)`。模型侧的工具声明需要一个扁平名称，recipe 会按模型版本转换。例如 V4 的 `ns/run` 可能变成 `ns__run`，V4.1 则使用 `ns::run`。

现在对话历史中有两次调用：

```json
[
  {"type":"function_call","call_id":"namespaced","namespace":"ns","name":"run","arguments":"{}"},
  {"type":"function_call","call_id":"bare","name":"ns__run","arguments":"{}"}
]
```

第二个本来就是一个顶层工具。它与 namespace 里的 `run` 不是同一个函数，即使两个 `call_id` 各不相同，转换后的工具名仍可能相同。

所以这里不是“返回结果串错了 call ID”的问题。调用与结果仍可按 ID 配对，但模型读到的函数身份已经合并了。保住关联 ID，不能代替保住工具身份。

## 只查本轮 `tools` 为什么不够

一个会话可以保留旧调用，但不再把旧工具放进本轮声明。比如第一次用了顶层 `ns__run`，后面只开放 namespace `ns` 下的 `run`，历史仍然必须重放。

如果名字分配器只把本轮顶层声明放进保留集合，旧的 `ns__run` 就看起来像一个空闲名字。当前 namespace 声明生成别名时占用了它，之后历史裸调用仍原样叫 `ns__run`，两种身份在同一份 prompt 里合到一起。

没有当前 namespace 声明时也有相似路径：转换历史遇到 namespace 调用，会临时分配一个扁平名。若尚未扫描到后面的裸调用，它仍不知道这个候选已经属于另一种历史身份。

这也是为什么只改某一条历史分支、或者只调整遍历顺序，并没有解决根因。把历史顺序调过来，错误不应该重新出现。

## 先保留身份，再分配别名

修复在 [工具转换初始化阶段](https://github.com/he-yufeng/deepseek-recipe/blob/32ca266005b8d63ed312683b29d8fe50875a3b20/deepseek-recipe/src/protocol/openai/responses/request/tools.rs#L71-L100)扩展已有的顶层保留集合：无 namespace 的历史 function call，以及历史 custom tool call 的名字，都在任何 namespace 扁平化之前加入。

裸历史名保持原样；namespace 名在现有 `unique_name` 规则下寻找不冲突的别名。如果基础候选已占用，就继续尝试带后缀的名称，而不是重新命名用户原有的顶层工具。

[历史 namespace 转换](https://github.com/he-yufeng/deepseek-recipe/blob/32ca266005b8d63ed312683b29d8fe50875a3b20/deepseek-recipe/src/protocol/openai/responses/request/tools.rs#L237-L283)还区分两种情况：当前已声明的 namespace 使用其已注册别名；未声明的旧身份则在本次转换中用同一份历史映射保持一致。当前声明、历史调用和 response 侧还原映射，不能各自临时拼一个名字。

这里保证的是同一次转换中的一致性，不承诺某个带后缀别名在所有未来请求里永久不变。跨协议需要保留的是结构化身份与映射，不是把某个字符串偶然当成全局主键。

## custom tool 也会留下历史身份

`apply_patch` 是一个容易看清的例子。当前 namespace 叫 `apply_`，本地名是 `patch`；V4 的连接规则会拼出 `apply_patch`。如果历史中已有同名 custom tool，而它已经不在本轮声明里，只保留 function 历史仍然会漏掉这一次冲突。

所以新增测试里有一例专门保留旧 custom call 的名字，并要求当前 namespace 声明避开它。这与“当前 synthetic/custom 工具之间是否互相避让”不是同一件事，后者另有 [PR #32](https://github.com/deepseek-ai/deepseek-recipe/pull/32)，不能把别人的方向也算进这次修复。

## 用什么验证

[三个原生 Rust 集成测试](https://github.com/he-yufeng/deepseek-recipe/blob/32ca266005b8d63ed312683b29d8fe50875a3b20/deepseek-recipe/tests/responses_history.rs)从公开 `ResponsesRequest` 接口构造完整输入，检查转换后的 assistant 调用名、当前声明以及已声明工具的 response 映射。它们内部覆盖 V4/V4.1、两种调用顺序、当前有无 namespace 声明，以及历史 custom call。

2026-10-20 的既有记录中，这三个测试放回基准工具转换源码全部失败，修复后全部通过。扩展到 core、recipe、encoding 和 demo 编译的原生测试，共 85 项实际通过：50+3+18+14。demo 和 rustdoc 的 0 测试只算编译或文档构建，不加入测试数。使用 Rust/Cargo 1.92.0，不替代仓库锁定的更新工具链验证；没有跑 OpenCV/Python 完整 workspace、GPU 或模型质量评估。

在包含这些测试的修复分支上，窄范围命令是：

```bash
cargo test -p deepseek-recipe --test responses_history --locked
```

本次补文读取并核对的是上述既有日志与源码，未把本轮未执行的重跑计为新验证。

这类问题值得从整个会话看，而不只盯着本轮声明。名字没有出现在当前 `tools` 中，不代表它已经从模型可见的历史里消失。协议适配层一旦把结构化身份拍平成字符串，就必须把这个代价算完整。
