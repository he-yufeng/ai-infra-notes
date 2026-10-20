# DeepInfer：批次排空以后，才轮到返回结果

2026-10-21。本文读的是离线生成入口，不讨论线上服务。原始代码固定在 [`8b161497`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b)；修复对照使用 [PR #85 的 `4130134`](https://github.com/he-yufeng/DeepInfer/commit/41301341fe9c9d0dec971b7f800a49ea242ed609)，该 PR 当前仍未合并，不能把分支行为说成上游主线行为。

`LLM.generate()` 最后返回几条字符串，看上去只是一个很薄的接口。真正要小心的却不是字符串，而是它把什么东西当成了输出：模型生成的 Token、请求终态通知，以及根本没有进入引擎的输入，不是同一种情况。

## `finished` 只说明停止了

先看 [`get_finished_reason`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/sequence.hpp#L30-L63)。正常结束有 `stop` 和 `length`，其余还包括主动中止、prompt 超长、资源不足、结构化输出失败和视觉处理失败。

这些状态都可以让 `finished` 为真。若收集器只按这个布尔值判断，然后把带着 `token_id` 的对象全部送去 tokenizer，就把“通知请求结束”误当成了“模型又生成了一个正常 Token”。

这条错误尤其隐蔽，因为通知带的 ID 未必越界。原始 [`outputs.hpp` 的 ignored-sequence 分支](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/outputs.hpp#L297-L303)使用输入的最后一个 Token 构造结束通知。拿一个合法词表 ID 去 decode，不会自动告诉调用方这是失败路径。

所以不能靠“tokenizer 没抛异常”推导生成成功。终态语义必须在通知还没有被拍平成字符串之前保留下来。

## 为什么不能看到失败就立刻抛

[收集循环](https://github.com/he-yufeng/DeepInfer/blob/41301341fe9c9d0dec971b7f800a49ea242ed609/deepinfer/entrypoints/llm.py#L181-L212)每轮先判断是否还有未完成请求，有则 `step()`，没有则调用 `flush_inflight_reqs()`。后者不是多余的一次空循环：异步路径仍可能有最后一批通知尚未交付。

如果 A 先失败，B 还在运行，在遍历 A 的通知时直接抛异常，会让这次批量调用提前离开，B 以及在途结果没有经过原本的收尾路径。下一次复用同一个入口时，旧工作和新工作还可能混在一起。

修复因此把两个阶段分开：先收集、克隆通知并排空批次，包括最后的 flush；再检查每个请求最后一条通知的结束原因，决定能不能返回解码文本。正常长度截断仍然是正常结果，不与资源失败混为一谈。

这里的克隆也有独立理由。底层返回的是池化对象的句柄，下一轮可以回收。入口在收集当下调用 `RequestOutput.clone`，不是把借用的句柄一直攒到结束再读取。生命周期正确与终态判断正确，两件事都要成立。

## 还有一类失败，没有任何通知

`LLMEngine.add_request` 会先检查 Token 取值。负数或超出当前可用词表范围时，它在 [Python 入队校验路径](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py#L580-L589)记录拒绝并返回 `None`，不会创建可供后续收集的 sequence。

假设输入依次是 A、B、C，B 在这里被拒绝。只等输出循环结束，得到的会是 A、C 两项。即使把 request ID 按数值排序，缺失的位置也不会自己恢复；调用方很容易把第二项继续当成 B 的结果。

因此，`generate` 还必须记下 `_add_request` 返回 `None` 的输入下标。接受的请求仍先排空，再明确报告被拒绝的下标，不能返回一份看起来成功、实际上缩短了的列表。

两类错误不应混用：已入队后失败靠结束通知识别，未入队靠 admission 返回值识别。对一个从未创建的请求等待终态，只会把接口契约想错。

## raw 模式为什么保留

解码模式已经丢掉结束原因，遇到异常终态需要显式报错。raw 模式则返回逐请求通知，让上层自行区分失败与正常结束；它不应把有意义的失败通知删掉。

但 raw 模式也不能忽略未入队输入，因为那类输入根本没有通知可返回。这就是为什么入队拒绝仍由 `generate` 统一报错，而终态通知的处理可以根据 `decode_sequence` 不同。

这不是一套返回所有成功和失败对象的新 API。它是在保持现有返回类型的前提下，避免把失败伪装成文本或输入顺序。后续如果设计具名结果结构，还可以更完整地表达同批部分成功，但那是另一个接口决定。

## 验证到哪一步

2026-10-20 留存的验证覆盖真实 Python 入口函数与测试替身，共 24 项：固定修复源码全部通过，同一套测试放到基准源码是 20 失败、4 通过。既有绿测源码 SHA-256 与当前 PR `4130134` 的入口文件一致。

测试包括混合批次排空、最后一次 flush、raw 保留、正常 stop/length、未入队下标，以及失败后再次调用入口。它没有运行原生调度器、CUDA/NPU 或完整模型，不能据此声称异步硬件执行已经完成端到端验证。本次写作读取并复核已有记录，没有把未执行的重新运行计入结果。

仓内测试位于 [PR 分支的 `test_llm_outputs.py`](https://github.com/he-yufeng/DeepInfer/blob/41301341fe9c9d0dec971b7f800a49ea242ed609/tests/entrypoints/test_llm_outputs.py)。有对应导入环境时可检查这份测试；原生扩展可导入，也不等于这些替身测试已经覆盖了原生计算。

这条路径给我的提醒很具体：批量推理接口的正确性，不只在采样的那一刻。状态在哪里变成普通文本，缺失项在哪里变成短列表，往往决定了失败能不能被调用方看见。
