# Kimi K3.1 工作流验证例子

[English](README.md) · 中文

模型说“完成了”，和系统有证据确认完成了，中间还隔着几步。这里把两件事做成可以直接运行的小例子：

- 原生Chat工具循环：保留推理与工具调用历史，核对流式结束状态，记录普通输入、缓存读、缓存写与输出。
- 一块带两个通孔的板：生成真实几何，连续修改两次，导出STL，再重新读入文件验收。

默认完全离线。协议使用合成SSE夹具，几何使用固定参数，不冒充模型输出。不需要权重、API key、浏览器或GPU，也不执行模型生成的任意代码。

![固定参数生成的三版真实几何，均通过STL导出重载检查，不是模型生成结果](assets/offline-edits.png)

## 从仓库根目录运行

锁定依赖环境使用Python 3.14。本轮本地验证环境为macOS arm64、Python 3.14.5。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r notes/examples/kimi_k31/requirements.txt
.venv/bin/python -m notes.examples.kimi_k31
```

默认在`.artifacts/`里创建带UTC时间的新目录。也可以指定`--output 新目录路径`。目标已经存在时会拒绝运行，不覆盖已有文件。

输出包含一份`offline/report.json`，以及初始版、扩大孔径版、加宽板件版各自的STL和PNG预览。报告明确标注夹具来源，离线模式的`live_requests`为0。

## 几何真的生成了什么

初稿长100、宽60、厚8毫米，两个直径6毫米的通孔，孔距60毫米，围绕中心对称。

第一轮只把孔径改成8毫米，第二轮只把板宽改成70毫米。每次都检查其他要求是否保留。生成器只接收五个有限范围的数值，不接收Python代码、shell命令或任意文件路径。

几何由trimesh和Manifold构建。验收看的是实际网格，而且STL导出后还会重新加载再验一次：

- 顶点有限、网格封闭、正体积、单一连通实体；
- 以原点为中心的长宽厚边界；
- 两个贯穿孔对应的拓扑，以及考虑离散圆柱误差后的体积；
- 指定孔位的贯穿圆柱探针不能被材料堵住；
- 覆盖完整板厚的环形探针必须仍有孔壁材料；
- 连续修改时没有顺手改变未要求修改的参数。

圆柱使用96段离散，孔壁探针留0.05毫米径向余量，包围盒容差为0.001毫米。这是有明确范围的几何验证，不是通用CAD等价证明，也不涉及强度、材料或制造认证。

反例测试不只有“忘了打孔”。还包括孔距错误但体积相同、孔径过大、盲孔、非封闭网格，以及通过缩小中段孔径来补偿体积的扩孔。最后这个反例的孔口直径9毫米，而要求是6毫米；只检查孔壁中间80%的高度，会把它放过，因此检查必须覆盖完整厚度。

预览图使用CPU逐像素深度缓冲，避免按三角形中心排序造成遮挡伪影。它只是几何投影，不承担独立验收职责。

## 协议里保留了什么

`ChatStream`处理跨字节块的UTF-8、推理文本、工具参数和空`choices`的usage尾包。只有收到完整`[DONE]`及支持的结束原因，才能返回完整结果。输出截断、断流、未知字段、调用ID冲突和用量计数倒退都会报错。

例子只提供一个只读工具，读取合成板件规格。工具调用完成后，把完整assistant消息和对应`tool_call_id`的工具结果放回下一轮历史，不拿一段自然语言摘要代替协议数据。

输入分项按原生Chat契约拆开：

```text
普通输入 = prompt_tokens - 缓存读取 - 缓存写入
输入总量 = 普通输入 + 缓存读取 + 缓存写入
输出总量已经包含reasoning tokens，不再额外相加
```

没有usage就记未知，不记零。多次usage更新按累计快照处理，不重复相加；同一响应的总输入或总输出倒退，会直接拒绝，不能把最后一个零包当成免费请求。

价格表固定为2026年10月24日的官方国际API价格，记录了默认5分钟缓存写入价。1小时缓存写入价在这个例子里保持未知；需要它时费用结果会标不完整，不造一个精确总价。这里是按用量估价，不是账户实际账单，也不含外部工具费用。

## 可选的原生API烟测

只有加`--live`才会访问原生端点`https://api.moonshot.ai/v1/chat/completions`。程序不会读取凭证文件、自动充值、下载权重、自动切模型或把OpenRouter key转发到原生接口。

运行环境已有合法原生key，通过`MOONSHOT_API_KEY`提供时：

```bash
.venv/bin/python -m notes.examples.kimi_k31 --live
```

默认使用`low`思考档，适合小额协议烟测。可选`--effort high`或`--effort max`，但不会因此扩大输出上限。另一支持型号通过`--model kimi-k3.1-pro-highspeed`选择。

第一次请求前，预算会写入`live/run-config.json`：最多6次请求，失败也计数；每次最多请求4096输出token，请求体最多16 KiB；整个live工作流600秒；保守费用预留最多2美元；HTTP自动重试为0。

正常流程是五次请求：两次完成只读工具循环，三次完成初稿和两次参数修改。发送内容只有公开的合成板件参数。key只放认证header，不进提示词；报告不保存请求header，并对出现的当前key做脱敏。

预留在请求发出前计算，使用请求字节数加封装余量、以及高于所支持模型的单价上界。失败请求不退回预留。它是在已知价格假设下的客户端保守估计，不是服务端账户消费硬限。

不跟随HTTP重定向。单操作超时与整个工作流的外层deadline共同约束停滞流和退出清理。服务端报错不会自动重试，清理失败也不会留下`accepted=true`。

`live/report.json`保留模型、端点、思考档、请求大小、输出限额、耗时、原始usage、费用估计和验收结果。即使任务输出正确，只要用量或费用不完整，整轮也不会报完全通过。凭证、权限或配额门会记录为资源阻塞，并返回非零退出码。

这只是协议与参数修改烟测，不是BenchCAD复现、视觉理解测试、模型排名或制造验证。仓内离线结果不代表已经取得成功的线上模型推理结果。

## 测试与静态检查

```bash
.venv/bin/python -m pip install -r notes/examples/kimi_k31/requirements-dev.txt
.venv/bin/python -m unittest discover -s notes/examples/kimi_k31/tests -v
.venv/bin/ruff check notes/examples/kimi_k31
.venv/bin/ruff format --check notes/examples/kimi_k31
```

测试使用真实几何布尔运算；网络边界用本地HTTP替身，不调用收费模型。测试临时产物只写仓库内被忽略的`.artifacts/`目录。

## 来源与上游关联

- [K3.1 Pro官方指南](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart.md)
- [原生Chat API契约](https://platform.kimi.ai/docs/api/chat.md)
- [国际API价格来源](https://platform.kimi.ai/)
- [Kimi Code #4368](https://github.com/MoonshotAI/kimi-code/issues/4368)：两套当前用量转换器丢失缓存写入分项的报告。其证据是源函数复现，不把问题报告说成已合并修复。

本目录新代码使用局部[MIT许可](LICENSE)。不重新授权第三方引用内容或仓库原有文章。
