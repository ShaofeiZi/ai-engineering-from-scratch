# 并行工具调用与工具流式传输

> 把三个彼此独立的天气查询串行执行，就需要三次往返。并行运行后，总耗时会缩短到最慢的单次调用耗时。如今，每家前沿提供商都能在一轮中输出多个工具调用。收益实实在在，但接线细节相当微妙。本课会讲解两部分：并行扇出与流式参数重组，并重点说明 ID 关联陷阱。

**Type:** 构建
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** 第 13 阶段 · 第 02 课（深入解析函数调用）
**Time:** 约 75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 存在的原因，以及何时应将其禁用。
- 在并行扇出期间，把流式参数块关联到正确的工具调用 ID。
- 在不过早解析的前提下，把局部 `arguments` 字符串重新组装为完整 JSON。
- 运行三城市天气基准，展示串行与并行延迟的差别。

## 问题

如果没有并行调用，智能体回答“班加罗尔、东京和苏黎世的天气怎么样”时会这样处理：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

这需要三次大语言模型往返，而且每次还要承担执行器延迟，总墙钟时间约为理想情况的 4 倍。

启用并行调用后：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

只需一次大语言模型往返。执行器耗时取三次调用的最大值，而不是总和。OpenAI、Anthropic 与 Gemini 的生产基准表明，在扇出工作负载上，墙钟时间可减少 60%～70%。

代价是关联复杂度。当三个调用乱序完成时，结果必须携带匹配的 `tool_call_id`，让模型能够正确对应。结果以流式方式到达时，必须先把局部参数片段组装成完整 JSON，再执行调用。Gemini 3 加入唯一 ID，部分原因正是为了解决两个并行调用使用同一工具时无法区分的现实问题。

## 概念

### 启用并行

- **OpenAI。** 默认启用 `parallel_tool_calls: true`。设置为 `false` 可强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用并行（Claude 3.5 及以上版本默认如此）。设置为 `true` 可强制串行。
- **Gemini。** 始终具备并行能力；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

当工具之间存在顺序依赖（先 `create_file`，再 `write_file`）、一次调用的输出会成为另一次调用的输入，或速率限制器无法承受扇出时，应禁用并行。

### ID 关联

模型输出的每个调用都有一个 `id`。宿主返回的每个结果都必须包含同一个 ID，否则结果会产生歧义。

- **OpenAI。** 每条工具角色消息使用 `tool_call_id`。
- **Anthropic。** 每个结果使用 `tool_use_id`，它位于对应的 `tool_result` 块上。
- **Gemini。** 每个结果使用 `id`，它位于对应的 `functionResponse` 上（Gemini 3 及以上；Gemini 2 按名称匹配，因此同名并行调用会出错）。

### 并发运行调用

宿主让每个调用的执行器运行在各自的线程、协程或远程工作器上。最简单的框架使用线程池；生产环境使用 asyncio 搭配 `asyncio.gather`，或使用结构化并发。完成顺序无法预测——ID 才是标识符。

一个常见错误是按调用列表顺序，而不是完成顺序回复结果。通常仍然有效，因为模型只关心 `tool_call_id`；但一旦结果丢失或重复，乱序提交会让调试更加困难。应优先按完成顺序回复，并显式携带 ID。

### 流式工具调用

模型流式输出时，`arguments` 会分段到达。三个并行调用各自的数据块会在线路上交错，你需要为每个 ID 建立一个累加器。

不同提供商的形态如下：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（局部字符串）。块中包含 `index`（调用列表中的位置）。按索引累积，在 `id` 首次出现时读取它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流式事件先是 `message_start`，接着每个块收到一个 `content_block_start`，其类型为 `tool_use`（包含 ID、名称、空输入）。`content_block_delta` 事件携带 `input_json_delta` 块，`content_block_stop` 关闭各块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）会输出带 `functionCallId` 的数据块，使多个并行调用能够干净地交错。Gemini 3 之前，流式传输每次只返回一个完整调用。

### 局部 JSON 与过早解析陷阱

在 `arguments` 完整之前，不能解析它。`{"city": "Beng` 这样的局部 JSON 无效，会抛出错误。正确的门禁是提供商给出的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`，或 Gemini 的流结束事件。只有这时才能尝试 `json.loads`。更稳健的做法是使用增量 JSON 解析器，在结构完成时生成事件；OpenAI 流式传输指南建议对展示实时“思考中”状态的用户体验采用这种方式。用括号计数判断完整性并不可靠（引号字符串或转义内容中的括号会导致误判），只能作为非正式调试启发式方法。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

宿主回复仍必须引用对应 ID：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

对 OpenAI 或 Anthropic 而言，回复顺序不影响正确性；只要 ID 匹配，Gemini 也接受任意顺序。

### 基准：串行与并行

`code/main.py` 中的框架模拟三个延迟分别为 400、600 和 800 毫秒的执行器。串行总耗时为 1800 毫秒；并行总耗时为 max(400, 600, 800) = 800 毫秒。差值是固定的，而不是成比例的，因此工具数量越多，节省越明显。

现实世界的注意事项：并行调用会给下游 API 造成压力。对存在速率限制的服务进行十路扇出会失败。阶段 13 · 17 会介绍网关级背压；重试语义计划在未来阶段介绍。

### 流式扇出的墙钟时间

如果模型本身也进行流式传输，就可以在某个调用的参数完整后立即开始执行，而不必等待所有调用结束。这项优化在 OpenAI 文档中有所介绍，但并非所有 SDK 都会暴露。本课框架实现了这一点：模拟数据流一旦生成完整参数对象，宿主就立即启动对应调用。

```figure
tp-parallel-fanout
```

## 投入使用

`code/main.py` 分成两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor`，以串行和并行方式运行三个模拟天气调用，并打印墙钟时间。第二部分重放模拟流式响应——三个并行调用的 `arguments` 数据块在同一数据流中交错——再使用 `StreamAccumulator` 按 ID 重新组装。不使用大语言模型，不访问网络，只有重组逻辑。

需要关注：

- 串行计时约为 1.8 秒；相同模拟延迟下，并行计时约为 0.8 秒。
- 累加器通过按 ID 缓冲，并仅在每个调用的 JSON 完整后解析，从而处理乱序到达的数据块。
- 某个 ID 的参数一旦完成，执行器就立即启动，而不是等待全部数据流结束。

## 交付成果

本课会产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，它会审查哪些工具可以安全并行、哪些具有顺序依赖、哪些会压垮下游速率限制，并返回带有逐工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比值约为 `max/sum`（真实运行会因线程调度、序列化与框架开销而稍微偏离理想值）。延迟呈现什么分布时，并行不再重要？

2. 扩展累加器，处理“调用在流式传输中途被取消”的情况：丢弃其缓冲区并生成 `cancelled` 事件。哪家提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池，并对两者进行基准测试。由于上下文切换成本较低，异步版本应略有优势，但前提是执行器执行真实 I/O。

4. 选择两个不应并行执行的工具（例如先 `create_file`，再 `write_file`）。为注册表加入 `ordering_dependency` 图，并依据该图控制并行扇出。这是依赖感知调度所需的最小机制，未来的智能体工程阶段会将其形式化。

5. 阅读 OpenAI 的并行函数调用章节与 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行的一种真实工具类型。（提示：对同一资源执行有后果的修改。）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 并行工具调用 | “一轮扇出” | 模型在一条助手消息中输出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 标志” | 启用或禁用多调用输出 |
| `disable_parallel_tool_use` | “Anthropic 反向标志” | 选择退出标志；默认启用并行 |
| 工具调用 ID | “关联句柄” | 结果消息必须原样返回的逐调用标识符 |
| 累加器 | “流缓冲区” | 按 ID 保存局部 `arguments` 数据块的字符串缓冲区 |
| 乱序完成 | “最快者先返回” | 并行调用以不可预测的顺序完成；ID 是关联胶水 |
| 依赖图 | “顺序约束” | 输出会成为其他工具输入的工具关系；不能并行执行 |
| 过早解析陷阱 | “JSON.parse 崩溃” | 尝试解析尚未完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3 功能” | 每个调用都带唯一 ID 的流式参数块 |
| 按完成顺序回复 | “不要等待全部结束” | 结果一到达就按 ID 返回 |

## 延伸阅读

- [OpenAI——并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling)——默认行为与选择退出标志
- [Anthropic——工具使用：实现工具使用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use)——`disable_parallel_tool_use` 与结果批处理
- [Google——Gemini 函数调用并行章节](https://ai.google.dev/gemini-api/docs/function-calling)——Gemini 3 起使用 ID 关联的并行调用
- [OpenAI——带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming)——OpenAI 数据流中的分块参数重组
- [Anthropic——流式消息](https://docs.anthropic.com/en/api/messages-streaming)——`content_block_delta` 中携带 `input_json_delta`
