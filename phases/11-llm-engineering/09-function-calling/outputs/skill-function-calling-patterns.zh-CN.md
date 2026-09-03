---
name: skill-function-calling-patterns
description: 在生产环境实现 function calling 的决策框架——涵盖工具设计、错误处理、安全性和提供商模式
version: 1.0.0
phase: 11
lesson: 09
tags: [function-calling, tool-use, agents, mcp, security, openai, anthropic]
---

# 函数调用模式

在构建使用工具的 LLM 应用时，请应用本决策框架。

## 何时使用函数调用

**在以下情况使用函数调用：**
- 模型需要实时数据（天气、股价、数据库查询）
- 任务需要副作用（发送邮件、创建记录、部署代码）
- 模型必须基于用户意图在多个动作之间做出选择
- 你正在构建与外部系统交互的智能体

**在以下情况改用结构化输出：**
- 你需要从文本中抽取数据（无需外部调用）
- 输出即最终产物，而非中间步骤
- 你只有一个 schema，而不是要在多个工具间选择

**在以下情况两者并用：**
- 模型先调用某个工具，再将工具结果结构化为特定的输出格式

## 工具设计准则

1. **一个工具只做一件事。** 名为 `manage_database` 却同时处理查询、插入、更新与删除的工具过于宽泛。拆分为 `query_records`、`insert_record`、`update_record`。面对具体工具时模型的选择会更准确。

2. **描述本身就是提示词。** 模型会阅读工具描述来决定选择。请像给初级开发者写指令一样来写它们，既要包括工具做什么，也要包括工具返回什么。

3. **用枚举加以约束。** 如果某个参数有 3-10 个合法值，就使用 enum。否则模型会自行编造字符串——"celsius"、"Celsius"、"C"、"metric"——除非你加以约束。

4. **工具越少越好。** GPT-4o 能很好地处理 5-10 个工具。达到 20+ 个工具时，选择准确率会下降。达到 50+ 个工具时，预期会出现 10-15% 的错误工具选择。可对相关功能进行分组，或使用路由层。

5. **必填就是必填。** 仅当工具确实缺少该参数就无法运行时，才将其标记为 required。带有合理默认值的可选参数能降低工具调用失败率。

## 各厂商的特定模式

### OpenAI（GPT-4o、o3、GPT-4o-mini）

```python
tools=[{"type": "function", "function": {"name": ..., "parameters": ...}}]
tool_choice="auto"       # model decides
tool_choice="required"   # must call at least one tool
tool_choice={"type": "function", "function": {"name": "specific_tool"}}
```

- 支持并行工具调用（一次响应中包含多个 `tool_calls`）
- 工具调用 ID 必须随结果一并传回
- `gpt-4o-mini` 便宜 10 倍，且能很好地处理简单的工具路由
- 结构化输出模式可与工具参数配合，保证 schema 合规

### Anthropic（Claude 3.5 Sonnet、Claude 4 Opus）

```python
tools=[{"name": ..., "description": ..., "input_schema": ...}]
tool_choice={"type": "auto"}     # model decides
tool_choice={"type": "any"}      # must call at least one tool
tool_choice={"type": "tool", "name": "specific_tool"}
```

- 工具调用以 `type: "tool_use"` 的内容块出现
- 结果放在 `type: "tool_result"` 的 user 消息中返回
- 字段名为 `input_schema`，而非 `parameters`（常见的迁移坑）
- 支持每次响应包含多个工具调用

### Google（Gemini 2.0 Flash、Gemini 2.0 Pro）

```python
function_declarations=[{"name": ..., "description": ..., "parameters": ...}]
function_calling_config={"mode": "AUTO"}   # or "ANY" or "NONE"
```

- 在顶层使用 `function_declarations`
- 结果通过 `function_response` 部分返回
- 支持并行函数调用

### 开源模型（Llama 3、Hermes、Qwen）

- 没有标准化格式——因模型与服务框架而异
- Hermes 格式（NousResearch）是最常见的微调约定
- vLLM 对受支持的模型提供与 OpenAI 兼容的工具调用
- Ollama 对兼容模型提供基础工具调用
- 上线前务必测试工具选择准确率——在 Berkeley Function Calling Leaderboard 上，开源模型比 GPT-4o 低 15-30%

## 错误处理模式

### 返回结构化错误

```json
{"error": true, "message": "City 'Toky' not found. Did you mean 'Tokyo'?", "code": "NOT_FOUND", "suggestions": ["Tokyo"]}
```

包含可操作的信息。"Not found"（未找到）是差的；"Not found, did you mean X?"（未找到，你是指 X 吗？）是好的。模型会利用错误信息进行自我纠正。

### 重试策略

1. 工具调用因可纠正的错误而失败（拼写错误、错误的枚举值）
2. 将错误作为工具结果回传给模型
3. 模型调整后重试
4. 每次工具调用最多重试 3 次
5. 3 次失败后，将错误返回给用户

### 超时处理

为所有工具执行设置超时。30 秒是一个合理的默认值。如果工具超时，返回一个结构化的超时错误，让模型能够告知用户，而不是一直挂起。

## 安全清单

| 检查项 | 原因 | 做法 |
|-------|-----|-----|
| 白名单函数 | 防止任意代码执行 | 仅注册用户需要的工具 |
| 校验参数类型 | 防范类型混淆攻击 | 执行前检查类型 |
| 清理字符串参数 | 防止注入 | 拒绝或转义特殊字符 |
| 参数化数据库查询 | 防止 SQL 注入 | 永不直接传入模型生成的 SQL |
| 过滤工具结果 | 防止数据泄露 | 移除 API 密钥、PII、内部错误 |
| 限制工具调用频率 | 防止失控循环 | 每次会话最多 10-20 次调用 |
| 记录所有工具调用 | 审计追溯 | 存储工具名、参数、结果、时间戳 |
| 阻断路径遍历 | 防止文件系统访问 | 在文件工具中拒绝 `..` 与绝对路径 |
| 沙箱化代码执行 | 防止系统访问 | 使用容器或受限的内置函数 |
| 校验返回大小 | 防止上下文塞满 | 截断超过 10KB 的结果 |

## 性能优化

- **并行调用：** 当模型请求多个相互独立的工具时，用 `asyncio.gather()` 或 `concurrent.futures` 并发执行
- **缓存：** 对同一会话内相同参数的工具结果进行缓存（天气在 60 秒内不会变化）
- **流式输出：** 在获取工具结果的同时，流式输出模型的最终响应
- **工具裁剪：** 若上下文紧张，只纳入与当前查询相关的工具定义（用分类器过滤）
- **用更小模型做路由：** 用 `gpt-4o-mini` 或 `claude-haiku-4-5` 做工具选择，再把结果交给更强的模型做综合

## 常见失败模式

| 失败 | 原因 | 修复 |
|---------|-------|-----|
| 选错工具 | 描述含糊 | 用具体的触发词重写描述 |
| 缺少必填参数 | 模型遗漏了某个参数 | 在参数描述中加入清晰的示例 |
| 无限工具循环 | 模型反复调用同一工具 | 设置最大迭代次数（5-10）并检测重复调用 |
| 幻觉参数 | 模型编造看似合理实则错误的值 | 使用枚举，并对照已知值校验 |
| 工具结果过大 | API 返回了 100KB 数据 | 回传前截断或摘要 |
| 模型忽略工具结果 | 结果格式混乱 | 返回字段名清晰的干净 JSON |
