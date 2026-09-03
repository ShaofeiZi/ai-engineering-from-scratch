# 工具使用与函数调用

> Toolformer（Schick 等，2023）开创了自监督工具标注。Berkeley Function Calling Leaderboard V4（Patil 等，2025）设定了 2026 年的基准：智能体任务占 40%，多轮占 30%，实时数据占 10%，非实时数据占 10%，幻觉检测占 10%。单轮调用已经解决；记忆、动态决策和长时程工具链仍未解决。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 01（智能体循环）、阶段 13 · 01（函数调用深入解析）
**Time:** 约 60 分钟

## 学习目标

- 解释 Toolformer 的自监督训练信号：只有当工具执行结果能降低下一个 token 的损失时，才保留工具标注。
- 说出 BFCL V4 的五类评估及其各自衡量的内容。
- 使用标准库实现包含 schema 验证、参数强制转换和执行沙箱的工具注册表。
- 诊断 2026 年的三个开放问题：长时程工具串联、动态决策和记忆。

## 问题

早期的工具使用只问：模型能否预测一次正确的函数调用？现代工具使用则会问：模型能否带着记忆串联 40 步工具调用，在部分可观测环境中根据结果动态决策，从工具失败中恢复，同时不虚构不存在的工具？

Toolformer 奠定了基线：模型能够通过自监督学习何时调用工具。BFCL V4 则定义了 2026 年的评估目标。二者之间的差距，正是生产智能体所处的空间。

## 核心概念

### Toolformer（Schick 等，NeurIPS 2023）

思路是让模型使用候选 API 调用为自己的预训练语料添加标注。对每个候选调用执行 API；只有当加入工具结果能够降低下一个 token 的损失时，才保留该标注。最后在筛选后的语料库上进行微调。

涵盖的工具包括计算器、问答系统、搜索引擎、翻译器和日历。自监督信号只关心工具是否有助于预测文本，不需要人工标签。

规模效应是：工具使用能力会随规模出现。较小模型会因工具标注而受损，较大模型则从中受益。因此，2026 年的前沿模型已内建强大的工具使用能力，而大多数 7B 模型仍需专门进行工具使用微调才能可靠工作。

### Berkeley Function Calling Leaderboard V4（Patil 等，ICML 2025）

BFCL 是 2026 年事实上的评估标准。V4 的构成如下：

- **Agentic（40%）**——完整的智能体轨迹：记忆、多轮交互、动态决策。
- **Multi-Turn（30%）**——包含工具链的交互式对话。
- **Live（10%）**——用户提交的真实提示（分布更难）。
- **Non-Live（10%）**——合成测试用例。
- **Hallucination（10%）**——检测不应调用任何工具的情况。

V3 引入了基于状态的评估：在一串工具调用后，检查 API 的实际状态（例如“文件是否已创建？”），而不是匹配工具调用的 AST。V4 增加了 Web 搜索、记忆和格式敏感性类别。

2026 年的关键发现是：单轮函数调用已接近解决。失败集中在记忆（跨轮次携带上下文）、动态决策（根据先前结果选择工具）、长时程工具链（超过 20 步后发生漂移）以及幻觉检测（没有合适工具时拒绝调用）。

### 工具 schema

每个提供商都有自己的 schema。细节有所不同，但基本结构相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`，OpenAI 使用 `function.parameters`，二者都接受 JSON Schema。描述是承重信息——模型依据描述选择正确工具。糟糕的工具描述是“选错工具”失败的首要根因。

### 参数验证

不要信任任何工具调用。应验证：

1. **类型强制转换。** schema 要求 int 时，模型可能返回字符串“5”。含义明确时可转换，否则拒绝。
2. **枚举验证。** 如果 schema 声明 `status in {"open", "closed"}`，而模型输出 `"in_progress"`，则以描述清楚的错误拒绝。
3. **必需字段。** 缺少必需字段时，应立即向模型返回错误观察，而不是让程序崩溃。
4. **格式验证。** 日期、电子邮件、URL 应使用具体解析器验证，而不是正则表达式。

每次验证失败都应返回结构化观察，让模型能够使用正确结构重试。

### 并行工具调用

现代提供商支持在一个 Assistant 轮次中并行发起工具调用。循环如下：

1. 模型发出 3 个工具调用，每个都带有不同的 `tool_use_id`。
2. 运行时执行这些调用（彼此独立时可并行）。
3. 每个结果以 `tool_result` 块返回，并通过 `tool_use_id` 建立关联。

工程规则：把关联 ID 视为承重信息。一旦对调，就会把错误工具的结果路由给另一个工具。

### 沙箱

工具执行就是沙箱边界。详情参见第 09 课。简而言之，每个工具都应声明读写范围、网络访问权限、超时时间和内存上限。通用的 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 更安全。

```figure
tool-routing
```

## 动手构建

`code/main.py` 实现了一个具有生产系统形态的工具注册表：

- JSON Schema 子集 Validator（仅使用标准库）。
- 注册工具及其描述、输入 schema、超时和 Executor。
- 参数强制转换与枚举验证。
- 使用关联 ID 并行分派工具。
- 将错误观察表示为结构化字符串。

运行：

```
python3 code/main.py
```

追踪会展示一个微型智能体如何在一个轮次中调用三个工具，其中有一个故意构造错误的调用。该调用会被拒绝，并返回模型可以据此修正的描述性错误。

## 实际使用

每个提供商都有自己的工具 schema——Anthropic、OpenAI、Gemini、Bedrock。如果需要支持多个提供商，可使用转换层（OpenAI Agents SDK、Vercel AI SDK、LangChain 工具适配器）。BFCL 是参考基准；如果工具使用是产品核心，应在发布前用它评估智能体。

## 交付成果

`outputs/skill-tool-registry.md` 会为给定任务领域生成工具目录、schema 与注册表，其中包括描述质量检查（每个工具的描述是否告诉模型何时使用它？）。

## 练习

1. 添加一个“no-op”工具，让模型能够明确拒绝使用其他任何工具。在类似 BFCL 的幻觉测试上进行测量。
2. 实现 int-as-string 与 float-as-string 的参数强制转换。强制转换从哪里开始掩盖真实缺陷？
3. 添加逐工具超时和断路器（连续失败 3 次后拒绝该工具 60 秒）。这会怎样改变模型的恢复方式？
4. 阅读 BFCL V4 描述，选择一个类别（例如“multi-turn”），让智能体处理 10 个示例提示，并报告通过率。
5. 将标准库 Validator 移植到 Pydantic 或 Zod。Pydantic / Zod 捕获了哪些玩具实现漏掉的问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 函数调用 | “工具使用” | 使用经过验证 schema 的结构化输出工具调用 |
| Toolformer | “自监督工具标注” | Schick，2023——保留结果能够降低下一个 token 损失的工具调用 |
| BFCL | “Berkeley Function Calling Leaderboard” | 2026 年基准：智能体 40%、多轮 30%、实时 10%、非实时 10%、幻觉 10% |
| 工具 schema | “提供给模型的函数签名” | 名称、描述、参数 JSON Schema |
| tool_use_id | “关联 ID” | 将工具调用与其结果关联；对并行分派至关重要 |
| 幻觉检测 | “知道何时不应调用” | V4 类别：没有合适工具时拒绝调用 |
| 参数强制转换 | “字符串到整数修复” | 针对可预测 schema 不匹配的有限修复；有歧义时拒绝 |
| 沙箱 | “工具执行边界” | 逐工具的读写范围、网络、超时与内存上限 |

## 延伸阅读

- [Schick 等，Toolformer（arXiv:2302.04761）](https://arxiv.org/abs/2302.04761)——自监督工具标注
- [Berkeley Function Calling Leaderboard（V4）](https://gorilla.cs.berkeley.edu/leaderboard.html)——2026 年评估基准
- [Anthropic 工具使用文档](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude Agent SDK 中的生产工具 schema
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/)——函数工具类型与 Guardrail
