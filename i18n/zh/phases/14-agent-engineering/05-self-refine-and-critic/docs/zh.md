# Self-Refine 与 CRITIC：迭代改进输出

> Self-Refine（Madaan 等，2023）让同一个 LLM 在循环中扮演三种角色——生成、反馈、改进——在 7 项任务上平均取得 20 个百分点的绝对提升。CRITIC（Gou 等，2023）通过外部工具完成验证，从而强化反馈步骤。到了 2026 年，每个框架都以“Evaluator–Optimizer”（Anthropic）或护栏循环（OpenAI Agents SDK）的形式提供了这种模式。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 01（智能体循环）、阶段 14 · 03（Reflexion）
**Time:** 约 60 分钟

## 学习目标

- 说出 Self-Refine 的三个提示（generate、feedback、refine），并解释为什么 refine 提示需要历史记录。
- 解释 CRITIC 的关键洞见：没有外部依据时，LLM 无法可靠地自我验证。
- 使用标准库实现一个带历史记录和可选外部 Verifier 的 Self-Refine 循环。
- 将这种模式映射到 Anthropic 的“Evaluator–Optimizer”工作流和 OpenAI Agents SDK 的输出护栏。

## 问题

智能体给出了一个几乎正确的答案。可能某行代码有语法错误，可能摘要太长，也可能计划漏掉了一个边界情况。你真正想要的是让智能体批评自己的输出，然后修复它。

Self-Refine 表明，只用一个模型、不需要训练数据或强化学习，也能做到这一点。但它有一个问题：LLM 不擅长对困难事实进行自我验证。CRITIC 给出了修复方案——让外部工具（搜索、代码解释器、计算器、测试运行器）执行验证步骤。

这两篇论文共同定义了 2026 年迭代改进的默认方式：生成，验证（条件允许时使用外部依据），改进，并在 Verifier 通过时停止。

## 核心概念

### Self-Refine（Madaan 等，NeurIPS 2023）

一个 LLM，三种角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节是：`refine` 能看到完整历史，即此前所有输出和批评，因此不会重复犯错。论文对此做过消融实验：移除历史后，质量会急剧下降。

核心结果：在数学、代码、缩写、对话等 7 项任务上，平均绝对提升 20 个百分点，其中也包括 GPT-4。无需训练、无需外部工具，只使用单个模型。

### CRITIC（Gou 等，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点在于反馈步骤由 LLM 为自己评分。对于事实性陈述，这种做法并不可靠——幻觉往往也会让生成它的模型觉得很有说服力。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包含：

- 用于核查事实陈述的搜索引擎。
- 用于检查代码正确性的代码解释器。
- 用于算术运算的计算器。
- 领域特定的 Verifier（单元测试、类型检查器、Linter）。

Verifier 根据工具结果产出结构化批评，Refiner 再以这份批评为条件进行改进。

核心结果：CRITIC 在事实类任务上超过 Self-Refine，因为批评具有外部依据。对于没有外部 Verifier 的任务（创意写作、格式调整），CRITIC 会退化为 Self-Refine。

### 停止条件

常见形式有两种：

1. **Verifier 通过。** 外部测试返回成功。条件允许时优先采用这种形式（单元测试、类型检查器、护栏断言）。
2. **未产生反馈。** 模型认为“输出没有问题”。这种方式更便宜但不可靠，应与最大迭代次数结合使用。

2026 年的默认做法是将两者组合：“Verifier 通过时停止；或者模型认为无问题且迭代次数至少为 2 时停止；或者达到 max_iterations 时停止。”

### Evaluator–Optimizer（Anthropic，2024）

Anthropic 2024 年 12 月的文章将其命名为五种工作流模式之一。其中包含两个角色：

- Evaluator：为输出评分并生成批评。
- Optimizer：根据批评修订输出。

循环持续到 Evaluator 判定通过。这就是 Anthropic 框架下的 Self-Refine / CRITIC。Anthropic 补充的关键工程细节是：Evaluator 和 Optimizer 的提示应有明显不同的结构，避免模型只是草率地盖章通过。

### OpenAI Agents SDK 输出护栏

OpenAI Agents SDK 以“输出护栏”的形式提供这种模式。Guardrail 是一个在智能体最终输出上运行的 Validator。如果 Guardrail 被触发（抛出 `OutputGuardrailTripwireTriggered`），输出会被拒绝，智能体可以重试。Guardrail 可以调用工具（CRITIC 风格），也可以是纯函数（Self-Refine 风格）。

### 2026 年的陷阱

- **盖章式循环。** 同一个模型以相同提示风格完成生成与批评，最终会收敛到“在我看来没问题”。应使用结构明显不同的提示，或让更小、更便宜的模型负责批评。
- **过度改进。** 每一轮 refine 都会增加延迟和 token。预算应设为 1–3 轮；超过后转交人工审查。
- **对简单任务使用 CRITIC。** 如果没有外部 Verifier，CRITIC 就会退化为 Self-Refine；不要为占位式 Verifier 支付额外延迟。

```figure
self-refine
```

## 动手构建

`code/main.py` 在一个玩具任务上实现 Self-Refine 与 CRITIC：给定主题，生成一份简短的项目符号列表。Verifier 检查格式（3 个项目符号，每项少于 60 个字符）。CRITIC 还增加了一个外部“事实 Verifier”，对已知幻觉施加惩罚。

组件如下：

- `generate`——脚本化的生成器。
- `feedback`——LLM 风格的自我批评。
- `verify_external`——CRITIC 风格、具有外部依据的 Verifier。
- `refine`——根据历史记录重写输出。
- 停止条件——Verifier 通过，或达到最多 4 次迭代。

运行：

```
python3 code/main.py
```

比较 Self-Refine 与 CRITIC 的运行结果。CRITIC 能发现 Self-Refine 漏掉的事实错误，因为外部 Verifier 拥有自我 Critic 所缺乏的依据。

## 实际使用

Anthropic 的 Evaluator–Optimizer 是使用 Claude 友好语言描述的这一模式。OpenAI Agents SDK 的输出护栏呈 CRITIC 形态（Guardrail 可以调用工具）。LangGraph 提供了一个与 Self-Refine 类似的反思节点。Google 的 Gemini 2.5 Computer Use 增加了逐步骤安全 Evaluator，它是 CRITIC 的一种变体：每个操作在提交前都会得到验证。

## 交付成果

`outputs/skill-refine-loop.md` 根据任务形态、Verifier 可用性和迭代预算，配置 Evaluator–Optimizer 循环。它会输出 Generator、Evaluator / Verifier 和 Optimizer 的提示，以及一套停止策略。

## 练习

1. 使用 max_iterations=1 运行玩具示例。CRITIC 仍有帮助吗？
2. 将外部 Verifier 替换为带噪声的版本（随机产生 30% 的误报）。循环会怎样运行？这正是 2026 年大多数护栏栈面对的现实。
3. 实现“Generator 与 Critic 使用不同模型”的变体：大模型负责生成，小模型负责批评。它是否优于同模型方案？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三类验证工具，并分别举例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的 Verifier 角色。SDK 做错了什么，又做对了什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | “能修复自己的 LLM” | 同一模型内带历史记录的生成 -> 反馈 -> 改进循环 |
| CRITIC | “工具落地验证” | 使用外部 Verifier（搜索、代码、计算、测试）替代反馈 |
| Evaluator–Optimizer | “Anthropic 工作流模式” | 两种角色——Evaluator 评分、Optimizer 修订——循环直至收敛 |
| 输出护栏 | “事后检查” | 智能体生成输出后运行的 OpenAI Agents SDK Validator |
| 验证步骤 | “批评阶段” | 承担关键判断：依据外部事实，还是自我评分 |
| 改进历史 | “模型已尝试过的内容” | 前置到 refine 提示中的先前输出与批评；移除后质量骤降 |
| 盖章式循环 | “自我认同失败” | 使用相同提示的批评返回“看起来不错”；通过结构不同的提示修复 |
| 停止条件 | “收敛测试” | Verifier 通过，或无反馈且达到迭代下限；绝不能只有单一条件 |

## 延伸阅读

- [Madaan 等，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651)——奠基论文
- [Gou 等，CRITIC（arXiv:2305.11738）](https://arxiv.org/abs/2305.11738)——工具落地验证
- [Anthropic，构建高效智能体](https://www.anthropic.com/research/building-effective-agents)——Evaluator–Optimizer 工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/)——以 CRITIC 形态实现的输出护栏
