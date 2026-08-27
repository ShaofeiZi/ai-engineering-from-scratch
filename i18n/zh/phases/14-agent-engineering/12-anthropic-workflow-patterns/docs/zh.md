# Anthropic 的工作流模式：先简单，后复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）与 agent（动态工具使用）。五种工作流模式已经覆盖了大多数场景。先从直接 API 调用开始，只有在步骤本身无法预判时，才引入 agent。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 01（Agent Loop）
**Time:** 约 60 分钟

## 学习目标

- 说出 Anthropic 提出的五种工作流模式：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 解释 agent 与 workflow 的区别，以及两者各自带来的工程成本。
- 判断什么时候应该优先选择 workflow，什么时候才该换成 agent。
- 仅用 stdlib 和一个脚本化 LLM 实现这五种模式。

## 问题

很多团队会为了一个本该用单次函数调用解决的问题，直接上多 agent 框架。代价并不小：框架会额外叠出多层抽象，提示词被包起来，控制流被藏起来，复杂度也往往被提前引入。Schluntz 和 Zhang 在 2024 年 12 月的文章里提出了业内最常被引用的反向提醒：先做简单方案，只有当复杂性确实创造了价值，再为它买单。

## 概念

### Workflow 与 agent

- **Workflow。** 通过预先写死的代码路径来编排 LLM 和工具，图由工程师掌控。
- **Agent。** 由 LLM 自己动态决定调用哪些工具、走哪些步骤，图由模型掌控。

两者都有适用位置。Workflow 更便宜、更快，也更容易调试。Agent 能处理开放式问题，但它的失败模式更难分析和约束。

### Augmented LLM

五种模式都建立在同一个基础上：一个接好了三类能力的 LLM，分别是搜索（retrieval）、工具（actions）和记忆（persistence）。任何一次 API 调用，本质上都可以带上这些增强能力。

### 五种模式

1. **Prompt chaining。** 第 1 次调用的输出作为第 2 次调用的输入。适合任务能被清楚地拆成线性步骤的场景。步骤之间还可以插入程序化 gate。

2. **Routing。** 先由一个分类器 LLM 决定，后面应该调用哪个下游 LLM 或工具。适合输入类别差异明显、需要不同处理路径的场景，比如 tier-1 support、refund、bug、sales。

3. **Parallelization。** 并发运行 N 次 LLM 调用，再汇总结果。常见有两种形态：sectioning（分不同片段处理）和 voting（对同一提示运行 N 次，再多数表决或综合）。

4. **Orchestrator-workers。** 由一个 orchestrator LLM 动态决定要调用哪些 worker（也都是 LLM），然后再综合它们的结果。它和 agent loop 很像，但 orchestrator 不会无限循环。

5. **Evaluator-optimizer。** 一个 LLM 先给出答案，另一个 LLM 负责评估；只要评估没通过，就继续迭代。这可以看作是对 Self-Refine（Lesson 05）的泛化。

### Workflow 什么时候优于 agent

- **任务可预测。** 如果步骤可以事先列出来，就应该列出来。
- **任务有成本上限。** Workflow 的步骤数有边界，agent 则可能一路失控膨胀。
- **任务受合规约束。** 审计需要能直接读懂图，而不是事后从执行轨迹里反推出流程。

### Agent 什么时候优于 workflow

- **开放式研究。** 下一步该做什么，取决于上一步实际返回了什么。
- **变长任务。** 任务可能持续几分钟到几小时，步骤数一开始并不明确。
- **陌生领域。** 你还不知道正确的 workflow 长什么样，必须先探索，再固化。

### 与之配套的上下文工程

Anthropic 2025 年的文章 “Effective context engineering for AI agents” 把一个相邻但关键的学科讲清楚了：200k context window 是预算，不是垃圾桶。什么该放进去，什么时候要压缩，什么时候允许上下文继续生长。这部分会在本课程关于上下文压缩的 Phase 14 课程中详细展开（早期编号里对应的是 Lesson 06）。

```figure
workflow-chain
```

## 动手构建

`code/main.py` 用 `ScriptedLLM` 实现了这五种工作流模式：

- `prompt_chain(input, steps)`：顺序执行。
- `route(input, classifier, handlers)`：分类后分派。
- `parallel_vote(prompt, n, aggregator)`：并发运行 N 次后聚合。
- `orchestrator_workers(task, workers)`：由 orchestrator 选择 worker。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)`：循环直到评估通过。

运行方式：

```
python3 code/main.py
```

每一种模式都会打印自己的 trace。单个模式的实现大约只有 10 到 15 行代码，而一个框架引入的额外复杂度，往往是以几千行为单位来衡量的。

## 如何使用

- 大多数任务直接用 API 调用就够了。
- 只有当模式真的需要持久状态（LangGraph）、actor-model 并发（AutoGen v0.4）或角色模板化（CrewAI）时，再考虑框架。
- 如果你想直接获得类似 Claude Code harness 的整体形状，而不是自己重搭一套，可以优先看 Claude Agent SDK。

## 交付成果

`outputs/skill-workflow-picker.md` 会根据任务描述选择合适的工作流模式，并给出决策理由；如果 workflow 不再够用，它还会说明如何把这套结构重构成 agent。

## 练习

1. 给 routing 加一个置信度阈值。低于阈值时升级给人工处理。对于 tier-1 support 场景，这个阈值应该设在哪里？
2. 给 `parallel_vote` 加超时。某一个调用卡住时会发生什么？缺票时你要怎么聚合结果？
3. 把 `evaluator_optimizer` 改成 bandit：在多轮迭代中始终保留当前最好的两个结果，避免后期一个差结果把先前的好结果覆盖掉。
4. 把 prompt chaining 和 routing 组合起来：先由 router 在三条 chain 里选一条。比较它与“单个大 prompt”方案之间的 token 成本。
5. 选一个你当前线上功能，画出它的 workflow graph，数一数步骤。这里真的需要 agent 吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Workflow | "Predefined flow" | 由工程师掌控的 LLM 与工具调用图 |
| Agent | "Autonomous AI" | 由模型掌控的图；工具方向动态决定 |
| Augmented LLM | "LLM with tools" | LLM + search + tools + memory；最小原子单元 |
| Prompt chaining | "Sequential calls" | 第 N 次调用的输出作为第 N+1 次调用的输入 |
| Routing | "Classifier dispatch" | 选择由哪条链或哪个模型处理输入 |
| Parallelization | "Fan out" | N 次并发调用，再按 sectioning 或 voting 聚合 |
| Orchestrator-workers | "Dispatcher agent" | 由 orchestrator LLM 动态挑选专家 LLM |
| Evaluator-optimizer | "Proposer + judge" | 迭代直到 evaluator 通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式的原始文章
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 与之配套的上下文工程
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 什么时候有状态图真的值得它的成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 被产品化后的 orchestrator-workers 形态
