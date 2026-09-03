# ReWOO 与 Plan-and-Execute：解耦规划

> ReAct 将思考与行动交织在同一条流中；ReWOO 则把两者拆开：先一次性制定完整计划，再执行。它的 token 用量减少 5 倍，在 HotpotQA 上的准确率提升 4%，而且可以把规划器蒸馏到 7B 模型中。Plan-and-Execute 对这一方法进行了泛化；Plan-and-Act 又将它扩展到了 Web 导航。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 01（智能体循环）
**Time:** 约 60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 拆分相比 ReAct 的交织循环更节省 token、更稳健。
- 仅使用标准库，实现计划 DAG、按依赖顺序执行的执行器，以及组合 Worker 输出的 Solver。
- 使用 Anthropic 在 2026 年所归纳的“五种工作流模式”框架，判断任务应该采用先规划后执行，还是交织式 ReAct。
- 识别长时程 Web 或移动端任务何时需要 Plan-and-Act 的合成计划数据。

## 问题

ReAct 的思考—行动—观察交织循环简单而灵活，但每次工具调用都必须携带此前的完整上下文，包括之前的每一次思考。token 用量会随深度呈平方增长。更糟的是，当工具在循环中途失败，模型必须根据错误观察重新推导整份计划。

ReWOO（Xu 等，arXiv:2305.18323，2023 年 5 月）注意到了这一问题，并作出一个取舍：预先规划全部工作，并行获取证据，最后组合答案。一次 LLM 调用负责规划，N 次工具调用获取证据（可以并行），再用一次 LLM 调用求解。它以较低的灵活性（计划是静态的）换取显著提升的 token 效率和更清晰的失败模式。

## 核心概念

### 三种角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点都指定工具、参数，以及它依赖的前置节点（例如 `#E1`、`#E2` 引用）。Workers 按拓扑顺序执行节点。Solver 将所有内容组合成最终答案。

### 为什么 token 用量能减少 5 倍

ReAct 的提示长度随步骤数线性增长。执行到第 10 步时，提示中包含思考 1、行动 1、观察 1、思考 2、行动 2、观察 2，依此类推。每个中间步骤还会重复包含原始提示。

ReWOO 只需支付一次较大的 Planner 提示、N 次较小的 Worker 提示（每次只包含工具调用，不含推理链），以及一次 Solver 提示。论文在 HotpotQA 上测得 token 用量约减少 5 倍，同时绝对准确率提高 4 个百分点。

### 为什么它更稳健

如果 ReAct 中的 Worker 3 失败，循环必须在流程中途从错误中重新推理。在 ReWOO 中，Worker 3 只需返回错误字符串；Solver 会同时看到错误和原始计划，从而优雅降级。失败定位落在具体节点，而不是模糊的流程步骤上。

### Planner 蒸馏

论文的第二项成果是：由于 Planner 看不到观察结果，可以使用 175B 教师模型产生的规划输出，对 7B 模型进行微调。小模型负责规划，推理时便不再需要大模型。如今这已成为常见做法——许多 2026 年的生产智能体会使用小型 Planner 搭配大型 Executor，或反过来搭配。

### Plan-and-Execute（2023）

LangChain 团队 2023 年 8 月的文章将 ReWOO 泛化成名为 Plan-and-Execute 的模式：前置 Planner 输出步骤列表，Executor 执行每一步，可选的 Replanner 在观察结果后修订计划。它比 ReWOO 更接近 ReAct（Replanner 会把观察重新引入规划），但仍保留了 token 节省优势。

### Plan-and-Act（Erdogan 等，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将这种模式扩展到长时程 Web 与移动端智能体。其关键贡献是合成计划数据：带标注的轨迹生成器产生显式计划训练数据，用于微调 Planner 模型。面对类似 WebArena、步骤超过 30–50 步的任务时，单条 ReAct 轨迹会失去连贯性，而这种 Planner 仍能持续工作。

### 如何选择

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、未知环境、需要响应式异常处理 |
| ReWOO | 工具已知的结构化任务、token 敏感、证据可并行获取 |
| Plan-and-Execute | 与 ReWOO 类似，但需要在部分执行后重新规划 |
| Plan-and-Act | 长时程（超过 30 步）、Web／移动端／计算机操作 |
| Tree of Thoughts | 搜索收益值得付出额外成本（第 04 课） |

Anthropic 在 2024 年 12 月给出的建议是：从最简单的方案开始。如果任务只是调用一次工具再做摘要，就不要构建 ReWOO；如果任务是一个包含 40 个步骤的研究项目，也不要只用 ReAct。

```figure
rewoo-plan
```

## 动手构建

`code/main.py` 实现了一个玩具版 ReWOO：

- `Planner`——根据提示生成计划 DAG 的脚本化策略。
- `Worker`——通过注册表分派每个节点的工具调用。
- `Solver`——读取证据并生成最终答案的脚本化组合器。
- 依赖解析——在分派时，将 `#E1` 之类的引用替换为先前 Worker 的输出。

演示使用两步计划回答“What is the population of the capital of France, rounded to millions?”：第一步查询首都，第二步查询人口，最后求解。

运行：

```
python3 code/main.py
```

追踪会先显示完整计划，再显示 Worker 结果，最后显示 Solver 的组合过程。将 token 数量（程序会打印粗略字符数）与 ReAct 风格的交织运行进行比较——在这类结构化任务上，ReWOO 更有优势。

## 实际使用

LangGraph 以配方形式提供 Plan-and-Execute（`create_react_agent` 用于 ReAct，自定义图用于 plan-execute）。CrewAI 的 Flows 则直接编码了这种模式：预先定义任务，再由 Flow DAG 执行。Plan-and-Act 的合成数据方法目前仍以研究为主；它的运行时模式（显式计划 DAG）已经通过 LangGraph 和 CrewAI Flows 用于生产。

## 交付成果

`outputs/skill-rewoo-planner.md` 根据用户请求和给定工具目录生成 ReWOO 计划 DAG。在交给执行器之前，它会验证计划是否无环、每个引用是否已解析、每个工具是否存在。

## 练习

1. 并行执行相互独立的计划节点。对于一个有 6 个节点、2 个并行组的 DAG，这能带来什么收益？
2. 添加一个 Replanner 节点，在任一 Worker 返回错误时触发。要让 ReWOO 变成 Plan-and-Execute，最小改动是什么？
3. 用小模型（7B 量级）替换 `Planner`，同时让 `Solver` 继续使用前沿模型。比较端到端质量——这种拆分会在哪里失效？
4. 阅读 ReWOO 论文中关于 Planner 蒸馏的第 4 节。从概念上复现 175B -> 7B 的结果：需要哪些训练数据？如何评估计划质量？
5. 将这个玩具实现移植为 Plan-and-Act 的轨迹结构：计划是序列，而不是 DAG。各项取舍会如何变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | “Reasoning without observations” | 先规划，再并行获取证据，最后求解——规划提示中不包含观察 |
| Plan-and-Execute | “LangChain 的 plan-execute 模式” | 在执行后增加可选 Replanner 节点的 ReWOO |
| Plan-and-Act | “扩展版 plan-execute” | 显式拆分 Planner / Executor，并使用长时程任务的合成计划训练数据 |
| 证据引用 | “#E1、#E2，……” | 在分派时替换为前置 Worker 输出的计划节点占位符 |
| Planner 蒸馏 | “小型 Planner，大型 Executor” | 使用大型教师模型的规划轨迹微调小模型 |
| Token 效率 | “更少的往返” | 论文中相比 ReAct，在 HotpotQA 上减少 5 倍 token |
| DAG 执行器 | “拓扑分派器” | 按依赖顺序运行计划节点；每一层可以并行 |

## 延伸阅读

- [Xu 等，ReWOO：将推理与观察解耦（arXiv:2305.18323）](https://arxiv.org/abs/2305.18323)——奠基论文
- [Erdogan 等，Plan-and-Act（arXiv:2503.09572）](https://arxiv.org/abs/2503.09572)——使用合成计划扩展 Planner–Executor
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview)——框架配方
- [Anthropic，构建高效智能体](https://www.anthropic.com/research/building-effective-agents)——选择能够奏效的最简单模式
