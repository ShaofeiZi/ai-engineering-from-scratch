# 评估驱动的 Agent 开发

> Anthropic 的建议是：“从简单提示词开始，用全面评估持续优化，只有在确有必要时才引入多步骤 agentic 系统。” 评估不是最后补上的一步，而是驱动 Phase 14 其他所有设计选择的外层循环。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段全部课程
**Time:** 约 60 分钟

## 学习目标

- 说出三层评估体系：静态 benchmark、定制 offline eval、在线 production eval，并理解每层各自的用途。
- 解释 evaluator-optimizer 的紧循环机制。
- 描述 2026 年的最佳实践：eval 要和代码放在一起、进入 CI、对 PR 起 gate 作用。
- 把 Phase 14 的每一课都关联到它应当生成的 eval case。

## 问题

代理很容易通过 demo，但会在生产环境里以 demo 完全预测不到的方式失败。Benchmark 回答的是“这个模型总体上强不强”，而不是“这个代理是否会给我的产品发出正确补丁”。真正有效的做法是：在三层上持续运行评估，并把每一条 guardrail、每一条学到的规则，都映射成一个可复现的 eval case。

## 概念

### 三层评估体系

1. **Static benchmarks**：代码场景看 SWE-bench Verified（Lesson 19），浏览器 / 桌面场景看 WebArena / OSWorld（Lesson 20），通用能力看 GAIA（Lesson 19），工具调用看 BFCL V4（Lesson 06）。它们主要用于跨模型比较和回归 gate。污染问题是真实存在的：SWE-bench+ 发现 32.67% 的解答泄漏。报告结果时应优先使用 Verified / +-audited 分数。

2. **Custom offline evals**：围绕你自己的产品形态来设计：
   - LLM-as-judge（Langfuse、Phoenix、Opik，见 Lesson 24）
   - Execution-based（真的运行 patch，再检查测试）
   - Trajectory-based（把动作序列和 gold trajectory 对比；OSWorld-Human 显示顶级代理的步数往往是 gold 的 1.4-2.7x）

3. **Online evals**：生产环境中的持续评估：
   - Session replay（Langfuse）
   - Guardrail 触发告警（Lesson 16、21）
   - 每一步的成本 / 延迟跟踪（Lesson 23 的 OTel spans）

### Evaluator-optimizer（Anthropic）

这个紧循环是：

1. proposer 先生成输出。
2. evaluator 负责评判。
3. 在 evaluator 通过前持续 refine。

这其实就是把 Self-Refine（Lesson 05）一般化后的形式。任何你真正关心可靠性的 agent flow，都可以外包一层 evaluator-optimizer。

### 2026 年的最佳实践

- eval 要和代码放在同一个仓库里。
- 每个 PR 都在 CI 中运行 eval。
- merge 要受 eval 分数约束，例如“不允许相对 main 回归超过 5%”。
- 每一条 guardrail 都应对应至少一个 eval case。
- 每一条 learned rule（如 Reflexion、pro-workflow learn-rule）也应对应到某个失败案例。

### 把 Phase 14 串起来

Phase 14 的每一课，都会自然产出相应的 eval case：

| 课程 | 生成的评估案例 |
|--------|------------------------|
| 01 代理循环 | 预算耗尽保护与无限循环保护 |
| 02 ReWOO | 工具失败时，规划器能正确地重新规划 |
| 03 Reflexion | 重试时会应用已学到的反思 |
| 05 Self-Refine/CRITIC | 评审器认可优化后的输出 |
| 06 工具使用 | 参数强制转换有效，并拒绝未知工具 |
| 07-10 记忆 | 检索引用与来源匹配，过时事实会失效 |
| 12 工作流模式 | 每种模式都能产生正确输出 |
| 13 LangGraph | 恢复后能精确复现状态 |
| 14 AutoGen Actor | DLQ 能捕获崩溃的处理器 |
| 16 OpenAI Agents SDK | 护栏能在正确的输入上触发 |
| 17 Claude Agent SDK | 子代理结果会返回编排器 |
| 19-20 基准测试 | SWE-bench Verified 分数、WebArena 成功率与 OSWorld 效率 |
| 21 计算机操作 | 逐步安全检查能捕获注入的 DOM |
| 23 OTel | Span 会发出必需属性 |
| 26 失效模式 | 检测器能标记已知失效 |
| 27 提示词注入 | PVE 会拒绝受到污染的检索结果 |
| 28 编排 | 监督者会把任务路由给正确的专家 |
| 29 运行时形态 | DLQ 能处理 N% 的失败 |

如果你的 eval suite 为这些点都准备了案例，那就说明你对 Phase 14 的覆盖已经相当完整。

### 评估驱动开发的常见失效点

- **没有 baseline。** 没有 last-known-good，eval 分数本身几乎不可读。baseline 必须存下来。
- **LLM-judge 没有 grounding。** judge 也会 hallucinate。Lesson 05 的 CRITIC 模式说明，judge 最好借助外部工具来落地判断。
- **过拟合 eval。** 如果你只优化 eval 分数，最后可能偏离真实生产价值。案例集需要轮换。
- **Flaky eval。** 非确定性案例会制造大量误报。应固定 seed、快照状态，尽量消除抖动。

```figure
ae-eval-three-layers
```

## 动手构建

`code/main.py` 是一个 stdlib eval harness：

- 带分类的 case registry（benchmark、custom、online）
- 一个待测的 scripted agent
- evaluator-optimizer loop：propose、judge、refine，直到通过或达到最大轮数
- 一个 CI gate：汇总通过率，并对比 baseline 判断是否回归

运行：

```
python3 code/main.py
```

输出会逐 case 展示 pass/fail、是否回归，以及最终的 CI gate verdict。

## 如何使用

- 把 eval case 和 agent 代码放在同一个 repo。
- 每个 PR 都通过 CI 跑 eval。
- 只要发生回归就让构建失败。
- 长期跟踪 pass rate。
- 每一个生产事故都回写成一个新的 eval case。

## 交付成果

`outputs/skill-eval-suite.md` 用于给一个代理产品搭建三层 eval suite，并配上 CI gate 与回归跟踪。

## 练习

1. 选一个你线上真实发生过的失败案例，写成一个 eval case。现在你的 agent 能通过它吗？
2. 为你的业务域设计一个三维度的 LLM-judge rubric，例如 factual、tone、scope。对 50 个 session 打分。
3. 把 eval suite 接进 CI。只要回归达到 >=5% 就让构建失败。
4. 增加一个 trajectory efficiency 指标：代理实际走了多少步，相对 gold trajectory 是多少倍？
5. 把 Phase 14 每一课都映射到你 eval suite 里的一个 case。缺了哪项，哪里就是空洞。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Static benchmark | "Off-the-shelf eval" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld 这类现成基准 |
| Custom offline eval | "Domain eval" | 围绕你产品形态设计的 LLM-as-judge / execution / trajectory 评估 |
| Online eval | "Production eval" | Session replay、guardrail alert、成本/延迟跟踪 |
| Evaluator-optimizer | "Propose-judge-refine" | 持续迭代，直到 judge 通过 |
| CI gate | "Merge blocker" | eval 回归时阻止合并 |
| Baseline | "Last-known-good" | 用于检测回归的参考分数 |
| Trajectory efficiency | "Steps over gold" | 代理步数除以人类专家最优步数 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — “start simple, optimize with evals”
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 经过策展的代码 benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — 工具调用 benchmark
- [Langfuse docs](https://langfuse.com/) — 生产里如何把 eval 和 session replay 结合起来
