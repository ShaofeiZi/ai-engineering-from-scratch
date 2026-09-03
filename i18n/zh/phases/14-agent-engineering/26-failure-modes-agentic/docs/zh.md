# 失败模式：为什么 Agent 会坏掉

> MASFT（Berkeley，2025）把 14 种多代理失败模式归入 3 大类别。微软的 Taxonomy 则说明，既有 AI 失败在 agentic 环境中会被进一步放大。行业实战数据最后汇聚成 5 类反复出现的问题：hallucinated actions、scope creep、cascading errors、context loss、tool misuse。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 05（Self-Refine 和 CRITIC），第 14 阶段 · 24（可观测性）
**Time:** 约 60 分钟

## 学习目标

- 说出 MASFT 的三大失败类别，并且每类至少举出四种具体模式。
- 解释为什么 agentic failure 会放大既有 AI failure modes，例如 bias 与 hallucination。
- 描述五类行业里反复出现的失败模式，以及对应的缓解手段。
- 实现一个 stdlib detector，为 agent traces 打上 failure-mode labels。

## 问题

团队经常会把一个在 90% traces 上能工作的 agent 推上线。但剩下 10% 的失败并不是随机噪声，它们通常会稳定落在少数几种重复模式里。一旦这些模式被明确命名，你就可以持续监控它们，并有针对性地修复。

## 概念

### MASFT (Berkeley, arXiv:2503.13657)

MASFT 即 Multi-Agent System Failure Taxonomy。它把 14 种 failure modes 聚成 3 大类别。研究中的 inter-annotator Cohen's Kappa 达到 0.88，说明这些类别之间具有较高可区分性。

这项工作的核心观点是：多代理系统里的失败，很多并不是“模型还不够强”，而是系统设计层面的根本缺陷，不能寄希望于只换一个更好的 base model 就自动消失。

### Microsoft 的 Agentic AI 故障模式分类

- 既有 AI failures，例如 bias、hallucination、data leakage，在 agentic setting 中会被进一步放大。
- autonomy 会引入新的失败方式，例如大规模 unintended action、tool misuse、mission drift。
- 这份 whitepaper 可以当作 agentic 产品的风险登记册来使用。

### Characterizing Faults in Agentic AI（arXiv:2603.06847）

- 失败可能来自 orchestration、本地状态演化，以及与环境的交互。
- 换句话说，它不只是“代码写坏了”或“模型输出差了一次”这么简单。

### LLM Agent Hallucinations Survey（arXiv:2509.18970）

这篇综述指出两类核心表现：

1. **Instruction-following Deviation**：agent 没有遵循 system prompt。
2. **Long-range Contextual Misuse**：agent 忘记了，或错误使用了更早轮次里的上下文。

此外还有 sub-intention errors，包括 Omission（漏步骤）、Redundancy（重复步骤）、Disorder（步骤顺序错乱）。

### 五类行业中反复出现的模式

Arize、Galileo、NimbleBrain 在 2024–2026 年的 field analysis 基本收敛到了五类高频问题：

1. **Hallucinated actions.** Agent 调用了根本不存在的工具，或者编造了调用参数。
2. **Scope creep.** Agent 把任务扩展到了用户并未要求的范围之外，例如多建 PR、多发邮件。
3. **Cascading errors.** 一次错误调用会引发连锁反应。比如一个幻觉出来的 SKU 触发了四次 API 调用，最后变成多系统事故。
4. **Context loss.** 长时任务里，agent 忘了前几轮里的关键约束。
5. **Tool misuse.** 用对了工具却传错参数，或者干脆选错了工具。

Cascading errors 往往最致命。Agent 常常分不清“我失败了”和“这个任务本来就做不到”，于是会在 400 错误之后仍然 hallucinate 一条成功消息，给整个 loop 假装收尾。

### 缓解手段：每一步都设 gate

应该在推理链的每一步都放自动验证 gate，并把环境状态当作事实依据来校验。具体来说：

- 每一步增加 safety classifier（Lesson 21）。
- 对 tool-call arguments 做 validation（Lesson 06）。
- 把检索结果与已知事实交叉核验（Lesson 05, CRITIC）。
- 通过重新探测真实状态来识别 success hallucination，例如文件到底有没有真的被创建。

### 失败监控常见的误区

- **只给 crash 打标签。** 大多数 agent failures 表面上看起来仍然“输出正常”，所以必须做内容级检查。
- **没有 baseline。** 漂移检测需要 last-known-good；没有基线，你就没法判断系统是不是在持续变差。
- **告警过量。** 每个失败都触发一页告警，会迅速把系统淹没。需要做聚类和限流。

```figure
failure-cascade
```

## 动手构建

`code/main.py` 实现了一个 stdlib failure-mode tagger，包括：

- 一个覆盖五类模式的 synthetic trace dataset。
- 针对每种模式的 detector functions，例如基于 tool calls、outputs、重复动作的 signature patterns。
- 一个会为每条 trace 打标签并输出 mode distribution 的 tagger。

运行方式：

```
python3 code/main.py
```

输出会展示每条 trace 的标签以及总体分布，可以看作是对 Phoenix trace clustering 能力的一种低成本复现。

## 如何使用

- **Phoenix**：适合在生产环境做 drift clustering（Lesson 24）。
- **Langfuse**：适合做 session replay 与 annotation。
- **Custom**：适合补足你所在业务领域的特定 signatures，这些往往不是通用 observability 平台默认能识别的。

## 交付成果

`outputs/skill-failure-detector.md` 用于生成贴合你业务域的 failure-mode detectors，并接入 trace store。

## 练习

1. 增加一个 “success hallucination” detector：agent 返回成功，但目标状态并未变化。
2. 给你做过的某个产品抽取 100 条真实 traces 并打标签。哪种模式最多？修复成本是多少？
3. 实现一个 “cascade radius” 指标：如果第 N 步失败，后续有多少步骤受到了影响？
4. 阅读 MASFT 的 14 种 failure modes，从中选出 3 个最适合你产品的，并写出 detectors。
5. 把一个 detector 接进 CI：如果有 >=5% 的 traces 被标成某种 mode，就让构建失败。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MASFT | "多智能体失败分类法" | Berkeley 提出的 14 模式分类 |
| Cascading error | "Ripple failure" | 一个早期错误沿着 N 个步骤继续传播 |
| Context loss | "忘了约束条件" | 长时任务中丢失了前几轮的关键事实 |
| Tool misuse | "工具错了 / 参数错了" | 调用形式合法，但实际调用错了 |
| Success hallucination | "Faked completion" | agent 在 400 错误后仍声称成功，且状态未变 |
| Scope creep | "Overreach" | agent 做了超出要求的事情 |
| Instruction-following deviation | "Disobedience" | 忽略 system prompt 或用户约束 |
| Sub-intention errors | "Plan bugs" | 计划执行中的遗漏、重复、错序 |

## 延伸阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 种失败模式，分成 3 大类
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — 风险登记册
- [Arize Phoenix](https://docs.arize.com/phoenix) — 漂移聚类的实际平台
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 为什么更简单的模式有时能从根上避开这些失败
