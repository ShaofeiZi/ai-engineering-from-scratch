# Reflexion：语言强化学习

> 基于梯度的强化学习需要数千次试验和一个 GPU 集群，才能修复某种失败模式。Reflexion（Shinn 等，NeurIPS 2023）用自然语言做到这一点：每次试验失败后，智能体写下一段反思，将它存入情景记忆，并以该记忆为条件进行下一次试验。Letta 的休眠时计算、Claude Code 的 CLAUDE.md 经验记录，以及 pro-workflow 的 learn-rule，背后都是这种模式。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 01（智能体循环）、阶段 14 · 02（ReWOO）
**Time:** 约 60 分钟

## 学习目标

- 说出 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector），以及情景记忆所扮演的角色。
- 使用标准库实现 Reflexion 循环，其中包含二元 Evaluator、反思缓冲区和全新重试。
- 针对给定任务，在标量、启发式和自我评估反馈源之间作出选择。
- 解释为什么语言强化能够捕获那些需要基于梯度的强化学习尝试数千次才能修复的错误。

## 问题

智能体执行任务失败了。在标准强化学习中，你会再运行数千次试验，计算梯度并更新权重。这个过程昂贵且缓慢，而且多数生产智能体没有足够训练预算来应对每一次失败。

Reflexion（Shinn 等，arXiv:2303.11366）提出了另一个问题：如果智能体只需想一想失败原因，再把这段想法放进提示中重新尝试，会怎样？无需更新权重，也不需要梯度，只需在多次试验之间保存自然语言。

结果是：它在 ALFWorld 上超过了 ReAct 与其他未经微调的基线，在 HotpotQA 上优于 ReAct，并在代码生成任务 HumanEval / MBPP 上取得当时最先进的结果。整个过程没有执行一次梯度更新。

## 核心概念

### 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

再加上一种数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次试验由 Actor 运行，Evaluator 对其评分。若分数较低，Self-Reflector 会生成一段反思（例如：“我选错了工具，因为我把问题误读成询问 X，而它实际询问的是 Y”）。这段反思进入情景记忆。下一次试验从头开始，但能够看到该反思。

### 三类 Evaluator

1. **标量型**——外部二元信号。ALFWorld 成功或失败，HumanEval 测试通过或失败。这是最简单、信号最强的形式。
2. **启发式**——预定义失败特征。例如：“若智能体连续两次产生相同行动，则标记为卡住。”“若轨迹超过 50 步，则标记为低效。”
3. **自我评估型**——由 LLM 为自身轨迹评分。没有真实标签时需要这种方式，但信号较弱；适合与基于工具的验证配合使用（第 05 课——CRITIC）。

2026 年的默认选择是混合使用：有标量信号时采用标量，没有时采用自我评估，再用启发式规则作为安全护栏。

### 为什么这种方法能够泛化

Reflexion 与其说是一种新算法，不如说是一种有了名字的模式。几乎所有生产环境中的“自愈”智能体都运行着它的某种变体：

- Letta 的休眠时计算（第 08 课）：一个独立智能体反思过去的对话，并写入记忆块。
- Claude Code 的 `CLAUDE.md`／“save memory”模式：把反思记录为经验，并前置到未来会话中。
- pro-workflow 的 `/learn-rule` 命令：把纠正意见记录为显式规则。
- LangGraph 的反思节点：节点为输出评分，并在需要时路由到 refine。

它们都源于同一个洞见：自然语言足以承载“我从失败中学到了什么”，并把这些经验传递到后续运行。

### 何时有效，何时无效

Reflexion 在以下情况下有效：

- 存在清晰的失败信号（测试失败、工具错误、答案错误）。
- 任务类别可复现（可以再次提出同类问题）。
- 反思有足够空间改进轨迹（行动预算充足）。

Reflexion 在以下情况下无济于事：

- 智能体第一次尝试已经成功。
- 失败来自外部因素（网络中断、工具损坏）——反思“网络中断了”无法改善后续运行。
- 反思演变成迷信——把某次偶发故障的叙事永久存下来。

2026 年的一个陷阱是记忆腐化。反思持续累积，其中一些会过时或出错；随着情景缓冲区增长，重新运行也会变慢。缓解方法包括定期压缩（第 06 课）、为反思设置 TTL，或使用独立的休眠时清理智能体（Letta）。

```figure
react-trace
```

## 动手构建

`code/main.py` 在一个玩具谜题上实现 Reflexion：生成一个由 3 个元素构成、总和等于目标值的列表。Actor 输出候选列表；Evaluator 检查总和；Self-Reflector 用一行文字诊断失败原因。该反思进入情景记忆，供下一次试验使用。

组件如下：

- `Actor`——看到反思后会改进的脚本化策略。
- `Evaluator.binary()`——根据是否达到目标总和给出通过／失败结果。
- `SelfReflector`——生成一行失败诊断。
- `EpisodicMemory`——带 TTL 语义的有界列表。

运行：

```
python3 code/main.py
```

追踪会显示三次试验。试验 1 失败并保存一段反思；试验 2 看到反思后有所改进，但仍然失败；试验 3 成功。将其与无反思的基线运行比较——基线会一直卡在试验 1 的答案上。

## 实际使用

LangGraph 将反思作为一种节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 会把情景缓冲区外部化为 Markdown 文件。Letta 的休眠时计算会在空闲期间运行 Self-Reflector，使主智能体仍受延迟目标约束。OpenAI Agents SDK 并不直接提供 Reflexion；你可以用自定义 Guardrail 按分数拒绝轨迹，再用跨运行持久存在的记忆 `Session` 构建它。

## 交付成果

`outputs/skill-reflexion-buffer.md` 用于创建和维护情景缓冲区，支持反思捕获、TTL 与去重。给定任务类别和一次失败，它会生成真正有助于下一次试验的反思，而不是泛泛地说“要更小心”。

## 练习

1. 从二元 Evaluator 切换到返回距离度量（偏离目标多远）的标量 Evaluator。它是否收敛得更快？
2. 为反思设置 10 次试验的 TTL。超过这个时间点后，较旧反思会带来帮助还是伤害？
3. 实现启发式 Evaluator：若同一行动重复出现，则把试验标记为卡住。它与 Self-Reflector 如何交互？
4. 使用一个忽略反思的对抗性 Actor 运行 Reflexion。要迫使 Actor 注意反思，最小的反思提示工程改动是什么？
5. 阅读 Reflexion 论文中关于 AlfWorld 的第 4 节。从概念上复现成功率提高 130% 的结果：相比原始 ReAct，关键变化是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | “自我纠正” | Shinn 等，2023——Actor、Evaluator、Self-Reflector 加情景记忆 |
| 语言强化 | “无梯度学习” | 前置到下一次试验提示中的自然语言反思 |
| 情景记忆 | “逐任务反思” | 针对一个任务类别保存先前反思的有界缓冲区 |
| 标量 Evaluator | “二元成功信号” | 来自真实标签的通过／失败或数值得分 |
| 启发式 Evaluator | “基于模式的检测器” | 预定义的失败特征（例如循环卡住、步骤过多） |
| 自我 Evaluator | “由 LLM 评判自己的轨迹” | 没有真实标签时使用的低信号后备方案——应配合基于工具的验证 |
| 记忆腐化 | “陈旧反思” | 情景缓冲区被过时条目填满；通过压缩／TTL 修复 |
| 休眠时反思 | “异步自我反思” | 在关键路径外运行 Self-Reflector，让主智能体保持低延迟 |

## 延伸阅读

- [Shinn 等，Reflexion：通过语言强化学习的语言智能体（arXiv:2303.11366）](https://arxiv.org/abs/2303.11366)——奠基论文
- [Letta，休眠时计算](https://www.letta.com/blog/sleep-time-compute)——生产环境中的异步反思
- [Anthropic，AI 智能体的有效上下文工程](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——把情景缓冲区作为上下文的一部分进行管理
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)——反思节点模式
