# 自主编码代理全景 (2026)

> 在不到三年的时间里，SWE-bench Verified 的成绩从 4% 提升到 80.9%。同样是 Claude Sonnet 4.5，在 SWE-agent v1 上得到 43.2%，而在 Cline autonomous 脚手架中达到 59.8%: 围绕模型构建的脚手架，如今与模型本身同样重要。OpenHands（前身为 OpenDevin）是目前最活跃的 MIT 许可平台，它的 CodeAct 循环会在沙盒里直接执行 Python 动作，而不是走 JSON 工具调用。那些醒目的高分背后还藏着一个方法论问题: 500 个 SWE-bench Verified 任务中有 161 个只需要改 1 到 2 行，而对于相同的前沿模型，SWE-bench Pro（要求改动 10 行以上的任务）得分仍然只有 23% 到 59%。

**Type:** 学习
**Languages:** Python（stdlib，CodeAct 与 JSON 工具调用对比）
**Prerequisites:** 阶段 14 · 07（工具使用），阶段 15 · 01（长时程代理）
**Time:** 约 45 分钟

## 问题

“哪个编码代理最好”这个问题本身就问错了。真正该问的是: 在与我的实际工作分布相匹配的任务集上，配合我准备在线上使用的脚手架，我最终能拿到多高的端到端可靠性？

从 2022 年到 2026 年，这个领域逐渐明白，脚手架是承重结构: 检索层、规划器、沙盒、编辑与验证循环、反馈格式，都会实质性地改变结果。Claude Sonnet 4.5 在 SWE-agent v1 的 SWE-bench Verified 上得到 43.2%；同样的模型放进 Cline 的 autonomous 脚手架后则达到 59.8%。同一组权重，绝对差了 16.6 个点。基础模型只是组件，循环本身才是产品。

另一个伴随而来的问题是，基准的“饱和”会掩盖回归。SWE-bench Verified 已经接近饱和，其中容易任务的长尾部分会把头部成绩整体往上抬: 500 个任务里有 161 个只需要改不超过 2 行。现实世界里的质量，更适合用 SWE-bench Pro 这类分布来衡量，因为它关注的是 10 行以上的改动，而同样的领先系统在这里依然只有 23% 到 59%。

## 概念

### 用一段话说明 SWE-bench

SWE-bench（Jimenez 等人）收集真实的 GitHub issue，并配有对应的真实补丁，要求代理生成一个补丁，让测试套件通过。SWE-bench Verified（OpenAI，2024）是在此基础上由人工筛选出的 500 个任务子集，去掉了含糊不清和本身损坏的题目。SWE-bench Pro 是更难的后继版本，只保留那些需要修改 10 行以上代码的任务，而当前前沿代理在这个集合上的分数仍然只有 23% 到 59%。

### 2022 到 2026 的曲线到底说明了什么

- **2022**: 研究型模型在原始 SWE-bench 上大约只有 4%。
- **2024**: GPT-4 + Devin 风格脚手架大约达到 14%；SWE-agent 大约是 12%。
- **2025**: Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中把成绩推进到 40% 到 55% 区间。
- **2026**: Claude Sonnet 4.5 和其他前沿竞争者在 SWE-bench Verified 上达到 70% 到 80% 以上。Epoch AI 的排行榜在持续追踪这些结果。

这条斜率来自三个相互叠加的来源: 更好的基础模型、更好的脚手架（CodeAct、反思、验证循环），以及更好的基准（Verified 去掉了噪声任务）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为 OpenDevin）押注了一个明确的架构方向: 模型不再输出要由宿主解码执行的 JSON 工具调用，而是直接输出 Python 代码，再由 Jupyter 风格的内核在沙盒里运行。这样一来，代理可以在一次动作里遍历文件、串联工具，还能自己捕获异常。

权衡在这里:

- **JSON 工具调用**: 每个动作就是一轮；易于审计；组合能力有限；默认更安全，因为每次调用都会经过显式校验器。
- **CodeAct**: 一个动作就可能是一整个程序；组合能力强；但需要更坚固的沙盒（OpenHands 使用 Docker 隔离）；失败模式则取决于沙盒运行时允许它做什么。

这两种架构都已经进入生产环境。CodeAct 在开放平台里更常见，例如 OpenHands、smolagents。JSON 工具调用则仍然主导托管服务，例如 Anthropic Managed Agents、OpenAI Assistants，因为这些服务由提供方直接控制执行器。

### 2026 年版图中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands（OpenDevin） | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | Agent-Computer Interface（ACI） | 第一个端到端的 SWE-bench 脚手架 |
| Aider | Apache-2 | 在本地仓库里通过 diff 编辑 | 脚手架极简，但回归稳定性很强 |
| Cline | Apache-2 | 带工具策略的 VS Code 代理 | 在 Sonnet 4.5 上得分最高的开放脚手架 |
| Devin（Cognition） | Proprietary | 托管 VM + 规划器 | 首个“AI 软件工程师”产品类别 |
| Claude Code | Proprietary | 权限模式 + routines | 第 10 课会详细讲它的代理循环 |

### 为什么脚手架占主导地位

一次编码运行就是一条长时程轨迹（见第 1 课）。可靠性会在多个步骤中连乘。脚手架能拉开差距的地方主要有三处:

1. **检索**: 找到该读哪些文件，往往才是隐形瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引、Aider 的 repo-map，都是在解决这个问题。
2. **验证循环**: 跑测试、看堆栈、再尝试一次，在 SWE-bench 上往往就是 10 个点以上的差距。
3. **失败遏制**: 能在出错时回滚的沙盒，可以阻止损害逐步累积。同一个模型，有没有验证循环，看起来就像两个完全不同的产品。

### 基准饱和与真实任务分布

OpenHands 的作者和 Epoch AI 都指出，SWE-bench Verified 有一段“容易尾部”: 500 个任务里有 161 个只需要改 1 到 2 行。高分一部分正是被这段尾部拉上去的。SWE-bench Pro 则只保留 10 行以上改动的任务，即使是前沿系统，在这里也只有 23% 到 59%。你的生产任务分布，几乎肯定更接近 Pro，而不是 Verified。

这对选代理的含义是: 应该从你自己的 bug 积压里抽出一个类似 Pro 的子集来跑。真正重要的分数，是它在与你实际交付相似的任务上的分数。

```figure
a5-scaffold-delta
```

## 学以致用

`code/main.py` 会在一个固定的迷你任务分布上，对比两种玩具版代理脚手架:

1. 一个 **JSON tool-call** 脚手架，每轮只能做一个动作。
2. 一个 **CodeAct** 脚手架，每次动作都可以输出一小段 Python 代码。

两者都使用一个桩替身“模型”（确定性规则），这样对比就把模型质量的影响隔离掉了，只看脚手架差异。输出会显示，CodeAct 脚手架能用更少的轮次解决更多任务，但代价是每次动作的爆炸半径更大。

## 交付成果

`outputs/skill-scaffold-audit.md` 会帮助你在采用某个编码代理脚手架之前做一次审计: 检索质量如何、有没有验证器、沙盒隔离是否足够、以及基准分布和你的实际任务分布是否匹配。

## 练习

1. 运行 `code/main.py`。面对同一组任务，每种脚手架分别需要多少轮？每次动作的爆炸半径又有多大？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上优于 JSON 工具调用。找出论文明确承认的一种失败模式，并用一句话说明这种失败模式在什么场景下会主导生产环境表现。

3. 从你的 bug 积压里挑一个任务，它需要跨两个文件改动 10 行以上。分别估计前沿模型在 (a) JSON 工具调用 与 (b) CodeAct 下的端到端成功率，并解释两者差距来自哪里。

4. SWE-bench Verified 中有 161 个单文件、只改 1 到 2 行的任务。构造一个排除这部分任务的分数。排行榜会怎样洗牌？

5. 阅读 “Introducing SWE-bench Verified”（OpenAI）。解释它用什么具体方法剔除了含糊不清的任务，并指出一类这种人工策展仍然可能漏掉的问题。

## 关键术语

| 术语 | 人们常说的叫法 | 实际含义 |
|---|---|---|
| SWE-bench | “编码基准” | 真实 GitHub issue，附带真实补丁和测试套件 |
| SWE-bench Verified | “清洗后的子集” | 500 个经人工筛选的任务，但仍保留了容易任务的尾部 |
| SWE-bench Pro | “更难的子集” | 要求 10 行以上改动；前沿系统仍只有 23% 到 59% |
| CodeAct | “代码即动作” | 代理输出 Python；由 Jupyter 风格内核在沙盒中执行 |
| JSON tool call | “函数调用” | 每个动作都是结构化 JSON 载荷，执行前要先校验 |
| Scaffold | “代理框架” | 围绕基础模型的检索、规划器、执行器与验证循环 |
| ACI（Agent-Computer Interface） | “SWE-agent 的格式” | 为 LLM 易用性设计的命令集合，不是给人类 shell 用的 |
| Verifier loop | “测试并重试” | 运行测试、读取输出、修正补丁；最大的非模型可靠性增益之一 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准与方法说明。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 介绍这个筛选子集是如何构建的。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构与事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时追踪的排行榜。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 用于理解长时程编码代理可靠性的框架。
