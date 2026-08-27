# 基准评测：SWE-bench、GAIA、AgentBench

> 到 2026 年，agent 评测主要由三类 benchmark 锚定：SWE-bench 用来测试代码补丁能力，GAIA 用来测试通用型工具使用能力，AgentBench 用来测试跨环境推理能力。你需要了解它们的构成、污染问题，以及它们没有测到什么。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 06（工具使用）
**Time:** 约 60 分钟

## 学习目标

- 说出 SWE-bench 的测试 harness（FAIL_TO_PASS），并解释为什么它以单元测试作为闸门。
- 解释 SWE-bench Verified（OpenAI，500 个任务）为什么存在，以及它排除了什么。
- 描述 GAIA 的设计：对人类简单、对 AI 困难，并分成三个难度等级。
- 说出 AgentBench 的八个环境，以及它对开源 LLM 的主要阻碍结论。
- 概述 SWE-bench+ 的污染发现及其影响。

## 问题

排行榜会告诉你哪个模型在某个 benchmark 上赢了，但它不会告诉你：

- 这个 benchmark 是否受到了污染（训练数据里已有答案、测试集泄漏）。
- 这个 benchmark 是否真的在衡量你关心的能力（代码、浏览还是通用能力）。
- 评估器是否足够稳健（AST 匹配、状态检查、人工审查）。

在你引用任何一个分数之前，先搞清楚这三个锚定 benchmark 以及它们各自的失败模式。

## 概念

### SWE-bench（Jimenez et al., ICLR 2024 oral）

- 来自 12 个热门 Python 仓库的 2,294 个真实 GitHub issue。
- agent 拿到的是：修复前 commit 上的代码库，加上一段自然语言 issue 描述。
- agent 产出的是：一个 patch。
- evaluator 的做法是：应用 patch，运行仓库测试套件。patch 必须让 FAIL_TO_PASS 测试翻转为通过（之前失败，现在通过），同时不能破坏 PASS_TO_PASS 测试。

SWE-agent（Yang et al., 2024）在发布时达到 12.5%，关键做法是强调 agent-computer interface，比如文件编辑器命令和模型更容易理解的搜索语法。

### SWE-bench Verified

由 OpenAI 在 2024 年 8 月推出，是一个人工整理过的 500 任务子集。它去掉了模糊 issue、不可靠测试，以及修复目标不清晰的任务。现在它是回答“你的 agent 是否真的能交付真实 patch？”这一问题时的主要 benchmark。

### 污染问题

- 超过 94% 的 SWE-bench issue 都早于大多数模型的训练截止时间。
- **SWE-bench+** 发现，32.67% 的成功 patch 在 issue 文本里就已经泄露了解法（模型直接在描述里看到了修复线索），另外 31.08% 的案例由于测试覆盖薄弱而可疑。
- Verified 更干净，但也不是完全无污染。

实际含义是：一个在 SWE-bench 上拿到 50% 的模型，在 SWE-bench+ 上可能只有 35%。如果你宣称自己在 SWE-bench 上表现很好，最好同时报告两者。

### GAIA（Mialon et al., Nov 2023）

- 总共 466 道题；其中 300 道保留给 huggingface.co/gaia-benchmark 上的私有排行榜。
- 设计理念是：“对人类来说概念上很简单（92%），对 AI 来说却很难（带插件的 GPT-4：15%）。”
- 测的是推理、多模态、web 与 tool use。
- 分为三个难度等级；Level 3 需要跨多种模态串接很长的工具链。

如果你想测“通用型能力”，就应该跑 GAIA。不要把它和代码类 benchmark 混为一谈。

### AgentBench（Liu et al., ICLR 2024）

- 包含 8 个环境，横跨代码（Bash、DB、KG）、游戏（Alfworld、LTP）、web（WebShop、Mind2Web）以及开放式生成。
- 多轮交互，每个 split 大约有 4k-13k 轮。
- 它的核心结论是：长期推理、决策能力和指令遵循，仍然是开源 LLM 追赶商业模型时的主要阻碍。

### 这些 benchmark 没有测什么

- 真实世界里的运营成本（token、wall-clock）。
- 对抗条件下的安全表现。
- 你自己业务领域中的表现（要靠自己的 eval，见 Lesson 30）。
- 长尾失败（benchmark 看平均值，但生产环境往往更在乎最差的 1%）。

### 做 benchmark 时常见的错误

- **执着于单一数字。** SWE-bench 50% 这个数字，信息量远小于 P50 / P75 / P95 成本和步骤分布。
- **忽略污染问题。** 只报 SWE-bench 分数，却不提 Verified 或 SWE-bench+，是会误导人的。
- **把 benchmark 当开发目标。** 一味优化 benchmark 分数，往往会偏离真实生产价值。

```figure
ae-swebench-gate
```

## 动手构建

`code/main.py` 实现了一个类 SWE-bench 的 toy harness：

- 合成的 bug-fix 任务（3 个）。
- 一个脚本化的 “agent” 来提出 patch。
- 一个测试运行器，用来检查 FAIL_TO_PASS（bug 是否修好）与 PASS_TO_PASS（是否没有引入新回归）。
- 一个基于问题分解深度的 GAIA 风格难度分类器。

运行它:

```
python3 code/main.py
```

输出会展示按任务和按难度划分的解决率，让 evaluator 的规则变得更具体。

## 如何使用

- **SWE-bench Verified**：用于代码 agent。报告成绩时优先给出 Verified 分数。
- **GAIA**：用于通用型 agent。使用它的私有排行榜 split。
- **AgentBench**：用于跨环境能力比较。
- **Custom evals**（Lesson 30）：用于你的产品真实形态。

## 交付成果

`outputs/skill-benchmark-harness.md` 可以为任意代码库-任务组合构建一个带 FAIL_TO_PASS / PASS_TO_PASS 闸门的 SWE-bench 风格 harness。

## 练习

1. 把这个 toy harness 移植到一个真实仓库上（选你自己的一个）。为已知 bug 写 3 个 FAIL_TO_PASS 测试。
2. 增加 step-count 指标。在你的 3 个任务里，每次成功解决平均用了多少 agent steps？
3. 阅读 SWE-bench+ 论文，实现一个 solution-leakage 检查（把 issue 文本和 diff 做模式匹配）。
4. 从公开 split 里下载一道 GAIA 题。跟踪一个 GPT-4 级别的 agent 会怎么做，它需要哪些工具？
5. 阅读 AgentBench 的分环境结果。哪个环境最像你的产品表面？那里的 “SOTA” 到底意味着什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SWE-bench | “Code agent benchmark” | 2,294 个 GitHub issue；patch 必须翻转 FAIL_TO_PASS 测试 |
| SWE-bench Verified | “Clean SWE-bench” | 500 个人工整理任务，由 OpenAI 发布 |
| FAIL_TO_PASS | “Fix gate” | 原本失败的测试，在打完 patch 后必须通过 |
| PASS_TO_PASS | “No-regression gate” | 原本通过的测试，打完 patch 后仍必须通过 |
| GAIA | “Generalist benchmark” | 466 道对人类简单、对 AI 困难的多工具题目 |
| AgentBench | “Multi-env benchmark” | 8 个环境；长时程、多轮交互 |
| Contamination | “Training-set leak” | benchmark 任务已经出现在模型训练数据中 |
| SWE-bench+ | “Contamination audit” | 在成功的 SWE-bench patch 中发现 32.67% 解法泄漏 |

## 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — 原始 benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 人工整理后的子集
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — 通用型 benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — 多环境 benchmark 套件
