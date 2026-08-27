# 红队测试：PAIR 与自动化攻击

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）提出的 PAIR，即 Prompt Automatic Iterative Refinement，是自动化黑盒 jailbreak 中最经典的一种。攻击者 LLM 带着红队系统提示，不断为目标 LLM 生成新的 jailbreak prompt，并把每一轮的尝试与回复积累在自己的对话历史中，作为 in-context 反馈继续优化。PAIR 通常能在 20 次查询以内成功；与 GCG（Zou 等人的 token-level gradient search）相比，它高效得多，而且不需要 white-box 访问。如今，PAIR 已经成为 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准基线之一，通常与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 一起比较。

**Type:** 构建
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** 阶段 18 · 01（指令遵循作为对齐信号）、阶段 14（智能体工程）
**Time:** 约 75 分钟

## 学习目标

- 描述 PAIR 算法的组成：攻击者系统提示、迭代式 refinement、以及 in-context feedback。
- 解释为什么在黑盒目标上，PAIR 会比 GCG 明显更高效。
- 说出另外四种自动攻击基线（GCG、AutoDAN、TAP、PAP），并说明它们各自的一个关键特征。
- 描述 JailbreakBench 和 HarmBench 的评测协议，以及 “attack success rate” 在各自框架下具体表示什么。

## 问题

red-teaming 过去大多是手工工作。少数有经验的测试者人工构造对抗性 prompt，再记录哪些攻击有效。这个流程很难扩展：要想估计攻击成功率，就需要足够大的统计样本；而目标模型每发布一个新版本，攻击面又会变化。PAIR 的意义在于，它把 red-teaming 变成了一个围绕黑盒目标持续优化的过程。

## 概念

### PAIR algorithm

输入包括：
- Target LLM T（被攻击的模型）。
- Judge LLM J（判断回复是否构成 jailbreak）。
- Attacker LLM A（红队优化器）。
- Goal string G，例如：“respond with [harmful instruction].”
- Budget K（通常是 20 次查询）。

循环过程如下，对 k in 1..K：
1. 将目标 G 和当前已有的 (prompt, response) 历史提供给 A。
2. A 生成一个新的 prompt p_k。
3. 把 p_k 提交给 T，获得回复 r_k。
4. J 根据目标对 (p_k, r_k) 打分。
5. 如果分数 >= threshold，则停止，说明已经找到 jailbreak。
6. 否则，把 (p_k, r_k) 追加到 A 的历史中，继续下一轮。

论文中的经验结果（NeurIPS 2023）显示：针对 GPT-3.5-turbo、Llama-2-7B-chat，PAIR 的攻击成功率超过 50%；平均成功所需查询次数大约在 10 到 20 次之间。

### 为什么 PAIR 效率高

GCG（Zou et al. 2023）通过梯度搜索对抗性 token suffix，需要 white-box 模型访问，并且常常生成不可读的后缀字符串。PAIR 则完全基于黑盒交互，而且生成的是自然语言攻击，更容易跨模型迁移。它真正的效率来源在于 in-context feedback：攻击者能够从每一次拒绝中学习，然后据此调整下一轮 prompt。GCG 没有这种等价机制，每次 token 更新都在重复发现前面已经找到的线索。

### 相关自动攻击

- **GCG (Zou et al. 2023, arXiv:2307.15043).** 基于梯度的 token-level 对抗后缀搜索。white-box、可迁移，但会产生不可读字符串。
- **AutoDAN (Liu et al. 2023).** 在层次化目标函数引导下，对 prompt 做 evolutionary search。
- **TAP (Mehrotra et al. 2024).** Tree-of-attacks with pruning，把多个 PAIR 式 rollout 组织成带剪枝的搜索树。
- **PAP (Zeng et al. 2024).** Persuasive Adversarial Prompts，把人类说服技巧编码成 prompt 模板。

### JailbreakBench and HarmBench

这两个 2024 年基准都在做标准化评估：

- **JailbreakBench (arXiv:2404.01318).** 覆盖 10 个 OpenAI policy category 下的 100 种有害行为。核心指标是 attack success rate (ASR)。评测通常需要一个 judge，例如 GPT-4-turbo、Llama Guard 或 StrongREJECT。
- **HarmBench (Mazeika et al. 2024).** 覆盖 7 个类别、510 种行为，并包含 semantic 和 functional harm 测试，可用于比较 18 种攻击对 33 个模型的效果。

ASR 一般都要在固定 query budget 下报告。比较两种攻击时，预算必须对齐；例如 200 次查询下 90% 的 ASR，和 20 次查询下 85% 的 ASR，并不能直接比较。

### 为什么它对 2026 年部署重要

到 2026 年，几乎所有前沿实验室在发布模型前，都会对生产模型跑 PAIR 和 TAP。ASR 曲线会出现在 model cards（第 26 课）和 safety-case 附录（第 18 课）中。也就是说，这已经不是“稀奇攻击”，而是标准基础设施。

### 它在 Phase 18 里的位置

第 12 课是自动攻击的基础。第 13 课（Many-Shot Jailbreaking）讨论的是长上下文长度利用。第 14 课（ASCII Art / Visual）是编码类攻击。第 15 课（Indirect Prompt Injection）则转向 2026 年生产环境里最核心的系统边界攻击。第 16 课会介绍对应的防御工具（Llama Guard、Garak、PyRIT）。

```figure
al-pair-loop
```

## 用它

`code/main.py` 构建了一个 toy PAIR loop。目标模型是一个 mock classifier，会拒绝“明显有害”的 prompt，例如依赖关键词过滤的简单防御。攻击者是一个基于规则的 refiner，会依次尝试 paraphrase、roleplay framing 和 encoding。judge 负责对回复打分。你可以看到：面对关键词过滤器时，攻击者通常会在约 5–15 次迭代内成功；而面对语义过滤器时，它会失败。

## 交付成果

这一课产出 `outputs/skill-attack-audit.md`。给定一份红队评估报告，它会审计：实际运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP）、每种攻击用了多少预算、采用了哪个 judge，以及使用的是哪一组有害行为集（JailbreakBench、HarmBench 或内部集合）。

## 练习

1. 运行 `code/main.py`。测量三种内置攻击策略的平均成功查询次数，并解释每种策略利用了目标防御的什么假设。

2. 实现第四种攻击策略，例如翻译到另一种语言，或使用 Base64 编码。报告它在关键词过滤目标和语义过滤目标上的新平均成功查询次数。

3. 阅读 Chao et al. 2023 的 Figure 5（PAIR 与 GCG 对比）。说明两个即使 PAIR 更高效、但仍应优先选择 GCG 的场景。

4. JailbreakBench 针对固定目标集报告 ASR。设计一个额外指标，用来衡量攻击多样性，例如成功 prompt 的方差，并解释为什么多样性会影响防御评估。

5. TAP（Mehrotra 2024）通过 branching + pruning 扩展 PAIR。为 `code/main.py` 草拟一个 TAP 风格的扩展，并说明它在计算成本与成功率之间的权衡。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| PAIR | “自动 jailbreak” | Prompt Automatic Iterative Refinement；由 attacker-LLM + judge-LLM 组成的循环攻击 |
| GCG | “梯度 jailbreak” | white-box 条件下，对对抗性后缀做 token-level gradient search |
| Attack success rate (ASR) | “k 次查询下的成功率” | 核心指标；必须与 query budget 和 judge 身份一起报告 |
| Judge LLM | “评分器” | 判断回复是否满足有害目标的 LLM |
| JailbreakBench | “那个评测基准” | 带类别标签的标准化有害行为集合 |
| HarmBench | “更大的基准” | 510 个行为，兼顾 functional 和 semantic harm 测试 |
| TAP | “攻击树” | 带 branching + pruning 的 PAIR 扩展；通常以更高算力换更高 ASR |

## 进一步阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG 论文
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 标准化评测基准
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — 更广泛的攻击评测基准
