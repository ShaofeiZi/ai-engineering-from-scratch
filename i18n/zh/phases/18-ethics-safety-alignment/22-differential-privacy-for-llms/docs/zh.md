# 面向 LLM 的差分隐私

> DP-SGD 仍然是标准方案，核心机制是对梯度更新注入噪声，并给出形式化的 (epsilon, delta) 保证。但它在计算、显存和效用上的代价都很高，因此到了 2025 年，更常见的落地配置已经变成参数高效的 DP 微调，也就是 LoRA + DP-SGD（ACM 2025）。当前有两组彼此紧张的证据：一方面，基于 canary 的 membership inference（Duan et al., 2024）在语言模型上报告的成功率有限；另一方面，训练数据提取（Carlini et al., 2021；Nasr et al., 2025）却能恢复大量逐字记忆。2025 年 3 月的工作（arXiv:2503.06808）给出的解释是，两者测量的对象并不相同，前者测的是插入式 canary，后者测的是“最容易被提取的数据”。新的 canary 设计让基于 loss 的 MIA 在没有 shadow models 的情况下也可实施，并首次对一个在真实数据上训练、且具有现实 DP 保证的 LLM 做出非平凡审计。替代路线还包括 PMixED（arXiv:2403.15638），也就是在推理时通过 next-token distribution 上的 mixture of experts 提供私有预测，以及 DP synthetic data generation（Google Research 2024）。新出现的攻击方向则是“通过 LLM 反馈逆转差分隐私”，也就是通过 confidence score 泄漏反推个体信息。

**Type:** 构建
**Languages:** Python (stdlib, DP-SGD noise-injection and ε-δ accountant demonstration)
**Prerequisites:** 阶段 01 · 09（信息论）、阶段 10 · 01（大模型训练）
**Time:** 约 60 分钟

## 学习目标

- 定义 (epsilon, delta)-differential privacy，并说出 DP-SGD 的标准配方。
- 解释 2024-2025 年的关键张力：canary MIA 与 training-data extraction 为什么会得出不同结论。
- 描述 PMixED，并说明为什么 inference-time private prediction 可以作为 DP training 的替代路线。
- 描述通过大模型反馈逆转差分隐私这一攻击。

## 问题

LLM 会记忆训练数据。Carlini 等人在 2021 年已经证明，生产级语言模型会在适当诱导下复现训练集中的原文文本。DP 是这类问题的形式化防线：通过训练方式保证，模型输出对任意单个训练样本的变化都不敏感。2024 到 2025 年的证据表明，DP-SGD 仍然是必要手段，但现实部署里的 ε 取值是否真正匹配威胁模型，仍然远未解决。

## 概念

### (ε, δ)-differential privacy

如果一个随机算法 M 满足：对任意只相差一个样本的数据集，以及任意事件 S：
P(M(D) in S) <= e^ε * P(M(D') in S) + δ.

它就称为 (ε, δ)-DP。

直观理解是：输出分布足够接近，接近程度由 ε 参数化，因此单个个体对输出的贡献无法被可靠推断出来，除非落入概率为 δ 的例外情况。

### DP-SGD

Abadi et al. 2016 提出的标准配方如下：
1. 抽样一个 mini-batch。
2. 计算每个样本各自的梯度。
3. 将每个样本的梯度裁剪到阈值 C。
4. 对裁剪后的梯度求和，并加入标准差为 σ * C 的 Gaussian noise。
5. 用这个带噪声的和来更新参数。

隐私成本通常由 accountant 跟踪，例如 Moments Accountant 或 Rényi DP accountant。LLM 文献里报告的 ε 值分布很广，取决于威胁模型、数据敏感度和目标效用；不存在一个放之四海而皆准的“安全默认 ε”。在部分 LLM 训练设置中，公开论文举过大约 ε ≈ 1–10 的例子，但这些只是示意区间，不应被当成推荐默认值。一般来说，ε 越低，就需要注入更多噪声，而效用损失也越大。

### LoRA + DP-SGD

对前沿大模型做完整的 DP-SGD 代价高得几乎不可承受。LoRA（Hu et al. 2022）把梯度更新限制在小规模 adapter 上，从而显著降低每样本梯度的存储成本。LoRA + DP-SGD 因此成为 2025 年最常见的现实配置。这里的 DP 保证只覆盖 adapter，而 base model 本身保持冻结。

### 2024-2025 年的张力

目前主要有两条证据链：

- **Canary MIA（Duan et al. 2024）。** 把独特的 canary 插入训练数据，然后检测 membership-inference attacker 是否能识别这些 canary。该路线在语言模型上报告的成功率有限，因此看起来像是在说 MIA 很难。
- **Training-data extraction（Carlini 2021, Nasr et al. 2025）。** 给模型一个前缀，测它能否把训练数据中的原文续写出来。该路线报告了显著的逐字记忆，因此看起来像是在说，在真正相关的意义上，MIA 并不难。

2025 年 3 月的解释性工作（arXiv:2503.06808）指出：这两者其实在测不同东西。MIA 问的是“样本 e 是否在 D 里”，而且常用对象是人为插入的 canary；数据提取问的是“D 中哪些内容能被我恢复出来”。对隐私而言，更关键的是“最容易被提取的样本”而不是任意插入的 canary。因为 canary 通常没有被专门优化成最易提取样本，所以它往往会低估真实风险。

新的 canary 设计、无需 shadow models 的 loss-based MIA，以及基于真实数据、现实 DP 保证的首个非平凡 LLM DP 审计，都是这一轮进展的关键成果。

### DP training 的替代路线

- **PMixED（arXiv:2403.15638）。** 在推理时实现 private prediction。做法是在 next-token distribution 上构造 mixture of experts；每个 expert 只看一部分训练数据；最终聚合时再加入噪声以满足 DP。这样就完全绕开了 DP training。
- **DP synthetic data generation（Google Research 2024）。** 先用 LoRA + DP-SGD 进行微调，再采样 synthetic data，最后让下游 classifier 在这些 synthetic data 上训练。

这两条路线都绕开了 full DP training 的高昂效用成本，但代价是它们针对的是不同的 threat model。

### 通过大模型反馈逆转差分隐私

这是 2025 年出现的一类新攻击。攻击者把 DP-trained model 的 confidence scores 当成 oracle，借此重新识别个体信息。也就是说，即便最终输出文本本身没有直接泄漏，confidence distribution 仍然可能成为泄漏通道。

对应的防守方式是：不要暴露 confidence，或者在对外暴露前先做截断、离散化或量化。这是 (ε, δ)-DP 训练之外仍需额外满足的部署要求。

### 它在 Phase 18 里的位置

Lessons 20-21 讨论 bias 与 fairness。Lesson 22 进入 privacy。Lesson 23 转向通过 watermarking 建立 provenance。Lesson 27 再继续覆盖监管层面的数据 provenance。

```figure
an-dp-clip-noise
```

## 用它

`code/main.py` 会在一个玩具二分类数据集上模拟 DP-SGD。你可以扫描 noise multiplier σ 和 clipping norm C，观察 (ε, δ) budget 与准确率成本之间的权衡。一个 “canary attack” 会插入唯一训练样本，并通过 log-loss test 比较它在 DP 之前和之后是否仍能被检测出来。

## 交付它

这一课会产出 `outputs/skill-dp-audit.md`。给定一个语言模型部署声称自己满足 DP，它会审计：声明中的 (ε, δ) 值、使用的是哪种 accountant、MIA 的评估协议，以及是否评估过 confidence exposure 这一类泄漏向量。

## 练习

1. 运行 `code/main.py`。把 σ 扫描到 {0.5, 1.0, 2.0}，报告 (ε, δ) 与准确率之间的权衡，并找出效用开始崩塌的点。

2. 实现 canary insertion 和 log-loss test。测量在 σ = 1.0 时，DP-SGD 前后的检测率变化。

3. 阅读 Nasr et al. 2025 关于 training-data extraction 的论文。为什么在中等 ε 下，提取成功率不会直接塌掉？这说明把 MIA 当作评估手段本身有什么局限？

4. 设计一个完全在推理时运行的 PMixED（arXiv:2403.15638）部署。它解决的是哪一种 DP-SGD 没有直接覆盖的 threat model？

5. 勾勒 Differential Privacy Reversal via LLM Feedback 的攻击路径。再设计一个限制 confidence-score leakage 的对策，并估算它的部署成本。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 差分隐私（DP） | “(ε, δ)-差分隐私” | 一种形式化隐私保证：相邻数据集发生变化时，输出分布仍保持接近 |
| 差分隐私随机梯度下降（DP-SGD） | “注入噪声的 SGD” | 先裁剪梯度，再加入高斯噪声；这是标准的差分隐私训练方法 |
| LoRA + DP-SGD | “高效的隐私微调” | 在低秩适配器上执行 DP-SGD；这是 2025 年的标准配置 |
| 成员推断攻击（MIA） | “成员推断” | 用于判断某个样本是否出现在训练数据中的攻击 |
| 金丝雀样本（Canary） | “插入的水印样本” | 为测量差分隐私泄漏而插入的唯一训练样本 |
| PMixED | “隐私推理混合” | 在下一词元分布上使用专家混合方法，实现推理阶段的差分隐私 |
| 差分隐私逆转（DP Reversal） | “置信度泄漏攻击” | 把模型置信度当作预言机，用于重新识别个体的攻击 |

## 延伸阅读

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — 标准的差分隐私训练算法
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — 训练数据提取研究的经典论文
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — 成功率有限的成员推断攻击
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) — 对上述矛盾的解释
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — 推理阶段的隐私预测
