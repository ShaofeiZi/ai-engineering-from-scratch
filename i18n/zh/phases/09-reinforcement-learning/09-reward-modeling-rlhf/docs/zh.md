# 奖励建模与 RLHF

> 人类无法为“优秀的助手回答”手写奖励函数，却可以比较两个回答并选出更好的一个。先根据这些比较拟合奖励模型，再用强化学习针对它优化语言模型。Christiano，2017；InstructGPT，2022。这套方案把 GPT-3 变成了 ChatGPT。到 2026 年，它大多正被 DPO 取代——但背后的心智模型依然成立。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 05（情感分析）、阶段 9 · 08（PPO）
**Time:** 约 45 分钟

## 问题

你已经用下一词元预测目标训练了一个语言模型。它能写出符合语法的英语，却也会撒谎、啰嗦，而且在应当拒绝时不肯拒绝。继续增加预训练无法解决这个问题——网络文本本身就是问题，而不是解药。

你想要一个*标量奖励*，它能表达“对于指令 X，回答 A 比回答 B 更好”。手写这样的奖励函数不可能做到。“有帮助”并不是词元上的闭式表达式。但人类可以比较两个输出并标记偏好，而且这种数据可以廉价地大规模采集。

RLHF（Christiano 等，2017；Ouyang 等，2022）把偏好转化为奖励模型，再使用 PPO 针对这份奖励优化语言模型。整个过程分为三步：SFT → RM → PPO。ChatGPT、Claude、Gemini 以及 2023～2025 年所有其他对齐大语言模型，都采用过这套方案。

到 2026 年，PPO 步骤大多被 DPO（阶段 10 · 08）取代，因为后者成本更低，做对齐微调时效果几乎相当。不过，*奖励模型*仍然支撑着每个 Best-of-N 采样器、每条可验证奖励强化学习流水线，以及每个使用过程奖励模型的推理模型。理解 RLHF，就理解了整套对齐技术栈。

## 概念

![RLHF 三阶段：SFT、基于成对偏好的 RM 训练、带 KL 惩罚的 PPO](../../../../../../phases/09-reinforcement-learning/09-reward-modeling-rlhf/assets/rlhf.svg)

**阶段 1：监督微调（SFT）。** 从预训练基础模型开始，在人类编写的目标行为示范上微调（遵循指令的回答、有帮助的回复等）。得到模型 `π_SFT`：它*偏向良好行为*，但动作空间仍然没有边界。

**阶段 2：训练奖励模型。**

- 收集回答对 `(y_+, y_-)`，它们对应提示词 `x`，再由人类标注“y_+ 优于 y_-”。
- 训练奖励模型 `R_φ(x, y)`，使它为 `y_+` 给出更高分数。
- 损失采用 **Bradley-Terry 成对逻辑损失**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 Sigmoid 函数。奖励之差代表偏好的对数几率。BT 自 1952 年提出以来一直是标准方法，也是现代 RLHF 的主流选择。

- `R_φ` 通常由 SFT 模型初始化，并在顶部添加一个标量输出头。Transformer 骨干网络保持相同，只用一个线性层输出奖励。

**阶段 3：使用 PPO 针对 RM 优化，并加入 KL 惩罚。**

- 初始化可训练策略 `π_θ`，其起点为 `π_SFT`；同时保留一个冻结的*参考*模型 `π_ref = π_SFT`。
- 回答 `y` 结束时的奖励为：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是一种*正则化项*，而不是严格的信赖域。`β` 通常取 `0.01`～`0.05`。
- 使用该奖励运行 PPO（第 08 课）。优势在词元级轨迹上计算，但 RM 只为完整回答打分。

**为什么需要 KL？** 如果没有它，PPO 会欣然找到奖励黑客策略——RM 只在分布内补全上训练，某个分布外回答可能得到比任何人类回答都高的分数。KL 让 `π_θ` 留在 RM 训练数据所在的流形附近。它是 RLHF 中最重要的单个调节旋钮。

**2026 年的现状：**

- **DPO**（Rafailov，2023）：通过闭式代数，把阶段 2+3 化简为偏好数据上的单个监督损失。不需要 RM，也不需要 PPO；只需一小部分计算量，就能在对齐基准上达到相近质量。详见阶段 10 · 08。
- **GRPO**（DeepSeek，2024～2025）：在 PPO 中用组相对基线取代评论家（价值网络），奖励由*验证器*（代码能否运行/数学答案是否匹配）给出，而不是由人类训练的 RM 给出。它已成为推理模型的主流方法。详见阶段 9 · 12。
- **过程奖励模型（PRM）：** 为部分解答（每个推理步骤）评分，用于面向推理的 RLHF 与 GRPO 变体。
- **宪法式 AI / RLAIF：** 使用已经对齐的大语言模型代替人类生成偏好，从而扩大偏好数据预算。

```figure
reward-model
```

## 动手构建

本课使用字符串表示微型合成“提示词”和“回答”。RM 是一个基于词袋表示的线性评分器。不使用真正的大语言模型——重要的是流水线的*形状*，而不是规模。参见 `code/main.py`。

### 第 1 步：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

真实 RLHF 会把这里替换为人类标注者。数据形状——`(prompt, preferred_response, rejected_response)`——完全相同。

### 第 2 步：Bradley-Terry 奖励模型

线性分数为 `R(x, y) = w · bag(y)`。训练时最小化 BT 成对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

经过几百次更新后，`w` 会为好词元赋予正权重，为坏词元赋予负权重。

### 第 3 步：建立在 RM 之上的类 PPO 策略

我们的玩具策略从词表中生成单个词元。用 RM 为该词元评分，计算 `log π_θ(token | prompt)`，加入相对于参考模型的 KL 惩罚，再应用经过裁剪的 PPO 代理目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 第 4 步：监控 KL

每次更新都跟踪平均 `KL(π_θ || π_ref)`。如果它逐渐超过 `~5-10`，策略就已远离 `π_SFT`——说明 `β` 太低，或奖励黑客正在出现。这是真实 RLHF 中最重要的诊断指标。

### 第 5 步：使用 TRL 的生产方案

理解玩具流水线后，再看真实库用户如何编写同样的循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——阶段 2 使用 `RewardTrainer`，阶段 3 使用内置相对于参考模型 KL 约束的 `PPOTrainer`。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

这个库会替你完成三件事。`adap_kl_ctrl=True` 实现了自适应 β 调度：若观测到的 KL 超过 `target_kl`，β 翻倍；若低于其一半，β 减半。按照惯例，参考模型应当冻结——绝不能意外地让它与 `policy` 共享参数。价值头与策略位于同一个骨干网络上（`AutoModelForCausalLMWithValueHead` 会挂接一个标量 MLP 头），因此 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## 陷阱

- **过度优化/奖励黑客。** RM 并不完美；`π_θ` 会找到得分很高但实际很差的对抗性补全。症状是奖励持续上升，人类评估分数却持平或下降。解决方法是提前停止、增大 `β`、扩充 RM 训练数据。
- **长度黑客。** 在有帮助回答上训练的 RM 往往会隐式奖励篇幅。策略于是学会填充回答。修复方法是使用长度归一化奖励，或采用带长度感知 RM 的 RLAIF。
- **RM 太小。** RM 至少需要与策略一样大。小型 RM 无法忠实评价策略的输出。
- **KL 调节。** β 太低 → 漂移与奖励黑客；β 太高 → 策略几乎不变。标准技巧是使用以固定的逐步 KL 为目标的*自适应* β。
- **偏好数据噪声。** 约 30% 的人类标签存在噪声或歧义。可以只用标注一致的数据训练 RM，或在 BT 中加入温度来校准。
- **离策略问题。** 第一轮之后，PPO 数据就会略微离策略。应按第 08 课所述监控裁剪比例。

## 学以致用

2026 年的 RLHF 分为多个层次：

| 层次 | 目标 | 方法 |
|-------|--------|--------|
| 指令遵循、有帮助、无伤害 | 对齐 | 优先使用 DPO（阶段 10 · 08），而非 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 使用验证器奖励的 GRPO（阶段 9 · 12）。 |
| 长视野多步任务 | 智能体 | PPO / GRPO + 针对各步骤的过程奖励模型。 |
| 安全/拒绝行为 | 安全 | 使用独立安全 RM 的 RLHF-PPO，或宪法式 AI。 |
| 推理时 Best-of-N | 快速对齐 | 解码时使用 RM；无须训练策略。 |
| 奖励蒸馏 | 推理计算 | 在冻结的大语言模型上训练小型“奖励头”。 |

RLHF 曾是 2022～2024 年的*主导*方法。到 2026 年，生产对齐流水线优先采用 DPO，只在高度依赖 RM 或安全关键的步骤中使用 PPO。

## 交付成果

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 在 `code/main.py` 中，用 500 对合成偏好数据训练 Bradley-Terry 奖励模型。在留出的 100 对数据上测量成对准确率，应超过 90%。
2. **中等。** 使用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每种取值绘制 RM 分数与相对于参考模型 KL 随更新次数变化的曲线。哪些运行出现了奖励黑客？
3. **困难。** 在相同偏好数据上实现 DPO（闭式偏好似然损失），并从计算量和最终 RM 分数两方面与 RLHF-PPO 流水线比较。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| RLHF | “对齐强化学习” | 三阶段 SFT + RM + PPO 流水线（Christiano，2017；Ouyang，2022）。 |
| 奖励模型（RM） | “评分网络” | 通过 Bradley-Terry 拟合成对偏好而得到的标量函数。 |
| Bradley-Terry | “成对逻辑损失” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | “靠近参考模型” | 奖励中的 `β · KL(π_θ \|\| π_ref)`；防止奖励黑客的正则化项。 |
| 奖励黑客 | “古德哈特定律” | 策略利用 RM 缺陷；症状是奖励上升，人类评估不变。 |
| RLAIF | “AI 标注的偏好” | 标签来自另一个语言模型而非人类的 RLHF。 |
| PRM | “过程奖励模型” | 为部分推理步骤评分；用于推理流水线。 |
| 宪法式 AI | “Anthropic 的方法” | 由明确规则引导，使用 AI 生成偏好。 |

## 延伸阅读

- [Christiano 等（2017），从人类偏好中进行深度强化学习](https://arxiv.org/abs/1706.03741)——开创 RLHF 的论文。
- [Ouyang 等（2022），InstructGPT——通过人类反馈训练语言模型遵循指令](https://arxiv.org/abs/2203.02155)——ChatGPT 背后的方案。
- [Stiennon 等（2020），通过人类反馈学习摘要](https://arxiv.org/abs/2009.01325)——更早用于摘要的 RLHF。
- [Rafailov 等（2023），直接偏好优化](https://arxiv.org/abs/2305.18290)——DPO；2026 年取代 RLHF 的默认选择。
- [Bai 等（2022），宪法式 AI：通过 AI 反馈实现无害性](https://arxiv.org/abs/2212.08073)——RLAIF 与自我批评循环。
- [Anthropic RLHF 论文（Bai 等，2022），训练有帮助且无害的助手](https://arxiv.org/abs/2204.05862)——HH 论文。
- [Hugging Face TRL 库](https://huggingface.co/docs/trl)——生产级 `RewardTrainer` 与 `PPOTrainer`。可阅读训练器源码，了解自适应 KL 与价值头细节。
- [Hugging Face——图解基于人类反馈的强化学习](https://huggingface.co/blog/rlhf)，作者 Lambert、Castricato、von Werra、Havrilla——带图讲解三阶段流水线的权威文章。
- [von Werra 等（2020），TRL：Transformer 强化学习](https://github.com/huggingface/trl)——该库的 `examples/` 包含用于 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton 与 Barto（2018），第 17.4 章——设计奖励信号](http://incompleteideas.net/book/RLbook2020.pdf)——奖励假说视角；思考奖励黑客问题的必要前置知识。
