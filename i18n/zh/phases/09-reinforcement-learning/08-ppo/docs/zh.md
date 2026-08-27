# 近端策略优化（PPO）

> A2C 每次更新后都会丢弃整段轨迹。PPO 在策略梯度外包上一层经过裁剪的重要性比率，让你可以在同一批数据上训练 10 轮以上，而不会让策略崩溃。Schulman 等人于 2017 年提出了它。到 2026 年，它仍是默认的策略梯度算法。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 9 · 06（REINFORCE）、阶段 9 · 07（Actor-Critic）
**Time:** 约 75 分钟

## 问题

A2C（第 07 课）是同策略方法：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据由*当前* `π_θ` 采样。执行一次更新后，`π_θ` 就变了；刚用过的数据此时已经变成离策略数据。再次使用它，梯度就会产生偏差。

运行轨迹的成本很高。在 Atari 上，8 个环境 × 128 步的一轮轨迹等于 1024 次转移，需要十几秒环境运行时间。只做一步梯度更新便将其丢弃，十分浪费。

信赖域策略优化（TRPO，Schulman，2015）首先解决了这个问题：限制每次更新，使新旧策略之间的 KL 散度不超过 `δ`。理论很清晰，但每次更新都需要求解一次共轭梯度。2026 年已经没有人运行 TRPO。

PPO（Schulman 等，2017）用简单的裁剪目标取代严格的信赖域约束，只多一行代码。每段轨迹可以训练十轮，无须共轭梯度，还有足够好的理论保证。九年后的今天，它仍是从 MuJoCo 到 RLHF 等各种任务默认采用的策略梯度算法。

## 概念

![PPO 裁剪代理目标：在 1 ± ε 处裁剪比率](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于收集数据的旧策略的似然比。`r_t = 1` 表示没有变化；`r_t = 2` 表示新策略采取 `a_t` 的概率是旧策略的两倍。

**裁剪代理目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

它包含两项：

- 如果优势 `A_t > 0`，且比率试图增长到 `1 + ε` 以上，裁剪会把梯度压平——不要让好动作的概率比旧概率高出超过 `+ε`。
- 如果优势 `A_t < 0`，且比率试图越过 `1 - ε`（意味着相对于裁剪后的下降幅度，我们会让坏动作更可能发生），裁剪会限制梯度——不要让坏动作的概率下降超过 `-ε`。

`min` 负责处理另一个方向：如果比率朝*有利*方向移动，仍然保留梯度（会伤害性能的一侧不进行裁剪）。

典型取值是 `ε = 0.2`。画出目标随 `r_t` 变化的函数，可以看到一条分段线性曲线，在“好的一侧”有平坦上沿，在“坏的一侧”有平坦下沿。

**完整 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

它与 A2C 采用相同的 Actor-Critic 结构。三个系数通常取 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 收集 `N × T` 次转移，即在 `N` 个并行环境中各运行 `T` 步。
2. 计算优势（GAE），并把它们冻结为常量。
3. 将 `π_{θ_old}` 冻结为当前 `π_θ` 的快照。
4. 训练 `K` 轮；对于每个 `(s, a, A, V_target, log π_old(a|s))` 小批次：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 执行梯度步骤。
5. 丢弃这段轨迹，返回第 1 步。

`K = 10`、小批次大小为 64，是一组标准超参数。PPO 很稳健：只要精确数值保持在 ±50% 范围内，通常影响不大。

**KL 惩罚变体。** 原论文还提出了另一种使用自适应 KL 惩罚的方案：`L = L^{PG} - β · KL(π_θ || π_old)`，根据观测到的 KL 调整 `β`。裁剪版本后来成为主流；KL 变体则保留在 RLHF 中，因为相对于参考策略的 KL 本来就是始终需要的独立约束。

```figure
ppo-clip
```

## 动手构建

### 第 1 步：运行轨迹时记录 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只在收集轨迹时获取一次，在后续各轮更新中不会变化。

### 第 2 步：计算 GAE 优势（第 07 课）

与 A2C 相同，并在整个批次上归一化。

### 第 3 步：裁剪代理目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“裁剪 → 梯度为零”的模式是 PPO 的核心。如果新策略已经在有利方向上偏离过远，更新就会停止。

### 第 4 步：价值与熵

与 A2C 一样，为评论家的目标加入标准 MSE，并为演员加入熵奖励。

### 第 5 步：诊断指标

每次更新都要观察三项指标：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果超过 `0.1`，应减小 `K_EPOCHS` 或 `LR`。
- **裁剪比例**——比率落在 `[1-ε, 1+ε]` 之外的样本比例。应约为 `~0.1-0.3`。如果约为 `~0`，说明裁剪从未触发 → 增大 `LR` 或 `K_EPOCHS`；如果达到 `~0.5+`，说明过拟合了这段轨迹 → 将它们调低。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。这是评论家质量指标。随着评论家学习，它应逐渐接近 1。

## 陷阱

- **裁剪系数设置不当。** `ε = 0.2` 是事实上的标准。降到 `0.1` 会让更新过于保守；提高到 `0.3+` 则容易导致不稳定。
- **训练轮数过多。** `K > 20` 经常导致不稳定，因为策略会偏离 `π_old` 太远。应限制训练轮数，尤其是对大型网络。
- **没有归一化奖励。** 较大的奖励尺度会侵蚀裁剪区间。在计算优势前，应归一化奖励（使用运行标准差）。
- **忘记归一化优势。** 标准做法是在每个批次内把优势归一化为均值 0、标准差 1。省略这一步会破坏 PPO 在大多数基准上的表现。
- **学习率没有衰减。** PPO 受益于把学习率线性衰减到零；固定学习率往往效果较差。
- **重要性比率计算错误。** 为保证数值稳定，始终使用 `exp(log_new - log_old)`，而不是 `new / old`。
- **梯度符号错误。** 最大化代理目标 = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO 错误。

## 学以致用

到 2026 年，PPO 已成为数量惊人的各类领域中的默认强化学习算法：

| 使用场景 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 使用高斯策略与 GAE(0.95) 的 PPO |
| Atari / 离散游戏 | 使用分类策略与滚动 128 步轨迹的 PPO |
| 大语言模型 RLHF | 相对于参考模型带 KL 惩罚的 PPO，在回答结束时由奖励模型打分 |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理大语言模型 | GRPO（第 12 课）——不使用评论家的 PPO 变体 |
| 只有偏好数据的场景 | DPO——把 PPO+KL 化简为闭式形式，无须在线采样 |

PPO 的*损失结构*——裁剪代理目标 + 价值 + 熵——是 DPO、GRPO 以及几乎每种 RLHF 流水线的脚手架。

## 交付成果

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 使用 `ε=0.2, K=4` 在 4×4 GridWorld 上运行 PPO。在环境步数相同的情况下，与 A2C（每段轨迹训练一轮）的样本效率比较。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`，绘制回报随环境步数变化的曲线，并跟踪每次更新的平均 KL。在这个任务上，KL 从哪个 `K` 开始爆炸？
3. **困难。** 用自适应 KL 惩罚替换裁剪代理目标（自适应系数 `β`：若 `KL > 2·target` 则翻倍，若 `KL < target/2` 则减半）。比较最终回报、稳定性以及不使用裁剪的效果。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 重要性比率 | “r_t(θ)” | `π_θ(a\|s) / π_old(a\|s)`；相对于收集数据时策略的偏离程度。 |
| 裁剪代理目标 | “PPO 的主要技巧” | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利一侧越过裁剪点后，梯度变平。 |
| 信赖域 | “TRPO / PPO 的意图” | 限制每次更新的 KL，以保证单调改进。 |
| KL 惩罚 | “软信赖域” | PPO 的另一版本：`L - β · KL(π_θ \|\| π_old)`，自适应调整 `β`。 |
| 裁剪比例 | “裁剪触发频率” | 诊断指标——应为 0.1～0.3；超出范围表示参数设置不当。 |
| 多轮训练 | “数据复用” | 每段轨迹训练 K 轮；以方差为代价换取样本效率。 |
| 近似同策略 | “基本属于同策略” | PPO 名义上是同策略算法，但 K>1 轮训练会安全地使用略微离策略的数据。 |
| PPO-KL | “另一种 PPO” | KL 惩罚变体；用于本就需要相对于参考策略施加 KL 约束的 RLHF。 |

## 延伸阅读

- [Schulman 等（2017），近端策略优化算法](https://arxiv.org/abs/1707.06347)——原始论文。
- [Schulman 等（2015），信赖域策略优化](https://arxiv.org/abs/1502.05477)——TRPO，PPO 的前身。
- [Andrychowicz 等（2021），同策略强化学习中什么最重要？一项大规模实证研究](https://arxiv.org/abs/2006.05990)——对每项 PPO 超参数进行消融。
- [Ouyang 等（2022），通过人类反馈训练语言模型遵循指令](https://arxiv.org/abs/2203.02155)——InstructGPT；RLHF 中的 PPO 方案。
- [OpenAI Spinning Up——PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)——带 PyTorch 的清晰现代讲解。
- [CleanRL PPO 实现](https://github.com/vwxyzjn/cleanrl)——许多论文采用的参考单文件 PPO。
- [Hugging Face TRL——PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer)——在语言模型上运行 PPO 的生产方案；适合与第 09 课（RLHF）一同阅读。
- [Engstrom 等（2020），深度策略梯度中的实现细节至关重要](https://arxiv.org/abs/2005.12729)——“37 项代码级优化”论文；分析哪些 PPO 技巧不可或缺，哪些只是传闻。
