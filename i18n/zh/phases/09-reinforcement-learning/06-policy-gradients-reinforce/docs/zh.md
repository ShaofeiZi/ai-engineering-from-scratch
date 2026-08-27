# 策略梯度——从零实现 REINFORCE

> 不再估计价值。直接参数化策略，计算期望回报的梯度，再沿上升方向迈步。Williams（1992）用一个定理写清了这一切。PPO、GRPO 以及每一种大语言模型强化学习循环，都源于此。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 3 · 03（反向传播）、阶段 9 · 03（蒙特卡洛）、阶段 9 · 04（TD 学习）
**Time:** 约 75 分钟

## 问题

Q-learning 与 DQN 参数化的是*价值*函数，并通过 `argmax Q` 选择动作。这对离散动作和离散状态没有问题，但当动作连续时（如何对十维力矩取 `argmax`？），或者你需要随机策略时（`argmax` 天生就是确定性的），这种方法就会失效。

策略梯度改为直接参数化*策略*。`π_θ(a | s)` 是一个输出动作分布的神经网络。执行时从中采样动作，计算期望回报相对于 `θ` 的梯度，再沿上升方向更新。不需要 `argmax`，不需要贝尔曼递归，只需对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams，1992）告诉我们，这个梯度可以计算：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合，计算回报，在每一步乘以 `∇ log π_θ(a | s)`，取平均值，再执行梯度上升，就完成了。

2026 年的每一种大语言模型强化学习算法——PPO、DPO、GRPO——都是 REINFORCE 的改进。熟练掌握它，是学习本阶段其余内容，以及阶段 10 · 07（RLHF 实现）和阶段 10 · 08（DPO）的前提。

## 概念

![策略梯度：Softmax 策略、log-π 梯度、回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任意策略 `π_θ`，若它由 `θ` 参数化：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步骤 `t` 开始的折扣回报。期望取自完整轨迹 `τ`，它从 `π_θ` 中采样得到。

**证明很短。** 在期望内部对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。利用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧）。再分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`。环境项消失。两行代数即可得到该定理。

**降低方差的技巧。** 普通 REINFORCE 的方差大得惊人——回报有噪声，`∇ log π` 有噪声，两者相乘后的噪声更大。有两种标准修复方法：

1. **减去基线。** 将 `G_t` 替换为 `G_t - b(s_t)`，其中基线 `b(s_t)` 可以是任何不依赖 `a_t` 的函数。它仍然无偏，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择是令 `b(s_t) = V̂(s_t)`，由评论家学习——这就得到 Actor-Critic（第 07 课）。
2. **从当前时刻起的回报。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。对于某个动作，只有未来回报才重要；过去的奖励只会贡献均值为零的噪声。

两者结合后得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（第 07 课）与 PPO（第 08 课）的直接祖先。

**Softmax 策略参数化。** 对离散动作而言，标准选择是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 可以是任何为每个动作输出一个分数的神经网络。其梯度具有简洁形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

也就是已选动作的分数减去该分数在策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式形式。阶段 9 · 07 的 SAC 所需内容仅此而已。

```figure
policy-gradient-landscape
```

## 动手构建

### 第 1 步：Softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对表格环境使用线性策略（每个动作对应一个权重向量）。若处理 Atari，只需换成 CNN，并保留 Softmax 输出头。

### 第 2 步：采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第 3 步：运行轨迹并记录对数概率

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第 4 步：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的独热向量减去概率）是 Softmax 策略梯度的核心。要把它练成肌肉记忆。

### 第 5 步：基线

对近期回合中的 `G` 使用运行均值，就足以降低方差，让 4×4 GridWorld 跑起来；大约 500 个回合即可收敛。把基线升级为学习得到的 `V̂(s)`，就得到 Actor-Critic。

## 陷阱

- **梯度爆炸。** 回报可能很大。始终应在批次内把 `G` 归一化为近似 `~N(0, 1)`，再乘以 `∇ log π`。
- **熵坍缩。** 策略过早收敛到近乎确定性的动作，于是停止探索并陷入局部。解决方法是在目标中加入熵奖励 `β · H(π(·|s))`。
- **高方差。** 普通 REINFORCE 需要数千个回合。标准修复方法是使用评论家基线（第 07 课）或 TRPO/PPO 的信赖域（第 08 课）。
- **样本效率低。** 同策略意味着每条转移在一次更新后就被丢弃。通过重要性采样进行离策略修正可以重新利用数据，但代价是方差增加（PPO 的比率就是经过裁剪的重要性采样权重）。
- **非平稳梯度。** 100 个回合前的同一个梯度使用的是旧 `π`。因此，同策略方法每收集少量轨迹就会更新一次。
- **信用分配。** 如果不使用从当前时刻起的回报，过去的奖励就会贡献噪声。始终使用从当前时刻起的回报。

## 学以致用

到 2026 年，REINFORCE 已很少直接运行，但它的梯度公式无处不在：

| 使用场景 | 衍生方法 |
|----------|---------------|
| 连续控制 | 使用高斯策略的 PPO / SAC |
| 大语言模型 RLHF | 在词元级策略上运行、带 KL 惩罚的 PPO |
| 大语言模型推理（DeepSeek） | GRPO——使用组相对基线、没有评论家的 REINFORCE |
| 多智能体 | 集中式评论家 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 只有偏好数据的场景 | DPO——把 REINFORCE 改写为偏好似然损失，无须采样 |

当你在 2026 年的训练脚本中看到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇论文所提出的方法（DPO、GRPO、RLOO），本质上都是叠加在这一行之上的方差缩减技巧。

## 交付成果

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 使用线性 Softmax 策略，在 4×4 GridWorld 上实现 REINFORCE。不使用基线训练 1000 个回合。绘制学习曲线并测量方差（回报的标准差）。
2. **中等。** 加入运行均值基线并重新训练。比较它与普通版本的样本效率和方差。基线使达到收敛所需的步数减少了多少？
3. **困难。** 加入熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`，绘制最终回报与策略熵。在这个任务上，最佳平衡点在哪里？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 策略梯度 | “直接训练策略” | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；由对数导数技巧推导。 |
| REINFORCE | “最初的策略梯度算法” | Williams（1992）；蒙特卡洛回报乘以对数策略梯度。 |
| 对数导数技巧 | “得分函数估计器” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；让期望的梯度变得易于计算。 |
| 基线 | “降低方差” | 任意 `b(s)` 都可从 `G` 中减去；它仍然无偏，因为 `E[b · ∇ log π] = 0`。 |
| 从当前时刻起的回报 | “只计算未来回报” | 使用 `G_t^{from t}`，而不是完整的 `G_0`；结果正确且方差更低。 |
| 熵奖励 | “鼓励探索” | `+β · H(π(·\|s))` 项可以防止策略坍缩。 |
| 同策略 | “用刚看到的数据训练” | 梯度期望针对当前策略计算——不能直接重复使用旧数据。 |
| 优势 | “比平均水平好多少” | `A(s, a) = G(s, a) - V(s)`；带基线 REINFORCE 所乘的有符号量。 |

## 延伸阅读

- [Williams（1992），用于连接主义强化学习的简单统计梯度跟随算法](https://link.springer.com/article/10.1007/BF00992696)——REINFORCE 原始论文。
- [Sutton 等（2000），使用函数近似的强化学习策略梯度方法](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)——结合函数近似的现代策略梯度定理。
- [Sutton 与 Barto（2018），第 13 章——策略梯度方法](http://incompleteideas.net/book/RLbook2020.pdf)——教材讲解。
- [OpenAI Spinning Up——VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html)——带 PyTorch 代码的清晰教学说明。
- [Peters 与 Schaal（2008），使用策略梯度进行运动技能强化学习](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf)——方差缩减与自然梯度视角，将 REINFORCE 与信赖域方法家族（TRPO、PPO）联系起来。
