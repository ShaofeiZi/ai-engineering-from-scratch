# Actor-Critic——A2C 与 A3C

> REINFORCE 的噪声很大。加入一个学习 `V̂(s)` 的评论家，用回报减去它，就得到期望相同但方差小得多的优势。这就是 Actor-Critic。A2C 以同步方式运行，A3C 则跨线程运行。两者共同构成理解所有现代深度强化学习方法的心智模型。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 9 · 04（TD 学习）、阶段 9 · 06（REINFORCE）
**Time:** 约 75 分钟

## 问题

普通 REINFORCE 可以工作，但方差非常糟糕。蒙特卡洛回报 `G_t` 在不同回合之间可能相差十倍以上。把这份噪声乘以 `∇ log π` 再取平均，会得到一个梯度估计器；它需要数千个回合，才能让策略移动到远少于此数量的 DQN 更新就能到达的位置。

方差源于直接使用原始回报。如果减去基线 `b(s_t)`——包括学习得到的价值在内的任意状态函数——期望不变，方差却会下降。可行的最佳基线是 `V̂(s_t)`。此时，乘以 `∇ log π` 的量就变成了*优势*：

`A(s, a) = G - V̂(s)`

如果一个动作产生了高于平均水平的回报，它就是好动作；低于平均水平则是坏动作。带学习式评论家的 REINFORCE 就是 *Actor-Critic*。评论家为演员提供低方差的教学信号。2015 年之后的每种深度策略方法（A2C、A3C、PPO、SAC、IMPALA）都采用这一思想。

## 概念

![Actor-Critic：策略网络加价值网络，以 TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，一个联合损失：**

- **演员** `π_θ(a | s)`：策略。从中采样动作，并使用策略梯度训练。
- **评论家** `V_φ(s)`：估计从状态出发的期望回报，通过最小化 `(V_φ(s) - target)²` 进行训练。

**优势。** 有两种标准形式：

- *蒙特卡洛优势：* `A_t = G_t - V_φ(s_t)`。无偏，但方差较高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用了 `V_φ`），但方差低得多。它也称为 *TD 残差* `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD，`n = ∞` 是蒙特卡洛。多数实现对 Atari 使用 `n = 5`，对 MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计（GAE）。** Schulman 等人（2016）提出，对所有 n 步优势进行指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差、高偏差），`λ = 1` 是蒙特卡洛（高方差、无偏）。`λ = 0.95` 是 2026 年的默认值——通过调节它，把偏差/方差旋钮设到需要的位置。

**A2C：同步优势 Actor-Critic。** 收集 `T` 步数据，覆盖 `N` 个并行环境，计算每一步的优势，在合并后的批次上更新演员和评论家，然后重复。这是 A3C 更简单、更易扩展的同胞方法。

**A3C：异步优势 Actor-Critic。** Mnih 等人（2016）提出。启动 `N` 个工作线程，每个线程运行一个环境。每个工作线程根据自己的轨迹在本地计算梯度，再异步应用到共享参数服务器。不需要回放缓冲区——各工作线程运行不同轨迹，自然降低了相关性。A3C 证明了可以大规模使用 CPU 进行训练。到 2026 年，基于 GPU 的 A2C（批量并行环境）占据主流，因为 GPU 需要大批次。

**联合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三个组成部分分别是策略梯度损失、价值回归和熵奖励。`c_v ~ 0.5`、`c_e ~ 0.01` 是标准起始值。

```figure
actor-critic
```

## 动手构建

### 第 1 步：评论家

线性评论家 `V_φ(s) = w · features(s)` 使用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境中，评论家经过几百个回合即可收敛。在 Atari 上，应将线性评论家替换为共享 CNN 主干 + 价值头。

### 第 2 步：n 步优势

给定长度为 `T` 的一段轨迹，以及经过自举得到的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是评论家的目标，`advantages` 则是乘以 `∇ log π` 的量。

### 第 3 步：联合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

这是同策略方法：每次更新使用一段轨迹，演员与评论家采用不同的学习率。

### 第 4 步：并行化（A3C 与 A2C）

- **A3C：** 启动 `N` 个线程。每个线程运行自己的环境和前向传播，并定期把梯度更新推送到共享主节点。主节点不加锁——竞态没有关系，只会增加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，把观测堆叠成 `[N, obs_dim]` 批次，执行批量前向传播与批量反向传播。GPU 利用率更高、结果确定，也更容易理解。这是 2026 年的默认选择。

为保持清晰，我们的玩具代码是单线程的；用三行 numpy 即可改写成批处理 A2C。

## 陷阱

- **演员梯度之前的评论家偏差。** 如果评论家是随机的，其基线就没有信息，训练信号仍是纯噪声。应先预热评论家几百步，再启用策略梯度；或者给演员使用较低的学习率。
- **优势归一化。** 在每个批次中把优势归一化为均值 0、标准差 1。这几乎不增加成本，却能大幅稳定训练。
- **共享主干。** 对图像输入，让演员和评论家共用特征提取器，再接各自的输出头。共享特征可以同时受益于两个损失。
- **同策略约束。** A2C 对数据恰好只复用一次。使用更多次会让梯度产生偏差（PPO 增加的重要性采样修正正是为了解决这一点）。
- **熵坍缩。** 如果没有 `c_e > 0`，策略会在几百次更新内变得近乎确定，并停止探索。
- **奖励尺度。** 优势幅度依赖奖励尺度。应归一化奖励（例如除以运行标准差），以便在不同任务间保持一致的梯度幅度。

## 学以致用

A2C/A3C 在 2026 年很少成为最终选择，却是后续所有方法改进的架构基础：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 裁剪的重要性比率，以支持多轮更新 |
| IMPALA | A3C + V-trace 离策略修正 |
| SAC（阶段 9 · 07） | 使用软价值评论家的离策略 A2C（下一课） |
| GRPO（阶段 9 · 12） | 不使用评论家的 A2C——组相对优势 |
| DPO | 把 A2C 压缩成偏好排序损失，无须采样 |
| AlphaStar / OpenAI Five | A2C + 联赛训练 + 模仿预训练 |

如果在 2026 年的论文中看到“优势”，就应想到 Actor-Critic。

## 交付成果

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上使用蒙特卡洛优势（`G_t - V(s_t)`）训练 Actor-Critic。将样本效率与第 06 课使用运行均值基线的 REINFORCE 比较。
2. **中等。** 改用 TD 残差优势（`r + γ V(s') - V(s)`），测量各优势批次的方差。它下降了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`，绘制最终回报与样本效率的关系。这个任务的偏差/方差最佳平衡点在哪里？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 演员 | “策略网络” | `π_θ(a\|s)`，通过策略梯度更新。 |
| 评论家 | “价值网络” | `V_φ(s)`，通过对回报/TD 目标进行 MSE 回归来更新。 |
| 优势 | “比平均水平好多少” | `A(s, a) = Q(s, a) - V(s)` 或其估计量；作为 `∇ log π` 的乘数。 |
| TD 残差 | “δ” | `δ_t = r + γ V(s') - V(s)`；单步优势估计。 |
| GAE | “插值旋钮” | 对 n 步优势进行指数加权求和，由 `λ` 参数化。 |
| A2C | “同步 Actor-Critic” | 跨环境批处理；每段轨迹执行一步梯度更新。 |
| A3C | “异步 Actor-Critic” | 工作线程把梯度推送到共享参数服务器。原始论文所用方法；2026 年已较少使用。 |
| 自举 | “在视野终点使用 V” | 截断轨迹，并加入 `γ^n V(s_{t+n})` 补全总和。 |

## 延伸阅读

- [Mnih 等（2016），深度强化学习的异步方法](https://arxiv.org/abs/1602.01783)——A3C，最初的异步 Actor-Critic 论文。
- [Schulman 等（2016），使用广义优势估计的高维连续控制](https://arxiv.org/abs/1506.02438)——GAE。
- [Sutton 与 Barto（2018），第 13 章——Actor-Critic 方法](http://incompleteideas.net/book/RLbook2020.pdf)——基础知识；评论家采用神经网络时，应结合第 9 章的函数近似内容阅读。
- [Espeholt 等（2018），IMPALA](https://arxiv.org/abs/1802.01561)——采用 V-trace 离策略修正的可扩展分布式 Actor-Critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/)——值得阅读的生产级 A2C/PPO 实现。
- [Konda 与 Tsitsiklis（2000），Actor-Critic 算法](https://papers.nips.cc/paper/1786-actor-critic-algorithms)——双时间尺度 Actor-Critic 分解的奠基性收敛结果。
