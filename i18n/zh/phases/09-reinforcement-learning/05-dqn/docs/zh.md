# 深度 Q 网络（DQN）

> 2013 年，Mnih 用原始像素训练了一个 Q-learning 网络，在 7 款 Atari 游戏上击败了所有经典强化学习智能体。2015 年，这项工作扩展到 49 款游戏、发表于《Nature》，并引爆了深度强化学习时代。DQN 就是 Q-learning 加上三项让函数近似保持稳定的技巧。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 3 · 03（反向传播）、阶段 9 · 04（Q-learning、SARSA）
**Time:** 约 75 分钟

## 问题

表格型 Q-learning 需要为每个（状态、动作）对保存一个独立 Q 值。国际象棋棋盘约有 10⁴³ 个状态，一帧 Atari 画面包含 210×160×3 = 100,800 个特征。表格型强化学习面对数千个状态就会失效，更不用说数十亿个。

事后看来，解决办法显而易见：用神经网络 `Q(s, a; θ)` 替换 Q 表。然而，这个“事后看来显然”的想法耗费了数十年才真正可用。Q-learning 直接结合函数近似时，会因“致命三角”而发散——函数近似 + 自举 + 离策略学习。Mnih 等人（2013、2015）找到了三项稳定训练的工程技巧：

1. **经验回放**打破转移之间的相关性。
2. **目标网络**冻结自举目标。
3. **奖励裁剪**归一化梯度幅度。

Atari DQN 首次用同一套架构和同一组超参数，从原始像素出发解决了数十个控制问题。此后所有“深度强化学习”方法——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都叠加在这三项技巧构成的底座之上。

## 概念

![DQN 训练循环：环境、回放缓冲区、在线网络、目标网络、贝尔曼 TD 损失](../../../../../../phases/09-reinforcement-learning/05-dqn/assets/dqn.svg)

**目标函数。** DQN 在神经 Q 函数上最小化单步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` 是在线网络，每一步都通过梯度下降更新。`θ^-` 是目标网络，每隔一段时间从 `θ` 复制一次（约每 10,000 步）。`D` 是存储历史转移的回放缓冲区。

**三项技巧，按重要性排序：**

**经验回放。** 使用容量约为 `~10⁶` 条转移的环形缓冲区。每个训练步骤从中均匀随机采样一个小批次。这样可以打破时间相关性（相邻帧几乎完全相同），让网络多次学习罕见的高奖励转移，并降低连续梯度更新之间的相关性。没有经验回放，在 Atari 上使用神经网络的同策略 TD 会发散。

**目标网络。** 在贝尔曼方程两侧使用同一个网络 `Q(·; θ)`，会让目标每次更新都移动——就像“追着自己的尾巴跑”。解决方法是保留第二个权重冻结的网络 `Q(·; θ^-)`。每隔 `C` 步，把 `θ → θ^-` 复制一次。这样，回归目标可以在数千个梯度步骤中保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是更平滑的变体。

**奖励裁剪。** 不同 Atari 游戏的奖励幅度从 1 到 1000 以上不等。把奖励裁剪到 `{-1, 0, +1}`，可以防止某一款游戏主导梯度。当奖励大小本身有意义时，这种做法并不正确；但对于只关心符号的 Atari 而言没有问题。

**Double DQN。** Hasselt（2016）修复了最大化偏差：由在线网络*选择*动作，再由目标网络*评估*该动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

它可以直接替换原目标，且效果始终更好。默认就应使用它。

**其他改进（Rainbow，2017）：** 优先经验回放（更频繁地采样 TD 误差较大的转移）、Dueling 架构（分离 `V(s)` 与优势头）、噪声网络（学习得到的探索）、n 步回报、分布式 Q（C51/QR-DQN）、多步自举。每项改进都能带来几个百分点的提升，而且收益大致可以叠加。

```figure
f3-dqn-stability
```

## 动手构建

本课代码只使用标准库、不使用 numpy——我们在一个微型连续 GridWorld 上手写单隐藏层 MLP，因此每个训练步骤只需数微秒。算法与扩展到 Atari 规模的 DQN 完全相同。

### 第 1 步：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 使用约 50,000 的容量；我们的玩具环境使用 5,000 就足够。

### 第 2 步：微型 Q 网络（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播是：线性 → ReLU → 线性。这就是整个网络。

### 第 3 步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

它的形状就是第 04 课的 Q-learning，只有两处区别：（a）我们对可微的 `Q(·; θ)` 进行反向传播，而不是索引一张表；（b）目标使用 `Q(·; θ^-)`。

### 第 4 步：外层循环

在每个回合中，针对 `Q(·; θ)` 采取 ε-贪心动作，把转移推入缓冲区，采样一个小批次并执行一步梯度更新，同时定期同步 `θ^- ← θ`。其模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在使用 16 维独热状态的微型 GridWorld 上，智能体约 500 个回合就能学到接近最优的策略。在 Atari 上，需要将训练扩展至 2 亿帧，并加入 CNN 特征提取器。

## 陷阱

- **致命三角。** 函数近似 + 离策略 + 自举可能发散。DQN 用目标网络 + 回放来缓解；两者都不能移除。
- **探索。** ε 必须衰减，通常在训练最初约 10% 的阶段中从 1.0 降到 0.01。早期探索不足会让 Q 网络收敛到局部盆地。
- **高估。** 对带噪 Q 值取 `max` 会产生向上偏差。生产环境中始终使用 Double DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成正比。
- **回放缓冲区冷启动。** 在缓冲区积累数千条转移前不要训练。用约 20 个样本计算的早期梯度会过拟合。
- **目标同步频率。** 过于频繁 ≈ 没有目标网络；过于稀疏 ≈ 目标陈旧。Atari DQN 每 10,000 个环境步骤同步一次。经验法则是每经过约 1/100 的训练视野同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧，使状态满足马尔可夫性质。任何需要速度信息的环境都必须堆叠帧或使用循环状态。

## 学以致用

到 2026 年，DQN 已经很少处于最佳水平，但仍是离策略算法的参照：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|------------------|--------------|
| 类 Atari 的离散动作任务 | Rainbow DQN 或 Muesli | 框架相同，技巧更多。 |
| 连续控制 | SAC / TD3（阶段 9 · 07） | DQN 没有策略网络。 |
| 同策略/高吞吐量 | PPO（阶段 9 · 08） | 没有回放缓冲区，更容易扩展。 |
| 离线强化学习 | CQL / IQL / Decision Transformer | 使用保守 Q 目标，没有自举爆炸。 |
| 大型离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | DQN 可以胜任；具体改进很重要。 |
| 大语言模型强化学习 | PPO / GRPO | 序列级而非步骤级；损失不同。 |

这些经验至今仍然通用。经验回放与目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 自我对弈缓冲区和每种离线强化学习方法中。奖励裁剪则以 PPO 中优势归一化的形式延续下来。这套架构就是蓝图。

## 交付成果

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`，绘制逐回合回报曲线。运行均值需要多少个回合才能超过 -10？
2. **中等。** 禁用目标网络（在贝尔曼目标两侧都使用在线网络）。测量训练的不稳定性——回报会振荡还是发散？
3. **困难。** 加入 Double DQN：使用在线网络选择 `argmax a'`，再用目标网络评估。对带噪奖励的 GridWorld 训练 1000 个回合，比较使用和不使用 Double DQN 时，`Q(s_0, best_a)` 相对于真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| DQN | “深度 Q-learning” | 使用神经 Q 函数、回放缓冲区和目标网络的 Q-learning。 |
| 经验回放 | “打乱的转移” | 每个梯度步骤都从环形缓冲区均匀采样；降低数据相关性。 |
| 目标网络 | “冻结的自举” | 在贝尔曼目标中使用的 Q 网络副本，定期更新；用于稳定训练。 |
| 致命三角 | “强化学习为何发散” | 函数近似 + 自举 + 离策略 = 没有收敛保证。 |
| Double DQN | “最大化偏差修复” | 在线网络选择动作，目标网络评估动作。 |
| Dueling DQN | “V 头与 A 头” | 将 Q 分解为 V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | “所有技巧的集合” | 把 DDQN + PER + Dueling + n 步 + 噪声网络 + 分布式方法合为一体。 |
| PER | “优先经验回放” | 按 TD 误差的大小成比例采样转移。 |

## 延伸阅读

- [Mnih 等（2013），使用深度强化学习玩 Atari](https://arxiv.org/abs/1312.5602)——开启深度强化学习浪潮的 2013 年 NeurIPS 研讨会论文。
- [Mnih 等（2015），通过深度强化学习实现人类水平的控制](https://www.nature.com/articles/nature14236)——发表于《Nature》的 49 款游戏 DQN 论文。
- [Hasselt、Guez、Silver（2016），使用 Double Q-learning 的深度强化学习](https://arxiv.org/abs/1509.06461)——DDQN。
- [Wang 等（2016），Dueling 网络架构](https://arxiv.org/abs/1511.06581)——Dueling DQN。
- [Hessel 等（2018），Rainbow：组合深度强化学习的多项改进](https://arxiv.org/abs/1710.02298)——叠加多种技巧的论文。
- [OpenAI Spinning Up——DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)——清晰的现代讲解。
- [Sutton 与 Barto（2018），第 9 章——使用近似的同策略预测](http://incompleteideas.net/book/RLbook2020.pdf)——教材对“致命三角”（函数近似 + 自举 + 离策略）的讲解；DQN 的目标网络与回放缓冲区正是为了驯服它。
- [CleanRL DQN 实现](https://docs.cleanrl.dev/rl-algorithms/dqn/)——用于消融研究的参考单文件 DQN；适合与本课从零实现的版本对照阅读。
