# 时序差分——Q-learning 与 SARSA

> 蒙特卡洛方法要等到回合结束。时序差分则利用下一个价值估计进行自举，在每一步之后立即更新。Q-learning 是离策略且乐观的，SARSA 是同策略且谨慎的。两者都只需一行代码，也都支撑着本阶段的每一种深度强化学习方法。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 9 · 01（MDP）、阶段 9 · 02（动态规划）、阶段 9 · 03（蒙特卡洛）
**Time:** 约 75 分钟

## 问题

蒙特卡洛方法可以工作，但有两个代价高昂的要求：回合必须终止，而且只能在获得最终回报后更新。如果一个回合有 1000 步，蒙特卡洛就要等 1000 步才能进行任何更新。它方差高、偏差低，实践中速度很慢。

动态规划的特性恰好相反——通过自举得到的备份方差为零——却要求模型已知。

时序差分（TD）学习折中两者。只需一次转移 `(s, a, r, s')`，就能构造单步目标 `r + γ V(s')`，并推动 `V(s)` 向它靠近。不需要模型，也不需要完整回合。等式右侧使用近似的 `V` 会引入偏差，但方差远低于蒙特卡洛，而且从第一步起就能在线更新。

这正是现代强化学习——DQN、A2C、PPO、SAC——赖以转动的支点。阶段 9 后续内容，都是建立在本课将要编写的单步 TD 更新之上的函数近似层和各种技巧。

## 概念

![Q-learning 与 SARSA：离策略最大值和同策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

方括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`，相当于蒙特卡洛中 `G_t - V(s_t)` 的在线版本。收敛要求 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`、`Σ α² < ∞`），并且所有状态都被无限次访问。

**Q-learning。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

无论智能体实际采取什么动作，`max` 都假设从 `s'` 开始会遵循*贪心*策略。正是这种解耦，让智能体可以通过 ε-贪心进行探索，同时让 Q-learning 学习 `Q*`。Mnih 等人（2015）将其发展为 Atari 上的深度 Q-learning（第 05 课）。

**SARSA。** 一种同策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

它的名称来自五元组 `(s, a, r, s', a')`。SARSA 使用智能体接下来*实际*采取的动作 `a'`，而不是贪心 `argmax`。它会收敛到 `Q^π`，对应当前运行的任意 ε-贪心策略 `π`；当 `ε → 0` 时，极限就是 `Q*`。

**悬崖行走中的差异。** 在经典悬崖行走任务中（跌落悬崖的奖励为 -100），Q-learning 会学会沿悬崖边缘行走的最优路径，却会在探索时偶尔遭受惩罚。SARSA 会选择离悬崖多一格的安全路径，因为它把探索噪声纳入 Q 值。随着训练推进，当 `ε → 0` 时，两者都会达到最优。实践中这一区别十分重要：如果部署时仍会进行探索，SARSA 的行为会更加保守。

**Expected SARSA。** 用 `Q(s', a')` 在 `π` 下的期望值替代它：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

它比 SARSA 方差更低（不需要采样 `a'`），使用的仍是同策略目标。现代教材经常把它作为默认方法。

**n 步 TD 与 TD(λ)。** 在自举前等待 `n` 步，从而在 TD(0) 与蒙特卡洛之间插值。`n=1` 就是 TD，`n=∞` 就是蒙特卡洛。TD(λ) 对所有 `n` 使用几何权重 `(1-λ)λ^{n-1}` 求平均。大多数深度强化学习方法使用 3 到 20 之间的 `n`。

```figure
qlearning-gridworld
```

## 动手构建

### 第 1 步：在 ε-贪心策略上运行 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

只有八行。与 Q-learning 的*唯一区别*就在目标值那一行。

### 第 2 步：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。就是这一个符号，区分了同策略与离策略。

### 第 3 步：学习曲线

跟踪每 100 个回合的平均回报。在简单的确定性 GridWorld 上，Q-learning 收敛得更快；在悬崖行走中，SARSA 更为保守。对于 `code/main.py` 中的 4×4 GridWorld，使用 `α=0.1, ε=0.1` 时，两者都能在约 2000 个回合后接近最优。

### 第 4 步：与动态规划真值比较

运行价值迭代（第 02 课）得到 `Q*`，检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。在 4×4 GridWorld 上训练 10,000 个回合后，一个健康的表格型 TD 智能体会落在 `~0.5` 以内。

## 陷阱

- **初始 Q 值很重要。** 对负奖励任务采用乐观初始化（`Q = 0`）会鼓励探索；悲观初始化可能让贪心策略永远陷在局部。
- **α 调度。** 固定 `α` 适用于非平稳问题。递减的 `α_n = 1/n` 在理论上可以收敛，但实践中太慢——将 `α` 固定在 `[0.05, 0.3]`，并监控学习曲线。
- **ε 调度。** 从较高取值（`ε=1.0`）开始，逐渐衰减到 `ε=0.05`。“GLIE”（在无限探索条件下极限贪心）是收敛条件。
- **Q-learning 的最大化偏差。** `max` 算子作用于带噪的 `Q` 时会产生向上偏差，导致过高估计。Hasselt 的 Double Q-learning（第 05 课中的 DDQN 会使用）以两张 Q 表修复这一问题。
- **不终止的回合。** TD 无须终止状态也能学习，但必须限制步数，或在达到上限时正确处理自举。标准做法是把上限视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，应使用可哈希键（用元组而非列表；使用经过取整的浮点数元组，而不是原始浮点数）。

## 学以致用

2026 年的 TD 格局：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 安全关键的同策略任务 | SARSA / Expected SARSA | 探索时更加保守。 |
| 高维状态 | DQN（阶段 9 · 05） | 使用经验回放与目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（阶段 9 · 07） | 在 Q 网络上执行 TD 更新；策略网络输出动作。 |
| 大语言模型强化学习（基于奖励模型） | PPO / GRPO（阶段 9 · 08、12） | Actor-Critic，通过 GAE 计算 TD 风格优势。 |
| 离线强化学习 | CQL / IQL（阶段 9 · 08） | 带保守正则化的 Q-learning。 |

你在 2026 年论文中读到的“强化学习”，有九成都是 Q-learning 或 SARSA 的某种扩展。在继续深入之前，应让自己熟练掌握表格更新。

## 交付成果

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 与 SARSA。绘制 2000 个回合的学习曲线（每 100 个回合的平均回报）。哪一种收敛更快？
2. **中等。** 构建悬崖行走环境（4×12，最后一行是奖励为 -100 的悬崖，跌落后重置到起点）。比较 Q-learning 与 SARSA 的最终策略，对各自路径截图。哪一条更靠近悬崖？
3. **困难。** 实现 Double Q-learning。在带噪奖励的 GridWorld 中（每步奖励加入高斯噪声 σ=5），证明 Q-learning 会显著高估 `V*(0,0)`，而 Double Q-learning 不会。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| TD 误差 | “更新信号” | `δ = r + γ V(s') - V(s)`，即自举残差。 |
| TD(0) | “单步 TD” | 每次转移后立刻更新，只使用下一状态的估计。 |
| Q-learning | “离策略强化学习入门” | 使用下一状态动作的 `max` 执行 TD 更新；无论行为策略如何，都学习 `Q*`。 |
| SARSA | “同策略 Q-learning” | 使用实际下一动作的 TD 更新；为当前 ε-贪心策略 π 学习 `Q^π`。 |
| Expected SARSA | “低方差 SARSA” | 用 π 下的期望替换采样得到的 `a'`。 |
| GLIE | “正确的探索调度” | 在无限探索条件下极限贪心；Q-learning 收敛所必需。 |
| 自举 | “在目标中使用当前估计” | TD 与蒙特卡洛的区别；会引入偏差，却能大幅降低方差。 |
| 最大化偏差 | “Q-learning 高估” | 对带噪估计取 `max` 会产生向上偏差；Double Q-learning 可以修复。 |

## 延伸阅读

- [Watkins 与 Dayan（1992），Q-learning](https://link.springer.com/article/10.1007/BF00992698)——原始论文与收敛性证明。
- [Sutton 与 Barto（2018），第 6 章——时序差分学习](http://incompleteideas.net/book/RLbook2020.pdf)——TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt（2010），Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)——最大化偏差的修复方法。
- [Seijen、Hasselt、Whiteson、Wiering（2009），Expected SARSA 的理论与实证分析](https://ieeexplore.ieee.org/document/4927542)——Expected SARSA 的动机。
- [Rummery 与 Niranjan（1994），使用连接主义系统进行在线 Q-learning](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems)——提出 SARSA 这一名称的论文（当时称为“修正连接主义 Q-learning”）。
- [Sutton 与 Barto（2018），第 7 章——n 步自举](http://incompleteideas.net/book/RLbook2020.pdf)——将 TD(0) 推广至 TD(n)，铺就了从 Q-learning 到资格迹、再到后来 PPO 中 GAE 的道路。
