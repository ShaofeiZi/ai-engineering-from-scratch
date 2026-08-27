# 蒙特卡洛方法——从完整回合中学习

> 动态规划需要模型，蒙特卡洛方法只需要回合。运行策略、观察回报、求取平均值。这是强化学习中最简单的思想，也是开启后续所有方法的钥匙。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 9 · 01（MDP）、阶段 9 · 02（动态规划）
**Time:** 约 75 分钟

## 问题

动态规划很优雅，但它假设你能对每个状态和动作查询 `P(s' | s, a)`。现实世界中几乎没有什么问题符合这一条件。机器人无法通过解析计算得到施加关节力矩后相机像素的分布；定价算法无法对客户所有可能的反应做积分；大语言模型也无法枚举一个词元之后的所有可能续写。

你需要一种只要求能从环境中*采样*的方法。运行策略，获得一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`，再用它估计价值。这就是蒙特卡洛方法。

从动态规划转向蒙特卡洛，在理念上十分重要：我们从*已知模型 + 精确备份*转向*采样轨迹 + 平均回报*。方差大幅增加，适用范围却也随之扩展。本课之后的每一种强化学习算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，只是有时会在其上叠加自举。

## 概念

![蒙特卡洛：运行轨迹、计算回报、取平均值；首次访问与每次访问](../assets/monte-carlo.svg)

**核心思想可以写成一行：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是访问 `s` 后、遵循策略 `π` 时观察到的回报。

**首次访问与每次访问蒙特卡洛。** 如果一个回合多次访问状态 `s`，首次访问蒙特卡洛只统计第一次访问后的回报；每次访问蒙特卡洛则统计所有访问。两者在极限情况下都是无偏的。首次访问更容易分析（样本独立同分布），每次访问则在每个回合中利用更多数据，实践中通常收敛更快。

**增量均值。** 无须存储所有回报，只需更新运行均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

将其改写为 `V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成固定步长 `α ∈ (0, 1)`，就得到一个能跟踪 `π` 变化的非平稳蒙特卡洛估计器。从蒙特卡洛跨越到 TD 乃至每种现代强化学习算法，关键就在这一步。

**探索现在成了问题。** 动态规划通过枚举接触每个状态，蒙特卡洛却只能看到策略实际访问的状态。如果 `π` 是确定性的，状态空间中的大片区域永远不会被采样，其价值估计也会永远停留在零。按历史顺序有三种解决办法：

1. **探索式起点。** 让每个回合从随机的 (s, a) 对开始。这样可以保证覆盖，却不符合实际情况（你无法把机器人“重置”到任意状态）。
2. **ε-贪心。** 相对于当前 Q 采取贪心动作，但以概率 `ε` 随机选择动作。渐近情况下，所有状态-动作对都会被采样。
3. **离策略蒙特卡洛。** 在行为策略 `μ` 下收集数据，再通过重要性采样学习目标策略 `π`。方差很高，但这是通往 DQN 等经验回放方法的桥梁。

**蒙特卡洛控制。** 与策略迭代一样，执行评估 → 改进 → 评估，只是评估以采样为基础：

1. 运行 `π`，获得一个回合。
2. 根据观察到的回报更新 `Q(s, a)`。
3. 令 `π` 成为相对于 `Q` 的 ε-贪心策略。
4. 重复。

它以概率 1 收敛到 `Q*` 与 `π*`，前提是满足一些温和条件：每个状态-动作对被无限次访问，且 `α` 满足 Robbins-Monro 条件。

```figure
epsilon-greedy
```

## 动手构建

### 第 1 步：运行轨迹 → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

不需要模型，只需要 `env.reset()` 和 `env.step(s, a)`。接口与 gym 环境相同，只是去掉了多余部分。

### 第 2 步：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

只需一次遍历，复杂度为 `O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免了重复求和。

### 第 3 步：首次访问蒙特卡洛评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

真正完成工作的只有三行：首次访问时把状态标记为已见，增加计数，再更新运行均值。

### 第 4 步：ε-贪心蒙特卡洛控制（同策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第 5 步：与动态规划金标准比较

随着回合数 → ∞，蒙特卡洛对 `V^π` 的估计应当与第 02 课的动态规划结果一致。实践中，在 4×4 GridWorld 上运行 50,000 个回合，就能把误差缩小到动态规划答案的 `~0.1` 以内。

## 陷阱

- **无限回合。** 蒙特卡洛要求回合必须*终止*。如果策略可能永远循环，就设置 `max_steps` 上限，并把触及上限视为隐式失败。GridWorld 中的随机策略经常超时，这是正常现象，只需确保统计方式正确。
- **方差。** 蒙特卡洛使用完整回报。长回合上的方差极大——末尾一次不走运的奖励，会同等幅度地改变 `V(s_0)`。TD 方法（第 04 课）通过自举降低方差。
- **状态覆盖。** 在初始 Q 值相同的情况下使用贪心蒙特卡洛，永远只会尝试一个动作。*必须*进行探索（ε-贪心、探索式起点、UCB）。
- **非平稳策略。** 如果 `π` 会变化（如蒙特卡洛控制），旧回报就来自不同的策略。固定 α 蒙特卡洛可以应对这种情况，样本平均蒙特卡洛则不能。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 会沿轨迹连乘，方差随视野长度爆炸。可以采用逐决策加权重要性采样设置上限，或改用 TD。

## 学以致用

蒙特卡洛方法在 2026 年的角色：

| 使用场景 | 为什么使用蒙特卡洛 |
|----------|--------|
| 短视野游戏（21 点、扑克） | 回合自然终止；回报清晰。 |
| 对日志策略进行离线评估 | 对存储轨迹上的折扣回报取平均值。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶开始的蒙特卡洛轨迹引导节点选择。 |
| 大语言模型强化学习评估 | 计算给定策略下采样补全的平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用蒙特卡洛 `G_t`。 |
| 强化学习教学 | 最简单而且真正有效的算法——去掉自举即可看到核心。 |

现代深度强化学习算法（PPO、SAC）通过 `n` 步回报或 GAE，在纯蒙特卡洛（完整回报）与纯 TD（单步自举）之间插值。两个端点都是同一种估计器的实例。

## 交付成果

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现均匀随机策略的首次访问蒙特卡洛评估。运行 10,000 个回合，绘制 `V(0,0)` 随回合数变化的曲线，并与动态规划答案比较。
2. **中等。** 使用 `ε ∈ {0.01, 0.1, 0.3}` 实现 ε-贪心蒙特卡洛控制。比较 20,000 个回合后的平均回报。曲线呈现什么形状？偏差-方差权衡体现在哪里？
3. **困难。** 使用重要性采样实现*离策略*蒙特卡洛：在均匀随机策略 `μ` 下收集数据，估计 `V^π`，其中 `π` 是确定性最优策略。比较普通重要性采样、逐决策重要性采样和加权重要性采样。哪一种方差最低？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 蒙特卡洛 | “随机采样” | 对来自分布的独立同分布样本求平均，以估计期望。 |
| 回报 `G_t` | “未来奖励” | 从步骤 `t` 到回合结束的折扣奖励之和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问蒙特卡洛 | “每个状态只统计一次” | 一个回合中只有第一次访问会贡献给价值估计。 |
| 每次访问蒙特卡洛 | “使用所有访问” | 每次访问都会贡献；略有偏差，但样本效率更高。 |
| ε-贪心 | “探索噪声” | 以概率 `1-ε` 选择贪心动作，以概率 `ε` 选择随机动作。 |
| 重要性采样 | “纠正从错误分布采样” | 用 `π(a\|s)/μ(a\|s)` 的连乘权重重新加权回报，以估计 `V^π`；数据来自 `μ`。 |
| 同策略 | “从自己的数据中学习” | 目标策略 = 行为策略。普通蒙特卡洛、PPO、SARSA 均属此类。 |
| 离策略 | “从其他策略的数据中学习” | 目标策略 ≠ 行为策略。重要性采样蒙特卡洛、Q-learning、DQN 均属此类。 |

## 延伸阅读

- [Sutton 与 Barto（2018），第 5 章——蒙特卡洛方法](http://incompleteideas.net/book/RLbook2020.pdf)——权威讲解。
- [Singh 与 Sutton（1996），使用替换式资格迹的强化学习](https://link.springer.com/article/10.1007/BF00114726)——首次访问与每次访问分析。
- [Precup、Sutton、Singh（2000），用于离策略策略评估的资格迹](http://incompleteideas.net/papers/PSS-00.pdf)——离策略蒙特卡洛与方差控制。
- [Mahmood 等（2014），用于离策略学习的加权重要性采样](https://arxiv.org/abs/1404.6362)——现代低方差重要性采样估计器。
- [Tesauro（1995），TD-Gammon：自学习西洋双陆棋程序](https://dl.acm.org/doi/10.1145/203330.203343)——首次大规模实证展示蒙特卡洛/TD 自我对弈如何收敛到超越人类的水平；也是本阶段后半部分所有课程在概念上的先驱。
