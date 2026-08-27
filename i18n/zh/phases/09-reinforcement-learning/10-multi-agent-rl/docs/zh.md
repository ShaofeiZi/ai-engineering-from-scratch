# 多智能体强化学习

> 单智能体强化学习假设环境是平稳的。把两个正在学习的智能体放进同一个世界，这项假设便会失效：每个智能体都是另一个智能体所处环境的一部分，而且双方都在变化。多智能体强化学习就是一套在马尔可夫假设不再成立时，仍能让学习收敛的方法。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 9 · 04（Q-learning）、阶段 9 · 06（REINFORCE）、阶段 9 · 07（Actor-Critic）
**Time:** 约 45 分钟

## 问题

机器人学习在房间中导航，是单智能体强化学习问题。足球队不是，AlphaStar 对战 StarCraft 对手不是，由竞价智能体组成的市场不是，两辆车协商如何通过十字路口也不是。现实世界中，许多多对多问题都不属于单智能体场景。

在任何多智能体环境中，从某一个智能体的视角看，其他智能体*就是*环境的一部分。随着它们学习并改变行为，环境变得非平稳。马尔可夫性质——“下一状态只取决于当前状态和我的动作”——遭到破坏，因为下一状态还取决于*其他*智能体选择了什么，而它们的策略又在不断变化。

这会破坏表格方法的收敛性证明（Q-learning 的保证假设环境平稳），也会破坏朴素深度强化学习：智能体循环追逐彼此，始终无法收敛到稳定策略。你需要多智能体专用技巧：集中式训练/分散式执行、反事实基线、联赛训练和自博弈。

2026 年的应用包括机器人群体、交通调度、自动驾驶车队、市场模拟器、多智能体大语言模型系统（阶段 16），以及任何包含多个智能参与者的游戏。

## 概念

![四种多智能体强化学习范式：独立、集中式评论家、自博弈、联赛](../assets/marl.svg)

**形式化：马尔可夫博弈。** 它是 MDP 的推广，包含状态 `S`、联合动作 `a = (a_1, …, a_n)`、转移 `P(s' | s, a)`，以及每个智能体各自的奖励 `R_i(s, a, s')`。每个智能体 `i` 都在自己的策略 `π_i` 下最大化自身回报。如果所有奖励相同，就是**完全合作**；如果是零和，就是**对抗**；如果兼而有之，就是**一般和**。

**核心挑战：**

- **非平稳性。** `P(s' | s, a_i)` 从智能体 `i` 的视角看，取决于不断变化的 `π_{-i}`。
- **信用分配。** 获得共享奖励时，究竟是哪一个智能体促成了它？
- **协调探索。** 智能体需要探索互补策略，而不能重复探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 指数增长。
- **部分可观测性。** 每个智能体只能看到自己的观测，全局状态是隐藏的。

**四种主流范式：**

**1. 独立 Q-learning / 独立 PPO（IQL、IPPO）。** 每个智能体学习自己的 Q 函数或策略，把其他智能体视为环境的一部分。这种方法简单，有时也能奏效（尤其是经验回放发挥平滑智能体建模的作用时）。理论收敛保证：没有。实践表现：适合松耦合任务，不适合紧耦合任务。

**2. 集中式训练、分散式执行（CTDE）。** 这是最常见的现代范式。每个智能体都有自己的*策略* `π_i`，仅以本地观测 `o_i` 为条件——部署时采用标准的分散式执行。*训练*时则使用集中式评论家 `Q(s, a_1, …, a_n)`，以完整全局状态和联合动作为条件。例如：
- **MADDPG**（Lowe 等，2017）：每个智能体都拥有集中式评论家的 DDPG。
- **COMA**（Foerster 等，2017）：反事实基线——询问“如果我改为采取动作 `a'`，奖励会是多少？”——从而分离我的贡献。
- **MAPPO** / 使用共享评论家的 **IPPO**（Yu 等，2022）：使用集中式价值函数的 PPO。它是 2026 年合作式多智能体强化学习的主流方法。
- **QMIX**（Rashid 等，2018）：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，并采用单调混合。

**3. 自博弈。** 让同一个智能体的两个副本彼此对战。对手策略就是自己过去某个时间点的策略快照。AlphaGo / AlphaZero / MuZero、OpenAI Five 都采用这种方法。它最适合零和游戏，训练信号是对称的。

**4. 联赛训练。** 将自博弈扩展到一般和/对抗环境：保留由历史与当前策略组成的种群，从联赛中采样对手，再针对它们训练。联赛会加入利用者（专门击败当前最佳策略）和主要利用者（专门击败利用者）。AlphaStar（StarCraft II）采用这种方法。当游戏会出现“石头剪刀布”式策略循环时，就需要联赛训练。

**通信。** 允许智能体互相发送学习得到的消息 `m_i`。这适用于合作场景。Foerster 等人（2016）证明，可微的智能体间通信可以端到端训练。如今基于大语言模型的多智能体系统（阶段 16），本质上就是使用自然语言通信。

```figure
f3-marl-orbit
```

## 动手构建

本课使用一个 6×6 GridWorld，其中有两个合作智能体。它们从相对的角落出发，必须到达同一个目标。共享奖励：只要仍有任一智能体在移动，每步奖励为 `-1`；二者都到达时奖励为 `+10`。参见 `code/main.py`。

### 第 1 步：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间为 `|A|² = 16`，全局状态由两个位置组成。

### 第 2 步：独立 Q-learning

每个智能体运行自己的 Q 表，并以联合状态为键。每一步中，两者分别选择 ε-贪心动作，收集联合转移，再使用共享奖励更新各自的 Q 值。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

这个任务的奖励密集且目标一致，因此该方法可以奏效。它会在紧耦合任务中失败，例如一个智能体必须*等待*另一个智能体时。

### 第 3 步：采用分解价值更新的集中式 Q

针对联合动作使用一个 Q 函数 `Q(s, a_1, a_2)`，并根据共享奖励更新。执行时通过边缘化实现分散决策：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。它以指数级联合动作空间为代价，换取*正确*的全局视角。

### 第 4 步：简单自博弈（对抗式双智能体）

同一个智能体承担两个角色。训练智能体 A 对抗智能体 B；每隔 `K` 个回合，把 A 的权重复制给 B。训练对称，进步也保持一致。这就是缩小版 AlphaZero 方案。

## 陷阱

- **非平稳回放。** 独立智能体使用经验回放时，比单智能体更糟，因为旧转移来自现已过时的对手。解决方法是重新标注，或按新近程度加权。
- **信用分配歧义。** 漫长回合后才得到共享奖励，无法明确判断哪个智能体做出了贡献。解决方法是反事实基线（COMA），或为每个智能体塑造奖励。
- **策略漂移/相互追逐。** 每个智能体的最佳响应都会随对方的更新而变化。解决方法是使用集中式评论家、降低学习率，或每次只冻结一方。
- **通过协调实施奖励黑客。** 智能体会找到设计者未曾预料的协同漏洞，例如拍卖智能体共同收敛到零报价。解决方法是谨慎设计奖励，并施加行为约束。
- **重复探索。** 两个智能体探索相同的状态-动作对。解决方法是为每个智能体设置熵奖励，或加入角色条件。
- **联赛循环。** 纯自博弈可能陷入优势关系循环。解决方法是使用包含多样化对手的联赛训练。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。应采用函数近似与分解动作空间（每个智能体一个策略输出头）进行近似。

## 学以致用

2026 年多智能体强化学习应用图谱：

| 领域 | 方法 | 说明 |
|--------|--------|-------|
| 合作导航/操作 | MAPPO / QMIX | CTDE；共享评论家 + 分散式演员。 |
| 双人游戏（国际象棋、围棋、扑克） | 结合 MCTS 的自博弈（AlphaZero） | 零和；对称训练。 |
| 复杂多人游戏（Dota、StarCraft） | 联赛训练 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | 使用注意力的 CTDE MAPPO / PPO | 部分可观测；团队规模可变。 |
| 拍卖市场 | 博弈论均衡 + 强化学习 | 当 `n` → ∞ 时使用平均场强化学习。 |
| 大语言模型多智能体系统（阶段 16） | 自然语言通信 + 角色条件 | 强化学习循环位于智能体规划层。 |

到 2026 年，多智能体强化学习增长最快的领域是基于大语言模型的系统：语言模型智能体群体通过协商、辩论来构建软件。强化学习作用于*轨迹级*输出的偏好优化，而非词元级输出（阶段 16 · 03）。

## 交付成果

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在双智能体合作 GridWorld 上训练独立 Q-learning。需要多少个回合，平均回报才能 > 0？绘制联合学习曲线。
2. **中等。** 加入“协调”任务：只有两个智能体在同一回合踏上目标，才算到达。独立 Q-learning 还能收敛吗？哪里出了问题？
3. **困难。** 为 MAPPO 风格训练实现集中式评论家，并在协调任务上将其收敛速度与独立 PPO 比较。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 马尔可夫博弈 | “多智能体 MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体都有自己的奖励。 |
| CTDE | “集中式训练、分散式执行” | 训练时使用联合评论家；每个智能体的策略只使用本地观测。 |
| IPPO | “独立 PPO” | 每个智能体分别运行 PPO。简单但常被低估的基线。 |
| MAPPO | “多智能体 PPO” | 使用以全局状态为条件的集中式价值函数的 PPO。 |
| QMIX | “单调价值分解” | `Q_tot = f_monotone(Q_1, …, Q_n)`，支持分散式 argmax。 |
| COMA | “反事实多智能体方法” | 优势 = 我的 Q 减去对我的动作进行边缘化后得到的期望 Q。 |
| 自博弈 | “智能体与过去的自己对战” | 一个智能体、两个角色；零和游戏的标准方法。 |
| 联赛训练 | “种群训练” | 缓存历史策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe 等（2017），混合合作-竞争环境的多智能体 Actor-Critic（MADDPG）](https://arxiv.org/abs/1706.02275)——使用集中式评论家的 CTDE。
- [Foerster 等（2017），反事实多智能体策略梯度（COMA）](https://arxiv.org/abs/1705.08926)——用于信用分配的反事实基线。
- [Rashid 等（2018），QMIX：单调价值函数分解](https://arxiv.org/abs/1803.11485)——具有单调性的价值分解。
- [Yu 等（2022），PPO 在合作式多智能体游戏中出人意料的有效性（MAPPO）](https://arxiv.org/abs/2103.01955)——PPO 在多智能体强化学习中表现出人意料地强。
- [Vinyals 等（2019），使用多智能体强化学习达到《星际争霸 II》宗师水平（AlphaStar）](https://www.nature.com/articles/s41586-019-1724-z)——大规模联赛训练。
- [Silver 等（2017），在没有人类知识的情况下掌握围棋（AlphaGo Zero）](https://www.nature.com/articles/nature24270)——零和游戏中的纯自博弈。
- [Sutton 与 Barto（2018），第 15 章——神经科学；第 17 章——前沿](http://incompleteideas.net/book/RLbook2020.pdf)——包含教材对多智能体环境的简要讨论，以及 CTDE 所要解决的非平稳问题。
- [Zhang、Yang 与 Başar（2021），多智能体强化学习：选择性综述](https://arxiv.org/abs/1911.10635)——涵盖合作、竞争和混合式多智能体强化学习及其收敛结果的综述。
