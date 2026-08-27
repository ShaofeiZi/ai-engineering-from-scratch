# MDP、状态、动作与奖励

> 马尔可夫决策过程由五样东西组成：状态、动作、转移、奖励和折扣。强化学习中的一切——Q-learning、PPO、DPO、GRPO——都在这个结构上做优化。只要学会一次，后面的强化学习内容都能轻松读懂。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 1 · 06（概率与分布）、阶段 2 · 01（机器学习分类体系）
**Time:** 约 45 分钟

## 问题

假设你正在编写国际象棋机器人、库存规划器、交易智能体，或训练推理模型的 PPO 循环。这四个领域截然不同，却有一个令人意外的共同点：都可以归结为同一个数学对象。

监督学习向你提供 `(x, y)` 数据对，并要求拟合一个函数。强化学习没有标签，只有一连串状态、采取的动作，以及一个标量奖励。那步棋赢得比赛了吗？补货决策省钱了吗？交易盈利了吗？大语言模型刚生成的词元是否让裁判给出了更高奖励？

在将这条数据流形式化之前，你无法从中学习。“我看到了什么”“我做了什么”“接下来发生了什么”“结果有多好”，都必须转化为可以推理的对象。这种形式化就是马尔可夫决策过程。本阶段中的每一种强化学习算法，包括最后的 RLHF 与 GRPO 循环，都在这个结构上进行优化。

## 概念

![马尔可夫决策过程：状态、动作、转移、奖励与折扣](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。智能体做决策所需的一切信息。在 GridWorld 中是所在格子，在国际象棋中是棋盘，在大语言模型中则是上下文窗口及所有记忆。
- **动作** `A`。可供选择的行为：向上/向下/向左/向右移动，走一步棋，或输出一个词元。
- **转移** `P(s' | s, a)`。给定状态 `s` 与动作 `a` 后，下一状态的概率分布。国际象棋中的转移是确定性的，库存管理中是随机的，大语言模型解码中则近似确定。
- **奖励** `R(s, a, s')`。标量反馈信号。胜利 = +1，失败 = -1；收入减成本；或者 GRPO 中的对数似然比项。
- **折扣** `γ ∈ [0, 1)`。未来奖励相对于当前奖励的重要程度。`γ = 0.99` 对应约 100 步的视野，`γ = 0.9` 则约为 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只取决于当前状态。若事实并非如此，说明状态表示不完整——这不是方法失效，而是状态定义失效。

**策略与回报。** 策略 `π(a | s)` 将状态映射为动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。价值 `V^π(s) = E[G_t | s_t = s]` 是从 `s` 出发、遵循策略 `π` 时的期望回报。Q 值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从某个特定动作开始的期望回报。每种强化学习算法都会估计这两者之一，再据此改进 `π`。

**贝尔曼方程。** 本阶段所有内容都会使用的固定点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程把期望回报拆成“当前步骤的奖励”加“到达状态的折扣价值”，并递归定义自身。本阶段中的每种算法，要么迭代这个方程直至收敛（动态规划），要么从中采样（蒙特卡洛），要么用一步结果自举（时序差分）。

```figure
discount-horizon
```

## 动手构建

### 第 1 步：一个微型确定性 MDP

构建一个 4×4 GridWorld。智能体从左上角出发，右下角为终止位置，每走一步获得 -1 奖励，动作集合为 `{up, down, left, right}`。参见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

只用五行，这就是完整环境：确定性转移、固定步进惩罚，以及吸收终止状态。

### 第 2 步：运行一次策略轨迹

策略是一个从状态到动作分布的函数。最简单的策略是均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。在这个 4×4 棋盘上，平均回报约为 -60 到 -80。最优回报是 -6（直接向下、向右到达终点）。缩小这段差距，就是阶段 9 的全部内容。

### 第 3 步：通过贝尔曼方程精确计算 `V^π`

对于小型 MDP，贝尔曼方程是一个线性方程组。枚举状态、计算期望并不断迭代，直到价值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。它是 Sutton 与 Barto 教材中的第一个算法，也是后续每种强化学习方法的理论基础。

### 第 4 步：`γ` 是具有物理含义的超参数

有效时域约为 `1 / (1 - γ)`。`γ = 0.9` → 10 步，`γ = 0.99` → 100 步，`γ = 0.999` → 1000 步。

取值太低，智能体会目光短浅；取值太高，信用分配会充满噪声，因为许多早期步骤都要为遥远未来的奖励共同负责。大语言模型 RLHF 通常使用 `γ = 1`，因为回合短且有明确上限。控制任务使用 `0.95–0.99`，长视野策略游戏则使用 `0.999`。

## 陷阱

- **非马尔可夫状态。** 如果必须参考最近三次观测才能决策，那么“状态”就不只是当前观测。解决办法是堆叠帧（Atari 上的 DQN 会堆叠 4 帧），或使用循环状态（在观测序列上运行 LSTM/GRU）。
- **稀疏奖励。** 在大型状态空间中，只有获胜时才给奖励会让学习几乎不可能。可以塑造奖励（提供中间信号），或先用模仿学习自举（阶段 9 · 09）。
- **奖励黑客。** 优化代理奖励经常会产生病态行为。OpenAI 的赛艇智能体没有完成比赛，而是不断原地转圈收集加速道具。奖励必须根据目标结果定义，而不是根据代理指标定义。
- **折扣设定错误。** 在无限视野任务中使用 `γ = 1`，会使所有价值变为无穷。必须设置有限视野或确保 `γ < 1`。
- **奖励尺度。** {+100, -100} 与 {+1, -1} 会产生相同的最优策略，却导致截然不同的梯度幅度。在送入 PPO/DQN 前，应将其归一化到近似 `[-1, 1]` 的范围。

## 学以致用

2026 年的技术栈会先把每个强化学习流水线归结为 MDP，再开始编写代码：

| 场景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（运动、操作） | 关节角度 + 速度 | 连续力矩 | 针对任务塑造 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘/牌桌 + 历史 | 合法动作 | 胜=+1 / 负=-1 | 1.0（有限） |
| 库存/定价 | 库存 + 需求 | 订购数量 | 收入 - 成本 | 0.95 |
| 大语言模型 RLHF | 上下文词元 | 下一个词元 | 结束时的奖励模型分数 | 1.0（回合约 200 个词元） |
| 推理任务 GRPO | 提示词 + 部分回答 | 下一个词元 | 结束时验证器给出 0/1 | 1.0 |

在编写任何训练循环之前，先写出这五元组。大多数“强化学习不起作用”的错误报告，最终都能追溯到纸面上就已经有问题的 MDP 定义。

## 交付成果

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 GridWorld 和随机策略轨迹。运行 10,000 个回合，报告回报的均值与标准差，并与最优回报（-6）比较。
2. **中等。** 对均匀随机策略运行 `policy_evaluation`，分别使用 `γ ∈ {0.5, 0.9, 0.99}`。将每种情况下的 `V` 打印成 4×4 网格。解释为什么 `γ` 越大，终止位置附近的状态价值增长得越快。
3. **困难。** 把 GridWorld 改成随机环境：每个动作都以 `p = 0.1` 的概率滑向相邻方向。重新评估均匀策略。`V[start]` 会变好还是变差？为什么？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MDP | “强化学习问题设置” | 满足马尔可夫性质的五元组 `(S, A, P, R, γ)`。 |
| 状态 | “智能体看到的内容” | 在所选策略类别下，足以描述未来动力学的统计量。 |
| 策略 | “智能体的行为” | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| 回报 | “总奖励” | 从当前步骤开始的折扣和 `Σ γ^t r_t`。 |
| 价值 | “一个状态有多好” | 在策略 `π` 下从 `s` 出发的期望回报。 |
| Q 值 | “一个动作有多好” | 在策略 `π` 下从 `s` 出发并以 `a` 为首个动作时的期望回报。 |
| 贝尔曼方程 | “动态规划递归” | 将价值/Q 分解为一步奖励加折扣后继价值的固定点关系。 |
| 折扣 `γ` | “未来与现在” | 施加于远期奖励的几何权重；有效时域为 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton 与 Barto（2018），《强化学习：导论》第 2 版](http://incompleteideas.net/book/RLbook2020.pdf)——权威教材。第 3 章介绍 MDP 与贝尔曼方程，第 1 章阐述贯穿后续所有课程的奖励假说。
- [Bellman（1957），《动态规划》](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming)——贝尔曼方程的起源。
- [OpenAI Spinning Up——第 1 部分：核心概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)——从深度强化学习角度给出的精炼 MDP 入门。
- [Puterman（2005），《马尔可夫决策过程》](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887)——关于 MDP 与精确求解方法的运筹学参考书。
- [Littman（1996），《序贯决策算法》（博士论文）](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf)——把 MDP 推导为动态规划特例的最清晰论述。
