# 动态规划——策略迭代与价值迭代

> 动态规划就像是可以“作弊”的强化学习。你已经知道转移函数和奖励函数，只需不断迭代贝尔曼方程，直到 `V` 或 `π` 不再变化。所有基于采样的方法都在努力逼近它给出的基准。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 9 · 01（MDP）
**Time:** 约 75 分钟

## 问题

现在有一个模型已知的 MDP：对于任意状态-动作对，都可以查询 `P(s' | s, a)` 与 `R(s, a, s')`。库存管理器知道需求分布；棋盘游戏拥有确定性转移；GridWorld 用四行 Python 就能描述。也就是说，你拥有一个*模型*。

无模型强化学习（Q-learning、PPO、REINFORCE）是为没有模型的情况发明的——你只能从环境中采样。但一旦拥有模型，就有更快、更好的方法：动态规划。Bellman 于 1957 年设计了这些方法，时至今日，它们仍然定义着正确答案。人们所说的“这个 MDP 的最优策略”，指的就是动态规划会返回的策略。

2026 年仍然需要动态规划，原因有三。第一，强化学习研究中的每个表格型环境（GridWorld、FrozenLake、CliffWalking）都会用动态规划求解，以得到金标准策略。第二，精确价值可以用来*调试*采样方法：如果 Q-learning 对 `V*(s_0)` 的估计与动态规划答案相差 30%，说明 Q-learning 实现有问题。第三，现代离线强化学习与规划方法（MCTS、AlphaZero 搜索、阶段 9 · 10 的基于模型强化学习）都会在学习得到或给定的模型上迭代执行贝尔曼备份。

## 概念

![策略迭代与价值迭代并列对比](../../../../../../phases/09-reinforcement-learning/02-dynamic-programming/assets/dp.svg)

**两种算法，本质上都是对贝尔曼方程进行固定点迭代。**

**策略迭代。** 交替执行以下两步，直到策略不再变化。

1. *评估：* 给定策略 `π`，计算 `V^π`：反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`，直至收敛。
2. *改进：* 给定 `V^π`，令 `π` 成为相对于 `V^π` 的贪心策略：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛之所以有保证，是因为：（a）每次改进要么保持 `π` 不变，要么严格提高某个状态的 `V^π`；（b）确定性策略的空间是有限的。即使状态空间很大，通常也只需约 5～20 次外层迭代就能收敛。

**价值迭代。** 把评估与改进合并为一次扫描。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

不断重复，直到 `max_s |V_{new}(s) - V(s)| < ε`。最后选择贪心动作，提取策略。每次迭代严格来说更快——没有内部评估循环——但通常需要更多次迭代才能收敛。

**广义策略迭代（GPI）。** 这是统一两者的视角。价值函数与策略被锁在双向改进循环中；任何让两者趋向相互一致的方法（异步价值迭代、修正策略迭代、Q-learning、Actor-Critic、PPO）都是 GPI 的一种实例。

**为什么 `γ < 1` 很重要。** 贝尔曼算子在上确界范数下是 `γ`-压缩映射：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩性意味着固定点唯一，而且会几何收敛。去掉 `γ < 1`，就会失去这项保证——此时必须设置有限视野或吸收终止状态。

```figure
value-iteration-gamma
```

## 动手构建

### 第 1 步：构建 GridWorld MDP 模型

沿用第 01 课的 4×4 GridWorld。我们再增加一种随机变体：智能体以 `0.1` 的概率滑向随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回由 `(s', r, p)` 组成的列表。这就是完整模型。

### 第 2 步：策略评估

给定策略 `π(s) = {action: prob}`，不断迭代贝尔曼方程，直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 第 3 步：策略改进

用 `π` 替换为相对于 `V` 的贪心策略。如果 `π` 没有变化，则返回结果——我们已经到达最优解。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 第 4 步：把两者组合起来

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在 4×4 网格上，通常经过 4～6 次外层迭代即可收敛。输出为 `V*(0,0) ≈ -6`，并得到一个能严格减少步数的策略。

### 第 5 步：价值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

固定点相同，代码行数更少。

## 陷阱

- **忘记处理终止状态。** 如果对吸收状态也应用贝尔曼方程，它仍会选出一个什么都不改变的“最佳动作”。应使用 `if s == terminal: V[s] = 0` 进行保护。
- **上确界范数与 L2 收敛。** 应使用 `max |V_new - V|`，而不是平均值。理论保证针对的是上确界范数。
- **原地更新与同步更新。** 原地更新 `V[s]`（Gauss-Seidel）比使用单独的 `V_new` 字典（Jacobi）收敛得更快。生产代码使用原地更新。
- **策略平局。** 如果两个动作的 Q 值相等，`argmax` 可能在每轮迭代中以不同方式打破平局，导致“策略稳定”检查来回振荡。应采用稳定的决胜规则，例如固定顺序中的第一个动作。
- **状态空间爆炸。** 动态规划每次扫描的复杂度是 `O(|S| · |A|)`，最多适用于约 10⁷ 个状态。超过这个规模，就需要函数近似（阶段 9 · 05 及后续课程）。

## 学以致用

2026 年，动态规划既是正确性基线，也是规划器的内部循环：

| 使用场景 | 方法 |
|----------|--------|
| 精确求解小型表格 MDP | 价值迭代（更简单）或策略迭代（外层步骤更少） |
| 验证 Q-learning / PPO 实现 | 在玩具环境中与动态规划最优 V* 比较 |
| 基于模型的强化学习（阶段 9 · 10） | 在学习得到的转移模型上执行贝尔曼备份 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索 = 异步贝尔曼备份 |
| 离线强化学习（CQL、IQL） | 保守 Q 迭代——对分布外动作施加惩罚的动态规划 |

每当有人提到“最优价值函数”，指的就是“动态规划固定点”。在论文中看到 `V*` 或 `Q*` 时，就应想到这个循环。

## 交付成果

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上分别以 `γ ∈ {0.9, 0.99}` 运行价值迭代。需要扫描多少轮才能满足 `max |ΔV| < 1e-6`？将 `V*` 打印为 4×4 网格。
2. **中等。** 在*随机* GridWorld（滑动概率 `0.1`）上比较策略迭代与价值迭代。统计扫描次数、实际运行时间和最终 `V*(0,0)`。哪一种在迭代次数上收敛更快？哪一种实际耗时更短？
3. **困难。** 构建修正策略迭代：评估步骤不再运行至收敛，只执行 `k` 次扫描。绘制 `V*(0,0)` 误差随 `k` 变化的曲线，其中 `k ∈ {1, 2, 5, 10, 50}`。这条曲线说明了评估与改进之间怎样的权衡？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 策略迭代 | “动态规划算法” | 交替执行评估（`V^π`）与改进（贪心 `π` 相对于 `V^π`），直到策略停止变化。 |
| 价值迭代 | “更快的动态规划” | 一次扫描中应用贝尔曼最优性备份；以几何速度收敛到 `V*`。 |
| 贝尔曼算子 | “递归式” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；它是 `γ`-压缩映射。 |
| 压缩映射 | “动态规划为何收敛” | 任何算子 `T`，只要满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|`，就有唯一固定点。 |
| GPI | “一切都是动态规划” | 广义策略迭代：任何推动 `V` 与 `π` 达成相互一致的方法。 |
| 同步更新 | “Jacobi 风格” | 一次扫描中始终使用旧的 `V`；易于分析，但速度较慢。 |
| 原地更新 | “Gauss-Seidel 风格” | 在更新过程中立即使用新的 `V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton 与 Barto（2018），第 4 章——动态规划](http://incompleteideas.net/book/RLbook2020.pdf)——策略迭代与价值迭代的权威讲解。
- [Bertsekas（2019），《强化学习与最优控制》](http://www.athenasc.com/rlbook.html)——压缩映射论证的严谨处理。
- [Puterman（2005），《马尔可夫决策过程》](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887)——修正策略迭代及其收敛分析。
- [Howard（1960），《动态规划与马尔可夫过程》](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/)——最初的策略迭代论文。
- [Bertsekas 与 Tsitsiklis（1996），《神经动态规划》](http://www.athenasc.com/ndpbook.html)——从动态规划通往近似动态规划/深度强化学习的桥梁，也是后续每课的基础。
