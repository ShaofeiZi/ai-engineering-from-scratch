# 游戏强化学习——AlphaZero、MuZero 与大语言模型推理时代

> 1992 年：TD-Gammon 仅凭 TD 就击败了西洋双陆棋人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始统治国际象棋、将棋与围棋。2024 年：DeepSeek-R1 证明，同一套方案将 PPO 换成 GRPO 后，也能用于推理。游戏是推动本阶段每一次突破的基准。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 9 · 05（DQN）、阶段 9 · 08（PPO）、阶段 9 · 09（RLHF）、阶段 9 · 10（MARL）
**Time:** 约 120 分钟

## 问题

游戏具备强化学习想要的一切：清晰的奖励（胜/负），无限的回合（自博弈可以重置），完美的仿真（游戏*本身就是*模拟器），离散或规模较小的连续动作空间，以及迫使智能体具备对抗鲁棒性的多智能体结构。

而且，每一次重大强化学习突破都通过游戏得到检验：TD-Gammon（西洋双陆棋，1992）、Atari-DQN（2013）、AlphaGo（2016）、AlphaZero（2017）、OpenAI Five（Dota 2，2019）、AlphaStar（StarCraft II，2019）、MuZero（学习式模型，2019）、AlphaTensor（矩阵乘法，2022）、AlphaDev（排序算法，2023）、DeepSeek-R1（数学推理，2025）——最新的例子再次证明，游戏强化学习技术也适用于文本。

这篇收官课通过同一个统一视角考察三种里程碑式架构——AlphaZero、MuZero 与 GRPO：**自博弈 + 搜索 + 策略改进**。每一种都是对上一种的推广；尤其是 GRPO，它把 AlphaZero 的方案应用于大语言模型推理，把词元视为动作，把数学验证视为获胜信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO：同一个循环，不同的环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017）。** Silver 等人提出。给定规则已知的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：使用一个主干 `f_θ(s) → (p, v)`。`p` 是合法着法的先验分布，`v` 是预期比赛结果。
- 蒙特卡洛树搜索（MCTS）：每一步都展开由可能后续局面组成的搜索树，用 `(p, v)` 作为先验 + 自举值。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自博弈：让智能体彼此对弈。在第 `t` 步，MCTS 的访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是比赛结果（+1 / 0 / -1）。

不使用任何人类知识，也没有手工启发式方法。仅靠一套方案，在每种游戏中进行数千万局自博弈后，就掌握了国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人提出，去掉了规则必须已知的要求。

- 不再使用固定环境，而是学习一个*潜在动力学模型* `(h, g, f)`：
  - `h(s)`：把观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习得到的潜在空间*中运行。搜索和训练循环都保持不变。
- 它既适用于围棋、国际象棋和将棋，也适用于 Atari——同一种算法，无须了解规则。

**随机 MuZero（2022）。** 加入随机动力学与机会节点，将方法扩展到西洋双陆棋一类的游戏。

**Muesli、Gumbel MuZero（2022～2024）。** 改进样本效率与确定性搜索。

**GRPO（2024～2025）。** DeepSeek-R1 的方案。它把形似 AlphaZero 的同一循环应用到语言模型推理：

- “游戏”：回答一道数学/编程/推理题。“获胜” = 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略：大语言模型。动作：词元。状态：提示词 + 已生成的部分回答。
- 不使用评论家（PPO 风格的 V_φ）。对于每个提示词，从策略中采样 `G` 个补全，分别计算奖励，再用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略施加 KL 惩罚，以防止漂移（与 RLHF 相同）。
- 完整损失为：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

不需要奖励模型、评论家或 MCTS，组相对基线取代了这三者。它只使用一小部分计算量，就能在推理基准上达到或超过 PPO-RLHF 的质量。

**完整的 R1 方案。** DeepSeek-R1（DeepSeek，2025）一篇论文中包含两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始，不做 SFT，直接使用 GRPO，并设置两种奖励：*准确性奖励*（基于规则——最终答案能否解析为正确数字/代码能否通过单元测试）和*格式奖励*（补全是否用 `<think>…</think>` 标签包裹思维链）。经过数千个步骤，平均回答长度从约 100 个词元增长到约 10,000 个，数学基准分数也上升到接近 o1-preview 的水平。模型从零学会了推理。缺点是思维链经常难以阅读、混用语言，而且缺乏行文润色。
- **R1。** 通过四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集数千条格式整洁的长思维链示范，对基础模型进行监督微调，从而得到可读的起点。
  2. **面向推理的 GRPO。** 使用准确性 + 格式奖励运行 GRPO，并增加*语言一致性*奖励，防止语言切换。
  3. **拒绝采样 + 第二轮 SFT。** 从强化学习检查点采样约 60 万条推理轨迹，只保留最终答案正确且思维链可读的样本，再与约 20 万条非推理 SFT 样本（写作、问答、自我认知）合并，重新微调基础模型。
  4. **全谱系 GRPO。** 再运行一轮强化学习，同时覆盖推理（基于规则的奖励）和通用对齐（基于偏好的有帮助/无伤害奖励）。

最终模型以开放权重在 AIME 与 MATH-500 上达到 o1 的水平，而且规模足够小，可以进行蒸馏。同一篇论文还发布了六个蒸馏后的稠密模型（从 Qwen-1.5B 到 Llama-70B），它们通过在 R1 的推理轨迹上执行 SFT 得到——学生模型不进行强化学习。对于学生模型的规模，蒸馏强大的强化学习教师始终优于从零开始做强化学习。

**为什么推理任务使用 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）给出了三个原因：（1）不需要训练价值网络，内存减半；（2）组基线天然适合推理任务产生的稀疏轨迹末端奖励；（3）逐提示词归一化使不同难度问题的优势可以相互比较，而 PPO 的单一评论家做不到这一点。

**无搜索与有搜索。** 游戏领域已经分叉：

- *视野很长的完全信息游戏*（围棋、国际象棋）：仍以搜索为基础，由 AlphaZero / MuZero 主导。
- *大语言模型推理*：生产环境尚未采用 MCTS；训练使用完整轨迹上的 GRPO，推理计算则使用 Best-of-N。过程奖励模型（PRM）预示着步骤级搜索将重新加入。

```figure
f3-selfplay-ladder
```

## 动手构建

`code/main.py` 实现一个**微型 GRPO**——带多组样本的多臂老虎机。其算法与用于大语言模型时相同，只是策略和环境更简单。它讲解的是*损失*与*组相对优势*，也就是 2025 年的创新。

### 第 1 步：微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中，验证器会运行单元测试或检查数学等式。

### 第 2 步：策略——每个提示词对应 K 个答案词元上的 Softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

这等价于以提示词为条件的大语言模型最后一层输出。

### 第 3 步：组采样与组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 DeepSeek 在 2024 年提出的技巧。不再需要评论家：“基线”就是组均值，归一化则使用组标准差。

### 第 4 步：与 REINFORCE 基线比较（不使用价值）

采用相同设置和相同计算量运行普通 REINFORCE。GRPO 收敛得更快、更稳定。

### 第 5 步：观察熵与 KL

使用与 RLHF 相同的诊断指标：相对于参考模型的平均 KL、策略熵、奖励随时间的变化。它们稳定后，训练即告完成。

## 陷阱

- **通过欺骗验证器实施奖励黑客。** GRPO 继承了 RLHF 的风险：如果验证器错误或可被利用，大语言模型就会找到漏洞。稳健的验证器（多组测试用例、形式化证明）至关重要。
- **组大小太小。** 组基线的方差按 `1/√G` 变化。低于 `G = 4` 时，优势信号噪声很大；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 长度不同的大语言模型补全具有不同的对数概率。可以按词元数归一化、使用序列级对数概率，或截断到最大长度。
- **纯自博弈循环。** AlphaZero 风格训练可能在一般和游戏中陷入优势关系循环。可用多样化对手池缓解（联赛训练，第 10 课）。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小，无法表示搜索产生的分布，训练就会停滞。
- **计算门槛。** MuZero / AlphaZero 需要庞大的计算量，一次消融往往就要数百 GPU 小时。学习时可以使用微型演示，例如在四子棋上运行 AlphaZero。
- **验证器覆盖不足。** 如果错误解法也能通过单元测试，模型就会强化这个错误。验证器设计必须覆盖边界情况。

## 学以致用

按领域划分的 2026 年游戏强化学习格局：

| 领域 | 主流方法 |
|--------|-----------------|
| 双人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 非完全信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略游戏（Dota、StarCraft） | PPO + 自博弈 + 联赛（OpenAI Five、AlphaStar） |
| 大语言模型数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开放复现） |
| 大语言模型对齐 | DPO / RLHF-PPO（不使用 GRPO；验证信号是偏好，而非可验证结果） |
| 机器人 | PPO + 领域随机化（并非游戏强化学习，但使用相同的策略梯度工具） |
| 组合问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

这套*方案*——自博弈、搜索增强改进、策略蒸馏——横跨文本、像素和物理控制。GRPO 是其中最年轻的实例，未来还会出现更多。

## 交付成果

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 多臂老虎机。针对 2 个提示词 × 每个 4 个答案词元进行训练，在 `G=8` 时用不到 1000 次更新收敛。
2. **中等。** 接入 PPO（裁剪版）与普通 REINFORCE。在同一个多臂老虎机上，比较它们与 GRPO 的样本效率和奖励方差。
3. **困难。** 扩展为长度为 2 的“推理链”：智能体输出两个词元，验证器为整个词元对提供奖励。测量 GRPO 如何处理两步序列中的信用分配。（提示：为每个*完整序列*计算组优势，再将其传播到两个词元位置。）

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | “使用学习式网络的树搜索” | 蒙特卡洛树搜索；通过使用学习所得 `(p, v)` 先验的 UCB1/PUCT 进行选择。 |
| AlphaZero | “自博弈 + MCTS” | 训练策略-价值网络，使其匹配 MCTS 访问次数与比赛结果。 |
| MuZero | “学习模型版 AlphaZero” | 相同循环，但通过学习得到的动力学在潜在空间中运行。 |
| GRPO | “无评论家的 PPO” | 组相对策略优化；使用组均值基线 + KL 的 REINFORCE。 |
| PUCT | “AlphaZero 的 UCB” | `Q + c · p · √N / (1 + N_a)`——在价值估计与先验之间平衡。 |
| 自博弈 | “智能体与过去的自己对战” | 零和游戏的标准方法；提供对称训练信号。 |
| 联赛训练 | “基于种群的自博弈” | 从历史策略 + 当前策略 + 利用者中采样对手。 |
| 验证器奖励 | “可验证强化学习” | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| 过程奖励 | “PRM” | 为每一个推理步骤评分，而不只是最终答案。 |

## 延伸阅读

- [Silver 等（2017），在没有人类知识的情况下掌握围棋（AlphaGo Zero）](https://www.nature.com/articles/nature24270)。
- [Silver 等（2018），通过自博弈掌握国际象棋、将棋和围棋的通用强化学习算法（AlphaZero）](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser 等（2020），通过使用学习所得模型进行规划来掌握 Atari、围棋、国际象棋和将棋（MuZero）](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals 等（2019），达到《星际争霸 II》宗师水平（AlphaStar）](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI（2024），DeepSeekMath：推进开放语言模型的数学推理极限（GRPO）](https://arxiv.org/abs/2402.03300)——提出 GRPO 与组相对基线的论文。
- [DeepSeek-AI（2025），DeepSeek-R1：通过强化学习激励大语言模型的推理能力](https://arxiv.org/abs/2501.12948)——完整的四阶段 R1 方案与 R1-Zero 消融。
- [Brown 等（2019），用于多人扑克的超人类 AI（Pluribus）](https://www.science.org/doi/10.1126/science.aay2400)——大规模 CFR + 深度学习。
- [Tesauro（1995），时序差分学习与 TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343)——一切的起点。
- [Hugging Face TRL——GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)——使用自定义奖励函数应用 GRPO 的生产参考。
- [Qwen 团队（2024），Qwen2.5-Math——GRPO 复现](https://github.com/QwenLM/Qwen2.5-Math)——在多种规模上开放复现 R1 方案。
- [Sutton 与 Barto（2018），第 17 章——强化学习前沿](http://incompleteideas.net/book/RLbook2020.pdf)——以教材视角阐述自博弈、搜索与“设计式奖励”，R1 则在大语言模型规模上实现了这些思想。
