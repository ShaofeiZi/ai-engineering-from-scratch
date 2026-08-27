# 面向 Agent 的共识与拜占庭容错

> 经典分布式系统里的 BFT 正在和带随机性的 LLM 节点相遇。2025-2026 年出现了三条代表性研究路线：**CP-WBFT**（arXiv:2511.10400）会根据 confidence probe 给每张票赋权；**DecentLLMs**（arXiv:2507.14928）采用无 leader 设计，让 worker 并行提案，再用 geometric median 聚合；**WBFT**（arXiv:2505.05103）把 weighted voting 与 Hierarchical Structure Clustering 结合起来，划分 Core 与 Edge 节点。而 “Can AI Agents Agree?”（arXiv:2603.01213）给出的经验事实很直接：即便只是让 agent 对一个标量达成一致，今天都很脆弱，一个欺骗性 agent 就足以破坏 Mixture-of-Agents。BFT 是必要的，但远远不够。本课会实现一个最小 BFT 协议，注入三类 agent 特有攻击（byzantine lie、sycophantic conformity、correlated-error monoculture），并测量不同 consensus 变体各自如何应对。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 16 · 07（心智社会与辩论），阶段 16 · 13（共享记忆）
**Time:** 约 75 分钟

## 问题

假设你有 N 个 LLM agents，每个都给出一个答案。它们彼此不一致。多数投票挑出的那个答案却是错的，因为其中两个 agent 共享同一基础模型、相近训练数据和相同失败模式，所以它们的错误高度相关。再加上第三个 agent 恰好以另一种新方式出错，最后多数就变成了“假多数”。

现在再引入一个欺骗性 agent：它故意撒谎。或者一个讨好型 agent：它总是附和最后一个发言的人。经典 BFT 假设拜占庭节点占比满足 `f < n/3`，并且它们的行为可以是任意的。但 2026 年的现实是：LLM 节点即使诚实，也仍然带随机性；不同节点之间往往高度相关；它们还会被彼此的输出影响。你不能再把这些节点视作彼此独立的伯努利投票器。

所以，经典 BFT（PBFT，1999）并没有错，但它不完整。它可以处理任意比特翻转，却无法处理“三个诚实 agent 因为共享训练数据而一起幻觉出同一个错误”。本课就是在 PBFT 的地基上，再往上叠加 2025-2026 年出现的三类适配思路。

## 概念

### 经典 BFT 能给你什么

Practical Byzantine Fault Tolerance（Castro & Liskov, OSDI 1999）可以容忍 `f < n/3` 的 Byzantine 节点。协议包含三个阶段（pre-prepare、prepare、commit）和两个关键原语（signed messages、quorum certificates），目标是在 `n >= 3f + 1` 的诚实或恶意节点之间，就单一值达成一致。

它的保证很强，但默认有几个前提：

1. **故障彼此独立。** Byzantine 节点之间不会系统性协同。
2. **诚实节点真的正确。** 经典协议只负责把分歧对齐，不负责质疑诚实节点输出本身的正确性。
3. **问题存在 ground truth。** 就算大家一致同意一个错误事实，那在协议层面仍然算“共识成功”。

LLM agents 会同时打破这三条。两个共享同一 base model 的 agent 会共享故障；一个“诚实”的 LLM 也依然可能 hallucinate；而对于含糊问题，“真相”很多时候就是 agent 集体决定出来的，并没有外部 oracle。

### 三类 LLM 特有攻击

**Byzantine lie。** 某个 agent 故意给出错误答案。只要 `f < n/3`，经典 BFT 理论上能处理这一类问题。

**Sycophantic conformity。** 某个 agent 在投票前先观察别人的答案，然后选择附和最后一个发言者。它未必恶意，但会系统性偏向最响亮的声音。经典 BFT 无法阻止这一点，因为它在签名和消息格式层面完全合法。

**Correlated-error monoculture。** 三个 agent 共用一个基础模型，于是一起 hallucinate 出相同的错误答案。多数错了，但它们又都是“诚实”地错。经典 BFT 在这里帮不上忙，因为它只能识别恶意异常，不能识别相关性故障。

### 2025-2026 年的三种应对路线

**CP-WBFT**（arXiv:2511.10400）即 Confidence-Probed Weighted BFT。每个 voter 都会附带一个 confidence probe，可以是自报概率，也可以是单独校准模型给出的置信度预测。投票权重随 confidence 变化。论文报告在 complete graph 上带来 +85.71% 的 BFT 提升。它对 sycophantic conformity 有一定缓解作用，因为这类附和型 agent 往往对自己跟风得来的立场并没有真正高置信度。

**DecentLLMs**（arXiv:2507.14928）采用无 leader 设计。多个 worker agent 并行提出候选答案，由 evaluator agents 评分，最终答案通过 geometric median 聚合得到。它在 `f < n/2` 时仍具备较强鲁棒性。它对 byzantine lie 和 correlated errors 都有缓解作用，因为 geometric median 对 outliers 更稳健，聚合结果会更接近密集簇，而不是被偏移均值带跑。

**WBFT**（arXiv:2505.05103）即 Weighted BFT with Hierarchical Structure Clustering。投票权重由 response quality 和历史学到的 trust score 决定，再把 agents 聚成 Core 与 Edge 两层：Core 先达成共识，Edge 再跟进。它主要缓解扩展性问题，也能在一定程度上缓解 monoculture，因为 Core 可以按多样性优先进行选择。

### 经验事实：“Can AI Agents Agree?”

这篇论文研究的是 scalar agreement，也就是多个 frontier models 在单一数值问题上能否达成一致。结论不太舒服：

- 即便没有对手，LLM agents 在许多 benchmark 上对标量问题的分歧率仍高于 30%。
- 只要有一个 agent 采用 deceptive persona，就能把 Mixture-of-Agents 的共识结果从诚实基线拉偏 40 多个百分点。
- 模型越异质，分歧率越高（好处是错误不相关），但也越难快速收敛（坏处是 time-to-agreement 变长）。

因此真正的结论是：BFT 只能给你“把输出对齐”的机制，却不能告诉你“对齐后的输出是否正确”。它必须和 verification（Phase 16 · 08 角色专业化）、diversity（Phase 16 · 15 辩论变体）以及 evaluator agents（Phase 16 · 24 benchmarks）联用。

### 剥离到最核心的协议

一个最小化的 LLM-agent BFT 轮次大致是这样：

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

这里最关键的 LLM 特有步骤是 semantic clustering。比如“the study reports 4.2%”和“4.2% improvement”应该视为同一簇。若只做朴素字符串相等比较，就会错过这类等价答案。生产环境里通常会用廉价 embedding model 或显式 canonicalization 来解决。

### 阈值调优

`threshold` 参数决定了什么时候接受结果，什么时候重试或升级。太低会接受弱多数，太高则什么都无法接受。经验上，当 agent 数量在 `n=5-7` 时，阈值常落在 0.5-0.67；如果 `n` 更小，阈值通常要更高。低于阈值时，系统应该升级到人工或另一个 agent ensemble。

### 共识也有帮不上忙的时候

- **Ambiguous questions。** 如果问题没有明确 ground truth，那共识本质上只是意见集合。应该明确把它当意见处理。
- **Compound questions。** 像“写代码并解释代码”这种复合问题，其实包含两个答案，应拆开投票。
- **Adversarial multi-round。** 如果 agent 能看到前几轮并持续模仿（类似 Du 2023 式辩论），它们最终会越来越趋同于彼此，而不一定趋同于真相。因此轮数必须有上界，通常是 2-3 轮。

```figure
swarm-consensus-wave
```

## 动手构建

`code/main.py` 实现了：

- `AgentVoter`：一个脚本化投票策略，输出 answer 与 confidence。
- `MajorityVote`：经典 plurality。
- `CPWBFT`：带 semantic clustering 的 confidence-weighted voting。
- `DecentLLMs`：基于评分提案的 geometric-median aggregation。
- `Scenario`：在三类攻击模式下分别运行不同 aggregator。

内置攻击模式包括：

1. `byzantine`：一个 agent 高置信度撒谎。
2. `sycophancy`：一个 agent 复制它看到的第一个答案，并继承相应 confidence。
3. `monoculture`：三个 agent 以中等置信度共享同一个错误答案，构成 correlated error。

运行：

```
python3 code/main.py
```

预期输出是一张表：形如 (attack, aggregator) -> final answer，其中正确答案会被高亮。Plurality 会在 monoculture 案例下失效；CPWBFT 的 confidence weighting 对 sycophancy 有缓解；当 monoculture 不超过半数人群时，DecentLLMs 的 geometric median 会把结果拉回诚实簇附近。

## 实际使用

`outputs/skill-consensus-designer.md` 用于为多代理 ensemble 设计共识协议：cluster 方法、weighting 规则、threshold，以及低于阈值时的 escalation policy。

## 交付上线

在任何 consensus mechanism 上线前：

- **至少用上面三种攻击模式做攻击测试。** 你的协议应当可预测地失败，而不是悄悄失败。
- **记录所有 minority clusters。** 少数簇是 correlated error 的早期预警信号。
- **强制 bounded rounds。** 不要“辩到一致为止”，那只会奖励 sycophancy。
- **把 agreement 和 correctness 分开。** 共识输出应该交给 verifier；而 verifier 必须独立于这个 ensemble。
- **持续监控 agreement rate。** 突然升高常意味着 conformity bias，突然下降则常意味着 model drift。

## 练习

1. 运行 `code/main.py`，确认 plurality 会在 monoculture attack 下失败，但当 monoculture confidence 低于 0.7 时，CPWBFT 能部分缓解。
2. 增加第四种攻击：**silent abstention**，即某个 agent 拒绝作答（“I don't know”）。不同 aggregator 应如何处理 abstentions？实现你的选择。
3. 把 semantic clustering 从字符串 canonicalization 换成 embedding similarity（用任意开源 embedding model）。sycophancy attack 的表现会发生什么变化？
4. 阅读 CP-WBFT（arXiv:2511.10400）。补上 confidence-probe calibration 步骤，也就是用单独校准模型检查每个 agent 的自报信心。测量 monoculture 场景下的准确率变化。
5. 阅读 “Can AI Agents Agree?”（arXiv:2603.01213）。复现一个简化版 scalar-agreement 实验：三个 agents、一个标量问题、一个 deceptive-persona prompt。CPWBFT 或 DecentLLMs 能抓住它吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| BFT | “拜占庭容错” | Castro-Liskov 1999 年提出的共识协议，可处理 `f < n/3` 的任意故障。 |
| 拜占庭节点 | “任意恶意行为” | 节点可以撒谎、丢弃消息或静默失败，只要不是安全崩溃，都属于此类。 |
| 置信度探针 | “你有多确定？” | 附加在投票上的自报概率或校准器预测概率。 |
| 语义聚类 | “同一答案，不同说法” | 在计票前把语义等价的答案分到同一组。 |
| 几何中位数 | “稳健中心” | 使所有样本到该点的距离之和最小的点；与均值相比，对离群值更稳健。 |
| 单一文化 | “同一模型，同类失败” | 多个智能体因共享训练数据或基座模型而发生相关错误。 |
| 谄媚式从众 | “附和最响亮的声音” | 智能体的投票偏向最先或最强势的发言者。 |
| Core/Edge | “分层 BFT” | WBFT 的两层结构：先由小型 Core 达成共识，再让 Edge 跟随。 |

## 延伸阅读

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — 基础协议
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — 按置信度对投票赋权
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — 使用几何中位数聚合
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — 通过核心层/边缘层拆分控制延迟
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — 标量一致性的脆弱性与欺骗性角色提示攻击
