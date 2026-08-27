# 投票、自洽性与辩论拓扑

> 最便宜的聚合方式，是采样 N 个独立 agents 然后做多数投票。Wang et al. 2022 的 self-consistency 就是在一个模型上采样 N 次完成这件事。多代理把它扩展成 **heterogeneous** agents，以摆脱 monoculture：不同模型、不同 prompts、不同 temperatures、不同上下文。超过多数投票之后，debate topology 本身也会决定结果。MultiAgentBench（arXiv:2503.01935, ACL 2025）比较了 star / chain / tree / graph 这些协调结构，发现 **graph 最适合 research**，但当 agent 数量超过大约 4 个后会出现明显的 “coordination tax”。AgentVerse（ICLR 2024）则记录了两类涌现行为：volunteer behaviors 与 conformity behaviors。后者既可能帮助形成共识，也可能把系统带进 groupthink。本课会把拓扑空间完整铺开，分别实现这些变体，并测量 coordination tax。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 16 · 07（心智社会与辩论），阶段 16 · 14（共识与 BFT）
**Time:** 约 75 分钟

## 问题

Debate 可能提高准确率（Du et al., arXiv:2305.14325），也可能把它拉低。决定 debate 是否有效的，通常不是一句“多搞几个 agent”就够了，而是四个结构性选择：

1. 谁和谁说话，也就是 topology。
2. 进行多少轮，也就是 rounds；Du 2023 的结论是轮数和 agent 数量是彼此独立的重要变量。
3. agents 是否异质，也就是是否使用不同 base models。
4. 是否存在 adversarial voice，例如是真正做 steel-manning，还是只是 straw-manning。

很多团队会在任务上直接“跑 5 个 agent 然后投票”，结果反而不如一个单 agent。失败并不是随机的，它通常能稳定追溯到 topology 与 heterogeneity。本课就是这张 topology 地图。

## 概念

### 自洽性：单模型基线

Wang et al. 2022 在 “Self-Consistency Improves Chain of Thought Reasoning” 中，做的是：对同一个模型，在 temperature > 0 的情况下采样 N 次，然后对推理路径对应的答案做多数投票。GSM8K 的结果表明，N=40 的 self-consistency 相比单次 greedy decode 有明显提升。它可以视为多代理投票出现之前的单 agent 前身。

但它的限制也很明确：self-consistency 使用的是同一个 base model，因此错误天然相关。如果这个模型存在系统性偏差，那么所有 N 个样本都会共享这类偏差。

### 多智能体投票：异质扩展

把 N 个样本换成 N 个 *不同的* agents。它们可以来自不同的 base models（Claude、GPT、Llama），也可以使用不同 prompts、不同工具权限、不同上下文来源。这样做的收益是不相关错误更多，代价则是不同 agent 的调用成本不同，协调它们本身也会带来开销。

到了 2026 年，异质辩论常见的一个名称是 **A-HMAD**，即 Adversarial Heterogeneous Multi-Agent Debate。它不是完全统一的正式术语，但很多论文会用它来表示“不同模型之间进行辩论，以降低 monoculture collapse 带来的相关错误”。

### 四种典型拓扑

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star：一个 hub，其他节点都只和 hub 对话。它相当于没有 back-channel 的 supervisor-worker。
Chain：线性结构，每个 agent 只看到前一个 agent 的输出，更像 pipeline。
Tree：层级结构，常见于 hierarchical agent systems（Lesson 06）。
Graph：任意节点可互连，既可以是 fully connected clique，也可以是一般 DAG。

### 协调成本（MultiAgentBench）

MultiAgentBench（MARBLE, ACL 2025, arXiv:2503.01935）在 research、coding、planning 等任务上比较了 star、chain、tree、graph。关键经验结果是：

- **Graph** 在 research tasks 上最好，因为信息可以 any-to-any 流动，agents 可以彼此交叉批评。
- **Star** 在快速事实问答上最好，因为 hub 能承担过滤与整合。
- **Chain** 在 stepwise pipeline 上最好，因为它天然匹配 staged refinement。
- **Coordination tax** 会在 graph topology 超过大约 4 个 agents 后出现，wall-clock 与 token 成本增长速度会快于质量提升。

这个 4-agent ceiling 不是理论极限，而是一个 2026 经验现象：随着大家互相可见，每个 agent 的上下文会迅速被 peers 的输出填满，而新增 agent N+1 的边际收益会很快下降。

### “Should we be going MAD?”

arXiv:2311.17371 是 2023 年对 MAD（Multi-Agent Debate）策略的代表性综述。后来很多工作都重复验证了它的一个关键发现：如果 MAD 结构上只是“自洽性 + 更贵”，也就是 independent sampling 加 aggregation，但预算相同，那么它经常还不如 self-consistency。MAD 真正开始有优势，是在 agents 具备真实 heterogeneity，且 debate 具有某种 adversarial structure 的情况下，例如明确安排某个 agent 负责反对意见。

### AgentVerse 的涌现行为

AgentVerse（ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）指出，即便没有显式设计，多代理辩论也会自然长出两类行为：

- **Volunteer。** 某个 agent 主动提出“我可以接下一步”。这类行为通常有利，因为它会把子任务分配给最有能力处理的人。
- **Conformity。** 某个 agent 会主动调整立场去贴近批评者，即便批评者本身是错的。

Conformity 正是为什么“辩到一致为止”很容易奖励 bullies。限制轮数并引入独立 judge，通常是更稳妥的做法。

### 异质性：真正能推动准确率的调节因素

2024-2026 年实践文献里很稳定的一个模式是：把 N 个 agents 中的一个换成不同 base model，带来的准确率提升，常常比在原模型上再额外加 1 个 agent 更大。直觉很简单：每增加一个真正独立的错误源，比增加一个高度相关的样本更有价值。

在极限情况下，heterogeneity 比 numerosity 更重要。对大多数有清晰 ground truth 的任务来说，三个不同模型往往优于五个同模型副本。

### 陪审团方法

Sibyl 框架在 Minsky-LLM 相关文献里形式化了“jury”方法：不是单纯多数投票，而是一个带角色分工的小型 panel。比如一个 agent 负责 cross-examine，一个负责补充 context，一个负责给 plausibility 打分。Jury methods 可以看作介于 plain vote 与 full MAD 之间的折中方案：比纯投票更强，也比完整多轮辩论更便宜、更不容易陷入 conformity。

### 何时值得结合投票与辩论

- 问题存在 ground truth，例如事实、数学或代码行为。
- agents 能访问不同来源或不同 tools，异质性是真实存在的。
- rounds 被明确限制在 2-3 轮，并且存在独立 judge 或 verifier。
- 预算允许 3-5 个 agents；超过 5-7 个 graph 节点后，coordination tax 往往开始主导。

### 何时结合投票与辩论反而有害

- 问题本质上是意见型，而不是事实型。大家最后会收敛到“最像正确答案的说法”，而不是最正确答案。
- 所有 agents 共享同一个 base model，monoculture 会让共识失去意义。
- rounds 无界，最后 conformity 一定会压过真实分歧。
- 任务本身很简单。一个 single agent 配合 N=5 的 self-consistency 通常更便宜，也一样准。

```figure
sw-debate-topology
```

## 动手构建

`code/main.py` 实现了：

- `run_star(agents, hub, question)`：hub 轮询所有 worker 并聚合结果。
- `run_chain(agents, question)`：串行精炼。
- `run_tree(root, children, question)`：深度为 2 的层次聚合。
- `run_graph(agents, question, rounds)`：bounded rounds 的 all-to-all debate。
- 一个脚本化 heterogeneity dial：每个 agent 都带一个 `error_bias`，表示它系统性出错的方向。
- 一个 measurement harness：对每种 topology，在 N=3、5、7 下运行，并报告 accuracy、total_tokens 与 wallclock_simulated。

运行：

```
python3 code/main.py
```

预期输出是一张 topology × N → (accuracy, tokens, latency) 表。Graph 在 N=3-5 的 research-style tasks 上通常最好；star 在 fast-factual tasks 上更有优势；而 N=7 的 graph 会明显暴露 coordination tax，也就是延迟膨胀快于准确率增长。

## 实际使用

`outputs/skill-topology-picker.md` 是一个根据任务描述来推荐 topology 的技能：它会给出 star / chain / tree / graph 中的选择、推荐 N（agent 数量）、建议的 heterogeneity profile（选哪些 base models），以及 round bound。

## 交付上线

对于任意 ensemble：

- 先从 **self-consistency at N=5** 开始，用一个强 base model 做最低成本基线。
- 如果准确率真的关键，再升级到 **heterogeneous voting at N=3**，并测量增量收益。
- 只有当任务确实具备结构性，例如 research 或 multi-step workflow，且 bounded rounds 可行时，才值得升级到 **debate topology**。
- 一定要记录 minority cluster。一个长期“少数但经常正确”的 minority，本质上就是 diversity signal。
- 评估时别只看 accuracy，也要一起看 wall-clock 与 tokens。“10 倍成本换一点点准确率”是否值得，是业务决策。

## 练习

1. 运行 `code/main.py`，画出 graph topology 的 coordination-tax 曲线：accuracy vs N、tokens vs N。曲线在哪个 N 开始拐弯？
2. 实现 A-HMAD：三个 agents 带有刻意不同的 biases。它与 Lesson 14 里的 monoculture attack 基线相比表现如何？
3. 给 graph topology 增加一个不投票、只对最终 consensus 打分的 “judge” 角色。它会改变 emergent conformity 吗？
4. 阅读 AgentVerse 论文。判断你的实现更强地表现出哪一种 emergent behavior。能否通过 prompt 改动诱发相反行为？
5. 阅读 MultiAgentBench（arXiv:2503.01935）第 4 节的 topology experiments。用你的 harness 在论文任务之一上复现 “graph-wins-research” 结论。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 自洽性 | “采样 N 次，再投票” | Wang 2022：单个模型在 temperature > 0 时采样 N 次，再对推理路径的答案做多数投票。 |
| 异质性 | “不同模型” | 由不同基座模型或提示家族组成的集成，用来打破单一文化。 |
| MAD | “多智能体辩论” | 多个智能体在若干轮中交换批评意见的统称。 |
| A-HMAD | “对抗式异质多智能体辩论” | 强调不同模型与对抗式结构的 MAD 变体。 |
| 拓扑 | “谁与谁交流” | 星形、链形、树形和图形结构，决定信息如何流动。 |
| 协调成本 | “收益递减” | 图形拓扑上的智能体超过约 4 个后，成本增长快于质量提升。 |
| 主动协助行为 | “未经提示便提供帮助” | AgentVerse 描述的涌现模式：智能体主动承担下一步。 |
| 从众行为 | “在压力下趋同” | AgentVerse 描述的涌现模式：智能体向批评者的观点靠拢。 |
| 陪审团 | “小型专家小组” | Sibyl 风格的小型角色化团队，例如质询者、上下文提供者和评分者。 |

## 延伸阅读

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — 单模型基线
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — agents 与 rounds 都是独立重要变量
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark，graph 适合 research，chain 适合 pipelines
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD 策略综述；等预算下 MAD 往往不如 self-consistency
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — volunteer 与 conformity 两类涌现行为
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — 基准实现参考
