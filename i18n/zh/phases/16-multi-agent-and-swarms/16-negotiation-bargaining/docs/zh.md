# 谈判与讨价还价

> Agents 会就资源、价格、任务分配和合作条款进行谈判。到 2026 年，相关 benchmark 的结论已经很清楚：NegotiationArena（arXiv:2402.05863）显示，LLM 可以通过 persona manipulation 把 payoff 提高约 20%；“Measuring Bargaining Abilities” （arXiv:2402.15813）则发现 buyer 比 seller 更难做，而且 scale 并不能解决结构性问题，他们提出的 **OG-Narrator**（deterministic offer generator + LLM narrator）把 deal rate 从 26.67% 推到了 88.88%；Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）运行了约 180k 次谈判，发现 **chain-of-thought concealment** 的 agent 会因为隐藏推理而更容易获胜；Bhattacharya et al. 2025 基于 Harvard Negotiation Project metrics 的排名中，Llama-3 最有效、Claude-3 最具攻击性、GPT-4 最公平。本课会实现 Contract Net Protocol（FIPA 的前身之一，Lesson 02），接上一组 LLM 风格 buyer / seller，再用 OG-Narrator 式分解跑实验，观察不同结构选择如何改变 deal rate。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 16 · 02（FIPA-ACL 渊源），阶段 16 · 09（并行群体网络）
**Time:** 约 75 分钟

## 问题

两个 agents 需要在价格上达成一致。若完全依赖自然语言 prompt，2024-2026 年的 LLM 在参数化较紧的 bargain 场景中，成交率其实低得惊人，大约只有 27%（见 arXiv:2402.15813）。更大的模型也没有从结构上解决这个问题：GPT-4 不一定比 GPT-3.5 更会“谈判”，它更多只是更会“说谈判的话”。

根本问题在于，LLM 把两个工作混在了一起：决定 offer 与叙述 offer。OG-Narrator 把这两件事拆开：由 deterministic offer generator 计算数值动作，再由 LLM 负责表达。结果是 deal rate 直接跃升到约 89%。

这和经典多代理研究中的结论一致：把 mechanism 与 communication layer 解耦，通常更稳。Contract Net Protocol（Smith, 1980；FIPA, 1996）就是经典的 task-market mechanism。你把 LLM 接在 narration 槽位上，本质上就得到一个现代化的、LLM 驱动的任务市场。

## 概念

### 一段话理解 Contract Net

Smith 在 1980 年提出的 Contract Net Protocol 是这样的：一个 **manager** 广播 **call for proposals (cfp)**；多个 **bidders** 返回 **propose** 消息，给出各自 offer；manager 选择一个 winner，向它发送 **accept-proposal**，同时给其他人发送 **reject-proposal**。winner 再去执行工作。还存在一个可选消息：**refuse**，表示 bidder 选择不参与提案。后来 FIPA 把这一交互模式标准化为 `fipa-contract-net`。

### 为什么 OG-Narrator 有效

“Measuring Bargaining Abilities of Language Models”（arXiv:2402.15813）指出了几个常见问题：

- LLM 经常违反 bargaining rules，例如报出完全没有意义的价格，或者忽略对方的 ZOPA。
- 它们不会做真正有策略的 anchoring，经常过早接受差的 first offer，或者给出象征性而非策略性的 counter-offer。
- scale 本身并不能修正这些问题。更大的模型只是把错误说得更像样。

OG-Narrator 的核心分解是：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

offer generator 可以是经典的 negotiation strategy，例如 Rubinstein bargaining model、Zeuthen strategy，或者一个简单的 tit-for-tat price concession 规则。LLM 只负责叙述，因此消息里会同时包含 deterministic price 与自然语言 framing。

成交率之所以会提升，是因为：
- 价格始终留在 bargaining zone 内。
- anchors 是策略性的，而不是情绪性的。
- LLM 回到了它真正擅长的地方：写作与表达。

### NegotiationArena 的发现

arXiv:2402.05863 是谈判 benchmark 的代表之一。它的几个 headline findings 很值得记住：

- LLM 通过采用 persona（例如 “I am desperate to sell this by Friday”）可以把 payoff 提高约 20%，说明 persona manipulation 是真实有效的策略。
- 公平或合作型的 agents 很容易被敌对型 agents 利用；想防守，必须明确进行 counter-posturing。
- 在大约 40% 的 benchmark scenarios 里，对称 pair-up 依然会收敛到明显不公平的结果。

这并不意味着“LLM 是糟糕的谈判者”，更准确的说法是：LLM 谈判得太像人类了，连那些容易被利用的部分也一起保留了下来。

### 隐藏思维链

Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）在多种策略上运行了约 180k 次谈判。最后胜出的 agent 有一个共性：它们会隐藏自己的推理。

- 如果某个 agent 在公开 scratchpad 里写出 “I will only go to $75; my reservation price is $70”，对手就能直接读取你的底线。
- 最优策略是私下计算，仅在 public channel 中输出报价与必要最小叙述。

这其实是经典博弈论在 2026 年的再次体现：暴露你的 private valuation，通常会伤害 payoff。LLM 天生不会意识到这一点，它们甚至很乐意把 reservation price 直接写进 reasoning trace，再被对手看到。

工程结论非常明确：private scratchpad context 与 public message context 必须严格分开，这不是可选优化，而是底线设计。

### Bhattacharya et al. 2025：模型风格排名

基于 Harvard Negotiation Project 的指标（原则性谈判、尊重 BATNA、利益互惠）：

- **Llama-3** 在达成 bargain 的有效性上表现最好，也就是 deal rate + payoff 最强。
- **Claude-3** 是最 aggressive 的谈判者，倾向高 anchor 与晚让步。
- **GPT-4** 是最 fair 的，pairing 间 payoff variance 最小。

这只是 2025 年的一个快照。重要的不是“2026 年 4 月到底谁最强”，而是不同 base models 会长期呈现稳定的 negotiation style，这本身就是一种 heterogeneity 来源（可与 Lesson 15 的 ensemble 设计结合）。

### 用 Contract Net + LLM 做任务分配

Contract Net 在 LLM 多代理里的现代复用方式通常是：

1. manager agent 先把任务拆成多个 unit。
2. 广播 `cfp`，把 task description 发给 worker agents。
3. 每个 worker 返回一个 offer，形式可能是 `(price, eta, confidence)`；price 可以是 tokens、compute units 或 dollars。
4. manager 选择 winner（也可以选多个，取决于任务）。
5. 被拒绝的 workers 可以继续去竞标其他任务。

这种模式可以扩展到 100+ workers，因为它本质上是 broadcast-and-respond，而不是同步群聊。Microsoft Agent Framework 的 orchestration patterns，以及一些 LangGraph 实现，都是这种思路。

### LLM-Stakeholders Interactive Negotiation

NeurIPS 2024 的工作（https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf）进一步把问题拓展到多方博弈：每个 stakeholder 都有 **secret scores** 与 **minimum-acceptance thresholds**，也就是私有 utility。LLM 必须从对话中推断这些隐藏偏好。它相当于把二方 bargaining 推广成 N 方 coalition formation，对真实生产里的 task market 很有参考价值。

### 表达与机制分离原则

综合 2024-2026 年的谈判 benchmark，最稳定的工程规则其实只有一句：

> 让 LLM 负责表达，不要让 LLM 计算报价。

如果 offer 是数值型的，例如 price、ETA、quantity，就应当由程序根据 negotiation state 进行 deterministic 计算，再让 LLM 负责 framing。若 offer 是结构性提案，例如 task decomposition 或 role assignment，可以让 LLM 起草，但在发送前必须用 schema 与 constraints 做验证。

```figure
a5-og-narrator
```

## 动手构建

`code/main.py` 实现了：

- `ContractNetManager`、`ContractNetTask`、`Bid`：manager + bidders，负责广播 cfp、收集 proposals、做 award。
- `og_narrator_bargain(state, rng)`：一个 OG-Narrator 风格 buyer，采用 deterministic 的 Zeuthen-style concession，逐步向中点让步。
- `seller_response(state, rng)`：一个 deterministic seller counter-offer policy，作为两种谈判风格共享的结构基线。
- `naive_llm_bargain(state, rng)`：模拟纯 LLM bargainer，它会以高方差报出价格，经常跑出 ZOPA。
- Measurement：在 1000 次试验中测量 deal rate，每次 trial 都重新采样 reservation prices。

运行：

```
python3 code/main.py
```

预期输出：naive-LLM 的 deal rate 大约在 65-75%，OG-Narrator 大约在 85-95%。这 15-25 个点的差距，正是将 offer generation 与 narration 解耦带来的结构性优势。脚本还会附带一个 Contract Net task-market allocation 示例，其中有三个 bidders 和一个任务。

## 实际使用

`outputs/skill-bargainer-designer.md` 用于设计 bargaining protocol：到底由谁生成 offer（deterministic 还是 LLM）、谁负责 narration、private scratchpad 如何与 public messages 隔离，以及应该如何监控 deal rate。

## 交付上线

生产谈判系统检查清单：

- **Separate scratchpad。** private state 绝不能出现在 counterpart 的上下文里。
- **Deterministic offer generation。** 对 price、quantity、ETA 等数值，应该计算，而不是 prompt。
- **Validate all incoming offers。** 在 protocol boundary 拒绝一切超出 ZOPA 的提案。
- **Bound rounds。** 轮数限制在 3-5 次；僵局时升级给 mediator。
- **持续测量 deal rate 与 payoff variance。** 成交率下降通常是症状，常见根因是 prompt drift 或对手侧攻击。
- **记录所有被拒绝的 proposals。** 对 Contract Net manager 来说，losing bidders 应该知道为什么失败。

## 练习

1. 运行 `code/main.py`，确认 OG-Narrator 在 deal rate 上优于 naive-LLM。差了多少？
2. 实现 **persona-based payoff improvement**（arXiv:2402.05863）：让 buyer 只在 narration 中采用 “desperate to buy this week” persona，而 offer generator 不变。deal rate 或 payoff 会改变吗？
3. 实现 chain-of-thought **concealment**：维护一个 private scratchpad string，但不把它传给 counterpart。若你故意泄露它，会发生什么？
4. 将 Contract Net 扩展到带 reserve price 的 N-bidder auction。当所有 bids 都高于 reserve 时，manager 应该如何在 lowest-price 与 highest-quality 之间做决策？你会选什么 award rule，为什么？
5. 阅读 Bhattacharya et al. 2025 关于 Harvard Negotiation Project metrics 的工作。实现两个不同风格的 bargainer（aggressive vs fair），测量它们在对称与非对称 pairing 下的 payoff variance。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 合同网 | “任务市场” | Smith 1980、FIPA 1996：由 cfp、propose 和 accept/reject 构成的经典任务市场机制。 |
| ZOPA | “可能达成协议的区间” | 买方最高接受价与卖方最低接受价的交集，区间外的报价无法成交。 |
| BATNA | “谈判协议的最佳替代方案” | 谈判失败时的最佳备选方案，它决定保留价格。 |
| OG-Narrator | “报价生成器 + 叙述器” | 一种职责分解：程序确定报价，LLM 负责自然语言表达。 |
| Zeuthen 策略 | “风险最小化让步” | 一种根据风险阈值决定让步幅度的经典策略。 |
| Rubinstein 讨价还价 | “交替报价均衡” | 带折扣因子的无限期讨价还价博弈论模型。 |
| 思维链隐藏 | “隐藏推理” | arXiv:2503.06416 中的获胜者隐藏私有草稿区，公开通道只发送报价。 |
| 人设操纵 | “情绪化姿态” | arXiv:2402.05863 指出，表现绝望或紧迫的人设可使收益提高约 20%。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — 代表性 benchmark；涵盖 persona manipulation 与 exploitation 现象
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator 与 buyer-harder-than-seller 结论
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — 约 180k 次谈判；CoT concealment 获胜
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 带 secret utilities 的多方可评分博弈
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — 经典机制，IEEE Transactions on Computers
