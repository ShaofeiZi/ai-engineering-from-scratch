# 多代理辩论与协作

> Du 等人在 ICML 2024 的论文 “Society of Minds” 中提出：让 N 个模型实例先独立给出答案，再在 R 轮里相互批评、逐步收敛。这种方法可以提升事实性、规则遵守与推理质量。与此同时，稀疏通信拓扑在 token 成本上往往优于全连接 full mesh。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 05（Self-Refine 和 CRITIC）
**Time:** 约 60 分钟

## 学习目标

- 解释 debate protocol：N 个 proposer、R 轮交叉批评，最终收敛到一个共享答案。
- 说明为什么 debate 能提升 factuality、rule-following 和 reasoning。
- 解释 sparse topology：并不是每个 debater 都必须看到所有其他人。
- 用一个脚本化 LLM 实现 stdlib debate，同时提供 full-mesh 与 sparse 版本，并比较 token cost 与 accuracy。

## 问题

Self-Refine（Lesson 05）本质上是一个模型批评自己，容易陷入 groupthink。CRITIC（Lesson 05）则把批评建立在外部工具之上，但现实里并不是每次都有可靠外部工具可用。Debate 提供了第三条路：多个实例、彼此交叉批评，并通过分歧来逼近更好的答案。

## 概念

### Society of Minds (Du et al., ICML 2024)

- N 个模型实例针对同一个问题独立提出答案。
- 在 R 轮过程中，每个模型会读取其他模型的提案并加以批评。
- 模型根据收到的批评更新自己的答案。
- 完成 R 轮后，返回最终收敛的答案。

原始实验因为成本限制，使用的是 N=3、R=2。但在更难的问题上，例如 MMLU、GSM8K、Chess Move Validity、biography generation，随着 agent 数量和轮数增加，准确率会继续提升。

跨模型组合往往比单模型内部辩论更强，例如 ChatGPT 与 Bard 联合使用，效果优于两者单独运行。

### 稀疏拓扑

论文 “Improving Multi-Agent Debate with Sparse Communication Topology” （arXiv:2406.11776，2024-2025）指出：full-mesh debate 并不总是最优。像 star、ring、hub-and-spoke 这样的 sparse topologies，往往能在更低 token 成本下做到接近的准确率。其核心差异在于，每个 debater 只会看到部分 peers，而不是所有 peers。

这会直接影响成本：

- Full mesh N=5, R=3 = 5 × 3 = 15 proposals，每个提案都要读 4 个 peers，总计 60 次 critique ops。
- Star N=5, R=3（1 个 hub + 4 个 spokes）= 15 proposals，而 spokes 只读 hub，因此只有 12 次 critique ops。

### Debate 何时有效

- **Factuality。** 多个独立提案彼此交叉核对，可以降低 hallucination。
- **Rule-following。** 例如棋步合法性判断，一个模型漏掉规则，其他模型可能补上。
- **Open-ended reasoning。** 不同 framing 的并行推进，往往更容易逼近正确答案。

### Debate 何时会拖后腿

- **Latency-sensitive UX。** N × R 的串行轮次，会带来你可能承受不起的延迟。
- **Cost-sensitive scale。** 每个问题都要额外支付 N × R 的 token 成本。
- **Simple factual lookups。** 对于简单事实查询，一次检索通常比五轮辩论便宜得多。

### 2026 年的实际实现形态

- **Anthropic orchestrator-workers**（Lesson 12）：可以视为一种带综合步骤的 debate 变体。
- **LangGraph supervisor**（Lesson 13）：中心路由 + 专家代理，完全可以把 debate 实现成一个节点。
- **OpenAI Agents SDK**（Lesson 16）：agents 之间可以来回 handoff，实现迭代式批评。
- **Multi-agent evals**：把 debate 与 evaluator-optimizer 配在一起，用于生成评估信号。

### 这种模式常见的失败点

- **Convergence collapse。** 所有 agents 很快收敛到第一个错误答案上。缓解方式是强制保留若干轮 disagreement。
- **Hub failure。** 在 star topology 中，一个错误 hub 会污染所有参与者。可以轮换 hub，或使用多个 hub。
- **Prompt homogenization。** 所有 agents 用同一套 prompt，最后产出高度同质化。需要引入 prompt 多样性和/或模型多样性。

```figure
debate-converge
```

## 动手构建

`code/main.py` 实现了一个 stdlib debate 系统，包括：

- `Debater` 类，表示一个带有个人意见漂移的脚本化 LLM。
- `FullMeshDebate` 和 `SparseDebate` 两个 runner。
- 三道问题：一个偏事实、一个偏规则、一个偏推理。
- 指标包括：最终收敛答案、收敛所需轮数、总 critique ops 数量。

运行方式：

```
python3 code/main.py
```

输出会展示不同协议下的准确率与成本；在示例中，sparse 以更低成本在 3 道题里追平了其中 2 道。

## 如何使用

- **Anthropic orchestrator-workers**：适合做简单的 2–3 worker debate。
- **LangGraph**：适合做带 checkpoint 的状态化多轮 debate。
- **Custom**：适合研究型场景，或对正确性保障有特殊需求的系统。

## 交付成果

`outputs/skill-debate.md` 提供一个多代理辩论脚手架，支持配置 topology、N、R，以及 convergence rule。

## 练习

1. 实现一个 “forced disagreement” 规则：第 1 轮要求每个 debater 必须给出不同提案。观察它对收敛速度的影响。
2. 增加一个按置信度加权的聚合器：debater 返回 (answer, confidence)，aggregator 按 confidence 加权。看看是否有帮助。
3. 把其中一个 “agent” 换成一个观点不同的脚本化 LLM。异质性能否提升准确率？
4. 针对你的 3 道题，分别测量 full mesh 与 sparse 的 token cost。画出 cost vs accuracy。
5. 阅读 Society of Minds 论文。把玩具系统扩展到 N=5、R=3。什么地方会坏？什么地方会变得更好？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Debate | "Multi-agent critique" | N 个 proposer，经过 R 轮交叉批评后收敛 |
| Full mesh | "Everyone reads everyone" | 每个 debater 每轮都读取所有 peers |
| Sparse topology | "Limited peer view" | debater 只读取部分 peers |
| Hub-and-spoke | "Star topology" | 一个中心 debater，N-1 个 spokes 只读取 hub |
| Convergence | "Agreement" | debaters 最终收敛到同一个共享答案 |
| Society of Minds | "Du et al. debate paper" | ICML 2024 的多代理辩论方法 |

## 延伸阅读

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 经典多代理辩论论文
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — 稀疏通信拓扑的结果
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 把 orchestrator-workers 视作 debate 变体
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 单模型自我批评的对应路线
