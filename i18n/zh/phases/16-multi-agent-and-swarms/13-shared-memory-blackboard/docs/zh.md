# 共享记忆与黑板模式

> 到了 2026 年，多代理系统里主要并存两条路线：**message pool**（所有人都能看到所有人的消息，例如 AutoGen GroupChat 或 MetaGPT），以及 **blackboard with subscription**（agent 只订阅与自己相关的事件，例如 Context-Aware MCP 或 Matrix framework）。这两者几乎就是多代理系统里唯一真正“有状态”的部分，因此也正是最容易藏复杂 bug 的地方。这里最典型的失败模式叫 **memory poisoning**：某个 agent 幻觉出了一个“事实”，其他 agent 把它当成已验证信息继续传播，准确率会以一种远比立即崩溃更难排查的方式缓慢衰退。本课会用 stdlib 分别实现这两种结构，注入一次 poisoning attack，并展示在生产环境里真正有效的三种缓解手段。

**Type:** 学习 + 构建
**Languages:** Python（标准库，`threading`）
**Prerequisites:** 第 16 阶段 · 04（原语模型），第 16 阶段 · 09（并行 / Swarm / 网络化架构）
**Time:** 约 75 分钟

## 问题

多代理系统需要一个地方让 agent 共享事实。最直观的方式是“什么都通过消息传”，但这其实是在用额外复制重新发明共享状态。另一种方式是“给所有人一个全局日志”，但全局日志会无限增长，也非常容易被污染。第三种方式是“给每个 agent 投射一个定制视图”，它更可扩展，但会对 schema 设计提出更高要求。

一旦某个 agent 产生幻觉，并把这条幻觉写进共享状态，后续所有读取这块状态的下游 agent 都会把它当成事实。等到人类察觉时，推理链可能已经深入了五层，而根因只是第三条写入的消息。调试多代理系统里的准确率衰退，通常比调试一次 crash 要难得多。

这就是 memory poisoning。根据 MAST taxonomy（Cemri et al., arXiv:2503.13657），它属于记录最充分的失败家族之一，而且它是结构性的：任何没有 provenance、又没有只读 verifier 的共享记忆设计，最终都会暴露出这一问题。

## 概念

### 两种主要拓扑

**Full message pool。** 每个 agent 都读取每一条消息。AutoGen GroupChat 和 MetaGPT 都采用这一思路。它的优点是简单、透明、容易检查；缺点是很难扩展到大约 10 个以上 agent，因为每个 agent 的上下文都会被其他人的工作塞满。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription。** 每个 agent 声明自己关心哪些 topics，底层系统只把相关消息路由给对应订阅者。CA-MCP（arXiv:2601.11595）和 Matrix 去中心化框架（arXiv:2511.21686）采用这一路线。它扩展性更好，但前提是你要先把 schema 设计清楚，让“订阅 topic”真正有意义。

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### 各自适合什么场景

- **Full pool** 适合 agent 数量少于 10、角色差异较大、会话跨度较短的场景。因为所有人都看到全部上下文，所以“谁说了什么”非常容易追踪。
- **Blackboard** 适合 agent 数量很多、角色较同质但实例数量庞大（例如 swarms）、任务周期较长的场景。路由机制能明显节省 token 成本并减少上下文污染。

很多生产系统会混合使用：上层规划层用一个较小的 full pool，下层 worker 层则用 blackboard。

### 一个记忆污染场景

假设三名 agent 正在做一个研究任务。Agent A 负责检索，Agent B 负责总结，Agent C 负责分析。

1. A 抓取网页后向共享状态写入：“这项研究报告了 42% 的准确率提升。”
2. 实际网页写的是 “4.2% improvement”，A 把小数点幻觉掉了。
3. B 读取共享状态后写入：“报告显示存在显著的 42% 准确率提升（来源：A）。”
4. C 继续读取共享状态后写入：“建议采用该方案，42% 的提升具有变革性。”
5. 最终报告引用了一个根本不存在的 42% 数字。

整个过程中，没有任何 agent 崩溃，没有任何测试失败，系统表面上“正常工作”。但这条幻觉通过共享状态，从 A 的上下文一路洗白成了所有下游 agent 的“事实”。

### 为什么这是结构性问题

如果没有共享状态，A 的幻觉最多停留在 A 的本地上下文里。其他 agent 可能会重新检索、重新推导，并在某个环节发现错误。但如果共享状态设计得过于天真，A 的上下文就会直接变成所有人的上下文，幻觉也就被洗成了事实。

问题并不在于共享状态本身，而在于共享状态**缺少来源信息，也缺少独立验证者**。真正有效的三种缓解手段是：

1. **每次写入都记录来源。** 共享状态中的每条记录都应包含谁写的、何时写的、基于什么提示，以及引用了什么来源。下游智能体应根据来源信息审慎读取。
2. **所有写入都仅追加并做版本化。** 更正不应覆盖原条目，而应作为替代旧版本的新条目追加写入，从而保留审计轨迹。
3. **至少保留一个不能写共享状态的智能体。** 这个只读验证者会抽样读取共享状态、重新获取来源并标记不一致。由于它没有写权限，因此不会把自身判断回灌到共享池。

### 黑板模式的历史先例（Hayes-Roth，1985）

黑板模式比 LLM agent 早了四十年。Hayes-Roth 在 1985 年的 “A Blackboard Architecture for Control” 中提出了 specialist Knowledge Sources：它们观察全局 blackboard、贡献部分解，并进一步触发其他知识源。2026 年的 blackboard（CA-MCP、Matrix）本质上还是同一个模式，只不过 Knowledge Sources 变成了 LLM agents，部分解变成了 JSON blobs。旧文献对写入争用、机会式控制、一致性维护的讨论，今天依然适用。

### 投影视图与完整视图

纯 blackboard 给所有订阅者的是相同的 topic-scoped 投影视图。更激进的设计则是 **per-agent projection**：每个 agent 都拿到一份专门按自己角色裁剪过的视图。LangGraph 的 state reducers 就是 2026 年这一思路的经典实现，它通过 reducer 把全局状态折叠成角色特定切片。

这种 per-agent projection 的扩展性更强，但它依赖 schema。没有 schema，你最后只会在每个 agent 的 prompt 里手工重复构建这些投影视图。

### 写入争用模式

多个 agent 同时写共享状态，本质上是并发问题，而不只是 LLM 问题。通常有三种可行模式：

- **Sequential writer（single producer）。** 所有写入都经由一个协调 agent 串行化。简单，但容易形成瓶颈。
- **Optimistic concurrency with versioning。** 每个 entry 带版本号；版本不匹配则写失败并重试。这是经典数据库策略。
- **Topic partitioning。** 不同 agent 各自拥有不同 topics，没有跨 topic 冲突。但这要求你先设计好边界。

大多数 2026 框架默认使用 sequential writer，因为 LLM 调用本来就慢，写冲突通常没那么频繁，瓶颈反而不明显。

### 只读验证者

这里最关键的缓解措施其实就是只读 verifier。实现上需要满足几条规则：

- Verifier 与团队共享状态，也就是它能读取 blackboard 或 message pool。
- Verifier 没有写共享状态的句柄，它只能向单独的 verification channel 输出。
- Verifier 会独立重取各条写入里引用的 source，并在发现矛盾时发出标记。
- Verifier 的输出应该送给人类或单独的决策 agent，而不是重新喂回共享池。

如果做不到这种隔离，verifier 的输出也会变成共享池中的新输入，最终结果就是：被污染的 pool 污染 verifier，verifier 再污染自己的验证输出。

```figure
swarm-blackboard
```

## 动手构建

`code/main.py` 用 stdlib Python 同时实现了两种拓扑、一个 toy poisoning attack，以及前面提到的三种缓解手段。

- `MessagePool`：线程安全、append-only 的全量日志。
- `Blackboard`：按 topic 做 pub/sub，并支持每个 agent 的 subscriptions。
- `ProvenanceEntry`：每次写入都会记录 writer、timestamp、prompt_hash、source_uri。
- `PoisoningScenario`：运行一个三 agent 研究流程，其中 agent A 幻觉掉一个小数点，并打印最终报告。
- `Verifier`：一个只读 agent，会重新抓取来源并标记不一致，然后在 verifier 存在的情况下再跑同一场景。

运行：

```
python3 code/main.py
```

预期输出：
- Run 1（no verifier）：幻觉的 42% 会一路传播进最终报告。
- Run 2（with verifier）：verifier 会标出不一致，pool 被标记为 “flagged”，最终报告会包含撤回说明。

## 实际使用

`outputs/skill-memory-auditor.md` 是一个审计技能，用来检查任意多代理系统的 shared-memory 设计是否具备 provenance、versioning 与 verifier separation。在新的多代理架构上线前，应该先跑一次这种审计。

## 交付成果

对于任何 shared-memory 设计：

- 每次写入都记录 provenance：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 让日志保持 append-only。所有更正都以引用旧条目的新 entry 形式追加。
- 至少部署一个具备独立 source access 的只读 verifier agent。
- 把 verifier 的输出路由到独立 channel，而不是回灌进共享 pool。
- 记录 supersession 写入的比例。这个比例上升，往往是 hallucination pattern 的早期信号。

## 练习

1. 运行 `code/main.py`，确认 run 1 会传播幻觉，run 2 能抓到它。
2. 再增加一个幻觉：让 agent B 编造一个 dataset size。Verifier 应该在不针对任何单独错误手工调参的情况下抓到这两类问题。
3. 把 full pool 改成按 topic 分区的 blackboard（`prices`、`summaries`、`analyses`）。哪些 poisoning 场景会因此更难实现，哪些则几乎不受影响？
4. 阅读 Hayes-Roth（1985, “A Blackboard Architecture for Control”）。指出文中两个本课没有展开、但 2026 系统仍值得借鉴的控制模式。
5. 阅读 CA-MCP（arXiv:2601.11595）。把它的 Shared Context Store 映射到 `code/main.py` 中的 MessagePool 或 Blackboard。CA-MCP 在这两个原语之上又增加了哪些机制？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 消息池 | “共享聊天历史” | 所有智能体都能读取的仅追加全局日志；透明，但扩展性差。 |
| 黑板 | “共享工作区” | 按主题划分的发布／订阅系统；智能体只订阅相关主题，扩展性更好。 |
| 来源信息 | “谁写了什么” | 每次写入的元数据，包括写入者、时间戳、提示和来源。 |
| 记忆污染 | “幻觉扩散” | 一个智能体的错误进入共享状态，下游智能体将其当作事实继续传播。 |
| 仅追加 | “不做原地更新” | 更正通过新增替代条目实现，而不覆盖原条目。 |
| 只读验证者 | “独立审计员” | 只读智能体，会重新获取来源并标记不一致。 |
| 投影视图 | “限定范围的视图” | 从全局状态中为每个智能体计算出的专属视图；LangGraph Reducer 是典型实现。 |
| 知识源 | “专家智能体” | Hayes-Roth 在 1985 年对黑板参与者使用的术语。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；memory poisoning 属于 coordination-failure 子类
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — 面向协同 MCP servers 的 Shared Context Store
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — 没有中心 orchestrator 的消息队列式 blackboard
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 生产环境中的 per-agent projection 模式
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 来自真实部署的 provenance 与 verification 实践
