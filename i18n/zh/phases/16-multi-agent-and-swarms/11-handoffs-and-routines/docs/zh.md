# 交接与例程：无状态编排

> OpenAI 的 Swarm（2024 年 10 月）把多智能体编排归纳为两个原语：**例程（routines）**，即系统提示中的指令与工具；以及**移交（handoffs）**，即返回另一个 Agent 的工具。这里没有状态机，也没有分支 DSL；LLM 通过调用相应的移交工具完成路由。它在生产环境中的后继者是 OpenAI Agents SDK（2025 年 3 月）。Swarm 本身仍是最清晰的概念参考，因为全部源码只有几百行。其 API 大致可概括为“智能体 = 提示 + 工具；移交 = 返回智能体的函数”，这也使该模式迅速普及。限制同样清楚：系统无状态，因此记忆由调用方负责。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 16 阶段 · 04（原语模型）
**Time:** 约 60 分钟

## 问题

几乎每个多智能体框架都要求你先学习自己的 DSL：LangGraph 的节点和边、CrewAI 的团队和任务、AutoGen 的 GroupChat 和管理器。这些 DSL 确实提供了抽象，但也会让系统显得比实际需要更沉重。

Swarm 则走了相反的路线：直接利用模型已有的工具调用能力。移交被建模为工具调用；当前持有对话的智能体本身就是编排者。状态机不再显式存在，而是隐含在各智能体的系统提示中。

## 概念

### 两个原语

**例程。** 定义智能体角色和可用工具的系统提示。可以把它看成一组边界明确的指令：“你是分诊智能体；如果用户询问退款，就把会话移交给退款智能体。”

**移交。** 智能体可以调用的一种工具，该工具会返回新的 Agent 对象。Swarm 运行时检测到这个返回值后，会在下一轮切换当前活跃的智能体。

这就是整个抽象层。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊智能体的系统提示会根据用户消息选择正确的移交。路由由 LLM 的工具调用能力完成。

### 为什么它会迅速传播

- **API 很小。** 只需要学会两个概念。
- **直接利用模型已有的能力。** 工具调用在各家模型提供商中已达到生产级。
- **没有状态机负担。** 你不需要描述整张图；由各个 agent 的提示词决定它们应该交接给谁。

### 无状态的代价

Swarm 明确是无状态的。框架在一次运行期间会保留消息历史，但不会持久化任何内容。记忆、连续性、长时间运行任务，全部都变成调用方的责任。

在生产版本 OpenAI Agents SDK（2025 年 3 月）中，这正是最主要的变化之一：SDK 在保留移交原语的同时，加入内置会话管理、防护栏和追踪功能。

### 适合 Swarm / 移交的场景

- **分诊模式。** 前线智能体负责把用户转给相应专家。
- **基于技能的交接。** “如果任务需要代码，就交给编码者；如果需要研究，就交给研究者。”
- **短而有边界的对话。** 例如客服支持、FAQ 转工单、简单工作流。

### Swarm 在哪些地方会吃力

- **带共享记忆的长会话。** 移交会把会话状态切换为“新智能体的提示加历史消息”。如果没有调用方管理的记忆，就很难维护跨智能体的持久共享状态。
- **并行执行。** 移交一次只会切换一个活跃智能体。要并行执行，必须由调用方同时编排多个 Swarm 运行。
- **审计与重放。** 无状态运行很难被完全重放；LLM 的移交决策本身也不确定。

### OpenAI Agents SDK（2025 年 3 月）

这个生产级后继者新增了以下能力：

- **会话状态。** 跨多次运行保留线程。
- **防护栏。** 输入/输出验证钩子。
- **追踪。** 记录每一次工具调用和移交。
- **移交过滤器。** 控制移交时究竟传递哪些上下文。

移交原语得以保留，新增的是围绕它的生产级易用能力。

### Swarm 与 GroupChat

两者都依赖 LLM 驱动的路由，但它们在 **谁来决定下一位发言者** 这件事上完全不同：

- GroupChat：由外部选择器（函数或 LLM）选择下一位发言者。
- Swarm：由当前活跃智能体调用移交工具来选择继任者。

Swarm 是“智能体决定下一步”；GroupChat 是“管理器决定下一步”。Swarm 的决策体现在当前活跃智能体的工具调用中；GroupChat 的决策则位于 `GroupChatManager` 中。

```figure
sw-handoff-routing
```

## 动手构建

`code/main.py` 从零实现 Swarm：包含 Agent 数据类、移交机制（工具返回 Agent），以及能够检测智能体切换的运行循环。

演示中，分诊智能体会把请求分流给退款、销售或支持专家。每位专家都有自己的工具，运行循环会打印每一次移交。

运行：

```
python3 code/main.py
```

## 实际使用

`outputs/skill-handoff-designer.md` 用来为具体任务设计移交拓扑：需要哪些智能体、它们可以调用哪些移交，以及交接时应传递哪些上下文。

## 交付成果

检查清单：

- **Handoff 日志。** 每一次 handoff 都要写入 trace event，包括 from-agent、to-agent 和 context snapshot。
- **上下文转移规则。** 要明确 handoff 时传递什么：完整历史（代价高）、最后 N 条消息，还是摘要。
- **Handoff 上的 guardrail。** 如果 handoff 的目标 specialist 具有不同的工具权限，就必须做认证与校验，否则 prompt injection 可能会强行触发不该发生的 handoff。
- **循环检测。** 两个 agent 来回交接是常见失败模式；用一个简单的 last-K 环形检查就能检测出来。
- **后备 agent。** 如果 handoff 目标不存在，应该退回到一个安全的默认 agent。

## 练习

1. 运行 `code/main.py`，把请求分流到 refund agent。确认第二轮的活跃 agent 的确是 refund。
2. 增加一个循环检测规则：如果同样两个 agent 连续 3 次互相 handoff，就强制退出。顺便设计一个 fallback。
3. 阅读 OpenAI Agents SDK 中关于 handoff filter 的文档。实现一个 “summarize-on-handoff” 版本：在 incoming agent 接手前，由 outgoing agent 先把上下文压缩成项目符号摘要。
4. 对比 Swarm 的 handoff 与 GroupChatManager 的 selector。哪一种模式更容易被 prompt injection 利用，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出一个 Swarm 明确做出的设计决定，以及 OpenAI Agents SDK 对它是保留了还是修改了。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 例程 | “智能体提示” | 系统提示加工具列表，用于定义角色与可用移交。 |
| 移交 | “转给另一个智能体” | 当前活跃智能体可以调用的工具；它返回新的 Agent，运行时随后切换活跃智能体。 |
| 无状态 | “运行之间没有记忆” | Swarm 不持久化任何内容，记忆由调用方负责。 |
| 活跃智能体 | “现在由谁发言” | 当前持有对话的智能体，移交会改变它。 |
| 上下文转移 | “交接时带过去什么” | 规定接手智能体可看到哪些历史：完整历史、最后 N 条或摘要。 |
| 移交循环 | “智能体来回踢皮球” | 两个智能体不断互相移交的失败模式。 |
| OpenAI Agents SDK | “生产版 Swarm” | 2025 年 3 月推出的后继者；在移交原语之上加入会话、防护栏和追踪。 |
| 移交过滤器 | “交接边界上的闸门” | SDK 中用于检查和修改移交边界上下文的功能。 |

## 进一步阅读

- [OpenAI Cookbook：编排智能体，Routines 与 Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 这套模式的经典表述
- [OpenAI Swarm 仓库](https://github.com/openai/swarm) — 原始实现，仍适合作为概念参考
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 带 session 和 tracing 的生产级后继者
- [Anthropic 关于 Claude 中 handoff 的说明](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 如何通过 `Task` 采用类似 handoff 的模式
