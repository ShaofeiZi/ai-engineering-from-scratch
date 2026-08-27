# 群聊与发言者选择

> 共享会话式编排会把 N 个智能体放进同一个对话，由选择器函数（LLM、轮询或自定义函数）决定下一个由谁发言。这是涌现式多智能体对话的典型形式：智能体并不知道自己在某张静态图中扮演哪个节点，只会对共享消息池作出反应。AutoGen GroupChat 和 AG2 GroupChat 是这一模式的参考实现：AutoGen v0.2 的 GroupChat 语义保留在 AG2 分支中；AutoGen v0.4 则将其重写为事件驱动的 Actor 模型。Microsoft 于 2026 年 2 月把 AutoGen 转入维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月发布候选版）。GroupChat 原语在 AG2 与 Microsoft Agent Framework 中都得以保留，因此学会一次便能在各处应用。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 16 阶段 · 04（原语模型）
**Time:** 约 60 分钟

## 问题

静态图（LangGraph）在工作流已知时很好用。但真实对话并非静态：编码者有时会询问审查者，有时会询问研究者或写作者。若把所有可能的移交都硬编码，边的数量会迅速爆炸。真正需要的是让*智能体对共享消息池作出反应*，同时由一个函数决定下一位发言者。

这正是 AutoGen GroupChat 要解决的问题。

## 概念

### 形态

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个智能体都能看到每一条消息。每轮结束后，系统都会调用选择器函数来决定下一位发言者。

### 三种选择器

**轮询。** 按固定顺序轮转，具有确定性，成本随 N 线性增长，但完全不考虑上下文。因此，即使当前话题是法律审查，轮到编码者时它仍会发言。

**LLM 选择。** 调用一个 LLM 读取最近的共享池内容，并返回最适合的下一位发言者。它能理解上下文，但速度较慢，因为每轮都会额外增加一次 LLM 调用。这是 AutoGen 的默认做法。

**自定义。** 编写一个包含任意逻辑的 Python 函数。常见形式是“LLM 选择加兜底规则”，例如“编码者发言后，下一轮必须交给验证者”。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。某个智能体完成当前轮发言后，管理器会调用选择器并取得下一个智能体。循环会持续到满足终止条件为止。

### 终止

常见的终止方式有三种：

- **最大轮数。** 对总轮数设置硬上限。
- **“TERMINATE”标记。** 智能体可以输出一个哨兵消息，管理器看到后便停止。
- **目标达成检查。** 每轮运行一个轻量验证器，在任务完成时终止对话。

### 系谱：分叉与合并

2025 年初，Microsoft 开始围绕事件驱动的 Actor 模型对 AutoGen（v0.4）进行大规模重写。社区随后将 AutoGen v0.2 的 GroupChat 语义分支为 AG2，保留早期用户已经集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 进入维护模式，并把事件驱动的 Actor 模型合并进 **Microsoft Agent Framework**（2026 年 2 月发布候选版，现已与 Semantic Kernel 合流）。GroupChat 概念在两条演进路径中都继续存在，只是实现细节不同。对于兼容 v0.2 的代码，AG2 是更合适的上游。

### 什么时候适合 GroupChat

- **涌现式对话。** 不希望预先把每一种可能的下一发言者关系都连接成图边。
- **角色混合型任务。** 编码者会询问研究者，研究者会询问档案员，档案员还可能回头询问编码者；流程不是 DAG。
- **探索式问题求解。** 更像“头脑风暴会议”，而不是“装配流水线”。

### 它会在什么情况下失效

- **严格确定性。** LLM 选择器可能前后不一致：同样的提示在不同运行中会选出不同的下一发言者。
- **谄媚级联。** 智能体会向最自信、最强势的前一位发言者让步，需要在提示中明确反制。
- **上下文膨胀。** 每个智能体都要读取每条消息，10 轮后上下文便会非常庞大。可以使用投影视图（第 15 课）限制可见范围。
- **热门发言者。** 某个智能体因选择器偏爱其专长而反复被选中，逐渐主导整场对话。可以把发言均衡度作为选择器特征。

### 群聊与监督者模式

底层原语其实相同，只是默认假设不同：

- 监督者模式：一个智能体负责规划，其他智能体负责执行。选择器的本质是“询问规划者下一步怎么做”。
- 群聊：所有智能体地位平等；选择器是作用于共享池的函数。

两者都建立在第 04 课的四个原语之上。群聊默认采用 LLM 选择式编排和完整消息池共享状态。

```figure
swarm-speaker
```

## 动手构建

`code/main.py` 使用标准库从零实现 GroupChat，其中包含三个智能体（编码者、审查者、管理者），提供轮询与 LLM 选择两种变体，并使用 `TERMINATE` 标记作为终止条件。

演示会打印完整对话记录，以及选择器的决策轨迹。

运行：

```
python3 code/main.py
```

## 实际使用

`outputs/skill-groupchat-selector.md` 用来为具体任务配置 GroupChat selector：该选 round-robin、LLM-selected 还是 custom，以及 selector 应该读取哪些输入（recent messages、agent specialties、turn counts）。

## 交付成果

检查清单：

- **Max rounds cap。** 一定要有。典型任务通常设为 10-20 轮。
- **Speaker-balance metric。** 跟踪每个 agent 的发言次数；当失衡超过阈值时报警。
- **Termination token。** 使用 `TERMINATE`，或者专门安排一个 verifier agent。
- **Projection or scoped memory。** 大约 10 条消息以后，就该考虑给每个 agent 只看一个局部视图，避免 context bloat。
- **Selector logging。** 对于 LLM-selected 变体，既要记录 selector 的输入，也要记录它的选择，否则几乎无法调试。

## 练习

1. 运行 `code/main.py`。比较 round-robin 和 LLM-selected 下的对话。哪一个 agent 在两种模式下更容易主导发言？
2. 在 selector 中加入 “max-speaks-per-agent” 规则。它会怎样改变 transcript？
3. 实现一个 goal-reached termination：当 reviewer 返回 “approved” 时就停止。它有多常在 round cap 之前触发？
4. 阅读 AutoGen stable docs 里的 GroupChat 文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。指出 `GroupChatManager` 使用的默认 selector 是什么。
5. 阅读 AG2 repo（https://github.com/ag2ai/ag2），并将它的 v0.2 GroupChat 与 v0.4 的 event-driven 版本对比。v0.4 具体增加了什么属性（throughput、fault-tolerance、composability）？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | “智能体同处一个聊天室” | 共享消息池加选择器函数，是 AutoGen / AG2 的基础原语。 |
| 发言者选择 | “下一个由谁发言” | 负责选择下一个智能体的函数，可以采用轮询、LLM 选择或自定义逻辑。 |
| GroupChatManager | “会议主持人” | AutoGen 中持有选择器并驱动轮次循环的组件。 |
| ConversableAgent | “基础智能体” | AutoGen 中可以发送和接收消息的基础智能体类。 |
| 终止标记 | “停止词” | 用来结束聊天的哨兵字符串，通常是 `TERMINATE`。 |
| 热门发言者 | “一个智能体主导对话” | 选择器不断选中同一个智能体的失败模式。 |
| 上下文膨胀 | “消息池无限增长” | 每个智能体都读取所有历史消息，因此上下文会随轮数不断增长。 |
| 投影视图 | “限定范围的视图” | 为不同角色提供共享池的局部视图，用来缓解上下文膨胀。 |

## 延伸阅读

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 repo](https://github.com/ag2ai/ag2) — 社区延续的 AutoGen v0.2 分支
- [Microsoft Agent Framework docs](https://learn.microsoft.com/en-us/agent-framework/) — 合并后的后继框架，2026 年 2 月 RC
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor model 重写的细节
