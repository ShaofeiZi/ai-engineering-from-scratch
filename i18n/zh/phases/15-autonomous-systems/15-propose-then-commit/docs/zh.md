# 人类在环：Propose-Then-Commit

> 到 2026 年，HITL 的共识已经非常具体。它不再是“代理发起请求，用户点击 Approve（批准）”这么简单，而是先提案后提交（propose-then-commit）：拟议动作先被持久化到持久存储（durable store），并绑定一个幂等键（idempotency key）；随后它会带着意图（intent）、数据血缘（data lineage）、触及权限（permissions touched）、影响范围（blast radius）和回滚方案（rollback plan）展示给审核人（reviewer）；只有得到明确正向确认之后才允许真正提交；执行完成后还要回读验证，确认副作用确实发生。LangGraph 的 `interrupt()` 加 PostgreSQL checkpointing、Microsoft Agent Framework 的 `RequestInfoEvent`，以及 Cloudflare 的 `waitForApproval()`，实现的都是同一个形状。这个模式最经典的失败方式，是橡皮图章式审批（rubber-stamp approval）：用户看到“Approve?（要批准吗？）”就机械点击。被反复记录的缓解手段，则是带明确检查项的质询式确认（challenge-and-response）。

**Type:** 学习
**Languages:** Python（stdlib，带幂等性的 propose-then-commit 状态机）
**Prerequisites:** 阶段 15 · 12（持久执行），阶段 15 · 14（跳线与诱饵）
**Time:** 约 60 分钟

## 问题

代理准备执行一个动作，而用户必须决定：批准，还是拒绝。如果这个决定几乎是瞬时做出的，那它大概率根本不算审查；如果这个决定是结构化的，它会慢一些，但也更可信。真正的工程问题是：如何让“结构化审查”成为阻力最小的默认路径。

2023 年前后的 HITL 模式，通常只是一个同步弹窗：“智能体想向 X 发送一封正文为 Y 的邮件，是否批准？”用户点一下“批准”，所有人都觉得系统很安全。但实际运行中，这种界面极易流于橡皮图章式审批：用户批得很快，审批本身几乎没有预测力；等代理真的出错时，审计轨迹里只会留下长长一串用户自己都回忆不起来的批准记录。

2026 年的模式，也就是 propose-then-commit，把 HITL 放到了持久化底座上，绑定结构化元数据，并要求明确的正向提交。每个托管代理 SDK 都有自己的名字：LangGraph 用 `interrupt()`，Microsoft Agent Framework 用 `RequestInfoEvent`，Cloudflare 用 `waitForApproval()`。API 名字不同，形状完全一样。

## 概念

### Propose-then-commit 状态机

1. **Propose（提案）。** 代理产生一个拟议动作，并把它持久化到持久存储，例如 PostgreSQL、Redis 或 Durable Object。记录中至少要包含：
   - intent（意图）：代理为什么要这样做
   - data lineage（数据血缘）：是哪段源内容导致了这个提案
   - permissions touched（触及权限）：会触及哪些 scope、文件或 endpoint
   - blast radius（影响范围）：最坏情况下影响有多大
   - rollback plan（回滚方案）：如果已经提交，如何撤销
   - idempotency key（幂等键）：每个提案唯一；重复提交返回同一条记录
2. **Surface（展示）。** 审核人（reviewer）会看到带完整元数据的提案。审核人必须是一个人，而不是代理自审。
3. **Commit（提交）。** 得到明确正向确认后，动作才真正执行。
4. **Verify（验证）。** 执行后回读目标，确认副作用确实发生。若验证失败，系统就进入一个“已知坏状态”，需要触发告警。

### Idempotency key

没有 idempotency key 时，一次短暂失败后的重试（retry）就可能把已经批准的动作执行两遍。一个具体例子是：用户批准“从 A 向 B 转 100 美元”；此时网络闪断；工作流（workflow）重试；用户其实只批准了一次，但转账却执行了两次。Idempotency key 会把这次批准和“唯一的一次副作用”绑定在一起，因此第二次执行会被识别为空操作（no-op）。

这和 Stripe、AWS API 采用的 idempotency 模式完全是同一种思路。Microsoft Agent Framework 文档也明确把它作为 agent approval 的基础机制。

### Durability：为什么审批要比进程更长寿

等待审批的那个“等待室（waiting room）”是一段代理并不拥有的状态。工作流（workflow）会在此处暂停（见第 12 课）。等审批真的到达时，工作流会从那个点精确恢复。这就是为什么 LangGraph 会把 `interrupt()` 和 PostgreSQL checkpointing 绑定在一起，而不是只靠内存状态。即使审批两天后才到，工作流也仍然能完好地接住它。

### 橡皮图章式审批与质询式确认缓解

HITL 的默认界面（UI），如果只是两个按钮“Approve / Reject（批准 / 拒绝）”，通常只会制造快速批准，而不是真正审查。被反复记录的缓解方式，是一份质询式确认清单（challenge-and-response checklist）：只有在审核人（reviewer）对若干具体问题给出明确肯定回答后，Approve 按钮才会解锁。典型问题可以是：

- “你理解这次操作会触及什么资源吗？”
- “你确认 blast radius 是可以接受的吗？”
- “如果这次动作失败，你已经有 rollback plan 吗？”

这不是为了形式主义增加官僚流程，而是一个强制触发机制（forcing function）。审核人如果连这些框都无法勾选，就应该要求澄清，或直接拒绝。Anthropic 的 agent-safety 研究也明确把“由清单驱动的 HITL”（checklist-driven HITL）列为对抗 rubber-stamp approval 的缓解手段。

### 什么算后果性动作（consequential action）

并不是每一个动作都需要 propose-then-commit。2026 年比较稳定的指导原则是：

- **后果性动作（Consequential actions）**：始终要走 HITL。例如不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **可逆动作（Reversible actions）**：有时需要 HITL。例如本地文件编辑、staging 环境改动、具备清晰回滚路径的可逆写入。
- **读取与检查（Reads and inspections）**：不需要 HITL。例如读取文件、列出资源、调用只读 API。

### 提交后的验证

“提交逻辑跑完了”并不等于“副作用真的发生了”。网络分区和竞争条件都可能让 workflow 误以为成功，而后端实际上没有持久化任何结果。Verify 这一步的意义，就是在 commit 之后重新读取目标资源进行确认。它和数据库里的 `RETURNING` 子句，或者 AWS 用 `GetObject` 在 `PutObject` 之后做回读验证，本质上是同一个模式。

### EU AI Act Article 14

EU AI Act Article 14 要求对欧盟高风险 AI 系统实施**有效的人类监督（effective human oversight）**。这里的“effective”不是装饰词。监管语境本身就明确排斥 rubber-stamp 这种装样子的审批模式。结合 challenge-and-response 的 propose-then-commit，才是 Microsoft Agent Governance Toolkit 合规文档里能够经得住 Article 14 审查的形状。

```figure
mx-propose-then-commit
```

## 用它

`code/main.py` 用 stdlib Python 实现了一个 propose-then-commit 状态机。持久存储（durable store）是一个 JSON 文件。幂等键（idempotency key）由 thread_id 和 action_signature 的哈希组成。驱动程序会模拟三种场景：一次正常的审批流程，一次短暂失败后的重试（retry，不得双重执行），以及 rubber-stamp 默认模式与 challenge-and-response 模式之间的对比。

## 交付成果

`outputs/skill-hitl-design.md` 用来审查一个拟议 HITL 工作流（workflow）是否具备 propose-then-commit 的完整形态，并标出缺失项，例如缺少元数据、缺少幂等性（idempotency）、缺少验证（verify），或缺少 challenge-and-response 层。

## 练习

1. 运行 `code/main.py`。确认一次已批准提案的 retry 会复用 durable record，而不是重复执行。然后把 idempotency key 改成包含 timestamp，展示 retry 如何导致双重执行。

2. 给 proposal record 加上一个 `rollback` 字段。模拟一次执行成功但 verify 失败的场景，并展示 rollback 会被自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。找出其中一个 toy engine 尚未包含的 metadata 字段，把它补进去，并解释它防御了什么风险。

4. 为一个具体动作设计 challenge-and-response checklist，例如“向公开 Twitter 账号发帖”。reviewer 必须回答哪三个问题？为什么偏偏是这三个？

5. 选一个同步“Approve?” 提示已经足够的场景，也就是不需要 durable store。解释为什么成立，并说明你在接受什么风险类别。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Propose-then-commit | “先提案后提交” | 先持久化 proposal，再 commit，并在之后 verify |
| Idempotency key | “可重试安全键” | 每个 proposal 唯一；第二次执行会变成空操作（no-op） |
| Data lineage | “来源链路” | 触发 proposal 的具体源内容 |
| Blast radius | “影响范围” | 动作出错时影响的范围 |
| Rubber-stamp | “走过场式批准” | 没有真正审查就点击了“Approve” |
| Challenge-and-response | “带质询的检查清单” | 审核人（reviewer）必须对具体问题做明确确认 |
| RequestInfoEvent | “Microsoft Agent Framework 请求原语” | 带结构化元数据的持久化 HITL 请求 |
| `interrupt()` / `waitForApproval()` | “框架原语” | LangGraph / Cloudflare 对同一形状的不同实现 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent` 与 durable approvals。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 与 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 把 HITL 视为长周期风险的缓解手段。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的人类监督监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督问题的 constitutional framing。
