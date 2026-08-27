# 检查点与回滚

> 图状态每发生一次迁移，都要持久化一次。某个工作进程（worker）崩溃后，它持有的租约（lease）会过期，新的工作进程会从最新检查点（checkpoint）接手。Cloudflare Durable Objects 可以把状态保留数小时甚至数周。propose-then-commit（第 15 课）要求为每个动作预先定义回滚（rollback）方案，动作执行后的验证则负责把整个闭环补齐。欧盟《AI Act》第 14 条要求高风险系统具备有效的人类监督，落到工程上，意味着检查点必须可查询、回滚必须演练过、审计轨迹必须在部署后依然存在。最尖锐的失败模式是：如果缺少幂等键（idempotency key）和前置条件检查（precondition check），那么一次瞬时故障后的重试，就可能把已经批准过的动作再执行一遍。真正能把这个问题抓出来的，是动作后验证（post-action verification）。

**Type:** 学习
**Languages:** Python（stdlib，检查点与回滚状态机）
**Prerequisites:** 阶段 15 · 12（持久执行），阶段 15 · 15（人类在环：Propose-Then-Commit）
**Time:** 约 60 分钟

## 问题

持久执行（durable execution，第 12 课）让崩溃后的 agent 可以恢复执行。propose-then-commit（第 15 课）让已经批准的动作具备可审计性。本课把两者接起来：如果某个已经批准的动作只执行了一半，随后崩溃并恢复，会发生什么？回滚（rollback）该在什么时候触发？又应该针对哪一份状态执行？

真实系统对这个问题的接线方式并不相同：

- **LangGraph** 会把每一次图状态迁移写入检查点（checkpoint）到 PostgreSQL。工作进程（worker）崩溃后，租约（lease）会释放，另一个工作进程会从最新检查点恢复。工作流在 `interrupt()` 处暂停，而这个暂停本身也是持久化的。
- **Cloudflare Durable Objects** 能按键（key）保留数小时或数周的状态。你可以把计算和已批准动作对应的存储放在一起。
- **Microsoft Agent Framework** 在工作流 API 里提供 `Checkpoint` 原语；重放配合幂等性（idempotency）可以覆盖重试路径。

无论实现细节怎样变化，真正可靠的组合始终是：幂等键（idempotency key，防止重复执行）+ 前置条件检查（precondition check，确认当前状态仍然与审批时一致）+ 动作后验证（post-action verify，确认副作用确实发生）+ 验证失败即回滚（rollback on verify-fail）。

## 概念

### 每一次迁移都要持久化

图状态迁移，是指工作流从一个具名状态进入另一个具名状态的任意一步。天真的实现只会在某几个“提交点”持久化；生产级实现会把每次迁移都持久化。代价只是多几次写入，但换来的可靠性提升非常大：重放可以落在任意位置，租约（lease）恢复也能精确到最近一步。

### Lease 恢复

当工作进程（worker）崩溃时，工作流本身并没有丢失；失效的只是租约（lease），也就是“当前工作进程正在执行这次运行（run）”的一个短期占有声明。租约过期后，另一个工作进程可以接过最新检查点（checkpoint）继续执行。正是这套租约机制，让生产系统可以在滚动部署时保住所有进行中的任务。

### Idempotency 加上 Preconditions

光有幂等性（idempotency）还不够。看一个例子：某个工作流获批执行“当 balance > $1000 时，从 A 向 B 转账 $100”。工作流已经提交（commit），执行中途崩溃，随后恢复。如果你只检查幂等键（idempotency key），那么恢复后这个转账只会执行一次，这一步看起来没问题。但如果在崩溃到恢复之间，A 的余额又被另一个工作流扣到了 $500 呢？此时幂等性检查（idempotency check）依然通过，前置条件（precondition）却已经不成立。没有前置条件检查（precondition check），你就把透支真正落地了。

每一个有后果的动作都必须同时具备：

- **Idempotency key（幂等键）**：防止重复执行。
- **Precondition check（前置条件检查）**：确认当前状态仍然与审批动作时的前提一致。

### 动作后的验证

“工具返回了 200”不等于验证。真正的验证，是重新读取目标状态，确认副作用确实在目标系统里发生了。常见模式包括：

- 数据库更新：先执行 `UPDATE ... RETURNING *`，再断言返回的行与预期状态一致。
- 发送邮件：提交后去 sent folder 检查对应 message ID 是否真的存在。
- 写文件：把文件读回，再做哈希校验。
- API 调用：对目标资源再发一次跟进 `GET`。

如果验证（verify）失败，说明工作流已经处在一个“已知有问题”的状态里，这时就必须启动回滚（rollback）。

### Rollback 方案

在 propose-then-commit（第 15 课）里，每个有后果的动作都必须自带回滚方案（rollback plan）。常见类型有：

- **In-band rollback（带内回滚）**：直接逆转副作用，例如 `DELETE` 之后做 `INSERT`，或邮件发错后补发 `Send-correction-email`。
- **Compensating transaction（补偿事务）**：发起一个新的动作去抵消原动作，这是标准的 SAGA 模式。
- **Out-of-band rollback（带外回滚）**：告警给人工，暂停工作流，把异常状态留给调查处理。

如果回滚（rollback）是空操作（no-op），也就是“这件事无法撤销”，那就必须在提案（proposal）里明确写出来。没有回滚的动作，在提交（commit）时需要更强的人在回路中监督，也就是第 15 课里的 challenge-and-response。

### 欧盟《AI Act》第 14 条的工程化理解

第 14 条要求高风险系统具备“effective human oversight”。工程团队通常会把它落成下面几条：

- 检查点（checkpoint）必须能被审计人员查询；
- 回滚（rollback）必须演练过，至少完整端到端测试一次；
- 审计轨迹必须在部署后依然存在，检查点后端（checkpoint backend）不能是易失的；
- 验证（verify）失败不能只悄悄记日志，必须触发告警。

一个工作流如果在提交（commit）中途崩溃、恢复后继续完成副作用，但整个链路里没有“验证加回滚”（verify + rollback）的通道，就经不起第 14 条的检验。

### 最锋利的失败模式：重复执行

这一类生产事故最常见的过程通常是：

1. 动作获批，幂等键（idempotency key）为 k。
2. 提交（Commit）开始，动作执行，返回 200。
3. 工作流在把“committed”状态持久化之前崩溃。
4. 工作流恢复后看到“approved but not committed”，于是再次执行。
5. 副作用触发了两次。

缓解方式是：先持久化一个“执行中（in-flight）”意图，再带着幂等键（idempotency key）执行动作，只有在动作后验证（post-action verification）成功后才标记为“已提交（committed）”。如果动作已经触发，但状态写入失败，那么恢复路径就知道要先验证（verify），必要时再决定是否重新触发；如果状态写入成功、动作本身失败，也可以通过恢复路径把动作精确补成一次。

```figure
checkpoint-replay
```

## 用起来

`code/main.py` 实现了一个带检查点（checkpoint）的工作流，包含幂等性（idempotency）、前置条件（preconditions）、验证（verify）和回滚（rollback）。驱动程序会模拟四种场景：正常执行、崩溃后重试（被幂等性捕获）、前置条件失败（工作流中止且动作不会触发）、验证失败（触发回滚）。

## 交付物

`outputs/skill-rollback-rehearsal.md` 用来为某个拟议工作流设计回滚演练测试，并审计检查点后端（checkpoint backend）是否真的能把审计轨迹持久保存下来。

## 练习

1. 运行 `code/main.py`，验证四种场景都出现。对于“commit 过程中崩溃”的场景，确认跨重试全过程里动作只触发一次。

2. 把“先标记为 done，再真正执行动作”的模式改成“动作之后再写状态”。重新运行崩溃场景，测量会多触发多少次重复动作。

3. 为一个具体的生产动作设计 rollback 方案，例如“向某个 Slack 频道发消息”。判断它属于 in-band、compensating 还是 out-of-band，并说明理由。

4. 选一个你熟悉的工作流，列出所有状态迁移。给每一步标上持久化要求（durability requirement：persist / do not persist），再数一数你目前实际上没有持久化的步骤有多少。

5. 设计一个“已演练 rollback”的端到端测试：启动真实工作流，让它中途崩溃，并确认 rollback 路径真的被触发。这个测试应该断言什么？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|---|---|---|
| Checkpoint | “检查点” | 每一次图状态迁移都持久化到耐久存储中 |
| Lease | “租约” | 工作进程正在执行某次运行的短期占用声明；崩溃时会过期 |
| Precondition | “前置条件” | 断言当前状态仍与已批准动作的前提一致 |
| Post-action verify | “动作后验证” | 重新确认目标系统中副作用确实发生了 |
| In-band rollback | “带内回滚” | 用逆操作直接撤销副作用 |
| Compensating transaction | “补偿事务” | 发起一个新的动作去抵消原动作 |
| Mark-as-done-first | “先写完成状态” | 在提交返回前先写入已提交状态 |
| Article 14 | “欧盟《人工智能法案》第 14 条” | 工程含义是：检查点可查询、回滚已演练、轨迹可审计 |

## 延伸阅读

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 了解 checkpoint 原语和 lease 恢复。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — 了解 Durable Objects 如何作为状态基底。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 监管层面的基线条文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时程工作流里的可靠性框架。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Claude Code Routines 的工作流形态。
