# 记忆块与休眠时计算

> 模型可以直接编辑离散的功能性记忆块；当主智能体空闲时，另一个休眠时智能体会异步整合记忆。这两个理念让记忆得以扩展到单次对话之外。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 07（MemGPT）
**Time:** 约 75 分钟

## 学习目标

- 说出 Letta 使用的三层记忆（core、recall、archival）及各自职责。
- 解释 memory block 模式：Human block、Persona block 与用户自定义 block 都是一等的 typed object。
- 说明 sleep-time compute 是什么、为何位于关键路径之外，以及为何可以使用比主智能体更强的模型。
- 实现一个脚本化的双智能体循环：主智能体提供响应，休眠时智能体在各 turn 之间整合 block。

## 问题

MemGPT（第 07 课）解决了虚拟内存的控制流，但生产环境出现了三个问题：

1. **延迟。** 每项记忆操作都位于关键路径上。如果智能体必须在用户等待时剪枝、摘要或核对，尾延迟会急剧上升。
2. **记忆腐化。** 写入不断累积，相互矛盾的事实仍然留存，检索结果被陈旧内容淹没。
3. **结构丢失。** 扁平的 archival store 无法表达“Human block 始终在 prompt 中；Persona block 始终在 prompt 中；Task block 则随 session 切换”。

Letta（letta.com）是原始 MemGPT 项目在 2024 年采用的平台名称——论文中的模式仍称为 MemGPT——而 2026 年的 Letta V1 重写则是后续独立的一步。Memory block 让结构显式化；sleep-time compute 把整合移出关键路径。

## 概念

### 三层结构

| 层 | 作用域 | 所在位置 | 写入方 |
|------|-------|----------------|------------|
| Core | 始终可见 | 主 prompt 内部 | 智能体工具调用 + 休眠时重写 |
| Recall | 对话历史 | 可检索 | 自动记录 turn |
| Archival | 任意事实 | Vector + KV + graph | 智能体工具调用 + 休眠时摄入 |

Core 对应 MemGPT core。Recall 是对话 buffer 及其被逐出的尾部。Archival 是外部存储。这种拆分消除了 MemGPT 两层结构承担过多职责的问题。

### Memory Block

block 是 core 层中带类型、持久化且可编辑的 section。最初的 MemGPT 论文定义了两种：

- **Human block**——关于用户的事实（姓名、角色、偏好、目标）。
- **Persona block**——智能体的自我概念（身份、语气、约束）。

Letta 将其推广为任意用户自定义 block：用于当前目标的 `Task` block、用于代码库事实的 `Project` block、用于硬性约束的 `Safety` block。每个 block 都有 `id`、`label`、`value`、`limit`（字符上限）和 `description`（让模型知道何时编辑它）。

block 可通过以下工具接口编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)`——压缩接近容量上限的 block。

### 休眠时计算

Letta 在 2025 年增加的能力：在关键路径之外，让第二个智能体在后台运行。休眠时智能体处理对话 transcript 和代码库上下文，将 `learned_context` 写入共享 block，并整合或使 archival 记录失效。

由此自然产生以下属性：

- **没有延迟成本。** 主智能体的响应无需等待记忆操作。
- **可以使用更强模型。** 休眠时智能体不受延迟约束，因此可以使用更昂贵、更慢的模型。
- **自然的整合窗口。** 在用户不等待时，对事实去重、摘要，并使矛盾事实失效。

这种结构与人类的工作方式相似：先完成任务，再“睡一觉”，长期记忆在夜间逐渐巩固。

### 原生推理

Letta V1（`letta_v1_agent`，2026）弃用 `send_message`/heartbeat 与内联 `Thought:` token，转而采用原生 reasoning。Responses API（OpenAI）与支持 extended thinking 的 Messages API（Anthropic）会通过独立 channel 发出 reasoning，并在多个 turn 间传递（生产环境中跨提供商传递时会加密）。控制循环仍然是 ReAct，但 thought trace 由系统结构表达，而不是由 prompt 格式表达。

### 这种模式会在哪里出错

- **Block 膨胀。** 无限执行 `block_append` 会很快触及上限。在一次写入即将越过 cap 前，应接入 block summarizer。
- **静默漂移。** 休眠时智能体重写 block，主智能体却毫不知情。应为 block 做版本控制，并在 trace 中呈现 diff。
- **受投毒的整合。** 休眠时智能体会把攻击者可触达的内容处理进 core。第 27 课的防护同样适用于休眠时接口。

```figure
memory-blocks
```

## 构建它

`code/main.py` 实现：

- `Block`——id、label、value、limit、description。
- `BlockStore`——CRUD 加 `near_limit(label)` 辅助方法。
- 两个脚本化智能体——`PrimaryAgent` 负责一个 turn，`SleepTimeAgent` 在 turn 之间整合。
- 一段三轮对话 trace，展示 block 写入，以及一次对 block 做摘要并使陈旧事实失效的休眠时处理。

运行：

```
python3 code/main.py
```

transcript 展示了职责分离：主智能体的 turn 快速响应并产生原始写入；休眠过程负责压缩和清理。

## 使用它

- 使用 **Letta**（letta.com）作为参考实现，可以自托管，也可以使用托管云。
- 把 **Claude Agent SDK Skill** 作为 block 形态的知识——Skill 是具名、带版本、可检索的指令 block，智能体按需加载。
- 需要控制存储后端的团队可以**自定义构建**。采用 Letta API 契约，以便日后迁移。

## 交付它

`outputs/skill-memory-blocks.md` 可为任意运行时生成 Letta 风格的 block 系统，其中包含 sleep-time hook、安全规则与 citation wiring。

## 练习

1. 添加 `block_summarize` 工具：当 `near_limit` 返回 true 时，用模型生成的摘要替换 block 值。怎样的触发阈值能同时减少摘要调用与 block 溢出？
2. 实现 archival 的休眠时去重：文本 token overlap 超过 90% 的两条记录合并为一条。只在休眠阶段执行，绝不能放进关键路径。
3. 为 block 做版本控制。每次写入都记录旧值和 diff。暴露 `block_history(label)`，让运维人员可以调试“智能体为什么忘了 X”。
4. 把休眠时智能体视为不可信写入方。当它们修改 Persona 或 Safety block 时，要求第二个智能体审查后才能提交。
5. 将示例移植到 Letta API（`letta_v1_agent`）。block schema 会怎样变化？原生 reasoning 又会如何改变 trace 形态？

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| Memory block | “可编辑的 prompt section” | core memory 中带类型、持久化、可由 LLM 编辑的片段 |
| Human block | “用户记忆” | 固定在 core 中的用户事实 |
| Persona block | “智能体身份” | 固定在 core 中的自我概念、语气与约束 |
| Sleep-time compute | “异步记忆工作” | 第二个智能体在关键路径之外执行整合 |
| Core / Recall / Archival | “层级” | 始终可见 / 对话 / 外部三层记忆划分 |
| Block limit | “上限” | 每个 block 的字符上限；触发摘要 |
| Native reasoning | “思考 channel” | 提供商层的 reasoning 输出，而非 prompt 层的 `Thought:` |
| Learned context | “休眠输出” | 休眠时智能体写入共享 block 的事实 |

## 延伸阅读

- [Letta，Memory Blocks 博客](https://www.letta.com/blog/memory-blocks)——memory block 模式
- [Letta，Sleep-time Compute 博客](https://www.letta.com/blog/sleep-time-compute)——异步整合
- [Letta，重构智能体循环](https://www.letta.com/blog/letta-v1-agent)——原生 reasoning 重写
- [Packer 等，MemGPT（arXiv:2310.08560）](https://arxiv.org/abs/2310.08560)——起源
