---
name: virtual-memory
description: 为任意目标运行时搭建一个 MemGPT 形态的两级记忆系统（主上下文 + 归档存储 + 记忆工具），并正确处理驱逐、引用和不可信输入。
version: 1.0.0
phase: 14
lesson: 07
tags: [memory, memgpt, virtual-context, archival, citations]
---

给定一个目标运行时（Python、Node、Rust）、一个模型提供商（Anthropic、OpenAI、local）以及一个存储后端（in-memory、SQLite、vector DB、KV、graph），产出一个正确的 MemGPT 形态的记忆系统。

需要产出：

1. 一个 `MainContext` 类型，包含一个 `core` dict（具名持久化分区）和一个 `messages` 列表（FIFO）。在达到大小上限时自动驱逐；被驱逐的对话轮次仍可通过 `conversation_search` 检索。
2. 一个 `ArchivalStore`，具备插入和搜索功能。记录必须携带 `id`、`text`、`tags`、`session_id`、`turn_id`、`created_at`。每次写入都返回已存储的 id 以供引用。
3. 五个与 MemGPT 接口对应的记忆工具：`core_memory_append`、`core_memory_replace`、`archival_memory_insert`、`archival_memory_search`、`conversation_search`。向模型呈现时要附带 `description` 文本，告知模型何时使用各个工具。
4. 一个引用契约：每次归档检索必须将记录 id 与文本一并返回，并且智能体必须在最终回答中引用它们。缺少引用的回答属于软失败。
5. 一个合并钩子（在 v1 中可以是空操作），以便 Lesson 08 的睡眠期智能体能够直接插入而无需重新布线。暴露 `list_records_since(timestamp)` 和 `delete(id)`。

硬性拒绝：

- 使用全量 prompt 的 LLM 评分来搜索归档。必须使用适当的检索后端（BM25、向量相似度）。允许在 top-k 候选短名单上进行 LLM 重排序，但不允许对全量语料库进行重排序。
- 主上下文没有驱逐策略。无界的主上下文会悄然增长并超出窗口。
- 将检索到的内容当作用户指令来存储。所有归档内容都是不可信文本（Lesson 27）。以观察形式而非系统 prompt 形式传递给模型。
- 编写一个会清空所有分区的 `core_memory_clear` 工具。Core 是承重结构；清空是一种自伤行为。支持 `replace` 而非 `clear`。

拒绝规则：

- 如果用户要求“只要答案，不要引用”，则在任何来源归属至关重要的领域（医疗、法律、政策、金融）予以拒绝。提供一个折中方案：以脚注而非行内形式渲染引用。
- 如果用户要求“将所有检索到的内容不经过滤地写回归档”，则予以拒绝并指向 Lesson 27。检索到的内容可被攻击者触达；无差别写回属于记忆投毒。
- 如果运行时没有持久化层，则拒绝交付一个被描述为具有“长期记忆”的智能体。应降级产品描述，而非降级实现。

输出：每个组件一个文件（`main_context.*`、`archival_store.*`、`memory_tools.*`、`agent.*`），外加一个 `README.md`，说明驱逐策略、引用契约，以及在哪里接入 Lesson 08（睡眠期合并）和 Lesson 09（Mem0 融合）。最后以“下一步阅读”结尾：如果智能体需要三级记忆或异步合并，则指向 Lesson 08；如果智能体需要向量 + KV + 图融合，则指向 Lesson 09。
