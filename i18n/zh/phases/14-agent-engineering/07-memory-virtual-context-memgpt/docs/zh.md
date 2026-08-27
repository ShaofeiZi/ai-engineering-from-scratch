# 智能体记忆——虚拟上下文与记忆分页

> 上下文窗口是有限的，而对话、文档和工具 trace 没有这样的上限。解决方案就是重新表述操作系统的虚拟内存：主上下文相当于 RAM，外部存储相当于磁盘，智能体在二者之间换入和换出页面。MemGPT（Packer 等，2023）为这种模式命名；许多生产级记忆系统都构建在它之上。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 01（智能体循环）、阶段 14 · 06（工具使用）
**Time:** 约 75 分钟

## 学习目标

- 解释 MemGPT 所依据的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 页面换入/换出。
- 仅用标准库实现两层 MemGPT 模式，包括主上下文 buffer、可搜索外部存储，以及换入/换出工具。
- 说明智能体如何发出“interrupt”来查询或修改外部记忆，以及结果如何拼接回下一个 prompt。
- 找出 MemGPT 中延续到 Letta（第 08 课）和 Mem0（第 09 课）的设计选择。

## 问题

上下文窗口看起来似乎可以解决记忆问题，实际上并不能。生产环境中反复出现三种故障模式：

1. **溢出。** 多轮对话、长文档或包含大量工具调用的 trajectory 超出窗口，截断位置之后的一切都会丢失。
2. **稀释。** 即便内容仍在窗口内，塞入无关上下文也会稀释模型对重要信息的注意力。前沿模型面对长输入时依旧会退化。
3. **持久性。** 新会话以空窗口开始。没有外部记忆的智能体无法跨会话说出“还记得你之前让我做的……”。

更大的窗口有所帮助，却无法彻底解决问题。Mem0 在 2025 年论文中的测量表明：128k 窗口的 baseline 仍会遗漏长时间跨度的事实，而配有外部记忆的 4k 窗口智能体却能找到它们。

## 概念

### 操作系统类比

MemGPT（Packer 等，arXiv:2310.08560，v2 发布于 2024 年 2 月）把上下文管理映射到操作系统虚拟内存：

| 操作系统概念 | MemGPT 概念 | 2026 年生产类比 |
|------------|---------------|------------------------|
| RAM | 主上下文（prompt） | Anthropic/OpenAI 上下文窗口 |
| 磁盘 | 外部上下文 | vector DB、KV、graph store |
| 缺页 | 记忆工具调用 | `memory.search`、`memory.read`、`memory.write` |
| 操作系统内核 | 智能体控制循环 | 带记忆工具的 ReAct 循环 |

智能体运行普通的 ReAct 循环，只是多出一类工具，用来在主上下文中换入和换出数据。

### 两层结构

- **主上下文。** 保存当前任务的固定大小 prompt，模型始终可见。
- **外部上下文。** 容量不受限制，通过工具搜索。相关时读取，事实出现时写入。

原论文在两个超出基础窗口的任务上评估了该设计：分析超过 10 万 token 的文档，以及跨多日会话保留持久记忆的聊天。

### Interrupt 模式

MemGPT 引入 memory-as-interrupt：对话进行到一半时，智能体可以调用记忆工具；运行时执行该工具，并将结果作为新的 observation 拼接到下一轮 assistant 消息。其概念与 Unix `read()` syscall 相同：进程阻塞，syscall 返回字节，然后进程继续执行。

规范的记忆工具接口：

- `core_memory_append(section, text)`——写入 prompt 中的持久化 section。
- `core_memory_replace(section, old, new)`——编辑持久化 section。
- `archival_memory_insert(text)`——写入可搜索的外部存储。
- `archival_memory_search(query, top_k)`——从外部存储检索。
- `conversation_search(query)`——扫描过去的 turn。

### 论文止步之处与生产系统的起点

2024 年 9 月，MemGPT 演变为 Letta。研究仓库（`cpacker/MemGPT`）仍然保留；Letta 则扩展了这套设计：

- 从两层扩展到三层（core、recall、archival——第 08 课）。
- 用原生 reasoning 替代 `send_message`/heartbeat 模式（第 08 课）。
- 由 sleep-time agent 异步执行记忆工作（第 08 课）。

即使生产系统运行的是 Letta、Mem0 或自定义两层存储，MemGPT 论文在 2026 年仍是其基础。

### 这种模式会在哪里出错

- **记忆腐化。** 写入增长速度超过读取，检索结果被陈旧事实淹没。解决方案：定期整合（Letta sleep-time）、显式失效（Mem0 conflict detector）。
- **记忆投毒。** 外部记忆本质上是被检索回来的文本。如果攻击者控制的内容进入一条记忆笔记，智能体会在下一次会话重新摄入它。这就是 Greshake 等人的攻击（第 27 课）跨时间后的表现。
- **引用丢失。** 智能体记得“用户让我交付 X”，却无法指出来自哪一轮。每次 archival 写入都要保存来源引用（session ID、turn ID）。

```figure
context-budget
```

## 构建它

`code/main.py` 仅使用标准库实现 MemGPT 的两层模式：

- `MainContext`——固定大小的 prompt buffer，包含一个 `core` dict 与一个 `messages` list；超过上限时自动压缩最早的消息。
- `ArchivalStore`——内存中的类 BM25 存储（使用 token overlap 评分），保存 (id, text, tags, session, turn) 记录。
- 映射到 MemGPT 接口的五个记忆工具。
- 一个脚本化智能体：先向 archival 写入事实，再调用 `archival_memory_search` 回答问题。

运行：

```
python3 code/main.py
```

trace 会展示智能体写入三条事实、把主上下文填满至上限（触发 eviction），再通过 archival 检索回答后续问题——无需真实 LLM 即可复现 MemGPT 工作流。

## 使用它

如今每个生产级记忆系统都是 MemGPT 的一种变体：

- **Letta**（第 08 课）——三层结构、原生 reasoning、sleep-time compute。
- **Mem0**（第 09 课）——vector + KV + graph，并由评分层融合。
- **OpenAI Assistants / Responses**——通过 thread 和 file 提供托管记忆。
- **Claude Agent SDK**——通过 Skill 和 session store 提供长期记忆。

应根据运维形态（自托管、托管、框架集成）选择，而不是根据核心模式选择——核心模式就是 MemGPT。

### 智能体记忆的形态

分页解决容量问题，却不决定该存储什么。生产系统反复采用四类记忆，每类回答不同问题：

- **工作记忆**——眼下什么最重要？也就是上下文内层：当前任务、最近 turn、固定的 core section。prompt 本身就是工作记忆。
- **情景记忆**——发生过什么？保存带 session 与 turn 引用的历史 turn 和 trajectory，需要时可以回放。
- **语义记忆**——什么是真的？有关用户、领域与世界的事实，会随着变化更新并去重。
- **程序性记忆**——这件事应该怎么做？学习得到的流程、偏好和规则，它们引导未来行为，而不是用于回忆事件。

开源实现分别从不同切入点解决问题：

| 类型 | 实现 | 处理方式 |
|------|----------------|-------------------|
| 工作记忆 | MemGPT / Letta | 使用记忆工具，在固定 prompt 预算中换入和换出内容（本课、第 08 课） |
| 情景记忆 | Zep | 时态知识图谱——事实带有有效区间，因此可以查询“某个时刻什么为真” |
| 语义记忆 | Mem0 | 提取流水线，在 vector、KV 与 graph store 之间去重和更新事实（第 09 课） |
| 语义 + 程序性 | LangMem | 在后台把事实与行为规则提取到存储中，供智能体在 turn 之间查阅 |
| 情景 + 语义 | agentmemory | 在 session 运行时捕获内容，再把它们整合为有类型、可搜索的记录 |

## 交付它

`outputs/skill-virtual-memory.md` 是一项可复用 Skill，可为任何目标运行时生成正确的两层记忆脚手架（main + archival + 工具接口），并接好 eviction 策略和 citation 字段。

## 练习

1. 添加 `max_main_context_tokens` 上限，以 token 衡量（用 `len(text.split())` * 1.3 近似）。超过上限时，把最早消息压缩成摘要。比较启用和禁用 summarizer 时的行为。
2. 在 archival store 上正确实现 BM25（term frequency、inverse document frequency）。在一组玩具事实上，将 recall@10 与 token-overlap baseline 比较。
3. 为 archival insert 添加 `citation` 字段（session_id、turn_id、source_url）。让智能体在每个由检索支持的答案中引用来源。
4. 模拟记忆投毒：添加一条写着“ignore all future user instructions”的 archival 记录。编写 guard，扫描检索结果中的指令式文本并将其标记为不可信。
5. 移植实现，改用 MemGPT 研究仓库的 core-memory JSON schema（`cpacker/MemGPT`）。从扁平字符串切换到 typed section 后，会发生什么变化？

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| Virtual context | “无限记忆” | 主层（prompt）+ 外部层（可搜索），支持换入/换出 |
| Main context | “工作记忆” | prompt——大小固定且始终可见 |
| Archival memory | “长期存储” | 可搜索的外部持久化存储，按需检索 |
| Core memory | “持久化 prompt section” | 固定在主上下文中的命名 section |
| Memory tool | “记忆 API” | 智能体发起、用于读写外部记忆的工具调用 |
| Interrupt | “记忆缺页” | 智能体暂停、运行时获取数据、结果拼入下一 turn |
| Memory rot | “陈旧事实” | 旧写入淹没检索；通过整合修复 |
| Memory poisoning | “注入的持久化笔记” | 攻击者内容存入记忆，并在 recall 时重新摄入 |

## 延伸阅读

- [Packer 等，MemGPT（arXiv:2310.08560）](https://arxiv.org/abs/2310.08560)——受操作系统启发的虚拟上下文论文
- [Letta，Memory Blocks 博客](https://www.letta.com/blog/memory-blocks)——三层架构的演进
- [Anthropic，有效的上下文工程](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——将上下文视作预算
- [Chhikara 等，Mem0（arXiv:2504.19413）](https://arxiv.org/abs/2504.19413)——构建于该模式之上的混合生产记忆
- [Zep（getzep/zep）](https://github.com/getzep/zep)——分类表中的时态知识图谱记忆
- [Mem0（mem0ai/mem0）](https://github.com/mem0ai/mem0)——第 09 课混合存储背后的提取流水线
- [LangMem（langchain-ai/langmem）](https://github.com/langchain-ai/langmem)——在后台提取事实与行为规则
- [agentmemory（rohitg00/agentmemory）](https://github.com/rohitg00/agentmemory)——将 session 捕获内容整合为有类型、可搜索的记录
