---
name: hybrid-memory
description: 生成一个 Mem0 形态的三存储记忆系统（向量 + KV + 图），包含融合评分器、作用域分类体系和时间失效机制。
version: 1.0.0
phase: 14
lesson: 09
tags: [memory, mem0, vector, graph, kv, fusion, scope]
---

给定目标运行时、一个向量后端（Qdrant、pgvector、Chroma、sqlite-vec）、一个 KV 后端（Postgres、Redis、dict）以及一个图后端（Neo4j、in-memory edges），产出一个融合记忆系统。

产出内容：

1. 三个存储类，位于 `add(text, user_id, session_id, scope, importance, tags)` 门面之后。写入时，提取器将 `text` 分解为记录、KV 三元组和图三元组。没有任何一个存储是可选的。
2. 一个融合评分器 `score = w_rel * relevance + w_imp * importance + w_rec * recency`。将三个权重全部作为配置项暴露。按产品调优，而非按调用调优。
3. 作用域分类体系：`user`、`session`、`agent`。检索必须遵守作用域。一个用户的查询绝不能泄露另一个用户的记录。
4. 时间失效。矛盾会将旧的边/记录标记为失效；绝不删除。暴露 `search(query, as_of=timestamp)` 用于历史查询。
5. 一个提取器接口。默认可以是 LLM 驱动的；允许在测试中使用确定性的 regex 回退。对每次 `add()` 的图边数量设置上限，以防止爆炸式增长。

硬性拒绝：

- 将单存储记忆描述为"Mem0 形态"。纯向量、纯 KV、纯图的产品没问题，但不属于混合记忆。不要错误命名它们。
- 在没有按作用域权重或显式 `scope=` 过滤器的情况下进行跨作用域检索。作用域泄露是一起合规与隐私事故。
- 在出现矛盾时删除。应失效并加盖时间戳。删除会掩盖 bug 并破坏审计。

拒绝规则：

- 如果用户要求"不进行重要性加权"，则拒绝。在百万条记录上进行扁平的相关性排序是迟早会发生的检索故障。
- 如果图后端没有冲突检测器，则拒绝将由此产生的系统称为"Mem0 形态"。降级其名称。
- 如果产品涉及 PII（医疗、法律、HR），则拒绝在提取器未经产品所有者审计的情况下交付。

输出：每个存储一个文件，外加 `memory.py`（门面）、`config.py`（权重）、`README.md`（解释融合权重、作用域策略、提取器契约和失效语义）。以"接下来读什么"结尾，指向 Lesson 10（如果智能体需要学习新技能）、Lesson 23（如果记忆操作需要 OTel spans）或 Lesson 27（用于检索时的不可信输入处理）。
