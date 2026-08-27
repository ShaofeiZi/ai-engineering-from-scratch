# 混合记忆：Vector + Graph + KV

> 混合记忆并行运行三种存储——vector 用于语义相似度，KV 用于快速事实查询，graph 用于实体关系推理——并在检索时通过评分层融合结果。这是外部记忆中广泛采用的生产模式；Mem0（Chhikara 等，2025）是一种参考实现。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 07（MemGPT）、阶段 14 · 08（Letta Block）
**Time:** 约 75 分钟

## 学习目标

- 解释为什么单一存储（仅 vector、仅 graph、仅 KV）不足以承载智能体记忆。
- 说出 Mem0 的三种并行存储，以及每种存储所优化的目标。
- 描述 Mem0 的融合评分——相关性、重要性、新近度——并解释它为什么是加权和，而不是层级。
- 仅用标准库实现玩具版三存储记忆，其中 `add()` 写入全部三种存储，`search()` 融合结果。

## 问题

面对以下三类查询，单一存储总有一类不适用：

- **语义相似度**——“上周我们讨论了哪些有关智能体漂移的内容？”Vector 胜出；KV 和 graph 会漏掉。
- **事实查询**——“用户的电话号码是什么？”KV 胜出；vector 浪费资源，graph 则大材小用。
- **关系推理**——“哪些客户共享同一个计费实体？”Graph 胜出；vector 与 KV 无法回答。

生产级智能体会在同一个 session 中发出全部三类查询。单一存储对于其中两类始终是不合适的。Mem0 的贡献在于把三者统一到一个 `add`/`search` 接口之后，再用评分函数融合结果。

## 概念

### 三种存储并行运行

Mem0（arXiv:2504.19413，2025 年 4 月）执行 `add(text, user_id, metadata)` 时：

1. 从文本中提取候选事实（由 LLM 驱动的步骤）。
2. 把每条事实写入 vector store（embedding），用于语义搜索。
3. 把每条事实写入以 (user_id, fact_type, entity) 为键的 KV store，实现 O(1) 查询。
4. 把每条事实作为 typed edge 写入 graph store（Mem0g），用于关系查询。

执行 `search(query, user_id)` 时：

1. Vector store 按 embedding cosine 返回 top-k。
2. KV store 根据从 query 推导出的 (user_id, type, entity) key 返回直接命中。
3. Graph store 返回从 query entity 可达的 subgraph。
4. 评分层融合三路结果。

### 融合评分

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **相关性**——vector cosine、KV 精确匹配、graph path weight。
- **重要性**——写入时标注或后续学习得到（某些事实更重要，例如姓名、ID、策略）。
- **新近度**——根据距上次写入或读取的时间做指数衰减。

权重根据产品调整。聊天智能体提高 `w_recency`；合规智能体提高 `w_importance`；检索智能体提高 `w_relevance`。

### Mem0g 与时态推理

Mem0g 增加了 conflict detector。当新事实与既有 edge 矛盾时，原 edge 会标记为无效，但不会删除。时态查询（“用户三月份住在哪座城市？”）会遍历在对应时间有效的 subgraph。

这是一种合规级行为，也是 Letta 失效模式所推广的做法。

### Benchmark 数字

Mem0 论文报告了以下结果（2025）：

- **LoCoMo**（长篇对话记忆）：91.6
- **LongMemEval**（长时间跨度的情景记忆）：93.4
- **BEAM 1M**（100 万 token 记忆 benchmark）：64.1

比较 baseline（完整上下文的 128k LLM、扁平 vector store、扁平 KV）都落后 10 分以上。benchmark 本身不足以决定选型——运维形态才是关键——但这些数字表明，融合设计带来的提升并非舍入误差。

### 作用域分类

Mem0 按作用域划分记忆：

- **用户记忆**——跨 session 持久化，以 `user_id` 为键。
- **Session 记忆**——在单个 thread 内持久化。
- **智能体记忆**——每个智能体实例独立的状态。

每次写入都要选择一个作用域。检索可以跨作用域查询，并为各作用域设置不同权重。未经思考就混合这些作用域，会导致“助理把 Bob 项目的信息告诉了 Alice”一类事故。

### 这种模式会在哪里出错

- **Embedding 漂移。** 前一百次查询中表现良好的 vector 结果，可能随 corpus 增长而退化。应定期为使用次数最多的 top-N 记录重新生成 embedding。
- **KV schema 膨胀。** `(user_id, type, entity)` 看似简单，直到每个团队都加入自己的 `type`。每季度审计一次 type 集合。
- **Graph 爆炸。** 一个噪声很大的 extractor 每条消息写入 50 条 edge。限制每次 `add` 调用的 graph 写入数量，并丢弃低置信度 edge。

```figure
ae-memory-fusion
```

## 构建它

`code/main.py` 仅用标准库实现三存储模式：

- `VectorStore`——使用朴素 token-overlap similarity 代替 embedding。
- `KVStore`——以 `(user_id, fact_type, entity)` 为键的 dict。
- `GraphStore`——typed edge（subject、relation、object、valid）。
- `Mem0`——顶层 facade，提供 `add()`、`search()`、融合评分和感知作用域的检索。
- 一段多用户、多 session 对话的完整 trace。

运行：

```
python3 code/main.py
```

输出会展示三条独立 recall 路径，以及融合后的 top-k。改变 `main()` 顶部的评分权重，观察排名变化。

## 使用它

- **Mem0（Apache 2.0）**——已具备生产可用性。可配合 Postgres + Qdrant + Neo4j 自托管，也可使用托管云。
- **Letta**——三层 core/recall/archival；可自选 vector 和 graph backend。
- **Zep**——带时态 KG 与事实提取的商业替代方案。
- **自定义构建**——适用于需要精确控制 extractor（合规场景）或融合权重（新近度占主导的语音智能体）的情况。

## 交付它

`outputs/skill-hybrid-memory.md` 会生成一套三存储记忆脚手架，并接入融合评分器、作用域分类与时态失效。

## 练习

1. 用真实 embedding 模型（sentence-transformers、Ollama、OpenAI embeddings）替换玩具版 vector similarity。在合成长对话上测量 recall@10。写入 1000 次后，排名是否漂移？
2. 添加时态查询：`search(query, as_of=timestamp)`。只返回在该时刻或之前有效的记录。哪种存储需要最多改造？
3. 实现 conflict detector：若传入事实与某条 graph edge 矛盾，就使旧 edge 失效，并记录新旧二者。使用“user lives in Berlin” -> “user lives in Lisbon”进行测试。
4. 扩展融合评分器，增加 `user_feedback` 维度（对检索记录点赞）。怎样防止投机取巧（智能体只返回自己曾经喜欢的记录）？
5. 阅读 Mem0 文档（`docs.mem0.ai`）。把玩具实现移植为 `mem0` 客户端调用，并在同样的 20 条测试 query 上比较检索质量。

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| Hybrid memory | “Vector 加 graph 加 KV” | 三种存储并行写入，在检索时融合 |
| Fact extraction | “记忆摄入” | 将文本拆成 (entity, relation, fact) tuple 的 LLM 步骤 |
| Fusion scoring | “相关性排名” | 相关性、重要性和新近度的加权和 |
| Scope | “记忆 namespace” | user / session / agent——决定谁能看到什么 |
| Mem0g | “记忆 graph” | 带时态有效性的 typed edge，用于关系查询 |
| Temporal invalidation | “软删除” | 把矛盾 edge 标记为无效，绝不删除 |
| Embedding drift | “检索腐化” | vector 质量随 corpus 增长而退化；应定期重新 embedding |

## 延伸阅读

- [Chhikara 等，Mem0（arXiv:2504.19413）](https://arxiv.org/abs/2504.19413)——原始论文
- [Mem0 文档](https://docs.mem0.ai/platform/overview)——生产 API、SDK 与托管云
- [Packer 等，MemGPT（arXiv:2310.08560）](https://arxiv.org/abs/2310.08560)——虚拟上下文的前身
- [Letta，Memory Blocks 博客](https://www.letta.com/blog/memory-blocks)——同源的三层设计
