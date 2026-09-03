---
name: prompt-caching-planner
description: 设计缓存友好的提示词布局，并选择合适的提供商缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定一个提示词（system + tools + few-shot + retrieval + history + user）以及使用画像（每小时请求数、所需 TTL、提供方），输出：

1. 布局。重排各段并标记单一缓存断点；说明哪些段是稳定的，哪些是易变的。
2. 提供方模式。Anthropic cache_control、OpenAI automatic 或 Gemini CachedContent。依据 TTL 与复用模式给出理由。
3. 盈亏平衡。TTL 内每次写入的预期读取次数；与不缓存方案相比的净成本对比及计算过程。
4. 验证计划。CI 断言：第二次相同请求的 cache_read_input_tokens > 0；仪表盘按缓存与未缓存 token 分拆展示。
5. 失效模式。列出在此配置下缓存未命中最可能的三种原因（动态时间戳、工具重排序、近似重复文本），以及你将如何逐一预防。

拒绝发布将易变字段置于断点之上的缓存方案。拒绝在未达到使 2 倍写入溢价回本的复用次数时启用 1 小时 TTL。
