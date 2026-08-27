# 综合项目 08——面向受监管垂直领域的生产级 RAG 聊天机器人

> 到 2026 年，Harvey、Glean、Mendable 和 LlamaCloud 的生产架构已经高度趋同：使用 docling 或 Unstructured 摄取文档，以 ColPali 处理视觉内容，执行混合检索，再通过 bge-reranker-v2-gemma 重排。生成阶段使用 Claude Sonnet 4.7，并将提示缓存（prompt caching）的命中率维持在 60%～80%；Llama Guard 4 和 NeMo Guardrails 提供防护，Langfuse 和 Phoenix 负责可观测性，RAGAS 则在包含 200 个问题的黄金评估集上评分。本综合项目要求你为法律、临床或保险等受监管领域构建这样一套系统，并通过黄金评估集、红队测试和漂移仪表盘三重检验。

**Type:** 综合项目
**Languages:** Python（流水线与 API）、TypeScript（聊天界面）
**Prerequisites:** 第 5 阶段（NLP）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）、第 18 阶段（安全）
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 小时

## 问题

受监管领域的 RAG（例如法律合同、临床试验方案和保险条款）是 2026 年落地最广的生产架构之一，因为它的投资回报清晰、风险也十分具体。Harvey 为 Allen & Overy 构建了法律领域方案，Mendable 提供面向开发者文档的版本，Glean 则服务于企业搜索。成熟模式已经很明确：高保真摄取文档，以混合检索配合重排，生成答案时强制附带引用并使用提示缓存，再叠加多层安全防护，持续监控质量漂移。

真正困难的并不是模型，而是合规与工程约束：系统必须识别不同司法辖区的要求（HIPAA、GDPR、SOC2），让每条引用都可审计，并控制成本——提示缓存命中率较高时可节省 60%～90% 的费用。它还要通过 RAGAS 的忠实度指标检测幻觉，并在源文档已经更新、索引却尚未同步时发现漂移。本项目要求你完整交付这些能力，并让系统同时通过包含 200 个问题的黄金评估集和一套红队测试。

## 概念

整套流水线分为两侧。**摄取侧**：使用 docling 或 Unstructured 解析结构化文档，以 ColPali 处理视觉内容丰富的文档；切分后的文本块会附带摘要、标签和基于角色的访问控制标记。向量规模低于 5000 万时写入 pgvector + pgvectorscale，否则使用 Qdrant Cloud；BM25 稀疏检索与向量检索并行运行。**对话侧**：LangGraph 管理记忆和多轮对话。每次查询先进行混合检索，再由 bge-reranker-v2-gemma-2b 重排，之后使用启用了提示缓存的 Claude Sonnet 4.7 合成答案。输出还要经过 Llama Guard 4 和 NeMo Guardrails 检查，最终返回带有引用锚点的回答。

评估体系分为四层。**黄金评估集（golden set）** 包含 200 组带标准答案和引用的标注问答，用于检验正确性。**红队测试（red team）** 通过越狱提示、PII 提取尝试和领域外问题检验安全性。**RAGAS** 自动为每一轮对话计算忠实度、答案相关性和上下文精确率。**漂移仪表盘（drift dashboard）** 使用 Arize Phoenix 每周监控检索质量与幻觉得分。

提示缓存是这套系统控制成本的关键手段。Claude 4.5+ 和 GPT-5+ 都支持缓存系统提示与检索上下文。命中率保持在 60%～80% 时，单次查询成本可降低到原来的三分之一至五分之一。为获得较高的缓存命中率，流水线必须采用稳定前缀：先放置系统提示和重排后的上下文，再将用户问题作为不缓存的后缀。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 文档摄取：Unstructured.io 或 docling 处理结构化文档，ColPali 处理视觉内容丰富的 PDF
- 向量数据库：5000 万向量以下优先 pgvector + pgvectorscale，否则使用 Qdrant Cloud
- 稀疏检索：带字段权重的 Tantivy BM25
- 编排：LlamaIndex Workflows 负责文档摄取，LangGraph 负责对话
- 重排模型：自托管 bge-reranker-v2-gemma-2b，或托管的 Voyage rerank-2
- 大模型：Claude Sonnet 4.7 + 提示缓存；回退模型可选自托管 Llama 3.3 70B
- 评估：RAGAS 0.2 在线评分，DeepEval 用于幻觉与越狱测试套件
- 可观测性：自托管 Langfuse 并配置标注队列，Arize Phoenix 用于漂移分析
- 防护：Llama Guard 4 输入输出分类器、NeMo Guardrails v0.12 策略层、Presidio PII 清洗
- 合规：文本块级角色访问标签，以及标明 GDPR/HIPAA 适用范围的司法辖区标签

```figure
canary-rollout
```

## 动手构建

1. **文档摄取。** 使用 Unstructured 或 docling 解析语料；一个具备实际意义的系统应包含 1000～10000 份文档。扫描件或视觉内容丰富的页面交给 ColPali 处理。输出的文本块应附带摘要、角色标签和司法辖区标签。

2. **建立索引。** 将稠密嵌入（Voyage-3 或 Nomic-embed-v2）写入 pgvector + pgvectorscale，同时使用 Tantivy 建立 BM25 辅助索引。把角色和司法辖区过滤条件存入 payload。

3. **混合检索。** 先按角色和司法辖区过滤，再并行执行稠密检索与 BM25，随后使用倒数排名融合（reciprocal rank fusion）合并结果。将 top-20 交给重排器，再用重排后的 top-5 生成答案。

4. **使用提示缓存合成答案。** 将系统提示和静态策略放入缓存头部，把重排后的上下文作为缓存扩展，再把用户问题放在不缓存的后缀中。稳定运行时，缓存命中率目标为 60%～80%。

5. **安全护栏。** 输入先经过 Llama Guard 4；NeMo Guardrails 拦截领域外问题和策略禁止的话题；Presidio 清除输出中意外泄露的 PII；最后使用后置过滤器强制答案附带引用。

6. **黄金评估集。** 准备 200 组由领域专家标注的问答，每组都包含答案和引用。使用该数据集评估精确引用匹配率、答案正确性和 RAGAS 忠实度。

7. **红队测试。** 设计 50 个对抗性提示，覆盖越狱攻击（PAIR、TAP）、PII 窃取尝试、领域外问题和跨司法辖区泄露。分别记录是否通过及问题严重程度。

8. **漂移仪表盘。** 使用 Arize Phoenix 每周追踪检索质量，包括 nDCG 和引用忠实度；下降 5% 时触发告警。

9. **成本报告。** 在 Langfuse 中记录提示缓存命中率、每次查询消耗的 token 数，以及按阶段拆分的 $/query（每次查询成本）。

## 实际运行

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付成果

`outputs/skill-production-rag.md` 描述了最终交付物：一套已经部署、带合规标签、通过评分标准并接入实时漂移监控的受监管领域聊天机器人。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度与答案相关性 | 在黄金评估集（200 组问答）上的在线评分 |
| 20 | 引用正确性 | 答案中带有可验证来源锚点的比例 |
| 20 | 防护覆盖率 | Llama Guard 4 通过率与越狱测试套件结果 |
| 20 | 成本与延迟工程 | 提示缓存命中率、p95 延迟、$/query（每次查询成本） |
| 15 | 漂移监控仪表板 | Phoenix 在线仪表板与每周检索质量趋势 |
| **100** | | |

## 练习

1. 再增加一部分适用于不同司法辖区的语料，例如在 GDPR 语料之外加入 HIPAA 语料。通过 20 个跨辖区问题，证明角色与司法辖区过滤能够防止越权泄露。

2. 统计一周生产流量中的提示缓存命中率，找出会破坏缓存前缀的查询，再重新调整提示结构。

3. 增加一个容量为 10k token 的摘要缓冲区来支持多轮记忆，并测量会话变长后忠实度是否下降。

4. 把 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B，比较 $/query（每次查询成本）与忠实度的变化。

5. 增加“不确定”模式：如果重排后的最高分低于阈值，智能体应回答“我没有足够可信的引用”，而不是勉强作答。测量虚假自信是否减少。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 提示缓存（Prompt caching） | “缓存系统提示与上下文” | Claude/OpenAI 的前缀缓存能力；命中时，缓存前缀词元的费用可降低 60%～90% |
| RAGAS | “RAG 评估器” | 自动评估忠实度、答案相关性和上下文精确率的框架 |
| 黄金评估集（Golden set） | “标注评估集” | 由专家标注且带引用的 200 组以上问答，是评估所依据的标准答案 |
| 司法辖区标签（Jurisdiction tag） | “合规标签” | 附着在文本块上的 GDPR/HIPAA/SOC2 适用范围标签，由检索过滤器强制执行 |
| 引用忠实度（Citation faithfulness） | “有据可依回答率” | 回答中的主张有多少能被检索到的原文片段支撑 |
| 漂移（Drift） | “检索质量衰退” | nDCG 或引用得分的周度变化；常用告警阈值是 5% |
| 红队测试（Red team） | “对抗性评估” | 上线前执行越狱、PII 提取和领域外提问等攻击测试 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 法律场景生产栈参考
- [Glean enterprise search](https://www.glean.com) — 企业级 RAG 的参考实现
- [Mendable documentation](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) — 托管式摄入方案
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本优化关键能力
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移监控参考实现
- [Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略护栏框架
