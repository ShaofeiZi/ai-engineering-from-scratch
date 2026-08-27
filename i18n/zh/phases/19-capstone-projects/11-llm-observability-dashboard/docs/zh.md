# 综合项目 11——LLM 可观测性与评估仪表盘

> Langfuse 转向了开放核心模式，Arize Phoenix 发布了 2026 年 GenAI 语义约定映射，Helicone 和 Braintrust 都进一步投入逐用户成本归因，Traceloop 的 OpenLLMetry 则成为事实上的 SDK 插桩方案。生产系统的典型形态已经明确：用 ClickHouse 存储追踪数据，用 Postgres 存储元数据，用 Next.js 构建界面，再让一批评估作业（DeepEval、RAGAS、LLM 裁判）在采样后的追踪记录上运行。请自行托管这样一套系统，从至少四类 SDK 摄取数据，并演示它能在五分钟内发现一项人为注入的回归。

**Type:** 综合项目
**Languages:** TypeScript（UI）、Python / TypeScript（摄取 + 评估）、SQL（ClickHouse）
**Prerequisites:** 第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 17 阶段（基础设施）、第 18 阶段（安全）
**Phases exercised:** P11 · P13 · P17 · P18
**Time:** 25 小时

## 问题

到 2026 年，所有承载生产流量的 AI 团队都会在模型旁运行一套可观测性平面，用于成本归因、幻觉检测、漂移监控、越狱信号识别、SLO 仪表盘和 PII 泄漏告警。Langfuse、Phoenix 与 OpenLLMetry 等开源参考实现已经围绕 OpenTelemetry GenAI 语义约定，统一了数据摄取模式。现在只需一套 SDK，就能为 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 插桩，并发送彼此兼容的跨度（span）数据。

你将构建一个自托管仪表盘，从至少四类 SDK 摄取数据，在采样后的追踪记录上运行一组精简的评估作业，检测漂移并发出告警。验收标准是：故意注入一项回归，例如让某个提示开始生成 PII，仪表盘必须在五分钟内发现问题并触发告警。

## 概念

系统通过 OTLP HTTP 摄取数据。SDK 会生成符合 GenAI 语义约定的 span，包括 `gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts` 和 `llm.completions`。这些 span 进入 ClickHouse 做列式分析，用户、会话和应用等元数据则进入 Postgres。

评估器以批处理作业的形式处理采样后的追踪记录。DeepEval 为忠实度、毒性和答案相关性评分；追踪记录包含检索上下文时，RAGAS 还会计算检索指标；自定义 LLM 裁判则执行领域专属检查，例如识别 PII 泄漏和违反策略的响应。评估结果会以评估 span 的形式写回同一个 ClickHouse，并链接到原始追踪记录的父 span。

漂移检测同时监控嵌入空间的分布变化和评估分数趋势：前者通过提示词嵌入上的 PSI 或 KL 散度衡量。告警先进入 Prometheus Alertmanager，再转发到 Slack 或 PagerDuty。界面使用 Next.js 15 和 Recharts 构建。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄取：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- 收集器：OpenTelemetry Collector，并启用用于控制成本的尾部采样处理器
- 存储：ClickHouse 存放 span，Postgres 存放元数据，S3 归档原始事件
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix 评估器套件，以及自定义 LLM 裁判
- 漂移：每周在池化后的提示词嵌入上计算 PSI / KL
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- 界面：Next.js 15 App Router + Recharts + Server Actions
- 开箱即用的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

```figure
ce-otel-drift
```

## 动手构建

1. **收集器配置。** 为 OpenTelemetry Collector 配置 OTLP HTTP 接收器；尾部采样器保留 100% 的错误追踪和 10% 的成功追踪；导出器将数据发送到 ClickHouse 与 S3。

2. **ClickHouse 数据表结构。** 创建 `spans` 表，其列与 GenAI 语义约定对应：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，另加一个保存长载荷的 JSON 字段。为 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖测试。** 分别使用 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM SDK 编写小型客户端应用，并通过 OpenLLMetry 自动插桩。验证每个客户端都能生成规范的 GenAI span，并成功写入 ClickHouse。

4. **评估作业。** 定时作业读取过去 15 分钟内采样的追踪记录，运行 DeepEval 的忠实度、毒性和答案相关性评估。输出以评估 span 写回，并链接到父追踪记录。

5. **自定义 LLM 裁判。** 构建一个 PII 泄漏裁判：给定响应后，调用防护 LLM 对泄漏 PII 的可能性评分，高分响应进入分诊队列。

6. **漂移检测。** 每周作业计算本周池化后的提示词嵌入与过去四周基线之间的 PSI，并在 PSI 超过阈值时发出告警。

7. **仪表盘。** 使用 Next.js 15 构建以下页面：概览（每秒 span 数、逐用户成本、p95 延迟）、追踪（搜索和瀑布图）、评估（忠实度趋势和毒性）、漂移（PSI 时间序列）以及告警。

8. **告警链。** Prometheus 导出器读取评估分数聚合值和延迟分位数；Alertmanager 将警告路由到 Slack，将严重事件路由到 PagerDuty。

9. **回归探针。** 注入一个缺陷：被评估的聊天机器人开始以 1% 的概率泄漏虚假 SSN。测量 MTTR，即从缺陷部署到 Slack 告警的时间。

## 运行示例

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付成果

`outputs/skill-llm-observability.md` 是最终交付物。接入一个 LLM 应用后，仪表盘会摄取其追踪数据、运行评估、对漂移发出告警，并在 Next.js 中呈现逐用户成本明细。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 追踪数据结构覆盖范围 | 能产生规范 GenAI span 的 SDK 类别数（目标：6 类以上） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数与人工标注集比较 |
| 20 | 仪表盘体验 | 注入回归后的 MTTR（目标低于 5 分钟） |
| 20 | 成本与规模 | 持续摄取速率达到每秒 1,000 个 span，且不产生积压 |
| 15 | 告警 + 漂移检测 | 端到端执行 Prometheus / Alertmanager 链路 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证规范的 span 已写入 ClickHouse，且 `gen_ai.*` 属性准确反映原始调用。

2. 在同一批追踪记录上用 Phoenix 评估器替换 DeepEval，测量两个评估引擎之间的分数差异。

3. 增强漂移检测器：按应用 ID 计算 PSI，而不是全局计算，并展示每个应用的漂移轨迹。

4. 添加“用户影响”页面：展示逐用户成本和逐用户失败率，并配备迷你趋势图（sparkline）。

5. 构建尾部采样策略，保留 100% 毒性分数大于 0.5 的追踪记录，并对其余记录做 10% 的分层采样。测量由此引入的采样偏差。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | “OTel LLM 属性” | 2025 年用于 LLM span 属性（系统、模型、token）的 OpenTelemetry 规范 |
| 尾部采样 | “追踪完成后采样” | Collector 在追踪完成后决定保留还是丢弃，因此可以先检查其中是否有错误 |
| PSI | “群体稳定性指数” | 比较两个分布的漂移指标；大于 0.2 通常表示显著漂移 |
| LLM 裁判（LLM-judge） | “以模型做评估” | LLM 按评分准则评估另一个 LLM 的输出（忠实度、毒性、PII） |
| 尾部采样策略 | “保留规则” | 根据错误条件与采样率，决定保存或丢弃哪些追踪记录 |
| 评估 span | “关联的评估追踪” | 携带评估分数，并链接到原始 LLM 调用 span 的子 span |
| 逐用户成本 | “单位经济效益” | 在一个时间窗口内归因到 user_id 的美元成本，是关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse)——参考性的开放核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)——在漂移支持方面表现突出的另一参考实现
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry)——自动插桩 SDK 家族
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——数据摄取结构
- [Helicone](https://www.helicone.ai)——另一种托管式可观测性方案
- [Braintrust](https://www.braintrust.dev)——另一种评估优先平台
- [ClickHouse documentation](https://clickhouse.com/docs)——span 的列式存储方案
- [DeepEval](https://github.com/confident-ai/deepeval)——评估器库
