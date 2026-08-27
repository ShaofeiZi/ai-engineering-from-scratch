# 生产运行时：Queue、Event、Cron

> 生产级代理通常运行在六种运行时形态上：request-response、streaming、durable execution、queue-based background、event-driven，以及 scheduled。先选运行时形态，再选框架。无论哪一种形态，可观测性都是承重结构。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 22（语音）
**Time:** 约 60 分钟

## 学习目标

- 说出六种生产运行时形态，并把每种形态对应到典型框架或产品模式。
- 解释为什么 durable execution（例如 LangGraph）对长时任务至关重要。
- 描述 event-driven 运行时，以及 Claude Managed Agents 在什么场景下合适。
- 解释为什么对多步骤代理来说，“可观测性是承重件”不是一句夸张说法。

## 问题

生产环境中的代理，会以 Jupyter notebook 根本暴露不出来的方式失败：第 37 步遇到网络超时、用户在语音通话中途挂断、机器重启导致 cron 任务中断、后台 worker 内存耗尽。你选择什么运行时形态，决定了这些失败里哪些是可恢复的，哪些会直接造成业务中断。

## 概念

### 请求-响应

- 同步 HTTP 模式，用户要一直等待到任务完成。
- 只适合短任务，通常要求在 <30s 内结束。
- 典型栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 需要的观测指标：标准 HTTP access log，加上 OTel spans。

### 流式输出

- 通过 SSE 或 WebSocket 持续输出增量结果。
- LiveKit 在此基础上进一步扩展到 WebRTC 语音 / 视频，见 Lesson 22。
- 典型栈：任意支持 streaming 的框架，加上一个能处理 SSE/WS 的前端。
- 需要的观测指标：每个 chunk 的时间、首 token 延迟、尾延迟。

### 持久执行

- 每一步之后都做状态 checkpoint；失败后可以自动恢复。
- AutoGen v0.4 的 actor model 会把故障隔离在单个 agent 内，见 Lesson 14。
- 这是 LangGraph 的核心差异化能力，见 Lesson 13。
- 当步骤数未知且失败后重跑成本很高时，它几乎是必需的。

### 基于队列 / 后台执行

- 任务先进入队列，再由 worker 拉取执行；结果通过 webhook 或 pub/sub 回传。
- 对于 long-horizon agent 尤其重要。Anthropic 在 computer use 公告里提到，一项任务可能包含几十到几百步。
- 典型栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS），或者自定义系统。
- 需要的观测指标：队列深度、单任务延迟分布、DLQ 大小。

### 事件驱动

- 代理订阅各种触发器：新邮件、PR 打开、cron 触发、文件变化等。
- Claude Managed Agents 开箱即覆盖了这类模式，见 Lesson 17。
- CrewAI Flows 则把事件驱动的确定性工作流结构化，见 Lesson 15。
- 需要的观测指标：触发源、从事件发生到代理启动的延迟、代理总时延。

### 定时调度

- 按 cron 形态定期运行的代理。
- 最好和 durable execution 结合，这样即使某次 nightly run 失败，下一个 tick 还能接着恢复。
- 典型栈：Kubernetes CronJob + 可持久化框架，或托管平台的 cron 服务，如 Render cron、Vercel cron。

### 2026 年常见部署模式

- **CrewAI Flows**：适合事件驱动的生产系统。
- **Agno**：适合无状态的 Python FastAPI 微服务。
- **Mastra**：通过 Express、Hono、Fastify、Koa 等 server adapter 嵌入现有服务。
- **Pipecat Cloud / LiveKit Cloud**：适合托管语音代理，见 Lesson 22。
- **Claude Managed Agents**：适合托管的、长时间运行的异步代理。

### 可观测性是承重结构

如果没有 OpenTelemetry GenAI spans（Lesson 23），再加上 Langfuse、Phoenix、Opik 这样的后端（Lesson 24），你几乎不可能有效调试一个在第 40 步失败的多步骤代理。这对生产不是“加分项”，而是基础设施。区别就在于：你是能够快速定位故障，还是只能从头重放并补更多日志。

### 生产运行时最常见的失误

- **运行时形态选错。** 把一个要跑 5 分钟的任务塞进 request-response。结果用户断开，worker 堆积，重试叠加放大。
- **没有 DLQ。** 队列 worker 没有 dead-letter queue，失败任务直接消失。
- **后台作业完全不透明。** 后台代理运行时没有导出 trace，用户不报错你根本不知道它已经失败。
- **跳过持久状态。** 任何超过 30 秒、且你承受不起从头重来的流程，都应该具备 durable execution。

```figure
wb-runtime-shapes
```

## 动手构建

`code/main.py` 提供了一个 stdlib 多形态演示：

- request-response endpoint（普通函数）
- 流式处理器（生成器）
- 带 DLQ 的 queue-based worker
- 事件触发器注册表
- cron 形态的 scheduler

运行：

```bash
python3 code/main.py
```

输出会给出五条 trace，展示同一个任务在不同运行时外壳下的行为。代理逻辑相同，外层运行时不同。durable execution 这个第六种形态，会在 Lesson 13 中通过 LangGraph checkpointing 单独展开。

## 如何使用

- **Request-response**：适合聊天式 UX。
- **Streaming**：适合渐进式响应。
- **Durable**：适合 long-horizon task。
- **Queue**：适合批处理、异步任务、长时间执行任务。
- **Event**：适合具备反应性的 agent。
- **Cron**：适合维护类工作，如 memory consolidation、eval、成本报告。

## 交付成果

`outputs/skill-runtime-shape.md` 用来为具体任务挑选运行时形态，并把可观测性要求一并接上。

## 练习

1. 把你的 Lesson 01 ReAct loop 移植到你自己的技术栈中的六种运行时形态里。哪一种最适合哪一种产品表面？
2. 给 queue-based demo 加上 DLQ。模拟 10% 的任务失败，并把 DLQ 大小暴露出来。
3. 写一个由 cron 触发的 eval agent，每晚针对当天最重要的 20 条 trace 运行评估。
4. 实现带 backpressure 的 streaming：如果客户端很慢，就让代理暂停输出。思考这和 turn budget 会怎么互相作用。
5. 读 Claude Managed Agents 的文档。什么情况下你会把一个自托管的 long-horizon agent 迁移到托管服务？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Request-response | "Synchronous" | 用户等待完成，只适合短任务 |
| Streaming | "SSE / WS" | 逐步输出，用户体验更好，并且可以逐块观测延迟 |
| Durable execution | "Resume from failure" | 状态有 checkpoint，失败后从上一步恢复 |
| Queue-based | "Background jobs" | Producer / worker pool / DLQ 结构 |
| Event-driven | "Trigger-based" | 代理对外部事件作出响应 |
| DLQ | "Dead-letter queue" | 存放失败任务的停车场 |
| Claude Managed Agents | "Hosted harness" | Anthropic 托管的长时异步代理，带 caching 和 compaction |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution 的细节
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管的长时异步代理
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — “dozens-to-hundreds of steps per task”
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor model 的故障隔离
