---
name: load-test-plan
description: 设计真实的 LLM 负载测试——选择工具（LLMPerf、k6、GenAI-Perf、guidellm），构建四种模式（稳态、爬坡、尖峰、浸泡），并在 CI 中设置门控。
version: 1.0.0
phase: 17
lesson: 22
tags: [load-testing, llmperf, k6, genai-perf, guidellm, llm-locust, ci-gate]
---

给定工作负载（端点、TTFT/TPOT/错误的 SLA）、目标规模（并发度、RPS）和 CI 姿态（PR 门控还是仅发布时），输出一份负载测试方案。

产出内容：

1. 工具。LLMPerf 用于基线运行；k6 + 流式扩展用于 CI 门控；GenAI-Perf 用于 NVIDIA 参考运行；guidellm 用于大规模合成。LLM-Locust 仅在已使用 Locust 时选用。
2. 提示词分布。从真实流量（若有）或公开分布（ShareGPT / HumanEval）获取输入 token 的均值 + 标准差。禁止使用单提示词循环。
3. 四种模式。稳态、爬坡、尖峰、浸泡。每种需指明：目标 RPS、持续时间、预期失败模式。
4. CI 门控。具体阈值：TTFT P95 < X，5xx < 5%，TPOT < Y。每个 PR 运行时长：3-5 分钟。
5. 指标对齐。注意报告工具是 GenAI-Perf 风格（ITL 不含 TTFT）还是 LLMPerf 风格（ITL 含 TTFT）。选定一种并保持一致。
6. 输出。提交到仓库的脚本文件（k6 JS、LLMPerf CLI）。

硬性拒绝条件：
- 使用均匀提示词做负载测试。拒绝——数据会失真。
- 不支持流式的负载测试。拒绝——LLM 端点默认为流式。
- 在未确认指标定义差异的情况下跨工具比较数据。拒绝。

拒绝规则：
- 若团队打算在原生 Locust 上运行而不用 LLM-Locust 扩展，拒绝——GIL 陷阱。
- 若 CI 门控预算 < 60 秒/PR，拒绝完整浸泡——建议快速稳态测试加独立夜间浸泡。
- 若提示词分布数据不可用，要求使用有文档记录的公开分布（ShareGPT）并标注假设。

输出：一页方案，包含工具、提示词分布、四种模式及其目标、CI 门控阈值、指标对齐。以唯一 CI 输出结尾：仅当所有阈值达标且 3 次运行稳定时 PR 方可通过。
