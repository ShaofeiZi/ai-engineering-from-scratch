---
name: multi-agent-team
description: 构建一个包含架构师、并行编码者、审查者和测试者的多智能体软件团队；在 SWE-bench Pro 上进行评估并产出交接事后复盘报告。
version: 1.0.0
phase: 19
lesson: 10
tags: [capstone, multi-agent, swe-bench, langgraph, a2a, worktree, roles]
---

给定一个 GitHub issue URL 和并行度级别，部署一个多智能体软件团队，产出可合并的 PR。在 50 个 SWE-bench Pro 问题上进行评估，并发布交接失败直方图。

构建计划：

1. 任务看板：基于文件（或 Redis）的 JSONL 类型化消息存储。消息种类：plan_request、subtask、diff_ready、review_needed、review_feedback、approved、test_needed、test_passed、test_failed、replan_needed。
2. 架构师（Opus 4.7）：阅读 issue，编写计划，发射具有显式接口（涉及文件、公共函数、测试影响）的子任务 DAG。
3. N 个编码者（Sonnet 4.7）：每个认领一个子任务，生成全新的 `git worktree add` + Daytona 沙盒，独立实现。
4. 合并协调器：三方合并；仅在文件级重叠时进行 LLM 介导的冲突解决。
5. 审查者（GPT-5.4）：阅读合并后的 diff；不能批准自己编写或提议的 diff；发射 approved 或 review_feedback，路由到相关编码者。
6. 测试者（Gemini 2.5 Pro）：在干净沙盒中运行测试套件；发射 test_passed 或 test_failed 并附带产物。
7. 交接核算：每条跨角色消息成为一条 Langfuse span，带有载荷大小和模型。计算 token 放大系数 = total_tokens / single_agent_baseline_tokens。
8. 注入明显的 bug 探针（10% 的运行）以测量审查者误批准率。
9. 在 50 个 SWE-bench Pro 问题上运行；发布 pass@1、挂钟时间 vs 单智能体基线、各角色 token 分解、交接失败直方图。

评估标准：

| 权重 | 标准 | 度量 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 50 题子集 pass@1 |
| 20 | 并行加速 | 挂钟时间 vs 单智能体基线 |
| 20 | 审查质量 | 注入 bug 探针的误批准率 |
| 20 | Token 效率 | 每个已解决问题总 token vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |

硬性拒绝：

- 审查者可以批准自己编写或提议的 diff。硬约束。
- 报告没有匹配的单智能体基线运行。多智能体必须在*每美元收益*上取胜，而非仅 pass@1。
- 任务看板使用自由格式字符串而非类型化 A2A 消息。
- 合并协调器静默丢弃冲突 diff 而非路由回以重新规划。

拒绝规则：

- 拒绝在没有每个角色预算上限（token + 美元）的情况下运行。
- 拒绝在测试者未在干净沙盒中验证的情况下打开 PR。
- 拒绝在单次运行中将编码者扩展到超过 8 个。超过此数值协调开销将占主导。

产出：一个包含任务看板 + 角色工作智能体的仓库、50 题 SWE-bench Pro 运行日志、匹配的单智能体基线运行、带角色标签 span 和各角色 token 分解的 Langfuse 仪表板、注入 bug 探针报告，以及一个事后复盘报告，指出最常失败的三种交接以及减少每种失败的消息模式或提示词变更。
