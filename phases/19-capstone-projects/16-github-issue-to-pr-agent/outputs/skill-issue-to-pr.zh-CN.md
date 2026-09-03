---
name: issue-to-pr
description: 构建一个异步的 GitHub Issue 到 PR 智能体，在云沙箱中运行，复现构建，验证测试，并在严格的每仓库预算内提交可供审查的 PR。
version: 1.0.0
phase: 19
lesson: 16
tags: [capstone, async-agent, github, fargate, daytona, swe-bench, budget, safety]
---

给定一个带有 `@agent fix this` 标签的 GitHub 仓库，上线一个自托管的云智能体，将每个被标记的 Issue 转化为可供审查的 PR，使用受限凭据并有界成本。

构建计划：

1. GitHub App 使用细粒度令牌：issues rw、PRs write、contents rw、workflows read。禁止强制推送。main 分支保护防止直接写入。
2. Webhook 接收器（Lambda 或 Fly.io）过滤标签 / PR 评论事件并入队到 SQS。
3. 调度器执行每仓库每天 $ 和 PR 数量上限；为每个允许的任务启动一个 ECS Fargate 任务。
4. 环境推断：从仓库内容检测语言 + 包管理器 + 运行时。如缺失则即时合成 Dockerfile。
5. 每个任务使用 Daytona 或 E2B 沙箱。将仓库克隆到全新的 `git worktree` + 智能体分支。
6. 智能体循环（mini-swe-agent 或基于 Claude Opus 4.7 或 GPT-5.4-Codex 的 SWE-agent v2）。工具：ripgrep、tree-sitter 仓库映射、read_file、edit_file、run_tests、git。上限：$20、30 轮、30 分钟。
7. 验证：沙箱内完整 CI；通过 jacoco / coverage.py 计算覆盖率增量；如增量 < -2% 则标记 `needs-review`；如 CI 红则停止。
8. 通过 GitHub API 开启 PR，附带理由、差异摘要、追踪 URL、成本、轮次。
9. 可观测性：每个 PR 的 Langfuse 追踪；日志脱敏密钥；每仓库预算仪表盘。
10. 在 30 个已标注的内部 Issue 上评估；在三个 Issue 的共享子集上与 Cursor Background Agents 和 AWS Remote SWE Agents 对比。

评估标准：

| 权重 | 标准 | 度量 |
|:-:|---|---|
| 25 | 30 个 Issue 的通过率 | 端到端成功（CI 绿 + 覆盖率达标） |
| 20 | PR 质量 | 差异大小、覆盖率增量、风格一致性 |
| 20 | 每个已解决 Issue 的成本和延迟 | $/PR 和挂钟时间/PR |
| 20 | 安全性 | 受限令牌、每仓库预算、无强制推送、凭据卫生 |
| 15 | 运维体验 | 理由评论、重试交互、@提及跟进 |

硬性否决项：

- 任何可以强制推送的智能体。硬性排除。
- 跳过预算检查的调度器。失控循环是典型的失败模式。
- 在沙箱内未通过完整 CI 即开启的 PR。
- 包含未脱敏令牌或 PII 的追踪归档。

拒绝规则：

- 拒绝在 main 上无分支保护的情况下安装。
- 拒绝在无每仓库每日预算（美元和 PR 数量）的情况下运行。
- 拒绝自动重试失败的运行；所有重试都需要人工重新打标签。

输出：一个代码仓库，包含 GitHub App、webhook 接收器、调度器 + 预算台账、Fargate 任务定义、沙箱生命周期管理器、mini-swe-agent 循环、30 个 Issue 的评估运行、与 Cursor Background Agents 和 AWS Remote SWE Agents 的并排对比，以及一篇写明排名前三的构建推断失败及各自减少失败的 Dockerfile 合成变更的报告。
