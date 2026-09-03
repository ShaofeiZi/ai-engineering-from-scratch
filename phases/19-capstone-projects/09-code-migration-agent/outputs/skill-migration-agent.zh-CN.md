---
name: migration-agent
description: 构建一个仓库级代码迁移智能体，结合确定性配方与智能体回退循环，通过 MigrationBench 测试，并发布失败分类法。
version: 1.0.0
phase: 19
lesson: 09
tags: [capstone, code-migration, openrewrite, libcst, migrationbench, agent, sandbox]
---

给定一个 Java 8 或 Python 2 仓库，产出一个已迁移的分支（迁移至 Java 17 或 Python 3.12），测试套件全绿且覆盖率回退最小。在 50 仓库 MigrationBench 子集上进行评估。

构建计划：

1. 确定性阶段：OpenRewrite（Java）或 libcst（Python）首先执行机械式重写。作为"配方"提交，附带干净的 diff。
2. Daytona 沙箱：预装目标运行时；按分支构建；源码只读挂载。
3. 智能体循环：基于 Claude Opus 4.7 + GPT-5.4-Codex 的 LangGraph 或 OpenAI Agents SDK。工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。分类失败类型（依赖、语法、测试、构建工具），应用针对性修复，重新运行。
4. 预算上限：30 分钟、$8、20 轮。超出任一限制即停止并归档为 `budget_exhausted`，附带当前 diff。
5. 测试 + 覆盖率门控：构建通过后测试全绿；覆盖率下降不得超过 2%。
6. 提交 PR，包含配方提交 + 智能体提交 + 摘要评论。
7. 失败分类法：按仓库打标签，取值来自 `{dep_upgrade_required, build_tool_drift, custom_annotation, test_flake, syntax_edge_case, budget_exhausted, coverage_regression}`。
8. 在 MigrationBench 上运行 50 个仓库；发布各类别通过率、每仓库成本和覆盖率保持情况；与确定性基线对比。

评分标准：

| 权重 | 评估维度 | 测量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集 pass@1 |
| 20 | 测试覆盖率保持 | 相对基线分支的平均覆盖率变化 |
| 20 | 每个已迁移仓库的成本 | 通过运行的平均 $/repo |
| 20 | 智能体 / 确定性工具集成 | 由 OpenRewrite 处理的修复占比 vs 智能体处理占比 |
| 15 | 失败分析报告 | 分类法完整性及示例 |

一票否决项：

- 跳过确定性阶段的流水线。OpenRewrite 处理机械式 70-80% 的工作比任何智能体都更便宜、更可靠。
- 覆盖率回退超过 2% 被视为通过。
- 将机械式重写和智能体编写的变更捆绑在一个提交中的 PR。必须分离。
- 未在同一 50 仓库上提供匹配的确定性基线运行便报告通过率。

拒绝规则：

- 拒绝将迁移分支强制推送到基线分支之上。必须始终创建新分支 + PR。
- 拒绝在沙箱中 CI 尚未翻绿时打开 PR。
- 拒绝在未获得明确修改许可的情况下对企业仓库执行操作。

交付物：一个仓库，包含双层迁移流水线、50 仓库 MigrationBench 运行日志、失败分类法看板、匹配的确定性基线运行，以及一篇关于三种最常见失败类别及其消除方法（配方变更）的报告。
