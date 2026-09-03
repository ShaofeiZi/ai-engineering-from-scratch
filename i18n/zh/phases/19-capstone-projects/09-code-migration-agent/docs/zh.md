# 综合项目 09——代码迁移智能体（仓库级语言 / 运行时升级）

> 到 2026 年，Amazon 的 MigrationBench（Java 8 升级到 Java 17）和 Google 的 App Engine Py2-to-Py3 migrator 已经为代码迁移智能体树立了标杆。Moderne 的 OpenRewrite 能够大规模执行确定性的 AST 重写，Grit 则以 codemod 风格的 DSL 解决同类问题。生产系统会将确定性底座与智能体层结合起来：前者负责安全的代码改写，后者处理有歧义的情况；按分支隔离的沙箱负责构建，测试工具则确保创建 PR 前所有检查已经通过。本综合项目要求你迁移 50 个真实仓库，并发布迁移通过率和失败分类体系。

**Type:** 综合项目
**Languages:** Python（智能体）、Java / Python（目标仓库）、TypeScript（仪表盘）
**Prerequisites:** 第 5 阶段（NLP）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 小时

## 问题

大规模代码迁移是 2026 年编码智能体最适合落地、也最容易验证价值的生产场景之一。判断标准十分明确：迁移后测试套件是否通过；收益也很实际，因为升级一批 Java 8 项目往往需要投入大量人力；此外还有公开基准可用，例如 MigrationBench 的 50 个仓库子集。OpenRewrite 负责确定性任务，智能体层则处理 OpenRewrite 配方无法覆盖的问题，例如有歧义的改写、构建系统漂移、长尾语法和传递依赖冲突。

你要构建一个智能体：输入 Java 8 或 Python 2 仓库，输出一个 CI 全部通过的迁移分支。你需要衡量迁移通过率、测试覆盖率保持情况和单仓库成本，并建立失败分类体系。将完整方案与“仅使用确定性工具”的基线并列比较，才能看清智能体究竟在哪些问题上产生了价值。

## 概念

整个流水线分为两层。**确定性底座**（Java 使用 OpenRewrite，Python 使用 libcst）安全地完成大部分机械改写，包括 import 语句、方法签名、空值安全改动、try-with-resources 和弃用 API 替换。它运行速度快，产生的代码差异也便于审计。**智能体层**（OpenAI Agents SDK，或由 Claude Opus 4.7 与 GPT-5.4-Codex 驱动的 LangGraph）负责处理配方无法覆盖的情况，包括构建文件升级（Maven、Gradle、pyproject）、传递依赖冲突、偶发测试失败和自定义注解。

每个仓库都在预装了目标运行时的 Daytona 沙箱中执行。智能体不断循环：运行构建、对失败分类、应用修复、重新运行。每个仓库都有明确上限：30 分钟、8 美元和 20 轮智能体交互。只有全部测试通过且测试覆盖率不低于迁移前时，才为该分支创建 PR；否则必须将仓库归入相应的失败类别，并附上证据。

失败分类体系本身就是交付物。50 个仓库中，究竟哪些问题最常见？传递依赖、自定义注解、构建工具版本漂移，还是与迁移无关的偶发测试失败？每个类别都要统计数量并提供一份具有代表性的代码差异，以便后续配方作者优先解决排名前三的问题。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性底座：OpenRewrite（Java）或 libcst（Python）
- 智能体：OpenAI Agents SDK，或由 Claude Opus 4.7 + GPT-5.4-Codex 驱动的 LangGraph
- 沙箱：按分支隔离的 Daytona devcontainer，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 50 仓库子集（Java 8 到 17），以及 Google App Engine Py2-to-Py3 仓库
- 测试框架：并行运行器，覆盖率使用 Jacoco（Java）或 coverage.py（Python）
- 可观测性：Langfuse + 每个仓库一份追踪包，记录每段代码差异
- 仪表盘：失败分类仪表盘，展示各类别数量与代表性代码差异

```figure
ce-migration-funnel
```

## 动手构建

1. **执行配方。** 先运行 OpenRewrite（Java）或 libcst（Python）配方，完成其中 70%～80% 的机械式迁移，并将这一步单独保存为 “recipe” commit。

2. **尝试构建。** 在 Daytona 沙箱中安装目标运行时并运行构建。如果构建通过，就直接进入测试阶段；如果失败，则交给智能体处理。

3. **智能体循环。** LangGraph 提供以下工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体先判断失败属于依赖、语法、测试还是构建工具问题，再进行针对性修复并重新运行。

4. **预算上限。** 每个仓库最多使用 30 分钟、8 美元和 20 轮智能体交互。任一上限耗尽后都要停止，并将仓库连同当前代码差异归入 “budget_exhausted” 类别。

5. **测试与覆盖率门禁。** 构建通过后运行完整测试套件，并与原始仓库的覆盖率比较。如果覆盖率下降超过 2%，则归入 “coverage_regression” 类别。

6. **创建 PR。** 迁移成功后，推送分支并创建 PR，其中附上已应用的配方、智能体产生的提交，以及改动摘要。

7. **失败分类。** 每个失败仓库都要归入以下某个类别：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`，再将结果汇总成仪表盘。

8. **迁移 50 个仓库。** 在 MigrationBench 子集上运行完整流程，报告各类别的通过率、单仓库成本、覆盖率保持情况，以及相对于“仅使用确定性工具”基线的改进。

## 实际运行

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付成果

`outputs/skill-migration-agent.md` 是最终交付物。给定一个仓库，它先执行确定性配方，再进入智能体循环，最终要么产出 CI 全部通过的迁移分支，要么将仓库归入某个失败类别。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集的 pass@1 |
| 20 | 测试覆盖率保持情况 | 相对原始仓库的平均覆盖率变化 |
| 20 | 单仓库迁移成本 | 迁移成功的运行中，$/repo（每个仓库的成本） |
| 20 | 智能体与确定性工具的集成 | OpenRewrite 完成和智能体完成的修复各自所占比例 |
| 15 | 失败分析报告 | 分类体系的完整度和代表性案例质量 |
| **100** | | |

## 练习

1. 只使用 OpenRewrite 运行一遍迁移流水线，不启用智能体。将通过率与完整流水线对比，找出只有智能体才能解决的案例。

2. 增加 “lint-clean” 检查：迁移后运行代码风格检查器，例如 Java 使用 spotless、Python 使用 ruff。如果出现新的 lint 错误，就令 PR 检查失败。测量“覆盖率保持不变、但代码风格发生回归”的比例。

3. 增加 “minimal-diff” 优化器：智能体分支通过测试后，再执行一轮精简，移除不必要的改动，并报告代码差异规模的缩减比例。

4. 扩展到第三种迁移：从 Node 18 升级到 Node 22。复用同一层沙箱封装，但将配方层替换为自定义 codemod。

5. 将首次构建通过时间（Time to First Green Build，TTFGB）作为一项用户体验指标。目标是 p50 低于 10 分钟。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 确定性底座（Deterministic substrate） | “配方引擎” | OpenRewrite / libcst 提供的声明式 AST 重写底座，并带有安全保证 |
| 代码迁移规则（Codemod） | “代码修改程序” | 一条能机械性重写源码的规则 |
| 构建漂移（Build drift） | “工具版本偏差” | Maven / Gradle / uv 跨大版本升级时出现的细微行为变化 |
| 失败类别（Failure class） | “分类项” | 仓库迁移失败的标注原因，例如依赖、语法、测试、构建工具或预算问题 |
| 覆盖率变化（Coverage delta） | “覆盖率保持情况” | 测试覆盖率从原始仓库到迁移分支的百分比变化 |
| 智能体轮次（Agent turn） | “工具调用轮次” | 智能体循环中一次计划、行动和观察的完整过程 |
| 预算耗尽（Budget exhaustion） | “达到上限” | 仓库耗尽 30 分钟、8 美元或 20 轮交互的限额后仍未迁移成功 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年代码迁移的标准基准
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — 确定性底座参考
- [OpenRewrite documentation](https://docs.openrewrite.org) — 配方编写文档
- [Grit.io](https://www.grit.io) — 另一套 codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 沙箱迁移参考
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — 另一套迁移基准来源
- [libcst](https://github.com/Instagram/LibCST) — Python 的确定性改写底座
- [Daytona sandboxes](https://daytona.io) — 按分支隔离的沙箱参考
