# 综合项目 16——GitHub Issue 到 PR 的自治智能体

> 给问题打个标签，就得到一个 PR。这是 2026 年自治编码智能体的典型产品形态：在云端沙箱里启动智能体，验证测试通过后，自动提交一份附带理由说明、可供审查的 PR。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都提供了这类产品。真正困难的是自动复现仓库的构建环境、防止凭据泄露、按仓库执行预算上限，并确保智能体无法强制推送。本综合项目要构建一个自托管版本，再与托管方案比较成本和通过率。

**Type:** 综合项目
**Languages:** Python（智能体）、TypeScript（GitHub App）、YAML（Actions）
**Prerequisites:** 第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30 小时

## 问题

异步云端编码智能体与交互式编码智能体（综合项目 01）已经是两类产品。前者的交互入口不是终端会话，而是 GitHub 标签。给某个问题打上 `@agent fix this` 后，后台工作进程会在云端沙箱中启动，克隆仓库、运行测试、编辑文件、完成验证，然后自动创建 PR，并在正文中说明修改理由。整个过程没有交互循环，也不需要打开终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都采用了类似形态。

这类系统的工程难点很具体：智能体必须从零构建仓库环境，不能依赖预先制作的开发镜像；不稳定测试需要重跑或隔离；GitHub App 的凭据权限必须尽可能小；预算要按仓库、按天限制；强制推送必须明确禁止。本项目要把这些约束落进一个可运行系统，并测量它相对托管方案的通过率、成本和安全性。

## 概念

入口是 GitHub webhook，可由问题标签或 PR 评论触发。调度器把任务发送到 ECS Fargate 或 Lambda。工作进程把仓库拉入 Daytona 或 E2B 沙箱，并根据仓库的语言、框架和包管理器推断通用 Dockerfile。智能体运行 mini-swe-agent 或 SWE-agent v2 循环，底层模型可以使用 Claude Opus 4.7 或 GPT-5.4-Codex。它反复执行四步：阅读代码、提出修复、应用补丁、运行测试。

验证是整个流程的闸门。创建 PR 之前，完整 CI 必须先在沙箱内通过。系统还要计算覆盖率变化；如果覆盖率下降超过阈值，仍可创建 PR，但必须自动添加 `needs-review` 标签。智能体还要在 PR 正文中解释修改理由，并维护一条 `@agent` 评论线程，供审查者继续追问。

安全边界由两层 GitHub 机制共同决定。GitHub App 只提供短期安装令牌，权限限制为 `workflows: read`，并尽可能缩小仓库内容和 PR 的权限范围。分支保护而非应用权限负责禁止直接写入 `main` 和强制推送，应用绝不能进入绕过名单。GitHub App 不支持对 `.github/workflows` 设置路径级只读权限，所以工作进程必须通过允许列表检查拟议差异。预算上限由调度器执行，例如每个仓库每天最多创建 5 个 PR，每个 PR 的成本上限为 20 美元。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发器：GitHub App + 细粒度令牌；Webhook 接收器由 Lambda 或 Fly.io 承载
- 工作进程：ECS Fargate 任务，或 GitHub Actions 自托管运行器
- 沙箱：每个任务使用一个 Daytona 开发容器或 E2B 沙箱
- 智能体循环：mini-swe-agent 基线，或运行在 Claude Opus 4.7 / GPT-5.4-Codex 之上的 SWE-agent v2
- 检索：tree-sitter 仓库映射 + ripgrep
- 验证：在沙箱内执行完整 CI，并以覆盖率变化作为门控条件
- 可观测性：使用 Langfuse，并把每个 PR 对应的跟踪归档链接写入 PR 正文
- 预算：按仓库限制每日美元开销和每天可创建的 PR 数量

```figure
cf-issue-to-pr
```

## 动手构建

1. **配置 GitHub App。** 创建细粒度安装令牌，权限限定为问题读写、pull_requests 写入、contents 读写和 workflows 读取。分支保护必须单独禁止直接推送到 `main` 和强制推送，应用本身不能进入绕过名单。工作进程还要对拟议差异执行允许列表检查，禁止写入 `.github/workflows`，因为 GitHub App 权限本身不支持路径级控制。

2. **实现 Webhook 接收器。** 用 Lambda 接收问题标签或 PR 评论 Webhook，只响应标签 `@agent fix this`，其余事件直接过滤。将任务写入 SQS。

3. **实现调度器。** 从 SQS 拉取任务，先检查仓库的每日预算是否超限，再启动 ECS Fargate 任务，并传入仓库 URL、问题正文和全新的 Daytona 沙箱。

4. **环境推断。** 自动识别语言栈，例如 Python、Node、Go、Rust，以及包管理器如 uv、pnpm、go mod、cargo。如果仓库里没有 Dockerfile，就在线生成一个。

5. **实现智能体循环。** 使用 mini-swe-agent 或 SWE-agent v2，底层模型可先选 Claude Opus 4.7。工具包括 ripgrep、tree-sitter 仓库映射、read_file、edit_file、run_tests 和 git。设置硬性预算：成本不超过 20 美元、实际耗时不超过 30 分钟、智能体轮次不超过 30。

6. **执行验证。** 智能体循环结束后，在沙箱内运行完整测试套件。用 jacoco 或 coverage.py 计算覆盖率变化。如果 CI 失败，直接停止，不得创建 PR。若覆盖率降幅超过 2%，可以创建 PR，但必须自动添加 `needs-review` 标签。

7. **发布 PR。** 推送智能体分支，并通过 GitHub API 创建 PR。创建时必须提供标题、修改理由、差异摘要、跟踪 URL、成本和轮次。

8. **做好凭据卫生。** 工作进程只能使用短期 GitHub App 安装令牌；日志归档前必须对密钥和敏感凭据脱敏。

9. **开展评估。** 准备 30 个预先选定且难度各异的内部问题，测量通过率、PR 质量（差异规模、风格、覆盖率）、成本和延迟，再用同一组问题与 Cursor Background Agents、AWS Remote SWE Agents 对比。

## 运行示例

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付成果

`outputs/skill-issue-to-pr.md` 是本课交付物：一个由 GitHub App 和异步云端工作进程组成的系统，可以把带标签的问题转成可供审查的 PR，同时控制成本并限定凭据权限范围。

| 权重 | 评判标准 | 衡量方式 |
|:-:|---|---|
| 25 | 30 个问题的通过率 | 端到端成功（CI 通过且覆盖率达标） |
| 20 | PR 质量 | 差异规模、覆盖率变化、代码风格符合度 |
| 20 | 每个已解决问题的成本和延迟 | 每个 PR 的美元成本和实际耗时 |
| 20 | 安全性 | 受限令牌、按仓库设置预算、禁止强制推送、凭据卫生 |
| 15 | 操作员体验 | 理由说明评论、重试入口、@ 提及后的交互 |
| **100** | | |

## 练习

1. 增加“修复不稳定测试”模式：标签 `@agent stabilize-flake TestX` 会在沙箱中把该测试运行 50 次，并提出能让它稳定下来的最小修改。

2. 选择三个相同的问题，比较本系统与 Cursor Background Agents 的成本，并说明各自更适合哪些场景。

3. 构建预算仪表板，统计每个仓库的每日成本和每个用户的成本，并在出现异常波动时告警。

4. 加入“试运行”（dry-run）模式：不运行 CI，只创建草稿 PR，让审查者先以较低成本查看方案。

5. 实现保留策略：自动删除超过 7 天仍未合并的 PR 分支。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | “受限权限的机器人身份” | 具备细粒度权限并使用短期安装令牌的应用 |
| 异步云智能体 | “后台智能体” | 在云端沙箱中运行的非交互式工作进程，而非终端内进程 |
| 环境推断 | “生成 Dockerfile” | 检测语言和包管理器；缺少 Dockerfile 时自动生成 |
| 验证 | “在沙箱里跑 CI” | 创建 PR 前，先在工作进程中运行完整测试套件 |
| 覆盖率变化 | “保持覆盖率” | 基线分支与智能体分支之间的测试覆盖率百分比变化 |
| 单仓库预算 | “每日上限” | 由调度器强制执行的美元开销和 PR 数量上限 |
| 理由说明 | “PR 正文中的说明” | 智能体总结改了什么以及为何改，是 PR 正文必填内容 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 权威异步云智能体参考实现
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考实现
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) — 托管式竞品
- [Google Jules](https://jules.google) — Google 的托管式实现
- [Factory Droids](https://www.factory.ai) — 另一种商业参考方案
- [GitHub App documentation](https://docs.github.com/en/apps) — 受限权限机器人身份的官方文档
- [Daytona cloud sandboxes](https://daytona.io) — 沙箱参考实现
