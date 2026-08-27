# 综合项目 10——多智能体软件工程团队

> 到 2026 年，多智能体工程团队的形态已基本收敛：架构师负责规划，N 个编码智能体在并行工作树中开发，审查智能体负责把关，测试智能体负责验证。SWE-AF 的工厂架构、MetaGPT 的角色式提示、AutoGen 0.4 的类型化参与者图（actor graph）、Cognition 的 Devin 和 Factory 的 Droids 各自走向了同一种结构。并行工作树以更多并发资源换取更短的实际耗时，共享状态与交接协议则成为主要故障面。本综合项目要构建这样一支团队，在 SWE-bench Pro 上进行评估，并报告哪些交接会失败以及失败频率。

**Type:** 综合项目
**Languages:** Python / TypeScript（智能体）、Shell（工作树脚本）
**Prerequisites:** 第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 16 阶段（多智能体）、第 17 阶段（基础设施）
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 小时

## 问题

面向大型任务时，单智能体编码执行框架会遇到能力上限。这并不是因为某个智能体不够强，而是因为 20 万 token 的上下文无法同时容纳架构方案、四路并行开发内容、审查意见和测试输出。多智能体工厂会拆开这些职责：架构师负责规划，编码智能体在并行工作树中实现，审查智能体把关，测试智能体验证。SWE-AF 的“工厂”架构、MetaGPT 的角色体系和 AutoGen 的类型化参与者图，描述的其实是同一种形态。

系统最容易出错的地方在角色交接（handoff）：架构师给出的方案可能无法落地，编码智能体可能产生相互冲突的代码差异，审查智能体可能放过一个并未奏效的修复，测试智能体也可能与仍在写代码的智能体发生竞态。你将构建这样一支团队，让它处理 50 个 SWE-bench Pro 问题，追踪每次交接，并发布事后分析。

## 核心概念

每个角色都是职责和消息类型明确的智能体。**架构师（Architect）** 使用 Claude Opus 4.7 阅读问题、编写方案，并把方案拆成接口清晰的子任务。**编码智能体（Coder）** 使用 Claude Sonnet 4.7；N 个实例并行运行，各自在一个 `git worktree` 与 Daytona 沙箱中独立实现子任务。**审查智能体（Reviewer）** 使用 GPT-5.4 阅读合并后的代码差异，决定批准还是提出具体修改要求。**测试智能体（Tester）** 使用 Gemini 2.5 Pro 在隔离环境中运行测试套件，并连同产物一起报告通过或失败。

各角色通过共享任务板通信，任务板可以由文件或 Redis 提供存储。每个角色只领取自己有权处理的任务，角色交接使用符合 A2A 协议的类型化消息。协调时要处理三类问题：由协调器角色或自动三方合并解决代码冲突；编码开始后冻结计划，把重新规划记录为独立事件；严格执行审查隔离，禁止审查智能体批准由自己编写或提议的改动。

Token 放大是隐藏成本。每跨越一次角色边界，就要额外生成摘要提示并传递交接上下文。一次 40 轮的单智能体任务，拆给四个角色后可能累计到 160 轮。评分标准会专门衡量相对单智能体基线的 token 效率，因为真正要回答的不是“多智能体能不能工作”，而是“按每美元成本计算，多智能体是否更划算”。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排：LangGraph，共享状态与每个智能体各自的子图
- 消息传递：A2A 协议（Google，2025），用于类型化的智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码智能体）、GPT-5.4（审查智能体）、Gemini 2.5 Pro（测试智能体）
- 工作树隔离：每个编码智能体使用 `git worktree add` 创建独立工作树，并配备 Daytona 沙箱
- 合并协调器：自定义三方合并，并在发生冲突时由 LLM 协助解决
- 评估：SWE-bench Pro（50 个问题）、SWE-AF 场景，以及用于单元测试的 HumanEval++
- 可观测性：Langfuse，记录按角色标记的 span，并分别核算每个智能体的 token
- 部署：K8s，每个角色使用独立 Deployment，并根据待处理任务量通过 HPA 扩缩容

```figure
ce-team-handoff
```

## 动手构建

1. **任务板。** 使用以文件为后端的 JSONL，记录以下类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体按标签订阅消息。

2. **架构师。** 读取 GitHub 问题，调用 Opus 4.7 并使用规划模板；模板要求明确每个子任务的接口，包括涉及的文件、公共函数和测试影响。随后发出一个包含子任务 DAG 的 `plan_request`。

3. **编码智能体。** N 个并行工作进程各自从任务板认领一项子任务。每个进程都通过 `git worktree add` 创建新分支和工作树，并启动 Daytona 沙箱；完成子任务后，发出包含补丁与测试变更的 `diff_ready`。

4. **合并协调器。** 所有编码智能体完成后，将 N 个分支通过三方合并汇入暂存分支。只有文件内容存在重叠时，才让 LLM 参与冲突解决。

5. **审查智能体。** GPT-5.4 阅读合并后的代码差异，并且不得批准自己编写的改动。它可以发出 `approved`（无需继续操作），也可以发出包含具体修改要求的 `review_feedback`，交回相关编码智能体。

6. **测试智能体。** Gemini 2.5 Pro 在干净的沙箱中运行测试套件并保存产物，再发出带堆栈跟踪的 `test_passed` 或 `test_failed`。测试失败后，任务会退回给负责相应子任务的编码智能体。

7. **交接核算。** 每条跨越角色边界的消息都在 Langfuse 中记录一个 span，其中包含载荷大小和所用模型。按子任务计算 token 放大率（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评估。** 处理 50 个 SWE-bench Pro 问题，并与单智能体基线比较 pass@1 和解决每个问题的美元成本；该基线只使用一个 Sonnet 4.7 和一个工作树。

9. **事后分析。** 对每个失败问题，找出出错的交接环节，例如计划过于模糊、合并冲突、审查智能体错误批准或测试偶发失败，并生成交接失败直方图。

## 实际使用

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付成果

`outputs/skill-multi-agent-team.md` 是最终交付物。给定问题 URL 和并行度后，团队会产出可直接合并的 PR，并附上各角色的 token 用量。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 在同一组 50 个问题上计算 pass@1 |
| 20 | 并行加速 | 与单智能体基线相比的实际耗时 |
| 20 | 审查质量 | 注入缺陷测试中的错误批准率 |
| 20 | Token 效率 | 每个已解决问题的总 token 数与单智能体基线相比 |
| 15 | 协调工程 | 合并冲突解决与交接失败直方图 |
| **100** | | |

## 练习

1. 在运行途中向代码差异注入一个明显缺陷（在主体前额外加入 `return None`）。测量审查智能体的错误批准率，并调优其提示，直到错误批准率低于 5%。

2. 将编码智能体数量减到两个（仍保留架构师、审查智能体和测试智能体，两个编码智能体各自顺序执行两个子任务），比较实际耗时和通过率。

3. 使用单写入者约束替换合并协调器，让各子任务只修改互不重叠的文件集合，并测量架构师因此承担的规划负担。

4. 将审查模型从 GPT-5.4 替换为 Claude Opus 4.7，测量错误批准率和 token 成本差异。

5. 添加第五个角色：文档智能体（Documenter，Haiku 4.5）。审查结束后，由它生成一条变更日志，并衡量文档质量的提升是否值得额外的 token 开销。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 并行工作树（Parallel worktree） | “隔离分支” | `git worktree add` 为每个编码智能体创建的全新工作树 |
| 任务板 | “共享消息总线” | 保存类型化消息、供智能体订阅的文件或 Redis 存储 |
| 交接（Handoff） | “角色边界” | 从一个角色的上下文跨入另一个角色上下文的任何消息 |
| Token 放大 | “多智能体开销” | 各角色总 token / 同一任务的单智能体 token |
| A2A 协议 | “智能体到智能体” | Google 2025 年为类型化智能体间消息制定的规范 |
| 合并协调器 | “集成器” | 执行三方合并并协调冲突的组件 |
| 错误批准 | “审查智能体幻觉” | 审查智能体批准包含已知缺陷的代码差异 |

## 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF)——2026 年多智能体工厂的参考实现
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT)——基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen)——Microsoft 的类型化 actor 框架
- [Cognition AI (Devin)](https://cognition.ai)——参考产品
- [Factory Droids](https://www.factory.ai)——另一项参考产品
- [Google A2A protocol](https://a2a-protocol.org/latest/)——智能体间消息规范
- [git worktree documentation](https://git-scm.com/docs/git-worktree)——工作树隔离机制的官方文档
- [SWE-bench Pro](https://www.swebench.com)——评估目标
