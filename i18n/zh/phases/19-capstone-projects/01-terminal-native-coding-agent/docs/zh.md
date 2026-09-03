# 综合项目 01——终端原生编程智能体

> 到 2026 年，编程智能体的形态已经基本定型：终端用户界面（TUI）执行框架、一份有状态的计划、受沙箱保护的工具接口，以及规划、行动、观察、恢复的循环。远看之下，Claude Code、Cursor 3 和 OpenCode 大同小异。本综合项目要求你从头搭建完整系统，让它接收命令行输入，最终创建拉取请求；然后在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 对比。做完后你会明白，真正棘手的不是调用模型，而是工具循环、沙箱，以及如何为 50 轮任务设定成本上限。

**Type:** 综合项目
**Languages:** TypeScript / Bun（执行框架）、Python（评测脚本）
**Prerequisites:** 第 11 阶段（LLM 工程）、第 13 阶段（工具与协议）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 小时

## 问题

到 2026 年，编程智能体已成为最主要的 AI 应用类别。Claude Code（Anthropic）、配备 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、拥有 11.2 万颗星的 OpenCode、Factory Droids 与 Google Jules，采用的都是同一套架构的变体：终端执行框架、受权限约束的工具接口、沙箱，以及围绕前沿模型运行的“规划—行动—观察”循环。前沿模型只是这套系统中很窄的一环：Live-SWE-agent 使用 Opus 4.5，在 SWE-bench Verified 上达到了 79.2%，但工程工作远不止模型本身。多数故障并非模型判断失误，而是工具循环不稳定、上下文污染、令牌成本失控，或文件系统操作带来破坏。

只从外部观察，无法真正理解这类智能体。你得亲手造一个，看着它在第 47 轮因 ripgrep 返回 8 MB 匹配结果而崩溃，再重新设计输出截断层。这正是本综合项目的用意。

## 核心概念

执行框架包含四个部分。**规划（Plan）**维护一个 TodoWrite 风格的状态对象，模型每轮都会重写它。**行动（Act）**负责分派读取、编辑、运行、搜索和 git 等工具调用。**观察（Observe）**捕获标准输出、标准错误与退出码，截断后再把摘要送回上下文。**恢复（Recover）**处理工具错误，同时避免撑爆上下文窗口或陷入死循环。2026 年的实现还多了一类机制：**钩子（hooks）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact` 都是可配置的扩展点，操作者可借此加入策略、遥测和防护规则。

沙箱选用 E2B 或 Daytona。每个任务都在全新的开发容器中运行，并挂载一个可读写的 git 工作树（worktree）。执行框架绝不接触宿主机文件系统；无论任务成功还是失败，工作树最终都会被销毁。成本控制有三道硬限制：每轮令牌上限、每次会话的美元预算，以及总轮数上限（通常为 50 轮）。可观测性由采用 GenAI 语义约定的 OpenTelemetry 跨度（span）提供，数据发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 执行框架运行时：Bun 1.2 + Ink 5（终端中的 React）
- 模型接入：通过 OpenRouter 统一 API 使用 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B 沙箱（JavaScript SDK）或 Daytona 开发容器
- 代码搜索：ripgrep 子进程，以及支持 17 种语言的预编译 tree-sitter 解析器
- 隔离：每个任务执行 `git worktree add`，成功或失败后都清理
- 评测框架：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 自建的 30 题留出集
- 可观测性：OpenTelemetry SDK 配合 `gen_ai.*` 语义约定，接入自托管 Langfuse
- PR 发布：GitHub App 使用细粒度令牌，权限范围仅限目标仓库

```figure
ce-agent-loop
```

## 动手构建

1. **TUI 与命令循环。** 使用 Ink 搭建 Bun 项目，接受 `agent run <repo> "<task>"` 命令。界面分为三栏：顶部显示计划，中部滚动显示工具调用，底部显示令牌预算。按 Ctrl-C 可以取消任务，但退出前必须先触发 `SessionEnd` 钩子。

2. **计划状态。** 定义带类型的 TodoWrite 模式，条目状态包括 pending、in_progress 和 done，并可附带备注。模型每轮都通过工具调用重写完整状态，不得增量修改。将计划持久化到 `.agent/state.json`，以便崩溃后继续执行。

3. **工具接口。** 定义六个工具：`read_file`、`edit_file`（可预览差异）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时限制）以及 `git`（支持 status / diff / commit / push）。通过 MCP StreamableHTTP 暴露这些工具，使执行框架不依赖具体传输方式。每个工具都必须截断返回内容，单次调用最多返回 4k 个令牌。

4. **沙箱封装。** 每个任务都启动一个 E2B 沙箱。执行 `git worktree add -b agent/$TASK_ID` 创建新分支。所有工具调用均在沙箱内执行，且无法访问宿主机文件系统。

5. **钩子。** 实现 2026 年的全部八种钩子。至少接入四个由用户编写的钩子：(a) `PreToolUse` 充当破坏性命令守卫，阻止针对工作树外部执行 `rm -rf`；(b) `PostToolUse` 统计令牌用量；(c) `SessionStart` 初始化预算；(d) `Stop` 写出最终追踪包。

6. **评测循环。** 克隆由 30 个 Python 问题组成的 SWE-bench Pro 子集，让执行框架逐题运行。以 mini-swe-agent 为最小基线，比较 pass@1、每题轮数和每题成本。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 设置三项硬上限：50 轮、200k 上下文、每个任务 5 美元。上下文达到 150k 时，`PreCompact` 钩子把较早轮次概括为一个先前状态块，在保留计划的同时为新的观察结果腾出空间。

8. **发布 PR。** 任务成功后，最后执行 `git push` 并调用 GitHub API 创建 PR，在正文中附上计划与差异摘要。

## 实际使用

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付成果

交付的技能文件位于 `outputs/skill-terminal-coding-agent.md`。输入仓库路径和任务描述后，它会在沙箱中运行完整的“规划—行动—观察”循环，并返回 PR URL 和追踪包。本综合项目按以下标准评分：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 相对基线表现 | 在 30 道相同的 Python 任务上比较你的执行框架与 mini-swe-agent |
| 20 | 架构清晰度 | 对照 Live-SWE-agent 的布局，审查规划、行动、观察三者的分离方式，以及钩子接口和工具模式定义 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示和破坏性命令守卫均通过红队测试 |
| 20 | 可观测性 | 追踪信息完整（100% 的工具调用都有对应跨度），且逐轮统计令牌用量 |
| 15 | 开发者体验 | 冷启动 < 2 秒；崩溃后能继续原计划；Ctrl-C 可在工具执行期间完整取消任务 |
| **100** | | |

## 练习

1. 把底层模型从 Claude Sonnet 4.7 切换为由 vLLM 提供的 Qwen3-Coder-30B。比较 pass@1 与每题成本，并报告开源模型具体在哪些任务上表现更差。

2. 增加一个 `reviewer` 子智能体，在发布 PR 前阅读差异，并可要求进入修改循环。测量误报审查是否会让 SWE-bench 通过率低于单智能体基线。提示：通常会。

3. 对沙箱做压力测试：编写一个尝试用 `curl` 访问外部 URL 的任务，再编写一个尝试向工作树外写入文件的任务。确认两者都被 PreToolUse 钩子拦截，并记录这些尝试。

4. 使用较小的模型 Haiku 4.5 实现 `PreCompact` 摘要。测量经过三次压缩后，计划内容的保留程度下降了多少。

5. 将 MCP StreamableHTTP 传输方式换成 stdio。对冷启动和单次调用延迟进行基准测试，并选出更适合纯本地场景的方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 执行框架（Harness） | “智能体循环” | 包围模型的一层代码，负责分派工具、维护计划状态并执行预算限制 |
| 钩子（Hook） | “智能体事件监听器” | 用户编写的脚本，由执行框架在八种生命周期事件之一发生时运行 |
| 工作树（Worktree） | “Git 沙箱” | 位于独立路径、与同一仓库相连的检出目录；丢弃它不会影响主克隆目录 |
| TodoWrite | “计划状态” | 带类型的 pending / in-progress / done 条目列表，由模型每轮重写 |
| StreamableHTTP | “MCP 传输方式” | MCP 2026 修订版中的长连接 HTTP 双向流传输方式，用于取代 SSE |
| 令牌上限（Token ceiling） | “上下文预算” | 每轮或每次会话的输入与输出令牌总上限；达到上限会触发压缩或终止 |
| pass@1 | “单次尝试通过率” | 无需重试且不查看测试集，第一次运行就能解决的 SWE-bench 任务比例 |

## 延伸阅读

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 提供的参考执行框架
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs 与 Composer 2 的产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — 用于比较 SWE-bench 执行框架的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) — 拥有 11.2 万颗星的开源执行框架
- [SWE-bench Pro leaderboard](https://www.swebench.com) — 本综合项目采用的目标评测
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP 与能力元数据
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用与令牌用量的跨度结构定义
