# 第 13 阶段：工具与协议

> AI 与真实世界之间的接口。

本阶段从函数调用和工具模式出发，逐步推进到可互操作协议、智能体技能、安全和生产治理。按数字顺序浏览很有用，但下方聚焦路线才是可靠的学习顺序。

## 在 GitHub 上开始本阶段

**前置条件：** 第 11 阶段 LLM 补全 API。对于 MCP 或智能体技能，请使用下方聚焦路线，而不要假定按数字顺序学习课程。

**第一个全阶段课程：** [工具接口](01-the-tool-interface/)

从仓库根目录运行以下命令：

```bash
python3 phases/13-tools-and-protocols/01-the-tool-interface/code/main.py
```

保留命令、退出码、描述-决策-执行-观察追踪记录、被拒绝输入证据，以及一句解释轮次限制的话。

**下一步：** 继续前往[函数调用深入剖析](02-function-calling-deep-dive/)，或选择下方的 Model Context Protocol (MCP) 或智能体技能路线。

浏览[第 13 阶段的完整课程列表](../../README.md#phase-13)或[跨阶段路线图](../../ROADMAP.md)。

## Model Context Protocol (MCP) 路径

聚焦的 MCP 路线包含 17 节课程，约 23 小时 15 分钟。它遵循 MCP `2026-07-28`，从一个自描述 JSON-RPC 请求一路推进到可运行的合规性关卡。

| 阶段 | 课程 | 你要证明的内容 | 时间 |
|---|---|---|---:|
| 核心 | [06](06-mcp-fundamentals/), [07](07-building-an-mcp-server/), [08](08-building-an-mcp-client/), [09](09-mcp-transports/), [10](10-mcp-resources-and-prompts/) | 信封、发现、客户端与服务器行为、传输、资源和提示词。 | 5 小时 50 分钟 |
| 双向 | [11](11-mcp-sampling/), [12](12-mcp-roots-and-elicitation/), [13](13-mcp-async-tasks/), [14](14-mcp-apps/) | MRTR 输入、显式作用域、持久化任务，以及无服务器端发起请求的应用边界。 | 5 小时 |
| 安全 | [15](15-mcp-security-tool-poisoning/), [16](16-mcp-security-oauth-2-1/), [18](18-mcp-auth-production/), [17](17-mcp-gateways-and-registries/) | 投毒防御、授权、生产令牌、网关路由和注册表准入。 | 5 小时 15 分钟 |
| 高级 | [28](28-mcp-tool-contracts-and-content/), [29](29-mcp-reliability-cancellation-and-flow-control/), [30](30-mcp-registry-supply-chain-and-drift/), [31](31-mcp-conformance-versioning-and-operations/) | 契约保真度、取消竞态、供应链漂移和发布证据。 | 7 小时 10 分钟 |

精确顺序是 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 18, 17, 28,
29, 30, 31。它定义在
[`learning-paths/model-context-protocol.json`](../../learning-paths/model-context-protocol.json) 中。
辅导器会创建 `MCP-LEARNING.md`，每次调用教授一节课程，并记录每个检查点所需的请求、响应、命令、工作目录、退出码和已脱敏的边界证据。

使用你的宿主环境所支持的调用方式开始：

| 宿主 | 调用方式 |
|---|---|
| Codex | `learn-mcp`，或从 `/skills` 中选择它 |
| Claude Code | `/learn-mcp` |
| 其他兼容宿主 | `Use learn-mcp to start or resume the Model Context Protocol (MCP) path.` |

### 你的前十分钟

从仓库根目录运行第 06 课的无状态转录：

```bash
python3 phases/13-tools-and-protocols/06-mcp-fundamentals/code/main.py
```

在输出中找到四样东西：重复的请求元数据、完整的 `server/discover` 结果、不支持的版本对应的错误 `-32022`，以及一个不创建或终止 MCP 协议会话的传输关闭。该转录是第一个检查点，而不仅仅是一个演示。

如果仓库或 Python 3 不可用，请阅读[第 06 课](06-mcp-fundamentals/)并手工追踪一次请求和响应。将检查点标记为概念性，并保留运行时、传输、授权和部署证据为待定状态。

在任何非回环绑定、共享入口、托管端点或注册表发布之前，完成第 15 课的可执行安全检查点。审查外部目标和所请求的授权范围，然后明确确认部署操作。完成的教程不授予部署权限。

旧版的 `initialize`、`Mcp-Session-Id`、独立 SSE `GET`、会话 `DELETE` 和服务器端发起的请求流仅出现在显式兼容性说明中。现代请求在 `params._meta` 中声明协议版本和客户端能力，使用 `server/discover`，并携带足够的信息以独立进行验证、授权、路由和重试。

[第 23 课](23-capstone-tool-ecosystem/)是 MCP 路线中唯一可选的综合实战课。在开始它之前，请先完成 17 节必修课程以及[第 19 课](19-a2a-protocol/)和[第 20 课](20-opentelemetry-genai/)。

## 智能体技能快速路径

聚焦路线包含五节课程，约 9 小时 30 分钟：

| 步骤 | 课程 | 产出 | 时间 |
|---:|---|---|---:|
| 1 | [22：可移植契约与运行时边界](22-skills-and-agent-sdks/) | 创建、安装、调用、验证并移除一个完整的技能包。 | 90 分钟 |
| 2 | [24：发现与渐进式披露](24-skill-discovery-and-progressive-disclosure/) | 追踪发现、编目、激活和资源加载。 | 105 分钟 |
| 3 | [25：调用与路由](25-skill-invocation-and-routing/) | 控制显式、隐式、人工、模型和弃权路径。 | 105 分钟 |
| 4 | [26：权限、沙箱与信任](26-skill-permissions-sandboxes-and-trust/) | 分离指令、权限、隔离和验证。 | 120 分钟 |
| 5 | [27：评估、打包与可移植性](27-skill-evals-packaging-and-portability/) | 构建发布关卡并在真实宿主中证明行为。 | 150 分钟 |

使用你的宿主环境所支持的调用方式开始：

| 宿主 | 调用方式 |
|---|---|
| Codex | `learn-agent-skills`，或从 `/skills` 中选择它 |
| Claude Code | `/learn-agent-skills` |
| 其他兼容宿主 | `Use learn-agent-skills to start or resume the Agent Skills Engineering path.` |

辅导器会创建或恢复 `AGENT-SKILLS-LEARNING.md`，每次调用教授一节课程，并记录每个检查点所需的证据。该路线定义在
[`learning-paths/agent-skills.json`](../../learning-paths/agent-skills.json) 中。

如果你更愿意先阅读，请从[第 22 课](22-skills-and-agent-sdks/)开始。它的第一个实验在约十分钟内即可将一个技能装入真实宿主。

### 前置快速通道

- 对于实际实验，你需要 `node`、`npx`、`python3`、一个选定的支持技能的宿主，以及对所选项目或用户技能作用域的写入权限。在安装前用 `node --version`、`npx --version` 和 `python3 --version` 验证这三个命令。
- 如果该前置检查不可用，请使用网站或手动阅读每个 `docs/en.md`。你可以完成概念性工作，但保留发现、调用、脚本、更新和卸载证据为待定状态。
- 如果工具契约对你来说是新概念，请略读[第 01 课](01-the-tool-interface/)和[第 05 课](05-tool-schema-design/)。
- 在第 26 课之前，确认你能解释工具投毒和不可信指令。[第 15 课](15-mcp-security-tool-poisoning/)是该前置检查的可选复习课程，而非本路线的第六节必修课。
- [第 23 课](23-capstone-tool-ecosystem/)是一个可选的系统综合实战课，而非第 22 课之后的下一节智能体技能课。在选修它之前，请先完成第 06 课到第 20 课。

## 完整阶段

完整课程计划请见 [ROADMAP.md](../../ROADMAP.md)。
