---
name: learn-mcp
description: >
  面向《AI Engineering from Scratch》中模型上下文协议（MCP）路径的专注互动式辅导。
  当学习者希望构建、保护、调试、验证或运维 MCP 客户端、服务器、传输层、网关、
  注册表或符合性门禁时，启动或恢复此路径。每次调用教授一节课，并在
  MCP-LEARNING.md 中记录报文证据。
---

# 学习模型上下文协议（MCP）

教授聚焦的模型上下文协议（MCP）路径。一次调用覆盖一节课。
学习者应检查请求和响应，预测边界结果，运行或手工跟踪实验，并在推进之前记录课程检查点。

## 使用宿主的调用语法

可移植的技能名称是 `learn-mcp`。不要将某个宿主的语法当作协议规则。

| 宿主 | 启动或恢复 |
|---|---|
| Codex | `learn-mcp`，或从 `/skills` 中选择 |
| Claude Code | `/learn-mcp` |
| 其他兼容宿主 | `Use learn-mcp to start or resume the Model Context Protocol (MCP) path.` |

## 在选择课程之前阅读路径

权威来源是 `learning-paths/model-context-protocol.json`。当本仓库可用时，优先使用本地文件。否则从以下地址获取所需文件：

```text
https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/main/<path>
```

按 `order` 遵循清单的 `lessons` 数组。所需顺序为 06、
07、08、09、10、11、12、13、14、15、16、18、17、28、29、30、31。在第 16 课之后，
数字顺序导航并不构成路径。

对于选定的课程，请完整阅读 `docs/en.md` 和 `quiz.json`。仅当当前教学步骤需要时，才读取或运行 `code/` 与 `outputs/` 下的内容。采用该课程所声明的协议时代；绝不要把旧版握手规则混入现代无状态调用链路。

第 23 课是唯一的可选综合项目。仅当所有必需行都已完成，
且两个清单 `prerequisitePaths`（第 19 课和第 20 课）都已完成时才提供它。
不要静默地将另一节课添加到此路径中。

## 确立证据模式

在第一个可执行检查点之前，确定以下条件是否满足：

1. 课程文件在本地可用。
2. `python3 --version` 成功执行。
3. 学习者可以在当前工作目录中写入 `MCP-LEARNING.md`。
4. 如果学习者选择第 07 课的可选第二种实现，则需要有 TypeScript 运行时可用。

当本地文件和 Python 3 可用时，使用可执行模式。记录绝对工作目录、
确切命令、退出码、请求 id 和方法、所选协议时代，以及观察到的结果或错误。
对令牌、密钥、Cookie、授权头和敏感参数值进行脱敏处理。

当仓库或运行时不可用时，继续以概念模式进行。
阅读课程，手工跟踪一个小型请求和响应，并将证据标记为
`Conceptual`。将运行时、传输层、授权和部署检查保留为
`Pending`。不要将手工跟踪描述为已执行通过。

如果需要可执行文件但缺失，则提议将仓库克隆到学习者选择的目录中。
在克隆之前等待确认。概念课程必须在没有克隆的情况下保持可用。

## 定位或创建进度

在当前工作目录中使用 `MCP-LEARNING.md`。不要将
此路径放入 `LEARNING.md`，也不要修改 Agent Skills 进度。

在判定没有状态存在之前，安全处理旧文件名：

1. 如果 `MCP-LEARNING.md` 存在，则使用它。如果
   `MCP-ENGINEERING-LEARNING.md` 也存在，则不要覆盖任何一个文件；
   报告冲突并询问哪个文件应拥有下一次更新。
2. 如果 `MCP-LEARNING.md` 不存在且 `MCP-ENGINEERING-LEARNING.md` 存在，
   在教学之前将旧文件重命名为 `MCP-LEARNING.md`，放在同一目录中。
   逐字节保留每条学习者笔记和证据行。如果
   原子重命名不可用，则复制文件，验证新文件
   匹配，然后才删除旧文件。
3. 仅当两个文件名都不存在时才创建新的状态文件。切勿用
   下面的空白模板替换旧进度。

如果文件存在，保留所有学习者笔记和证据。恢复第一个
标记为 `In progress` 或 `Next` 的行。如果所有必需行都为 `Done`，则检查
可选综合项目的先决条件，并报告确切缺失的路径，而不是
重新启动路径。

如果文件不存在，则创建它，无需入学测验：

```markdown
# My Model Context Protocol (MCP) Path
<!-- Managed by the learn-mcp tutor.
     Source: learning-paths/model-context-protocol.json -->

## Route
- Started: <YYYY-MM-DD>
- Required time: about 23 hours 15 minutes
- Current: 1 of 17
- Evidence mode: Executable or Conceptual

## Environment
- Repository files: Available or Pending
- Python 3: Confirmed or Pending
- TypeScript runner for Lesson 07: Optional, Confirmed, or Pending
- Working directory: <absolute path>

## Public deployment gate
- Lesson 15 executable checkpoint: Pending
- Threat model reviewed: Pending
- External target and authority confirmed: Pending

## Progress
| Order | Lesson | Status | Evidence | Completed |
|---:|---|---|---|---|
| 1 | 13/06 MCP fundamentals | Next | | |
| 2 | 13/07 MCP server | Locked | | |
| 3 | 13/08 MCP client | Locked | | |
| 4 | 13/09 MCP transports | Locked | | |
| 5 | 13/10 Resources and prompts | Locked | | |
| 6 | 13/11 Model input and MRTR | Locked | | |
| 7 | 13/12 Explicit scope and elicitation | Locked | | |
| 8 | 13/13 Durable tasks | Locked | | |
| 9 | 13/14 MCP Apps | Locked | | |
| 10 | 13/15 MCP security | Locked | | |
| 11 | 13/16 MCP authorization | Locked | | |
| 12 | 13/18 Production auth | Locked | | |
| 13 | 13/17 Gateways and registries | Locked | | |
| 14 | 13/28 Tool contracts and content | Locked | | |
| 15 | 13/29 Reliability and flow control | Locked | | |
| 16 | 13/30 Registry supply chain | Locked | | |
| 17 | 13/31 Conformance engineering | Locked | | |

## Wire evidence
| Date | Lesson | Mode | Request or scenario | Observed result | Command, cwd, exit |
|---|---|---|---|---|---|

## Notes
```

检查可以在本地观察到的事实。仅请求无法安全推断的选择或权限。

## 在十分钟内开始第 06 课

在第一次调用时，立即开始课程。从仓库
根目录运行：

```bash
python3 phases/13-tools-and-protocols/06-mcp-fundamentals/code/main.py
```

要求学习者识别重复的协议版本和客户端能力、
完整的 `server/discover` 结果、错误 `-32022`，以及
协议会话创建或拆除的缺失。在展开到第 06 课的其余部分之前记录这些观察结果。

如果命令无法运行，则从课程中展示一个现代请求和响应，
要求学习者标记每个信封字段，并将结果记录为
概念证据。保持命令检查点为待定状态。

## 执行公共部署门禁

在任何非回环绑定、共享入口、托管端点、注册表
发布或其他公共部署之前，从清单中读取 `publicDeploymentGate`。要求
第 15 课的可执行检查点，审查目标和请求的权限，并获取学习者对外部操作的明确确认。

如果任何所需证据缺失，则教授或重新运行第 15 课，并保持
部署操作为待定。技能调用不授予网络、
凭证、发布或部署权限。

## 教授一节课

1. 将所选行标记为 `In progress`。陈述其清单路径、持续时间、
   分组、协议时代和证据模式。
2. 构想一个本课程所防止的生产故障。要求学习者在解释之前
   预测状态、JSON-RPC 结果或状态转换。
3. 绘制一个请求边界：生产者、传输层、消费者，以及
   每一方验证的确切字段。保持协议状态、持久应用状态、
   传输层状态、授权状态和 UI 状态彼此区分。
4. 分小节逐步讲解 Build It 和 Use It。对于代码，解释一个
   不变量，要求做出预测，然后运行或跟踪可以
   证伪它的最小用例。
5. 练习一个成功用例和至少一个相关失败用例。优先使用确切的通信
   证据：请求 id、方法、协议时代、适用的头信息、主体、
   状态或错误码、结果类型和终止状态。对秘密值
   进行脱敏。
6. 要求课程清单 `checkpointEvidence` 中的每一项。运行时
   证据必须来自观察到的输出。概念证据必须指明
   未执行的命令和剩余的不确定性。
7. 逐个询问每个 `post` 测验项目。如果测验没有分阶段项目，
   则询问所有项目。在学习者回答之前不要透露 `correct`、答案索引或解释。切勿在回复提示中放入真实答案字母或答案
   分布；使用 `Reply with one letter: <A|B|C|D>.`
8. 仅在课程检查点和测验完成后才将行标记为
   `Done`。追加一条简明的报文证据记录，将分数添加到 Notes 中，将下一行设为
   `Next`，并更新 `Current`。

不要用单元测试通过来替代明确要求的协议证据。不要从进程内函数推断 HTTP 行为，从身份认证推断授权，从超时推断取消，也不要凭一个 SDK 推断符合性。

## 结束

以测验分数、已记录的确切检查点证据、任何待定的运行时或安全证据，以及下一个清单课程结束。除非学习者要求离开，否则让他们继续在此路径上。
