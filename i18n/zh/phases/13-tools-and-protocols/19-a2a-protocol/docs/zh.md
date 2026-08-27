# A2A——智能体间协议

> MCP 面向智能体与工具之间的交互。A2A（Agent2Agent）则面向智能体与智能体之间的交互——它是一项开放协议，让基于不同框架构建、内部实现不透明的智能体也能协同工作。A2A 由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，并在 2026 年 4 月发展到 v1.0，获得 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 等 150 多家组织的支持。它吸收了 IBM 的 ACP，并新增 AP2 支付扩展。本课将依次讲解 Agent Card、Task 生命周期以及两种传输绑定。

**Type:** 构建
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** 第 13 阶段 · 第 06 课（MCP 基础）、第 13 阶段 · 第 08 课（MCP 客户端）
**Time:** 约 75 分钟

## 学习目标

- 区分智能体到工具（MCP）与智能体到智能体（A2A）各自适用的场景。
- 在 `/.well-known/agent.json` 发布包含技能与端点元数据的 Agent Card。
- 走通 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用由 Part（text、file、data）组成的 Message，并用 Artifact 表达输出。

## 问题

一个客服智能体需要把报告撰写工作委派给专门的写作智能体。在 A2A 出现之前，可选方案包括：

- 自定义 REST API。可以工作，但每一对智能体都需要单独适配。
- 共享代码库。要求两个智能体运行在同一种框架之上。
- MCP。并不适合：MCP 用来调用工具，而不是让两个智能体在保留各自不透明内部推理的同时协作。

A2A 填补了这一空白。它把交互建模为一个智能体向另一个智能体发送 Task；Task 拥有生命周期、消息和产物。被调用智能体的内部状态始终不透明——调用方只能看到任务状态变化和最终输出。

A2A 是一项“让不同框架中的智能体彼此通信”的协议。它不会取代 MCP；二者相互补充。

## 概念

### Agent Card

每个兼容 A2A 的智能体都会在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现过程以 URL 为基础：获取卡片，读取 A2A 端点 URL，再枚举它提供的技能。

### 签名 Agent Card（AP2）

AP2 扩展（2025 年 9 月）为 Agent Card 增加了密码学签名。发布方用 JWT 签署自己的卡片，使用方负责验证，从而防止身份冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户端通过 `tasks/send` 发起任务。被调用的智能体推动任务在各状态之间流转；客户端可以通过 SSE 订阅状态更新，也可以轮询。

### Message 与 Part

一条消息携带一个或多个 Part：

- `text`——纯文本内容。
- `file`——带 mimeType 的 base64 二进制数据。
- `data`——有类型的 JSON 载荷（作为被调用智能体的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifact

输出是 Artifact，而不是未经封装的字符串。Artifact 是带名称、带类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifact 可以按数据块流式传输，由调用方逐步累积。

### 两种传输绑定

1. **HTTP 上的 JSON-RPC。** 使用 `/a2a` 端点，以 POST 发送请求，并可选择用 SSE 流式返回。这是默认绑定。
2. **gRPC。** 适用于原生采用 gRPC 的企业环境。

两种绑定承载完全相同的逻辑消息结构。

### 保持不透明性

一个关键设计原则是：被调用智能体的内部状态保持不透明。调用方只能看到任务状态与 Artifact。被调用智能体的思维链、工具调用和子智能体委派过程均不可见。这与 MCP 不同，在 MCP 中，工具调用是透明的。

其原因在于：A2A 让竞争者之间也能协作，而不必暴露内部实现。A2A 可以表达“调用这个客服智能体”，同时不让调用方获知该智能体具体如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** A2A 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，获得 150 多家组织支持。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 用例 | 智能体到工具 | 智能体到智能体 |
| 不透明性 | 工具调用透明 | 内部推理不透明 |
| 典型调用方 | 智能体运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 具有生命周期的 Task |
| 授权 | OAuth 2.1（阶段 13 · 16） | JWT 签名的 Agent Card（AP2） |
| 传输 | Stdio / Streamable HTTP | HTTP 上的 JSON-RPC / gRPC |

当你需要调用某个具体工具时使用 MCP；当你需要把一项完整任务委派给另一个智能体时使用 A2A。许多生产系统会同时使用二者：智能体用 MCP 构建工具层，用 A2A 构建协作层。

```figure
a2a-task-lifecycle
```

## 使用它

`code/main.py` 实现了一个最小 A2A 测试框架：研究智能体发布自己的卡片；写作智能体收到一条包含 PDF 和文本指令等 Part 的 `tasks/send` 请求，依次经历 working → input_required → working → completed 状态，最后返回文本 Artifact。实现全部基于标准库，并使用内存传输，使注意力集中在消息结构上。

阅读代码时请重点观察：

- Agent Card 的 JSON 结构。
- Task ID 的分配与状态转换。
- 包含多种类型 Part 的 Message。
- 任务执行中途的 input-required 分支。
- 完成时返回的 Artifact。

## 交付它

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个需要供其他智能体调用的新智能体，该技能会生成 Agent Card JSON、技能 schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。跟踪完整的 Task 生命周期，包括被调用智能体请求澄清、任务暂停在 input-required 的阶段。

2. 添加签名 Agent Card。对卡片的规范化 JSON 计算 HMAC 签名。编写验证器，并确认卡片被篡改后验证失败。

3. 实现任务流式传输：写作智能体通过 SSE 发出三个增量 Artifact 数据块，由调用方累积。

4. 设计一个封装 MCP 服务器的 A2A 智能体。把每个 MCP 工具映射成一个 A2A 技能，并分析取舍——这种映射损失了哪些不透明性？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月仍未被任何框架实现的那项功能。（提示：它与多跳任务委派有关。）

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| A2A | “智能体间协议” | 用于不透明智能体协作的开放协议 |
| Agent Card | “`.well-known/agent.json`” | 描述智能体技能与端点的公开元数据 |
| Skill | “可调用单元” | 智能体支持的一项命名操作（类似 MCP 工具） |
| Task | “委派单元” | 具有生命周期和最终 Artifact 的工作项 |
| Message | “任务输入” | 携带 Part（text、file、data） |
| Part | “有类型的数据块” | 消息中的 `text` / `file` / `data` 元素 |
| Artifact | “任务输出” | 任务完成时返回的具名、有类型输出 |
| AP2 | “Agent Payments Protocol” | 用于信任和支付的签名 Agent Card 扩展 |
| Opacity | “黑盒协作” | 被调用智能体的内部实现对调用方隐藏 |
| Input-required | “任务暂停” | 智能体需要更多信息时进入的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/)——A2A 的规范定义
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A)——参考实现与 SDK
- [Linux Foundation — A2A 发布新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)——2025 年 6 月的治理权移交
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade)——路线图与合作伙伴发展势头
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258)——v1.0 发布说明与向后兼容指南
