# A2A：智能体间协议

> Google 在 2025 年 4 月发布 A2A；到 2026 年 4 月，规范已经稳定发布在 https://a2a-protocol.org/latest/specification/，并且有 150 多家组织参与支持。A2A 是 MCP（Lesson 13）的横向补充：MCP 解决的是垂直方向的 agent ↔ tools，而 A2A 解决的是对等方向的 agent ↔ agent。它定义了 Agent Card（发现）、带 artifacts 的 tasks（文本、结构化数据、视频等）、不透明的 task lifecycle，以及 auth。越来越多的生产系统会把 MCP 与 A2A 一起使用。Google Cloud 也在 2025-2026 年间把 A2A 支持纳入了 Vertex AI Agent Builder。

**Type:** 学习 + 构建
**Languages:** Python（标准库，`http.server`、`json`）
**Prerequisites:** 第 16 阶段 · 04（原语模型）
**Time:** 约 75 分钟

## 问题

你的 agent 需要去调用另一套系统里的另一个 agent。该怎么做？你当然可以手写一个 HTTP endpoint，再定义一套定制 JSON schema，然后祈祷对方也恰好按你的格式来实现。问题在于，这样每一对 agent 之间都会变成一次私有集成。

A2A 试图成为这种调用的通用 wire protocol。它提供标准发现、标准任务模型、标准传输方式、标准 artifacts。可以把它理解成“面向 agent 的 HTTP+REST”，只是这里的一等公民不再是网页或资源，而是 agent 本身。

## 概念

### 四个核心元素

**Agent Card。** 一个放在 `/.well-known/agent.json` 的 JSON 文档，用来描述 agent：名字、skills、endpoints、支持的 modalities、auth 要求。发现流程就是先读取这张卡。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task。** 工作单元。它是一个异步、带状态的对象，生命周期通常是 `submitted → working → completed / failed / canceled`。客户端提交 task 后，可以轮询，也可以订阅更新。

**Artifact。** Task 产出的结果类型。可以是 text、structured JSON、image、video、audio。A2A 把这些不同模态都当成一等输出。

**Opaque lifecycle。** A2A 并不规定远端 agent *如何* 完成这个任务。客户端只看到状态迁移和 artifacts；底层实现可以自由选择任何框架。

### MCP 与 A2A 的分工

- **MCP**（Lesson 13）：agent ↔ tool。agent 通过 JSON-RPC 读写 tool server，默认偏无状态。
- **A2A**：agent ↔ agent。双方都是拥有自身推理能力的 agent，协议关注的是对等协作。

真实生产多代理系统往往两者并用。一个 A2A peer 在自己的那一侧去调用 MCP tools。把这两个问题拆开，可以让体系结构更清晰。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

如果要流式推送，则可以通过 SSE 订阅 `/tasks/{id}/events` 来接收更新。

### 身份认证

A2A 常见的三种认证方式：

- **Bearer token**：例如 OAuth2 或 opaque token。
- **mTLS**：双向 TLS，由组织彼此证明身份。
- **Signed requests**：对 payload 做 HMAC 签名。

认证要求会直接写在 Agent Card 里，由客户端发现并遵守。

### 到 2026 年 4 月已有 150+ 组织支持

企业级采用推动了 A2A 的扩张。一个重要变化是：A2A 开始成为企业 agent 系统跨信任边界协作的标准方式。Google Cloud 在 Vertex AI Agent Builder 中加入了 A2A 支持；Microsoft Agent Framework 支持它；LangGraph、CrewAI、AutoGen 等主流框架也都开始提供 A2A adapters。

### A2A 适合的场景

- **跨组织调用。** A 公司里的 agent 去调用 B 公司里的 agent。没有 A2A，这类对接很快会退化成一堆定制合同。
- **异构框架互通。** LangGraph agent 调 CrewAI agent，再调一个自定义 Python agent。A2A 提供了统一层。
- **带类型的 artifacts。** 视频结果、结构化 JSON、音频，不需要再人为塞进纯文本。
- **长生命周期任务。** Opaque lifecycle + polling 很适合持续数小时的任务。

### A2A 不适合的场景

- **极度延迟敏感的微调用。** A2A 的生命周期是异步式的，不适合亚毫秒级 agent-to-agent 通讯；这种情况更适合直接 RPC。
- **同进程内紧耦合 agent。** 如果两个 agent 本来就在同一个 Python 进程内运行，A2A 的 HTTP 往返会显得过重。
- **小团队内部系统。** 规范带来的形式化成本是真实存在的；如果全是内部 agent，未必需要这么完整的协议外壳。

### A2A 与 ACP、ANP、NLIP

2024-2026 年间还出现了几类邻近规范：

- **ACP**（IBM/Linux Foundation）：A2A 的前身之一，范围更窄。
- **ANP**（Agent Network Protocol）：更强调 peer discovery，偏去中心化。
- **NLIP**（Ecma Natural Language Interaction Protocol，于 2025 年 12 月标准化）：更聚焦自然语言内容类型。

截至 2026 年 4 月，A2A 是采用度最高的对等 agent 协议。可参考 arXiv:2505.02279（Liu et al., “A Survey of Agent Interoperability Protocols”）里的对比。

```figure
sw-agent-card-discovery
```

## 动手构建

`code/main.py` 用 `http.server` 与 JSON 实现了一个最小可运行的 A2A server 和 client。服务端：

- 暴露 `/.well-known/agent.json`，
- 接收 `POST /tasks`，
- 管理 task state，
- 在 `GET /tasks/{id}` 返回 artifacts。

客户端：

- 获取 Agent Card，
- 提交 task，
- 轮询直到完成，
- 读取 artifact。

运行：

```
python3 code/main.py
```

脚本会先在后台线程里启动 server，再让 client 对它发起调用。你会看到完整流程：discovery、submit、poll、artifact。

## 实际使用

`outputs/skill-a2a-integrator.md` 用来设计 A2A 集成方案：Agent Card 里放什么、task schemas 怎么定义、auth 如何选择、以及 streaming 与 polling 之间如何取舍。

## 交付成果

上线前检查清单：

- **固定 spec version。** A2A 仍在持续演进，Agent Card 最好显式声明协议版本。
- **让 task creation 幂等。** 网络重试造成的重复提交，应尽量落成同一个 task。
- **明确 artifact schemas。** 声明 agent 会返回哪些结构，消费方应该做校验。
- **加 rate limits 与 auth。** A2A 是对外接口，按标准 web security 做防护。
- **为 failed tasks 设计 dead-letter 机制。** 长期观察重复失败的模式。

## 练习

1. 运行 `code/main.py`，确认 client 能发现 server，并收到正确 artifact。
2. 给 server 增加第二个 skill，例如 “summarize”。更新 Agent Card，再写一个 client 按 task type 自动选 skill。
3. 实现一个 SSE 流式 endpoint：`/tasks/{id}/events`，用于持续推送 state changes。客户端需要做哪些不同处理？
4. 阅读 A2A spec（https://a2a-protocol.org/latest/specification/），指出这个 demo 没实现的三个规范要求。
5. 对比 A2A（通过 Agent Card 做 discovery）与 MCP（通过 `listTools` 做能力探测）。自描述 agent 与 capability probing 之间的取舍是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| A2A | “智能体到智能体” | 用于跨系统调用其他智能体的对等协议，由 Google 于 2025 年推出。 |
| Agent Card | “智能体名片” | 位于 `/.well-known/agent.json` 的 JSON 文档，描述技能、端点和认证要求。 |
| Task | “工作单元” | 带生命周期的异步有状态对象；完成后产出工件。 |
| Artifact | “结果” | 带类型的输出，例如文本、结构化 JSON、图像、视频和音频。 |
| 不透明生命周期 | “如何完成由智能体自行决定” | 客户端只查看状态转换；服务端内部可以自由选择框架和工具。 |
| 发现 | “找到智能体” | 通过 `GET /.well-known/agent.json` 获取 Agent Card。 |
| MCP 与 A2A | “工具与对等方” | MCP 是纵向的智能体 ↔ 工具；A2A 是横向的智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | “同类协议” | 若干相邻规范；截至 2026 年，A2A 的采用度最高。 |

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — 官方规范
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布公告
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — 参考实现与 SDK
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 对比
