# 为什么需要多 Agent？

> 一个 Agent 撞上了能力边界。更明智的做法不是换成更大的 Agent，而是使用更多 Agent。

**Type:** 学习
**Languages:** TypeScript
**Prerequisites:** 第 14 阶段（智能体工程）
**Time:** 约 60 分钟

## 学习目标

- 识别单 Agent 的能力上限（上下文溢出、专业能力混杂、串行瓶颈），并解释何时应当把工作拆给多个 Agent
- 比较不同编排模式（流水线、并行扇出、监督者、分层），并根据任务结构选择合适的模式
- 设计一个具有明确角色边界、共享状态和通信契约的多 Agent 系统
- 分析多 Agent 的复杂性（延迟、成本、调试难度）与单 Agent 简洁性之间的权衡

## 问题

你在阶段 14 中构建了一个单 Agent。它运转良好，可以读取文件、执行命令、调用 API，并对结果进行推理。接着，你让它处理一个真实代码库：200 个文件、三种语言、依赖基础设施的测试，以及一项要求——写代码前还必须研究外部 API。

Agent 卡住了。不是因为 LLM 不够聪明，而是因为任务超出了一个 Agent 循环所能处理的范围。上下文窗口被文件内容塞满；Agent 忘记了 40 次工具调用前读过的内容；它试图同时扮演研究员、程序员和审查者，结果三件事都做不好。

这就是单 Agent 的能力上限。只要任务具备以下特征，你就会碰到它：

- **所需上下文超过单个窗口容量**——读取 50 个文件会轻易突破 20 万 token
- **不同阶段需要不同专业能力**——研究工作需要的提示方式与代码生成不同
- **工作可以并行开展**——既然三个文件可以同时读取，为什么要依次读取？

## 概念

### 单 Agent 的能力上限

单 Agent 意味着一套循环、一个上下文窗口和一条系统提示。可以把它想象成这样：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

会有三个方面失效：

1. **上下文饱和**——工具结果不断累积。到第 30 轮时，Agent 可能已经消耗了 15 万 token 的文件内容、命令输出和先前推理，第 5 轮出现的关键细节会被淹没。

2. **角色混乱**——一条写着“你既是研究员、程序员、审查者，又是测试人员”的系统提示，会让 Agent 每项研究和编码都只做一半，审查更是永远无法收尾。

3. **串行瓶颈**——Agent 先读文件 A，再读文件 B，然后读文件 C。这意味着三次串行 LLM 调用和三次串行工具执行，完全没有并行性。

### 多 Agent 解决方案

把工作拆开。给每个 Agent 分配一项工作、一个独立上下文窗口，以及一条为该工作定制的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个 Agent 都拥有：
- 一条聚焦的系统提示（“你是一名代码审查者，唯一职责是发现缺陷。”）
- 自己的上下文窗口（不会被其他 Agent 的工作污染）
- 明确的输入/输出契约（接收研究笔记，输出代码）

### 采用这种方式的真实系统

**Claude Code 子 Agent**——Claude Code 使用 `Task` 派生子 Agent 时，会创建一个任务范围明确的子 Agent。父 Agent 的上下文保持整洁；子 Agent 专注完成工作并返回摘要。

**Devin**——运行规划 Agent、编码 Agent 和浏览器 Agent。规划者把工作拆成步骤，编码者编写代码，浏览器则研究文档。它们各自拥有独立上下文。

**多 Agent 编码团队（SWE-bench）**——SWE-bench 上表现最好的系统会使用研究员阅读代码库、规划者设计修复方案，再由编码者完成实现。单 Agent 系统的得分较低。

**ChatGPT Deep Research**——并行派生多个搜索 Agent，让每个 Agent 探索不同角度，再综合所有结果。

### 连续谱

是否采用多 Agent 并不是非黑即白的选择，而是一条连续谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单 Agent**——一套循环，一条提示。适合简单任务。

**子 Agent**——父 Agent 为聚焦的子任务派生子 Agent。父 Agent 维护计划，子 Agent 汇报结果。Claude Code 采用的就是这种方式。

**流水线**——Agent 按顺序运行，Agent A 的输出成为 Agent B 的输入。适合研究 → 编码 → 审查 → 测试等分阶段工作流。

**团队**——多个 Agent 借助共享消息总线并行运行，各自承担一种角色，由编排器协调。适合需要同时运用不同技能的任务。

**集群**——许多相同或近似的 Agent 共享状态，没有固定的编排器。Agent 从队列中领取工作，适合高吞吐量并行任务。

### 四种多 Agent 模式

#### 模式 1：流水线

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个 Agent 对数据进行转换，再向后传递。它易于理解，但任一阶段失败都会阻塞后续阶段。

#### 模式 2：扇出/扇入

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

先将工作拆给多个 Agent 并行执行，再合并结果。它适合能够分解为多个独立子任务的工作。

#### 模式 3：编排器—工作者

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个智能编排器决定要做什么、将任务委派给工作者，再综合各项结果。编排器本身也是 Agent，并配有用于派生工作者的工具。

#### 模式 4：对等集群

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排器。Agent 之间进行点对点通信，决策从交互中涌现。它更难调试，却可以扩展到大量 Agent。

### 何时不应使用多 Agent

多 Agent 会增加复杂度。Agent 之间的每条消息都可能成为故障点。调试工作会从“阅读一段对话”变成“跨五个 Agent 追踪消息”。

**在以下情况下坚持使用单 Agent：**
- 任务能放进一个上下文窗口（工作数据少于约 10 万 token）
- 不同阶段不需要不同的系统提示
- 串行执行已经足够快
- 任务非常简单，拆分带来的开销大于收益

**复杂度成本：**
- 每个 Agent 边界都是一次有损压缩：Agent A 的完整上下文会被摘要成发给 Agent B 的消息
- 协调逻辑（谁在何时以何种顺序做什么）本身就会引入缺陷
- 延迟会上升：N 个 Agent 至少意味着 N 次串行 LLM 调用；如果它们需要来回沟通，调用次数还会更多
- 成本会成倍增加：每个 Agent 都会独立消耗 token

经验法则：如果任务不到 20 次工具调用就能完成，并且能放进 10 万 token，请坚持使用单 Agent。

```figure
swarm-messages
```

## 动手构建

### 第 1 步：负担过重的单 Agent

下面这个单 Agent 试图包办所有事情。它只有一条庞大的系统提示，并用同一个上下文窗口容纳研究、代码和审查内容：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方式存在以下问题：
- 上下文窗口会随每个阶段不断增长。到审查步骤时，其中既包含研究笔记，也包含代码和先前的推理。
- 系统提示过于笼统，无法针对每个阶段分别调优。
- 所有工作都无法并行运行。

### 第 2 步：专家 Agent

现在将它拆开，让每个 Agent 只负责一项工作：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有一条聚焦的提示，并且获得一个干净的上下文窗口，其中只包含它完成工作所需的输入。

### 第 3 步：通过消息协调

使用显式消息传递把这些专家连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个 Agent 只接收发送给自己的消息，因此不存在上下文污染。研究员阅读文档时消耗的 5 万 token 永远不会进入审查者的上下文。

### 第 4 步：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多 Agent 版本使用的 token 总量更多（三个 Agent、三次独立的 LLM 调用），但每个 Agent 的上下文都能保持干净。由于系统提示针对各个阶段进行了专门设计，每个阶段的产出质量都会提高。

## 实际应用

本课会产出一个可复用的提示词，帮助你判断何时应采用多 Agent。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四位专家——“测试” Agent。它接收编码者输出的代码和审查者给出的反馈，然后编写测试
2. 修改流水线，使审查者可以把反馈发回编码者，形成修订循环（最多 2 轮）
3. 将串行流水线改成扇出模式：让研究员和一个“需求分析” Agent 并行运行，合并两者的输出后再交给编码者

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|----------------------|
| Swarm | “AI Agent 的蜂群思维” | 一组共享状态且没有固定领导者的对等 Agent。其行为从局部交互中涌现。 |
| 编排器 | “老板 Agent” | 一种可通过工具派生和管理其他 Agent 的 Agent。它负责规划和委派，但不一定亲自完成实际工作。 |
| 协调器 | “交通警察” | 一个非 Agent 组件（通常只是代码，而不是 LLM），根据规则在 Agent 之间路由消息。 |
| 共识 | “Agent 达成一致” | 一种要求多个 Agent 在继续执行前达成一致的协议，用于解决彼此冲突的输出。 |
| 涌现行为 | “Agent 自己想明白了” | 由 Agent 交互产生、但未被显式编程的系统级模式；既可能有益，也可能有害。 |
| 扇出/扇入 | “Agent 版 MapReduce” | 将任务拆给多个 Agent 并行执行（扇出），再合并它们的结果（扇入）。 |
| 消息传递 | “Agent 相互交谈” | Agent 之间的通信机制：将结构化数据从一个 Agent 发送到另一个 Agent，以此取代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)——多 Agent 模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)——微软的多 Agent 对话框架
- [Claude Code 子 Agent 文档](https://docs.anthropic.com/en/docs/claude-code)——Claude Code 如何通过 Task 进行委派
- [CrewAI 文档](https://docs.crewai.com/)——基于角色的多 Agent 框架
