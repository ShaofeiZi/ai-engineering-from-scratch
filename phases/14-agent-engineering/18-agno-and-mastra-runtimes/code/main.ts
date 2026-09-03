// 第14阶段 · 第18课 — Agno 与 Mastra 运行时对比（TypeScript 移植版）。
// 最小化的 Mastra 形状草图：Agent + 工具注册表 + 工作流，并带有
// 模拟的 LLM 步骤。另附一个 Agno 形状的草图作为对比。仅用标准库 ——
// 真正的 Mastra 包会接入 Zod、Vercel AI SDK 和遥测。
// 参考：https://mastra.ai/docs/agents/overview
//       https://mastra.ai/docs/workflows/overview
//       https://docs.agno.com/introduction
//       https://sdk.vercel.ai/docs/foundations/agents

import process from "node:process";

// --- 共享 LLM 桩。Mastra 在此接入 Vercel AI SDK 的 `generateText`。

type LLMResponse = { text: string; inputTokens: number; outputTokens: number };

async function mockLLM(systemPrompt: string, userMessage: string): Promise<LLMResponse> {
  const inputTokens = Math.ceil((systemPrompt.length + userMessage.length) / 4);
  // 模拟网络延迟，不使用真实模型。
  await new Promise((r) => setTimeout(r, 5));
  return {
    text: `[mock reply to ${userMessage.slice(0, 60)}]`,
    inputTokens,
    outputTokens: 32,
  };
}

// --- Agno 形状：无状态 agent + 会话存储。每个请求创建一个全新的 agent，
// 历史记录保存在会话存储中（生产环境中即你的数据库）。

type AgnoAgent = {
  name: string;
  run: (prompt: string) => Promise<string>;
};

class AgnoSession {
  private turns = new Map<string, string[]>();
  append(sessionId: string, turn: string): void {
    const list = this.turns.get(sessionId) ?? [];
    list.push(turn);
    this.turns.set(sessionId, list);
  }
  history(sessionId: string): string[] {
    return [...(this.turns.get(sessionId) ?? [])];
  }
}

async function agnoHandler(
  session: AgnoSession,
  agent: AgnoAgent,
  sessionId: string,
  prompt: string,
): Promise<{ reply: string; elapsedUs: number }> {
  const start = process.hrtime.bigint();
  session.append(sessionId, `user: ${prompt}`);
  const reply = await agent.run(prompt);
  session.append(sessionId, `assistant: ${reply}`);
  const elapsedUs = Number((process.hrtime.bigint() - start) / 1000n);
  return { reply, elapsedUs };
}

// --- Mastra 形状：Agents + Tools + Workflows。

type ToolInputSchema = Record<string, "string" | "number" | "boolean">;
type ToolInput = Record<string, string | number | boolean>;
type ToolResult = { output: string };

type MastraTool = {
  id: string;
  description: string;
  inputSchema: ToolInputSchema;
  execute: (input: ToolInput) => Promise<ToolResult>;
};

// 轻量运行时检查，使工具可以拒绝形状不匹配的调用。真正的 Mastra
// 在此使用 zod schema + 推断的 TS 类型。
function checkSchema(schema: ToolInputSchema, input: ToolInput): string | null {
  for (const [key, expected] of Object.entries(schema)) {
    if (!(key in input)) return `missing field ${key}`;
    if (typeof input[key] !== expected) return `field ${key}: expected ${expected}, got ${typeof input[key]}`;
  }
  return null;
}

type ToolCall = { tool: string; input: ToolInput };
type AgentTrace = { tool: string; result: string }[];

class MastraAgent {
  constructor(
    readonly name: string,
    readonly instructions: string,
    private readonly tools: Map<string, MastraTool>,
  ) {}

  static withTools(name: string, instructions: string, tools: MastraTool[]): MastraAgent {
    const map = new Map<string, MastraTool>();
    for (const t of tools) map.set(t.id, t);
    return new MastraAgent(name, instructions, map);
  }

  async run(userMessage: string, calls: ToolCall[]): Promise<{ output: string; trace: AgentTrace; tokens: number }> {
    const trace: AgentTrace = [];
    let tokens = 0;

    // Agent 决定工具调用（此处为预先提供）。每次成功的调用会
    // 向轨迹追加一步；错误的调用则记录错误信息。
    for (const call of calls) {
      const tool = this.tools.get(call.tool);
      if (!tool) {
        trace.push({ tool: call.tool, result: "error: unknown tool" });
        continue;
      }
      const schemaError = checkSchema(tool.inputSchema, call.input);
      if (schemaError) {
        trace.push({ tool: call.tool, result: `error: ${schemaError}` });
        continue;
      }
      const { output } = await tool.execute(call.input);
      trace.push({ tool: call.tool, result: output });
    }

    // 最终的 LLM 步骤将轨迹 + 用户消息组合成一条回复。
    const traceText = trace.map((t) => `${t.tool}: ${t.result}`).join("\n");
    const reply = await mockLLM(this.instructions, `${userMessage}\n\nTool results:\n${traceText}`);
    tokens = reply.inputTokens + reply.outputTokens;
    return { output: reply.text, trace, tokens };
  }
}

// 工作流：一个有序的步骤列表。每个步骤接收上一步的输出。
type WorkflowStep<I, O> = { name: string; run: (input: I) => Promise<O> | O };

class MastraWorkflow {
  private steps: WorkflowStep<unknown, unknown>[] = [];
  addStep<I, O>(name: string, run: (input: I) => Promise<O> | O): MastraWorkflow {
    this.steps.push({ name, run: run as (input: unknown) => unknown });
    return this;
  }
  async run(initial: unknown): Promise<{ name: string; output: unknown }[]> {
    const trace: { name: string; output: unknown }[] = [];
    let current: unknown = initial;
    for (const step of this.steps) {
      current = await step.run(current);
      trace.push({ name: step.name, output: current });
    }
    return trace;
  }
}

// --- 演示

const searchTool: MastraTool = {
  id: "search",
  description: "在固定语料库上进行网络搜索",
  inputSchema: { query: "string" },
  execute: async (input) => ({ output: `3 results for ${String(input.query)}` }),
};

const summariseTool: MastraTool = {
  id: "summarise",
  description: "将文本压缩为一句话",
  inputSchema: { text: "string" },
  execute: async (input) => ({ output: `summary: ${String(input.text).slice(0, 40)}...` }),
};

async function main(): Promise<void> {
  process.stdout.write("=".repeat(70) + "\nAgno 与 Mastra 运行时对比 — 第14阶段 · 18\n" + "=".repeat(70) + "\n");

  // 1. Agno 形状 — 衡量 agent 创建 + 处理器延迟。
  process.stdout.write("\n1. Agno 形状（无状态 FastAPI 风格处理器）\n");
  const session = new AgnoSession();
  const agnoAgent: AgnoAgent = {
    name: "agno_a",
    run: async (prompt) => `[agno reply] ${prompt.slice(0, 40)}`,
  };
  for (let i = 0; i < 3; i += 1) {
    const { reply, elapsedUs } = await agnoHandler(session, agnoAgent, "s001", `query ${i}: how do I ship an agent`);
    process.stdout.write(`  第 ${i} 轮：${reply}  （处理器 ${elapsedUs} 微秒）\n`);
  }
  process.stdout.write(`  会话历史长度：${session.history("s001").length}\n`);
  process.stdout.write("  模式：每个请求创建全新 agent，会话保存状态，FastAPI/Hono 本身无状态。\n");

  // 2. Mastra 形状 — agent 先运行工具再进行总结。
  process.stdout.write("\n2. Mastra 形状（Agents + Tools + Workflows）\n");
  const mastraAgent = MastraAgent.withTools(
    "research_agent",
    "搜索、总结、引用",
    [searchTool, summariseTool],
  );
  const result = await mastraAgent.run("research agent engineering", [
    { tool: "search", input: { query: "agent engineering 2026" } },
    { tool: "search", input: { query: "BFCL V4 benchmarks" } },
    { tool: "unknown_tool", input: { query: "fails on purpose" } },
  ]);
  process.stdout.write(`  agent 输出：${result.output}  （约 ${result.tokens} tokens）\n`);
  for (const t of result.trace) process.stdout.write(`    工具 ${t.tool}：${t.result}\n`);

  // 3. 工作流 — 归一化 → 搜索 → 总结。
  process.stdout.write("\n3. 工作流运行\n");
  const workflow = new MastraWorkflow()
    .addStep<string, string>("normalise", (p) => p.trim().toLowerCase())
    .addStep<string, string>("search", async (p) => (await searchTool.execute({ query: p })).output)
    .addStep<string, string>("summarise", async (p) => (await summariseTool.execute({ text: p })).output);
  const workflowTrace = await workflow.run("  Agent Engineering 2026  ");
  for (const { name, output } of workflowTrace) process.stdout.write(`    ${name}：${String(output)}\n`);

  process.stdout.write("\n按技术栈选择：python+fastapi → Agno；typescript+next/vercel → Mastra。\n");
}

main();
