type LLMResponse = {
  output: string;
  tokens: number;
  calls: number;
};

type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

async function fakeLLMCall(
  systemPrompt: string,
  userMessage: string
): Promise<LLMResponse> {
  const inputLength = systemPrompt.length + userMessage.length;
  const simulatedTokens = Math.floor(inputLength / 4) + 500;

  await new Promise((resolve) => setTimeout(resolve, 50));

  return {
    output: `[响应：${userMessage.slice(0, 80)}...]`,
    tokens: simulatedTokens,
    calls: Math.floor(Math.random() * 5) + 1,
  };
}

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `你是一名全栈开发者。你必须：
1. 调研需求
2. 编写代码
3. 审查代码中的缺陷
4. 编写测试
在一次对话中完成所有这些工作。`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `调研：${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `基于以下调研：\n${contextWindow.join("\n")}\n\n现在为这项任务编写代码：${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `基于此前的全部上下文：\n${contextWindow.join("\n")}\n\n审查代码。`
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

function createSpecialist(
  name: string,
  systemPrompt: string
): SpecialistAgent {
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
  "你是一名技术研究员。阅读文档、发现规律并总结结论。只输出实现所需的事实。"
);

const coder = createSpecialist(
  "coder",
  "你是一名资深 TypeScript 开发者。根据需求和调研笔记，编写整洁且经过测试的代码。不要输出其他内容。"
);

const reviewer = createSpecialist(
  "reviewer",
  "你是一名代码审查员。找出缺陷、安全问题和逻辑错误。说明要具体，并引用行号。"
);

async function multiAgentPipeline(task: string): Promise<AgentResult> {
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
    .map((m) => `[来自 ${m.from}]：${m.content}`)
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
    .map((m) => `[来自 ${m.from}]：${m.content}`)
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
    content: messages
      .map((m) => `[${m.from} -> ${m.to}]：${m.content}`)
      .join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}

async function multiAgentFanOut(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const [researchResult, requirementsResult] = await Promise.all([
    researcher.run(`调研以下任务的技术方案：${task}`),
    createSpecialist(
      "requirements",
      "你是一名需求分析师。提取功能性需求和非功能性需求，不要遗漏。"
    ).run(`分析以下任务的需求：${task}`),
  ]);

  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  messages.push({
    from: "requirements",
    to: "coder",
    content: requirementsResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed + requirementsResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls + requirementsResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[来自 ${m.from}]：${m.content}`)
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

  const reviewResult = await reviewer.run(codeResult.content);
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages
      .map((m) => `[${m.from} -> ${m.to}]：${m.content}`)
      .join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}

async function main() {
  const task = "为 Express.js API 构建限流中间件";

  console.log("=== 单 Agent 方案 ===\n");
  const singleResult = await singleAgentApproach(task);
  console.log(`使用的 token：${singleResult.tokensUsed}`);
  console.log(`工具调用：${singleResult.toolCalls}`);
  console.log(`上下文：所有内容都在一个窗口中\n`);

  console.log("=== 多 Agent 流水线 ===\n");
  const pipelineResult = await multiAgentPipeline(task);
  console.log(`使用的 token：${pipelineResult.tokensUsed}`);
  console.log(`工具调用：${pipelineResult.toolCalls}`);
  console.log(`上下文：每个 Agent 只获得自身所需内容\n`);

  console.log("=== 多 Agent 扇出 ===\n");
  const fanOutResult = await multiAgentFanOut(task);
  console.log(`使用的 token：${fanOutResult.tokensUsed}`);
  console.log(`工具调用：${fanOutResult.toolCalls}`);
  console.log(`上下文：researcher 与 requirements 并行运行\n`);

  console.log("=== 对比 ===\n");
  console.log(
    `单 Agent 上下文污染：全部 ${singleResult.tokensUsed} 个 token 都在一个窗口中`
  );
  console.log(
    `多 Agent 隔离：${pipelineResult.tokensUsed} 个 token 分布在 3 个隔离窗口中`
  );
  console.log(
    `扇出并行：research 与 requirements 同时运行`
  );
}

main();
