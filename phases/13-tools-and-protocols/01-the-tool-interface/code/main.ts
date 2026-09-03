// 第 13 阶段第 01 课——工具接口，TypeScript 版。
//
// 对应 code/main.py：描述 -> 决策 -> 执行 -> 观察。
// "决策" 步骤用关键词路由模拟，以便离线运行循环；
// 替换为任意真实 provider 客户端后，整体结构保持不变。
//
// 规范参考：
//   OpenAI 工具调用     https://platform.openai.com/docs/guides/function-calling
//   Anthropic 工具使用  https://docs.anthropic.com/en/docs/build-with-claude/tool-use
//   MCP 工具原语       https://modelcontextprotocol.io/specification/2026-07-28
//
// 运行：npx tsx code/main.ts

import { randomUUID } from "node:crypto";

const MAX_TURNS = 5;

type JsonSchema = {
  type?: "object" | "string" | "number" | "integer" | "boolean" | "array";
  properties?: Record<string, JsonSchema>;
  required?: string[];
  enum?: unknown[];
};

type ToolArgs = Record<string, unknown>;
type ToolResult = Record<string, unknown>;

type Tool = {
  name: string;
  description: string;
  inputSchema: JsonSchema;
  executor: (args: ToolArgs) => ToolResult;
  consequential?: boolean;
};

type HistoryEntry =
  | { role: "user"; content: string }
  | { role: "tool"; id: string; name: string; content: string };

type ToolCall = {
  id: string;
  name: string;
  arguments: ToolArgs;
};

type Decision = { content: string } | { toolCalls: ToolCall[] };

function toolAdd(args: ToolArgs): ToolResult {
  const a = args.a as number;
  const b = args.b as number;
  return { sum: a + b };
}

function toolGetTime(args: ToolArgs): ToolResult {
  const timezone = (args.timezone as string | undefined) ?? "UTC";
  const now = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
  return { now, timezone };
}

function toolGetWeather(args: ToolArgs): ToolResult {
  const fake: Record<string, number> = {
    Bengaluru: 28,
    Tokyo: 12,
    Zurich: 4,
    Lagos: 31,
  };
  const city = args.city as string;
  const units = (args.units as string | undefined) ?? "celsius";
  const temp = fake[city] ?? 20;
  return { city, temp, units };
}

const REGISTRY: Tool[] = [
  {
    name: "add",
    description:
      "当用户请求两个数字之和时使用。" +
      "不要用于减法、乘法或符号代数运算。",
    inputSchema: {
      type: "object",
      properties: {
        a: { type: "number" },
        b: { type: "number" },
      },
      required: ["a", "b"],
    },
    executor: toolAdd,
  },
  {
    name: "get_time",
    description:
      "当用户询问当前时间时使用。" +
      "不要用于历史日期或未来日程安排。",
    inputSchema: {
      type: "object",
      properties: {
        timezone: { type: "string" },
      },
      required: [],
    },
    executor: toolGetTime,
  },
  {
    name: "get_weather",
    description:
      "当用户询问某个指定城市的当前天气状况时使用。" +
      "不要用于天气预报或历史气象数据。",
    inputSchema: {
      type: "object",
      properties: {
        city: { type: "string" },
        units: { type: "string", enum: ["celsius", "fahrenheit"] },
      },
      required: ["city"],
    },
    executor: toolGetWeather,
  },
];

function validate(schema: JsonSchema, value: unknown): string[] {
  const errors: string[] = [];
  const t = schema.type;

  if (t === "object") {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return [`期望 object，实际为 ${describeType(value)}`];
    }
    const obj = value as Record<string, unknown>;
    for (const field of schema.required ?? []) {
      if (!(field in obj)) errors.push(`缺少必填字段 '${field}'`);
    }
    for (const [key, sub] of Object.entries(schema.properties ?? {})) {
      if (key in obj) errors.push(...validate(sub, obj[key]));
    }
    return errors;
  }

  if (t === "number" && typeof value !== "number") {
    errors.push(`期望 number，实际为 ${describeType(value)}`);
  }
  if (t === "string" && typeof value !== "string") {
    errors.push(`期望 string，实际为 ${describeType(value)}`);
  }
  if (schema.enum && !schema.enum.includes(value as never)) {
    errors.push(`值 ${JSON.stringify(value)} 不在枚举 ${JSON.stringify(schema.enum)} 中`);
  }
  return errors;
}

function describeType(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function newCallId(): string {
  return `call_${randomUUID().replace(/-/g, "").slice(0, 8)}`;
}

// 模型的替代实现。通过关键词路由，使循环可离线运行。
// 生产环境替代方案：替换为返回相同结构的 provider 调用。
export function fakeDecide(userMsg: string, history: HistoryEntry[]): Decision {
  const last = history[history.length - 1];
  if (last && last.role === "tool") {
    return { content: `基于工具输出构建的最终答案：${last.content}` };
  }
  const msg = userMsg.toLowerCase();

  if (/\b(add|sum|plus)\b/.test(msg) || msg.includes("加") || msg.includes("求和")) {
    const nums = (msg.match(/-?\d+\.?\d*/g) ?? []).map((n) => Number(n));
    if (nums.length >= 2) {
      return {
        toolCalls: [
          { id: newCallId(), name: "add", arguments: { a: nums[0], b: nums[1] } },
        ],
      };
    }
  }

  if (msg.includes("time") || msg.includes("几点") || msg.includes("时间")) {
    return {
      toolCalls: [
        { id: newCallId(), name: "get_time", arguments: { timezone: "UTC" } },
      ],
    };
  }

  const weatherMatch =
    msg.match(/weather in (\w+)/) ?? userMsg.match(/([\p{L}\p{M}\p{N}_]+) 的天气/u);
  if (weatherMatch) {
    const city = weatherMatch[1][0].toUpperCase() + weatherMatch[1].slice(1);
    return {
      toolCalls: [
        {
          id: newCallId(),
          name: "get_weather",
          arguments: { city, units: "celsius" },
        },
      ],
    };
  }

  return { content: "我无法将该请求路由到任何已注册的工具。" };
}

function runLoop(userMsg: string): void {
  console.log("=".repeat(72));
  console.log(`用户 : ${userMsg}`);
  console.log("-".repeat(72));

  const toolsByName = new Map(REGISTRY.map((t) => [t.name, t]));
  const history: HistoryEntry[] = [{ role: "user", content: userMsg }];

  for (let turn = 1; turn <= MAX_TURNS; turn++) {
    const decision = fakeDecide(userMsg, history);

    if ("content" in decision) {
      console.log(`第 ${turn} 轮 决策 : 最终答案`);
      console.log(`模型 : ${decision.content}`);
      return;
    }

    for (const call of decision.toolCalls) {
      const tool = toolsByName.get(call.name);
      console.log(`第 ${turn} 轮 决策 : 调用 ${call.name} id=${call.id}`);
      console.log(`           参数 = ${JSON.stringify(call.arguments)}`);

      if (!tool) {
        console.log(`           错误 : 未知工具 ${call.name}`);
        return;
      }
      const errs = validate(tool.inputSchema, call.arguments);
      if (errs.length > 0) {
        console.log(`           校验错误 : ${JSON.stringify(errs)}`);
        return;
      }
      if (tool.consequential) {
        console.log("           关卡 : 工具有副作用，需确认");
      }

      const start = performance.now();
      const result = tool.executor(call.arguments);
      const ms = performance.now() - start;
      console.log(
        `第 ${turn} 轮 执行: ${tool.name} -> ${JSON.stringify(result)} [${ms.toFixed(2)} ms]`,
      );
      history.push({
        role: "tool",
        id: call.id,
        name: tool.name,
        content: JSON.stringify(result),
      });
    }
    console.log(`第 ${turn} 轮 观察: 历史长度 = ${history.length}`);
  }
  console.log("循环终止 : 触发 MAX_TURNS 熔断器");
}

function describeRegistry(): void {
  console.log("工具注册表");
  console.log("-".repeat(72));
  for (const t of REGISTRY) {
    const kind = t.consequential ? "有副作用" : "纯函数";
    console.log(`  ${t.name.padEnd(14)} [${kind}] - ${t.description}`);
  }
  console.log();
}

function main(): void {
  console.log("=".repeat(72));
  console.log("第 13 阶段第 01 课 - 工具接口（TypeScript 移植版）");
  console.log("=".repeat(72));
  describeRegistry();
  const queries = [
    "请计算 7 加 35",
    "现在几点？",
    "请告诉我 Bengaluru 的天气",
    "写一首关于茶的俳句",
  ];
  for (const q of queries) {
    runLoop(q);
    console.log();
  }
}

if (require.main === module) {
  main();
}
