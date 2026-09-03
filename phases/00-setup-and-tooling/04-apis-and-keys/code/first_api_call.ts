// 阶段 0 · 课程 04 — API 与密钥（TypeScript 移植版）。
// 从环境变量读取 ANTHROPIC_API_KEY，解析一个最小化的 .env 文件，
// 然后用全局 fetch 发起一次 /v1/messages 调用。设置 MOCK=1 可完全跳过网络。
// 参考：https://docs.anthropic.com/en/api/messages
//       https://nodejs.org/api/process.html#processenv
//       https://nodejs.org/api/globals.html#fetch（Node 18+ 自带 fetch）

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";


type MessagesRequest = {
  model: string;
  max_tokens: number;
  messages: { role: "user" | "assistant"; content: string }[];
};

type MessagesResponse = {
  content: { type: string; text: string }[];
  usage: { input_tokens: number; output_tokens: number };
};

// .env 加载器。各框架的格式都一致；这里为了保持可移植性而不引入依赖。
// 每行 KEY=VALUE，以 # 开头为注释，值两侧可选地带有引号。
function loadDotenv(path: string): Record<string, string> {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    return {};
  }
  const out: Record<string, string> = {};
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

function mergeEnv(): NodeJS.ProcessEnv {
  // process.env 优先，这样用户无需编辑文件即可覆盖其中的配置。
  const fromFile = loadDotenv(resolve(process.cwd(), ".env"));
  return { ...fromFile, ...process.env };
}

// 这个固定响应的形状与真实的 /v1/messages 响应一致，
// 因此无论是否设置 MOCK=1，外层代码都完全相同。
const MOCK_RESPONSE: MessagesResponse = {
  content: [
    {
      type: "text",
      text: "A neural network is a stack of differentiable functions that learns patterns by adjusting weights against a loss signal.",
    },
  ],
  usage: { input_tokens: 12, output_tokens: 28 },
};

async function callMessages(apiKey: string, request: MessagesRequest): Promise<MessagesResponse> {
  if (process.env.MOCK === "1" || apiKey === "mock") {
    return MOCK_RESPONSE;
  }

  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(request),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`anthropic ${resp.status}: ${body.slice(0, 200)}`);
  }
  return (await resp.json()) as MessagesResponse;
}

async function main(): Promise<number> {
  const env = mergeEnv();
  const model = (env.LLM_MODEL ?? "").trim() || "claude-sonnet-5";
  const apiKey = env.ANTHROPIC_API_KEY ?? "mock";
  const usingMock = process.env.MOCK === "1" || apiKey === "mock";

  process.stdout.write("=== API 调用 ===\n\n");
  process.stdout.write(
    usingMock
      ? "模式：MOCK（不访问网络）。如需实时调用，请取消设置 MOCK 并导出 ANTHROPIC_API_KEY。\n\n"
      : "模式：LIVE。\n\n",
  );

  const request: MessagesRequest = {
    model,
    max_tokens: 256,
    messages: [{ role: "user", content: "What is a neural network in one sentence?" }],
  };

  try {
    const response = await callMessages(apiKey, request);
    const text = response.content[0]?.text ?? "";
    process.stdout.write(`响应：${text}\n`);
    process.stdout.write(
      `Token 用量：输入 ${response.usage.input_tokens}，输出 ${response.usage.output_tokens}\n`,
    );
    return 0;
  } catch (err) {
    process.stderr.write(`请求失败：${(err as Error).message}\n`);
    return 1;
  }
}

main().then((code) => process.exit(code));
