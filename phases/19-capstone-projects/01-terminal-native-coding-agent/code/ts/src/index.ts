// 综合项目 19/01：终端原生编码智能体运行框架（多文件 TypeScript）。
//
// 资料来源：
//   本课程的 docs/en.md（带有八个 2026 hook 的 Bun + Ink TUI 运行框架）
//   Claude Code 文档            https://docs.anthropic.com/en/docs/claude-code
//   Model Context Protocol      https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/
//   OpenTelemetry GenAI semconv https://opentelemetry.io/docs/specs/semconv/gen-ai/
//
// 综合项目的运行框架部分：REPL 命令解析器（repl.ts）、包含 read_file/run_shell
// 的工具分发器（tools.ts）、脚本化离线模型（model.ts）、八事件 hook 总线
//（hooks.ts）、每轮整体重写的计划状态（plan.ts），以及一个小型通过/失败评测
// 计数器（eval.ts）。非交互路径在退出前断言评测通过，因此程序可以自验证。

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { runAgent } from "./harness.ts";
import { runEval } from "./eval.ts";
import { isInteractive, repl } from "./repl.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main(): Promise<void> {
  const sandbox = path.resolve(__dirname, "..");
  if (isInteractive()) {
    await repl(sandbox);
    return;
  }
  const task = "演示无需网络调用的 plan-act-observe 循环";
  const result = runAgent(task, sandbox);
  console.log(result.plan);
  console.log("---");
  console.log(
    `turns=${result.budget.turnsUsed} tokens=${result.budget.tokensUsed} ` +
      `dollars=$${result.budget.dollarsUsed.toFixed(3)}`,
  );
  console.log("---");
  console.log(`追踪事件数：${result.trace.length}`);
  for (const ev of result.trace) console.log(" ", JSON.stringify(ev));
  console.log("---");
  const e = runEval(sandbox);
  console.log(`评测：通过=${e.passed} 失败=${e.failed}`);
  if (e.passed !== 3 || e.failed !== 0) {
    throw new Error(`评测回归：通过=${e.passed} 失败=${e.failed}`);
  }
  if (!result.passed) {
    throw new Error("脚本化演示运行未收敛到全部完成的计划");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
