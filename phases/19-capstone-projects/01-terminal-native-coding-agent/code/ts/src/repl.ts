import * as readline from "node:readline";
import { runAgent } from "./harness.ts";
import { runEval } from "./eval.ts";

export type Command =
  | { kind: "run"; task: string }
  | { kind: "eval" }
  | { kind: "help" }
  | { kind: "quit" }
  | { kind: "unknown"; raw: string };

export function parseCommand(line: string): Command {
  const trimmed = line.trim();
  if (!trimmed) return { kind: "help" };
  if (trimmed === "quit" || trimmed === "exit") return { kind: "quit" };
  if (trimmed === "help" || trimmed === "?") return { kind: "help" };
  if (trimmed === "eval") return { kind: "eval" };
  const m = /^run\s+(.+)$/.exec(trimmed);
  if (m) return { kind: "run", task: m[1] };
  return { kind: "unknown", raw: trimmed };
}

export function helpText(): string {
  return [
    "运行框架命令：",
    "  run <task>   针对脚本化模型运行单个任务的 plan/act/observe 循环",
    "  eval         运行离线评测并输出通过/失败计数",
    "  help         显示此帮助信息",
    "  quit         退出",
  ].join("\n");
}

export function isInteractive(): boolean {
  return process.stdin.isTTY === true && process.argv.includes("--repl");
}

export async function repl(sandbox: string): Promise<void> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log(helpText());
  const ask = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));
  while (true) {
    const line = await ask("agent> ");
    const cmd = parseCommand(line);
    if (cmd.kind === "quit") break;
    if (cmd.kind === "help") {
      console.log(helpText());
      continue;
    }
    if (cmd.kind === "eval") {
      const e = runEval(sandbox);
      console.log(`评测：通过=${e.passed} 失败=${e.failed}`);
      continue;
    }
    if (cmd.kind === "run") {
      const r = runAgent(cmd.task, sandbox);
      console.log(r.plan);
      console.log("---");
      console.log(
        `turns=${r.budget.turnsUsed} tokens=${r.budget.tokensUsed} ` +
          `dollars=$${r.budget.dollarsUsed.toFixed(3)} passed=${r.passed}`,
      );
      continue;
    }
    console.log(`未知命令：${cmd.raw}；请输入 'help'`);
  }
  rl.close();
}
