// 阶段 0 · 课程 01 — 开发环境校验器（TypeScript 移植版）。
// 探测 node 版本以及 git、python3、cargo、deno 是否存在；与 verify.py 行为一致。
// 参考：https://nodejs.org/api/process.html  https://nodejs.org/api/child_process.html

import { execFileSync } from "node:child_process";
import process from "node:process";

type ProbeFn = () => { ok: boolean; detail?: string };

type Probe = {
  name: string;
  required: boolean;
  run: ProbeFn;
};

function whichVersion(cmd: string, args: string[] = ["--version"]): ReturnType<ProbeFn> {
  // 使用 execFile（而非 exec）以避免启动 shell，从而防止用户 PATH 查找被二次解释。
  try {
    const out = execFileSync(cmd, args, {
      stdio: ["ignore", "pipe", "ignore"],
      encoding: "utf8",
      timeout: 4000,
    });
    return { ok: true, detail: out.trim().split("\n")[0] };
  } catch {
    return { ok: false };
  }
}

const PROBES: Probe[] = [
  {
    name: "Node.js 20+",
    required: true,
    run: () => {
      const major = Number.parseInt(process.versions.node.split(".")[0]!, 10);
      return { ok: major >= 20, detail: `v${process.versions.node}` };
    },
  },
  {
    name: "TypeScript 运行器 (tsx)",
    required: false,
    run: () => whichVersion("npx", ["-y", "tsx", "--version"]),
  },
  {
    name: "Git",
    required: true,
    run: () => whichVersion("git"),
  },
  {
    name: "Python 3.10+",
    required: true,
    run: () => {
      const probe = whichVersion("python3");
      if (!probe.ok || !probe.detail) return probe;
      // detail 形如 "Python 3.11.7"；这里提取主版本号与次版本号。
      const match = probe.detail.match(/(\d+)\.(\d+)/);
      if (!match) return { ok: false, detail: probe.detail };
      const [major, minor] = [Number(match[1]), Number(match[2])];
      const ok = major > 3 || (major === 3 && minor >= 10);
      return { ok, detail: probe.detail };
    },
  },
  {
    name: "Rust (cargo)",
    required: false,
    run: () => whichVersion("cargo"),
  },
  {
    name: "Deno",
    required: false,
    run: () => whichVersion("deno"),
  },
];

function run(): number {
  process.stdout.write("\n=== AI Engineering from Scratch —— 环境检查 ===\n\n");

  let requiredPassed = 0;
  let requiredTotal = 0;

  for (const probe of PROBES) {
    const result = probe.run();
    const tag = result.ok ? "PASS" : "FAIL";
    const detail = result.detail ? ` (${result.detail})` : "";
    const flag = probe.required ? "" : "  [可选]";
    process.stdout.write(`  [${tag}] ${probe.name}${detail}${flag}\n`);
    if (probe.required) {
      requiredTotal += 1;
      if (result.ok) requiredPassed += 1;
    }
  }

  process.stdout.write(`\n结果：${requiredPassed}/${requiredTotal} 项必需检查通过\n`);
  if (requiredPassed === requiredTotal) {
    process.stdout.write("\n环境已就绪。请从阶段 1 开始。\n\n");
    return 0;
  }
  process.stdout.write("\n请修复上方失败的必需检查项，然后重新运行。\n\n");
  return 1;
}

process.exit(run());
