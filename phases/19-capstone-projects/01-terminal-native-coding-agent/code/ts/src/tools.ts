import { readFileSync, realpathSync } from "node:fs";
import * as path from "node:path";
import { z } from "zod";
import type { ToolArgs, ToolFn } from "./types.ts";

export const TRUNCATE_BYTES = 4096;

export const ReadFileArgs = z.object({ path: z.string().min(1) });
export const RunShellArgs = z.object({ cmd: z.string().min(1) });

export class SandboxPathError extends Error {
  readonly code = "PATH_OUTSIDE_SANDBOX";

  constructor(detail?: string) {
    super(detail ? `路径越出沙箱范围：${detail}` : "路径越出沙箱范围");
    this.name = "SandboxPathError";
  }
}

export function toolReadFile(sandbox: string, args: ToolArgs): string {
  const parsed = ReadFileArgs.parse(args);
  const candidate = path.resolve(sandbox, parsed.path);
  const sandboxResolved = path.resolve(sandbox);
  let full: string;
  let root: string;
  try {
    full = realpathSync(candidate);
    root = realpathSync(sandboxResolved);
  } catch (err) {
    throw new SandboxPathError((err as Error).message);
  }
  if (full !== root && !full.startsWith(root + path.sep)) {
    throw new SandboxPathError();
  }
  const data = readFileSync(full, "utf8");
  return data.slice(0, TRUNCATE_BYTES);
}

export function toolRunShell(_sandbox: string, args: ToolArgs): string {
  const parsed = RunShellArgs.parse(args);
  const stub: Record<string, string> = {
    ls: "README.md\nsrc\ntests",
    "git status": "位于分支 agent/demo\n无内容可提交，工作树干净",
  };
  const out = stub[parsed.cmd] ?? `(stub) 已运行：${parsed.cmd}`;
  return `exit=0\n${out.slice(0, TRUNCATE_BYTES)}`;
}

export const TOOLS: Record<string, ToolFn> = {
  read_file: toolReadFile,
  run_shell: toolRunShell,
};
