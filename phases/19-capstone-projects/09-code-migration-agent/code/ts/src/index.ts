/**
 * 代码迁移智能体：看板骨架入口（TypeScript）。
 *
 * 对应 docs/en.md 中的看板层：智能体在沙箱中运行；此服务器为操作人员渲染进度。
 * Hono 路由提供 HTML 根页面、/migrations 和 /migrations/:id。状态机位于
 * migrations.ts；预算 + 成本位于 cost.ts；类型位于 types.ts。
 *
 * 来源：phases/19-capstone-projects/09-code-migration-agent/docs/en.md
 * Recipe 规范：https://docs.openrewrite.org 和 libcst Python 解析器。
 */

import { serve } from "@hono/node-server";
import { buildApp } from "./server.js";
import { defaultSeed, rolledUpStats, tickAll } from "./migrations.js";

function summarise(migrations: ReturnType<typeof defaultSeed>): void {
  const stats = rolledUpStats(migrations);
  console.log("[看板] 已初始化迁移：", migrations.length);
  for (const m of migrations) {
    const passed = m.files.filter((f) => f.status === "passed").length;
    console.log(
      `[看板] ${m.repo} ${m.sourceRuntime}->${m.targetRuntime} ` +
        `状态=${m.state} 文件=${passed}/${m.files.length} ` +
        `轮次=${m.turns}/${m.maxTurns} 成本=$${m.spentUsd.toFixed(2)}`,
    );
  }
  console.log("[看板] 汇总：", stats);
}

export function runDemoTicks(rounds: number): ReturnType<typeof defaultSeed> {
  const migrations = defaultSeed();
  for (let i = 0; i < rounds; i++) tickAll(migrations);
  return migrations;
}

function main(): void {
  console.log("[看板] 正在模拟智能体进度的 40 个 tick……");
  const migrations = runDemoTicks(40);
  summarise(migrations);
  if (process.env["SERVE"] === "1") {
    const port = Number(process.env["PORT"] ?? 8009);
    const app = buildApp(migrations);
    serve({ fetch: app.fetch, port }, (info) => {
      console.log(`[看板] 正在提供服务：http://localhost:${info.port}`);
    });
    setInterval(() => tickAll(migrations), 750).unref();
  } else {
    console.log(
      "[看板] 设置 SERVE=1 可在 PORT（默认 8009）上启动 HTTP 看板",
    );
  }
}

main();
