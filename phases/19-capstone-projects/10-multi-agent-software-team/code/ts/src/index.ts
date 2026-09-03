/**
 * 多智能体软件团队：综合项目骨架入口（TypeScript）。
 *
 * 对应 docs/en.md 中的角色拆分（planner / coder / reviewer，加上以轮询方式调度
 * 他们的 coordinator）和工作树启动步骤（生产环境中每个分支使用 Daytona 沙箱；
 * 此处使用拒绝 denylist shell 命令的 execFile stub）。共享工作区位于内存中。
 *
 * 来源：phases/19-capstone-projects/10-multi-agent-software-team/docs/en.md
 * 技术栈参考：SWE-AF factory、MetaGPT roles、AutoGen 0.4 actor graph。
 */

import { Coordinator } from "./coordinator.js";
import { launchWorktree } from "./runtime.js";

async function worktreeDemo(): Promise<void> {
  console.log("[团队] 工作树 stub：带 denylist 的 execFile");
  const ok = await launchWorktree({
    branch: "feature/refund-rounding",
    command: "node",
    argv: ["-e", "console.log('编码者沙箱已就绪：' + process.env.BRANCH)"],
  });
  console.log("  node 标准输出：", ok.stdout.trim());
  if (ok.stderr) console.log("  node 标准错误：", ok.stderr.trim());

  const refused = await launchWorktree({
    branch: "feature/refund-rounding",
    command: "rm",
    argv: ["-rf", "/"],
  });
  console.log("  rm 已被拒绝：", refused.refused);

  const shellInjected = await launchWorktree({
    branch: "feature/refund-rounding",
    command: "node",
    argv: ["-e", "1", ";", "echo", "pwned"],
  });
  console.log("  注入已被拒绝：", shellInjected.refused);
}

function teamDemo(): void {
  console.log("[团队] coordinator 演示：从 issue 到合并后的 diff");
  const coordinator = new Coordinator();
  const result = coordinator.run({
    from: "user",
    to: "planner",
    topic: "issue.opened",
    body: "refund amounts off-by-one cent on edge rounding cases",
    ts: Date.now(),
  });
  console.log("  已批准：", result.approved, "轮次：", result.turns);
  console.log("  文件：");
  for (const file of coordinator.workspaceFiles()) {
    console.log(
      `    ${file.path}（编写者=${file.lastWriter} 修订=${file.revisions}）`,
    );
  }
  console.log("  消息日志：");
  for (const m of coordinator.messageLog()) {
    console.log(`    ${m.from} -> ${m.to} :: ${m.topic}`);
  }
  console.log("  统计：", coordinator.stats());
}

async function main(): Promise<void> {
  teamDemo();
  console.log();
  await worktreeDemo();
}

main().catch((err) => {
  console.error("[团队] 致命错误：", err);
  process.exit(1);
});
