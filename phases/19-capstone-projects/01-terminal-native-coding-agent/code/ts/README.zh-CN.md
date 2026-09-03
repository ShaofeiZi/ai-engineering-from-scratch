# 毕业项目 19/01 — 终端原生编程智能体（TypeScript）

多文件 TypeScript 测试框架，实现了 `../docs/en.md` 中描述的
计划/执行/观察循环。离线、确定性、零网络调用。

## 目录结构

```text
src/
  index.ts     entry point; runs a scripted demo and the eval, then exits 0
  repl.ts      interactive command parser (run / eval / help / quit)
  harness.ts   the plan-act-observe loop, wired through the hook bus
  hooks.ts     eight-event hook bus plus a destructive-command guard
  model.ts     scripted offline LLM that drives the demo
  tools.ts     read_file + run_shell with zod-validated args
  plan.ts     PlanState (todo rewrite) + Budget (turn / token / dollar ceilings)
  eval.ts      tiny pass/fail counter across three offline tasks
  types.ts     shared shape definitions
tests/
  harness.test.ts
  tools.test.ts
```

## 运行

```bash
npm install
npm start                # 运行脚本化演示和离线评测，随后以状态码 0 退出
npm start -- --repl      # 打开交互式智能体运行框架 REPL
npm test                 # 通过 tsx 运行 node --test
npm run typecheck        # 运行 tsc --noEmit
```

非交互式 `npm start` 路径会断言评估报告 `passed=3
failed=0`，且脚本化运行收敛到全部完成的计划。任何漂移
都会导致运行失败。
