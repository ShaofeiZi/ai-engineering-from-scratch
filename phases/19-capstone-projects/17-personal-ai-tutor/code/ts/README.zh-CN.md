# 第 17 课 — 个人 AI 导师（TypeScript Web 应用）

毕业项目的 TypeScript 部分。Python 侧负责交付学习者模型与导师策略；本项目提供 Web 应用接口：一个课程 DAG 遍历器、一个 BKT 风格的学习者模型，以及一个 FSRS-lite 间隔重复调度器，通过两个 HTTP 路由暴露出来。

## 目录结构

```text
src/
  index.ts       entry: demo (default) or HTTP server (--serve)
  server.ts      Hono routes (GET /lesson/next, POST /lesson/:id/submit)
  curriculum.ts  DAG fixture + Kahn topo sort + next-lesson picker
  mastery.ts     MasteryStore (per-lesson BKT-ish update)
  repetition.ts  scheduleNextDue (interval doubling / halving, clamped)
  types.ts       Lesson, Mastery, Pick
tests/
  curriculum.test.ts  topo order, BKT update, FSRS scheduling
```

## 运行

```bash
npm install
npm run typecheck
npm test
npm start            # 运行可自行终止的课程遍历
npm run serve        # 在 :8090 上启动 HTTP 服务器
```
