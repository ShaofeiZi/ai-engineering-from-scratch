# 第 12 课 - 视频理解管线（TypeScript UI）

毕业项目的 TypeScript 部分。Python 端（`code/main.py`）拥有
多向量索引和时间定位。本项目提供仪表板部分：
一个基于 Hono 的应用，覆盖四个管线阶段（分块、嵌入、索引、问答）。

## 布局

```text
src/
  index.ts     entry: demo (default) or HTTP server (--serve)
  server.ts    Hono routes (/, /jobs, /job/:id) + HTML index
  jobs.ts     JobStore + fixture seeder
  stages.ts    stage advance + overall status
  types.ts     Stage, StageState, Job
tests/
  stages.test.ts  job state transitions + store
```

## 运行

```bash
npm install
npm run typecheck
npm test
npm start              # 运行可自行终止的演示
npm run serve          # 在 :8123 上启动 HTTP 服务器
```
