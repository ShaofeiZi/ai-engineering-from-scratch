# 第 16 课 - GitHub Issue 到 PR 智能体（TypeScript Webhook 接收器）

这是毕业项目的 TypeScript 部分。Python 端负责智能体循环和
调度器；YAML 端负责 Actions 工作流。本项目是 GitHub
App 的 webhook 接收器：对原始请求体进行 HMAC 校验，按事件类型路由，
为 `issues.opened` 分发一个桩智能体。

## 目录结构

```text
src/
  index.ts    entry: demo (default) or HTTP server (--serve)
  server.ts   Hono webhook receiver (POST /webhook)
  verify.ts   X-Hub-Signature-256 HMAC, timing-safe
  router.ts   event-type routing (ping, issues, pull_request)
  agent.ts    stub agent + audit log
  types.ts    payload + audit shapes
tests/
  verify.test.ts  signature pass, tampered, router pathing
```

## 运行

```bash
npm install
npm run typecheck
npm test
npm start            # 运行可自行终止的演示（进程内重放）
npm run serve        # 在 :8081 上启动 HTTP 服务器
```

HMAC 密钥从 `GH_WEBHOOK_SECRET` 读取（演示默认为
`demo-shared-secret`）。
