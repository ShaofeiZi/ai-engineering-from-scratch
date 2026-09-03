# 毕业项目 19/03 — 实时语音助手（TypeScript）

多文件 TypeScript Web 客户端测试框架，用于 `../docs/en.md` 中描述的流式语音流水线。包含离线状态机模拟以及基于 `ws` 包的实时 WebSocket 服务器。

## 目录结构

```text
src/
  index.ts        entry point; runs two offline sessions, probes the live ws, exits 0
  server.ts       hono /healthz + ws upgrade via WebSocketServer
  orchestrator.ts IDLE -> LISTENING -> WAITING -> THINKING -> SPEAKING with barge-in
  vad.ts          turn-completion scorer + synthetic 20ms-frame generator
  protocol.ts     zod-validated frame envelope (event / summary)
  types.ts        AudioChunk, Metrics, SessionOptions, SessionSummary
tests/
  vad.test.ts
  orchestrator.test.ts
  protocol.test.ts
```

## 运行

```bash
npm install
npm start                # 运行两个离线会话和 WebSocket 自探测，随后以状态码 0 退出
npm start -- --serve     # 保持 WebSocket 服务器运行；按 Ctrl-C 停止
npm test                 # 通过 tsx 运行 node --test
npm run typecheck        # 运行 tsc --noEmit
```

非交互式 `npm start` 路径会断言：干净会话到达 `first_audio_out`，打断会话至少注册一个打断事件，以及实时 WebSocket 探测在关闭前收到一个 `summary` 帧。
