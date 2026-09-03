# 毕业项目 08 — 生产级 RAG 聊天机器人（TypeScript）

聊天 UI 骨架，通过 Server-Sent Events 流式传输带有引用锚点的响应。与
`../main.py` 中的 Python 流水线配合使用。会话状态存储在一个以
`sessionId` 为键的进程内 Map 中，因此同一 session id 可以驱动多轮
对话。

## 目录结构

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts        # 入口，包含演示和 HTTP 服务器
    server.ts      # Hono 应用，提供 /、/chat/stream（SSE）、/sessions、/health
    session.ts     # 会话存储 SessionStore（Map<sessionId, Session>）
    stream.ts      # SSE 帧编码器、解析器、模拟检索器和 tokenizer
    types.ts        # 类型定义：Session、Turn、Citation、KbEntry、SseEvent
  tests/
    session.test.ts
    stream.test.ts
    server.test.ts
```

## 运行

```bash
npm install
npm run typecheck
npm test
npm start          # 执行一次自检，随后以状态码 0 退出
npm run serve      # 在 127.0.0.1:<port> 上启动交互式 HTTP 服务器
```

交互式服务器在 `PORT` 未设置时自动选取空闲端口，将聊天 HTML 客户端挂载到
`/`，并通过 `GET /chat/stream?sessionId=...&q=...` 进行流式传输。演示
客户端使用 `EventSource` 并监听 `session`、`citations`、`token` 和
`done` 事件。

## 测试

通过 tsx 使用 `node --test` 运行器。覆盖范围：

- SessionStore：创建、查找、追加、列表、对缺失 id 的无操作处理。
- SSE 编码器 + 解析器往返；按司法管辖区标签的检索加权；
  分词器回退 + "See also" 尾部。
- 服务器：`/`、`/health`、`/chat/stream` 正常路径（session +
  citations + token + done），缺少 q 时返回 400，多轮会话持久化，
  `/sessions` 列表。
