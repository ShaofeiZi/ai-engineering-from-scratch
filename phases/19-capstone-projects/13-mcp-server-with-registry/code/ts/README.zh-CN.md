# 第 13 课 - 无状态 MCP 服务器（TypeScript）

这是毕业项目的 TypeScript 部分。Python 端（`code/main.py`）负责注册表与策略门；
本项目实现 MCP 传输层：手写的、基于 stdio 的换行分隔 JSON-RPC 2.0，
附带三个模拟事件工具。它直接实现 MCP `2026-07-28` 规范，不依赖
`@modelcontextprotocol/sdk`，因此你可以逐字节检查线上传输的每一个字节。

尽管模拟事件存储会持久化数据，协议本身是无状态的。每个请求都在
`params._meta` 中重复其协议版本和客户端能力；不存在任何连接、进程或先前的
请求来建立会话。服务器提供必需的 `server/discover`，在每个成功结果中
标识自身，并发布确定性、可缓存的工具列表。`tools/call` 依据
`tools/list` 返回的同一有界模式校验参数；对已知工具的格式错误参数会返回
一个完整的工具结果，其中 `isError: true`，且永远不会到达其执行器。

运行时标识为 `com.example/internal-incidents`。它使用已验证发布者
`example.com` 的反向 DNS 命名空间。与之匹配的已发布 `server.json`
必须使用相同的名称，即使本地 npm 包有自己的私有项目名称。

## 目录结构

```text
src/
  index.ts      entry: fixture demo (default) or stdio loop (--serve)
  transport.ts  stdin readline + fixture replay
  protocol.ts   request validation / server/discover / tools/list / tools/call
  tools.ts      three incident tools + executors
  types.ts      JSON-RPC + tool shapes
tests/
  protocol.test.ts  stateless metadata, discovery, tools, errors, roundtrip
```

## 运行

```bash
npm install
npm run typecheck
npm test
npm start            # 运行可自行终止的夹具演示
npm run serve        # 运行真实的 stdio 循环（等待 stdin 输入）
```

演示会自行终止。真实的 stdio 服务器在输入流关闭前会一直保持运行；
不存在 MCP 关闭请求或初始化握手。
