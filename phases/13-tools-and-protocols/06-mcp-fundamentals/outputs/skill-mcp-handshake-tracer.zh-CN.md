---
name: mcp-request-tracer
description: 逐条审计 MCP 转录记录，覆盖现代无状态协议时代与显式遗留协议时代。
version: 2.0.0
phase: 13
lesson: 06
tags: [mcp, json-rpc, stateless, metadata, compatibility]
---

给定一组 MCP JSON-RPC 封包，依据 MCP `2026-07-28` 对每条消息进行独立审计。检测遗留流量，但绝不假定握手或协议会话存在。

产出：

1. 消息标注。标明方向、JSON-RPC 类型、方法、原语、请求 id，以及检测到的时代。
2. 现代元数据检查。对每个请求，验证 `params._meta.io.modelcontextprotocol/protocolVersion` 和 `params._meta.io.modelcontextprotocol/clientCapabilities`。记录是否存在推荐的 `clientInfo`。
3. 结果检查。验证每个现代成功响应具有 `resultType: "complete"` 或其他已指定的结果类型，并在结果 `_meta` 中包含推荐的服务端身份信息。
4. 发现与版本检查。验证现代服务端实现了 `server/discover`。将 `-32022` 解读为现代证据，并检查 `data.requested` 和 `data.supported`。
5. 缓存检查。对于 `server/discover`、list 方法和 `resources/read`，要求 `ttlMs` 和 `cacheScope`。对非确定性的列表排序进行标记。
6. 方向检查。拒绝现代流量中由服务端发起的 JSON-RPC 请求。允许与请求相关的通知以及客户端开启的 `subscriptions/listen` 流。
7. 兼容性检查。将 `initialize` 和 `notifications/initialized` 标记为仅遗留。不要求它们出现在现代流量中。

硬性拒绝：

- 将 stdio 进程、HTTP 连接或 `Mcp-Session-Id` 视为现代协议状态。
- 从更早的请求中推断客户端能力。
- 在识别到 `-32020`、`-32021` 或 `-32022` 等现代错误后回退到遗留协议。
- 接受没有 `resultType` 的现代成功响应。

拒绝规则：

- 如果转录记录不是 JSON-RPC 2.0，停止并指明不兼容的封包。
- 如果被要求静默重写证据，拒绝。保留原始转录记录，并单独产出一个修正后的示例。

按到达顺序，每条消息输出一行：

```text
[request/modern/tools] id=7 tools/list metadata=valid
```

以现代、遗留、无效和模糊消息的计数结尾，随后给出第一个纠正措施。
