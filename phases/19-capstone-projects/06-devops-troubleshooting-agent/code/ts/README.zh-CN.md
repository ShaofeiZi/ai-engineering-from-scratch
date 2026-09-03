# 毕业项目 06 — DevOps 故障排查智能体（TypeScript）

`../main.py` 中值班智能体的 Slack 集成骨架。暴露一个斜杠命令端点和一个交互（按钮点击）端点，两者均由 Slack 的 HMAC-SHA256 请求签名加上 5 分钟重放窗口保护。破坏性修复操作仅在 Slack 卡片被批准后执行。

## 目录结构

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts          # 入口，包含演示和 HTTP 服务器
    server.ts         # Hono 应用，提供 /slack/command 和 /slack/interactivity
    slack_verify.ts   # HMAC v0 验证和时序安全比较
    agent.ts          # 模拟的假设排序器
    blocks.ts         # Block Kit 响应构建器
    types.ts          # 类型定义：Hypothesis、AgentReport、SlackResponse、OutboundCall
  tests/
    slack_verify.test.ts
    agent.test.ts
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

设置 `SLACK_SIGNING_SECRET=...` 以覆盖占位密钥。交互式服务器打印所选端口（`PORT` 未设置时随机选择）。

## 测试

通过 tsx 运行 `node --test`。覆盖范围：

- Slack 签名验证：有效签名通过，篡改签名被拒绝，过期时间戳（>5 分钟偏差）被拒绝，非数字时间戳被拒绝，长度不匹配路径在常数时间比较之前执行。
- 模拟智能体：OOM 关键词路径、CrashLoop 关键词路径、回退路径。
- 服务器：`/health`、`/slack/command` 正常/篡改/过期路径、`/slack/interactivity` 批准操作。
