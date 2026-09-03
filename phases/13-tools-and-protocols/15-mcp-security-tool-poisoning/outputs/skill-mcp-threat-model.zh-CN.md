---
name: mcp-threat-model
description: 针对 MCP 2026-07-28 部署，围绕元数据、路由、授权、MRTR 和兼容性边界进行基于证据的威胁建模。
version: 2.0.0
phase: 13
lesson: 15
tags: [mcp, security, stateless, tool-poisoning, mrtr]
---

给定一个 MCP 部署，产出一份基于证据的威胁模型。假设任何服务器、包、缓存、注册表条目或网关路由都可能已被攻破。

## 必需输入

- 客户端、网关、服务器、授权服务器和注册表的信任边界。
- 完整的归一化工具描述符与已批准的摘要。
- 认证主体、签发方、受众、作用域和工具策略。
- 当前接受的协议修订版本与遗留版本。
- MRTR 操作、输入模式、状态保护和重放策略。
- 缓存作用域、TTL、订阅路由和审计留存。

## 需产出

1. 线路验证。先验证每次请求的版本与能力，再在版本支持之前验证路由头的一致性。不匹配时要求返回 HTTP 400 `-32020`；当匹配的版本不受支持时返回 HTTP 400 `-32022` 并附带确切的支持和请求数据；未知方法返回 HTTP 404 `-32601`；通知被接受时返回 202 和空响应体。
2. 描述符审查。报告投毒指标、完整描述符摘要变更、未知工具以及模式或注解变更。
3. 命名空间映射。为每个后端工具给出一个限定公共名称，并拒绝静默的冲突消解。
4. 授权矩阵。将已认证的主体和签发方映射到资源、工具、参数约束和作用域。不得将 `clientInfo` 或 `serverInfo` 用作身份标识。
5. MRTR 审查。确认每个 `inputRequests` 条目都是客户端所声明能力支持的完整嵌入请求。将 `elicitation: {}` 视为隐式表单支持，将 `elicitation: {form: {}}` 视为显式表单支持。拒绝仅 URL 的 elicitation，返回 HTTP 400 `-32021` 和 `data.requiredCapabilities.elicitation.form`。将受保护的 `requestState` 绑定到方法、工具、确切参数、主体、目的、过期时间和 nonce。在由所有处理器实例共享的有界且按 TTL 清理的重放存储中，先按键匹配并验证每个 `inputResponses` 条目，再原子性地消费 nonce。
6. 风险轴审查。标记任何将不可信输入、敏感数据和有重大影响的操作结合在一起的自动化步骤。
7. 缓存与订阅审查。确保依赖于用户的结果是私有的，且长期存活的通知使用 `subscriptions/listen`。
8. 兼容性边界。将任何较旧的握手、会话、GET 流、服务器回调或实验性任务行为隔离在显式版本门控之后。
9. 传输边界。识别实现是完整的 HTTP 适配器还是进程内协议模型。将模型关联到 Lesson 09 的 JSON Content-Type 以及 JSON 加 SSE Accept 验证。
10. 修复顺序。给出三个最高杠杆的修复措施，包含负责人和验收证据。

## 硬性拒绝

- 静默的工具覆盖或按发现顺序选择路由。
- 在未经人工或策略重新批准的情况下更新描述符摘要。
- 将自报告的客户端或服务器信息视为认证。
- 将声明的能力视为权限。
- 为有重大影响的操作信任明文或未签名的 `requestState`。
- 将唯一的重放账本仅保存在单个网关或服务器实例中。
- 仅以 `Mcp-Session-Id` 作为速率限制或审批状态的键。
- 将已弃用的 Sampling、Roots、Logging 或遗留的 HTTP 加 SSE 作为新的实现路径来呈现。

## 输出格式

返回名为 Trust Boundaries、Wire Findings、Descriptor Findings、Route Map、Authorization Matrix、MRTR Findings、Compatibility Findings 和 Remediation 的章节。将已确认的证据与假设分开。最后给出当前跨越最多边界的单一攻击路径。
