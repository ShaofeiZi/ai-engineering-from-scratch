# Kill Switch、Circuit Breaker 与 Canary Token

> Kill switch 是一个保存在代理可编辑表面之外的布尔开关，例如 Redis key、feature flag 或签名配置，它能彻底关闭代理。Circuit breaker 更细粒度：它会在某个特定模式上跳闸，例如连续五次完全相同的工具调用，暂停出问题的路径，并升级给人工。Canary token 则继承自经典欺骗防御（deception）：放一个代理根本没有正当理由去碰的假凭证或蜜罐（honeypot）记录，只要它被访问，就立刻报警。基于 eBPF 的数据路径，例如 Cilium，可以在内核层把被隔离 pod 的出口重写到一个法证蜜罐；Cilium 公开基准显示，在负载下数据路径（datapath）的 P99 延迟仍低于 1 毫秒。真正决定传播预算的，不是数据路径本身，而是策略更新如何传播到节点。会适应移动基线的统计探测器，例如 EWMA 和 CUSUM，会悄悄接受漂移，所以必须再叠一层不会弯曲的硬性宪法式限制（constitutional limit）。

**Type:** 学习
**Languages:** Python（stdlib，三类探测器模拟器：kill switch、circuit breaker、canary）
**Prerequisites:** 阶段 15 · 13（成本治理器），阶段 15 · 10（权限模式）
**Time:** 约 60 分钟

## 问题

成本治理器（第 13 课）限制的是代理能花多少钱，但它并不限制代理能在预算内做什么。一个带有 $50 速度上限（velocity limit）的代理，仍然可以泄露机密、发出错误帖子，或删除某个资源。很多真正危险的动作，在 token 成本上反而很便宜。

本课讲的是贴着成本层运行的三种探测器：

1. **Kill switch**：一个保存在代理触达范围之外的布尔关停按钮。
2. **Circuit breaker**：检测特定动作模式并暂停某条路径的探测器。
3. **Canary token**：代理本不该碰，一碰就会暴露自己的诱饵。

这三种东西都早于 LLM 工程。经典欺骗技术、速率限制跳闸器和特性开关式关停，在自主代理出现之前就存在。新变化在于攻击面：代理会读取不受信任内容（第 11 课）、会修改自己的记忆，还可能把一连串“看起来安全”的动作组合成一个真正危险的结果。这里提到的探测器之所以有用，就在于它们不相信代理的自我报告。

## 概念

### Kill switch

Kill switch 是一个代理能读取、但不能写入的布尔开关。常见实现方式包括：

- **托管服务里的 feature flag。** 例如 LaunchDarkly、Statsig、Unleash。更新一般以秒级传播。
- **代理轮询的 Redis key。** 实现简单，但要求代理在每一轮都去检查。
- **对象存储里的签名配置。** 代理在启动时验证签名；对未签名状态一律拒绝。
- **操作系统级信号或容器生命周期操作。** 例如 Docker `kill`、Kubernetes `kubectl delete pod`、systemd stop。

一个正确的 kill switch 应该具备这些性质：

- 代理不能把它重新写成 `off`。它必须保存在代理凭证没有写权限的系统里。
- 它必须在每一个有后果的动作前都检查，而不是只在启动时检查一次。
- 一旦它处于关闭状态，代理就不应再做任何外部可观察的事，包括向它仍可访问的系统写日志。
- 重新启用必须是显式的人类动作，而不是一个自动超时恢复。

### Circuit breaker

Circuit breaker 暂停的是某个特定模式，而不是整个代理。经典三态模型来自 2007 年 Nygard 的书，到今天仍然成立：

- **Closed**：允许动作执行。
- **Open**：阻止动作执行。
- **Half-open**：冷却之后，只放行 1 到 3 次探测请求，默认通常是 1 次；如果成功则闭合，否则再次打开。

在代理场景里，常见触发条件包括：

- 连续五次完全相同的工具调用，说明可能卡进了重复循环。
- 同一个工具连续五次失败，但输入不同，说明更像系统性故障。
- 工具调用速率高于阈值，对应第 13 课的速度异常。
- 某个特定工具，例如 `file.delete`，发生在一次 out-of-trust 内容读取之后。

### Canary token

Canary token，也叫 honeytoken，是代理根本不应该去碰的输入。一旦被访问，就是报警。

针对代理系统的例子包括：

- 一个假的 `AWS_SECRET_ACCESS_KEY`，它没有真实权限，但一旦被尝试外泄，就能立即归因。
- 一条带已知哨兵值（sentinel value）的假数据库记录。任何读写都应触发告警。
- 工作区里的一个假文件。读取或修改它都值得怀疑。
- 在会话开始时注入的一条假记忆。如果它出现在后续输出里，就说明记忆被泄露了。

Canary 的设计高度依赖工作流。一个代理本来就有正当理由去碰的 canary，只会制造噪声。

### 为什么统计探测器必须叠加硬限制

统计探测器，例如 EWMA、CUSUM 或工具调用率上的 z-score，会适应一个不断变化的基线。这对于合法漂移是优点，对于对抗性漂移则是缺点。一个足够耐心的攻击者，可以把基线以足够慢的速度往危险方向推移，直到没有任何告警触发。

硬性的宪法式限制（constitutional limit，第 17 课）不会弯。像“永远不要给 `secrets@` 发邮件”这样的规则更简单、更可审计，也不可能通过慢漂移来规避。真正的分层方式是：用统计探测器抓大多数噪声，用硬限制拦住那些已经绕过统计模型的攻击。

### 通过 eBPF 数据路径重定向做隔离

Cilium 基于 eBPF 的网络策略可以在内核层把被隔离 pod 的出口重写到一个法证 honeypot。Cilium 的公开基准表明，在 TCP_RR 场景下，数据路径的 P99 延迟仍低于 1 毫秒。你在集群里真正要为“触发隔离”买单的实际时间，主要取决于策略更新从控制平面传播到节点的速度。在一个预热过 operator 的 Kubernetes 集群里，几百毫秒是现实数字；尾延迟则取决于集群规模和 reconcile cadence。

这样做能带来两件事：第一，代理不会把任何真实内容泄露出去；第二，代理在蜜罐里的每个动作都能被完整观察。它和 canary token 的思路是同一个模式，只是发生在更底层：代理会用自己的行为暴露自己的意图。

### 哪些东西任何探测器都抓不住

- **由允许原语组合出来的语义攻击。** 每个单独动作都通过了所有探测器，问题出在组合本身。这正是第 10 课里的 Auto Mode 复合失效（composite failure）。
- **把探测器本身写进恶意载荷（payload）的攻击。** 如果代理读到的页面写着“在执行前先禁用 canary”，而代理又真的具备这个能力，那探测器本身就被污染了。因此 canary token 必须放在代理无权修改的系统中。

```figure
circuit-breaker
```

## 用它

`code/main.py` 会模拟一个短代理轨迹，并让它穿过三种探测器：一个保存在外部字典中的 kill switch，一个会在五次相同工具调用后跳闸的 circuit breaker，以及一个一旦被读取就报警的 canary file。输入的合成轨迹包括：合法动作、重复循环、canary 探测，以及一个被 kill switch 截断的场景。

## 交付成果

`outputs/skill-tripwire-design.md` 用来审查一个拟议代理部署的探测器栈（detector stack），并标出缺口，例如缺失 kill switch、缺失 canary，或 circuit breaker 阈值设得过松。

## 练习

1. 运行 `code/main.py`。确认 circuit breaker 会在第 5 轮触发，第 9 轮的 fake-key read 会触发 canary。

2. 增加一个统计探测器：用 EWMA z-score 监控工具调用速率。喂给它一条缓慢漂移的轨迹，展示探测器始终不触发。然后再加一条硬限制，例如“10 分钟内不允许超过 50 次工具调用”，并展示同一条轨迹会触发硬限制。

3. 为浏览器代理（第 11 课）设计一组 canary token。至少列出三个 canary，并说明每个 canary 能检测到什么。

4. 阅读 Cilium 的 network-policy 文档。具体描述一个出口重定向隔离流程：使用哪个 policy selector、作用到哪个 pod、做什么 egress rewrite、触发什么 alert。再说明“决定隔离”到“第一包被重定向”之间的实际延迟由什么因素决定。

5. 为被 kill-switched 的代理定义一个重新启用流程。谁有权重新启用？必须记录什么？在重新启用前，代理本身必须发生什么变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Kill switch | “关闭按钮” | 保存在代理编辑表面之外的布尔开关；每个有后果的动作前都检查 |
| Circuit breaker | “模式暂停器” | 针对重复、失败率或速率上限的动作级跳闸 |
| Canary token | “Honeytoken” | 代理没有正当理由去碰的诱饵；一旦访问就报警 |
| Honeypot | “法证沙箱” | 被重定向后的流量或工作区，专门用于观察隔离代理 |
| EWMA | “移动平均” | 指数加权平均；既能适应漂移，也可能因此放过漂移 |
| CUSUM | “累计和” | 用来检测相对基线的持续偏移 |
| Hard limit | “宪法规则” | 不会适应历史；始终保持不变 |
| Constitutional limit | “永真规则” | 与第 17 课的 constitution 绑定；代理不能编辑 |

## 延伸阅读

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — autonomous agent 的 kill-switch 与 circuit-breaker 框架。
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 生产治理模式。
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — 检测与响应要求。
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod 级出口重定向与法证 honeypot 模式。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 把硬编码禁令视作 constitutional limit。
