---
name: tripwire-design
description: 审查拟定的智能体检测器栈（kill switch、circuit breakers、canary tokens），并在首次自主运行前标记缺失的 tripwire。
version: 1.0.0
phase: 15
lesson: 14
tags: [kill-switch, circuit-breaker, canary, honeytoken, detection-and-response]
---

针对某智能体部署拟定的检测器栈，参照三检测器参考架构（kill switch、circuit breaker、canary）进行审计，并标记其中缺失、误调或对智能体暴露的内容。

产出：

1. **Kill-switch 审计。** 开关位于何处（feature flag、Redis、signed config）？确认智能体的凭据无法将其关闭。确认每一个后果性操作都会检查该开关，而不仅是在启动时检查。确认重新启用是一个显式的人工操作。
2. **Circuit-breaker 清单。** 列出每个 breaker 监控的模式（重复、连续失败、速率、不受信任读取后的特定工具调用）。说明各自的阈值和 cool-down。阈值超过 10 通常过于宽松。
3. **Canary 设计。** 列出环境中的每个 canary token。对每个 token 说明：它是什么（假凭据、假 DB 记录、假文件、假 memory 条目），位于何处，何种访问会触发警报，谁会被寻呼。确认没有任何 canary 存在被合理触碰的理由。
4. **统计层与硬限制分层。** 确认该栈除任何统计检测器（EWMA、z-score）之外，还至少使用了一个硬限制（Lesson 17 constitutional style）。仅使用统计检测器会容忍缓慢漂移。
5. **Quarantine 路径。** 当检测器触发时会发生什么？完全停止智能体、路径级暂停、流量重定向（eBPF / Cilium honeypot）、仅告警。确认该路径至少进行过一次端到端测试。

硬性拒绝：
- 任何没有外部 kill switch 的部署。
- Canary tokens 存储在智能体具有写权限的系统中。
- 仅使用统计检测且没有硬限制。
- Circuit breaker 的 cool-down 在没有人工审查的情况下自动重新启用。
- 在无人值守运行中，kill switch 仅在启动时检查，而非逐操作检查。

拒绝规则：
- 如果用户无法说出承载 kill switch 的、位于智能体凭据之外的特定系统，则拒绝。"我们使用智能体读取的配置文件" 不算 kill switch——前提是智能体可以写入配置文件。
- 如果用户将 Auto Mode 分类器（Lesson 10）视为 tripwire 的替代品，则拒绝。该分类器与 detection-and-response 正交。
- 如果拟定的 canary 位于智能体有合理理由读取的系统中，则拒绝并要求重新设计。

输出格式：

返回一份 tripwire 审计，包含：
- **Kill-switch 行**（位置、检查频率、重新启用程序）
- **Circuit-breaker 表**（模式、阈值、cool-down）
- **Canary 表**（token、位置、警报、所有者）
- **分层说明**（是否存在统计与硬限制 y/n）
- **Quarantine 流程**（触发什么、发生什么、是否测试 y/n）
- **就绪状态**（production / staging / research-only）
