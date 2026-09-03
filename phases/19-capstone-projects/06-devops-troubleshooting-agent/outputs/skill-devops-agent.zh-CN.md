---
name: devops-agent
description: 构建一个 Kubernetes 故障排查智能体，遍历集群知识图谱，对根因进行排序，并通过 Slack 对每一步修复操作进行审批门控。
version: 1.0.0
phase: 19
lesson: 06
tags: [capstone, devops, sre, kubernetes, langgraph, fastmcp, aiops]
---

给定一个 K8s 集群和一个告警源（PagerDuty 或 Alertmanager），构建一个智能体，在五分钟内产出排序后的根因假设，并通过 Slack 审批卡片对每一步修复操作进行门控。

构建计划：

1. 每 30 秒将 kube-state-metrics 导入 Neo4j 或 kuzu。构建由 Pod、Deployment、Service、Node、PVC、HPA 以及遥测叠加边组成的图谱，遥测叠加边指向 Prometheus、Loki 和 Tempo 数据源。
2. 搭建一个 FastAPI webhook 接收器，用于接收 PagerDuty 和 Alertmanager 的告警。
3. 通过 FastMCP 以 StreamableHTTP 传输方式暴露只读工具：kubectl get/describe、promql、logql、traceql。
4. 构建 LangGraph 根因分析智能体，包含三个节点：`sample`（拉取 15 分钟遥测数据）、`walk`（遍历图谱邻居）、`hypothesize`（按时间近因 × 特异性 × 引用数量对候选项排序）。
5. 将排名前三的假设及图路径可视化发布到 Slack，并附带审批按钮。
6. 将破坏性工具（scale、rollback、delete）部署在独立的 FastMCP 服务器上，该服务器受审批令牌保护，智能体仅在获得 Slack 批准后才能获取该令牌。
7. 维护一个仅追加的审计日志：记录每一条*考虑过的*命令、是否获批、是否执行、由谁批准。
8. 构建 20 个合成故障场景（OOMKill、DNS 抖动、HPA 振荡、PVC 填满、吵闹邻居、故障 sidecar、ConfigMap 错误滚动、证书轮换、镜像拉取退避、探针失败，以及另外 10 个）。在 20 个合成故障场景上评估智能体的根因分析准确率和假设生成时间。

评分标准：

| 权重 | 评估维度 | 测量方式 |
|:-:|---|---|
| 25 | 根因分析准确率 | 在 20 个合成故障场景中至少 80% 正确定位根因 |
| 20 | 安全性 | 审计日志中破坏性操作防护在未获 Slack 批准时绝不触发 |
| 20 | 假设生成时间 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都包含图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端联通 |

一票否决项：

- 使用单一 MCP 服务器混合提供只读工具和破坏性工具的智能体。
- 任何未附带遥测引用的根因分析结果。无引用的假设必须被拒绝。
- 审计日志仅记录已执行操作的。必须记录每一条考虑过的命令。
- 未在 20 个场景测试套件（带种子）上运行智能体便声称准确率。

拒绝规则：

- 拒绝在未获得人工值班人员 Slack 批准的情况下执行修复。即使根因显而易见。
- 拒绝通过只读 MCP 暴露 `kubectl exec`、`kubectl port-forward` 或任何交互式工具。这些操作实质上具有破坏性。
- 拒绝在未为每个 Deployment 生成独立审批卡片的情况下跨多个 Deployment 批量应用修复。

交付物：一个仓库，包含 FastAPI 接收器、LangGraph 智能体、只读和破坏性 MCP 服务器、Slack 集成、20 场景测试套件、与 AWS DevOps Agent 在三个共享故障上的并排对比，以及一篇关于一周观测窗口内近误命令（智能体*考虑过*但未执行的命令）的报告。
