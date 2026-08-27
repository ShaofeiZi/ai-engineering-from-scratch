# 综合项目 06——面向 Kubernetes 的 DevOps 故障排查智能体

> AWS DevOps Agent 已正式发布，Resolve AI 公布了 K8s 故障排查手册，NeuBird 演示了语义监控，Metoro 则将 AI SRE 与每项服务的 SLO 关联起来。到 2026 年，这类系统的生产形态已经基本定型：告警 Webhook 触发后，智能体读取遥测数据，沿 K8s 对象图逐步排查，对根因假设排序，再向 Slack 发送带审批按钮的简报。系统默认只读，任何补救操作都要由人工批准。本综合项目就是构建这样一个智能体：用 20 起合成故障评估它，并选取 3 个相同案例与 AWS DevOps Agent 并排比较。

**Type:** 综合项目
**Languages:** Python（智能体）、TypeScript（Slack 集成）
**Prerequisites:** 第 11 阶段（LLM 工程）、第 13 阶段（工具与 MCP）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）、第 18 阶段（安全）
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 小时

## 问题

2025 至 2026 年，SRE 领域逐渐形成了一项共识：“由 AI 智能体初步处置故障，由人类批准补救操作。”AWS DevOps Agent、Resolve AI、NeuBird、Metoro 和 PagerDuty AIOps 都已将这套模式用于生产环境。智能体读取 Prometheus 指标、Loki 日志、Tempo 追踪数据、kube-state-metrics，以及 K8s 对象知识图谱，在 5 分钟内给出按可能性排序的根因假设，并附上遥测证据。破坏性命令不得由智能体自行执行，必须先在 Slack 中取得明确的人工批准。

真正困难的不是推理，而是确定权限边界并保障安全。智能体需要默认只读的 RBAC 权限、经过加固的 MCP 工具服务器，以及一份同时记录“考虑过”和“实际执行过”的命令审计日志。它还必须识别自己无法处理的情况，并及时升级给人类。运行成本也要受控，不能让一次 OOMKill 连锁故障又产生 5000 美元的智能体账单。

## 概念

智能体以知识图谱为基础。图中的节点既包括 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC），也包括遥测源（Prometheus 时间序列、Loki 日志流、Tempo 追踪数据）。边用于表示归属关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）和观测关系（Pod -> Prometheus 时间序列）。kube-state-metrics 会持续同步这张图，每次出现告警时还会重新采样。

告警触发后，智能体从受影响的对象开始定位根因。它沿图中的边向外遍历，提取最近 15 分钟的相关遥测数据，并据此提出假设。排序取决于证据：支持假设的遥测引用数量、证据的新近程度，以及证据的具体程度。排名前三的假设会连同图路径可视化和补救操作审批按钮一起发送到 Slack。

所有补救操作都要经过审批。默认只允许读取数据；缩容、回滚和删除 Pod 等破坏性操作必须先在 Slack 中获得批准。ArgoCD 回滚钩子还要求提供智能体永远不会持有的认证令牌。审计日志要记录智能体*考虑过*的每条命令，而不只是*实际执行过*的命令，这样复盘时才能发现险些造成事故的操作。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性来源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式），存储 K8s 对象及其遥测关系边
- 智能体：LangGraph，为每项工具配置允许列表，默认只读
- 工具传输：基于 StreamableHTTP 的 FastMCP；破坏性工具放到审批闸门后的独立服务器
- 模型：Claude Sonnet 4.7 负责根因推理，Gemini 2.5 Flash 负责日志摘要
- 补救：ArgoCD 回滚 Webhook、PagerDuty 升级处理、Slack 审批卡片
- 审计：仅追加的结构化日志（考虑过、已执行、已批准、结果）
- 部署：在独立命名空间中部署 K8s 工作负载，并使用专属的受限 RBAC 角色

```figure
ce-rootcause-walk
```

## 动手构建

1. **导入图数据。** 每 30 秒将 kube-state-metrics 同步到 Neo4j 或 kuzu。节点至少包括 Pod、Deployment、Node、Service、PVC 和 HPA；边包括 OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS 和 SCALES。再叠加一层遥测关系边 OBSERVED_BY，用于表示观测某个 Pod 的 Prometheus 时间序列。

2. **告警接收器。** 编写 FastAPI 端点，接收 PagerDuty 或 Alertmanager Webhook，并提取受影响的对象和 SLO 未达标信息。

3. **只读工具接口。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki LogQL 和 Tempo TraceQL。每项工具只开放范围严格的 RBAC 动词，例如“get”“list”和“describe”。默认服务器不得提供“delete”“exec”或“scale”。

4. **根因分析智能体。** 用 LangGraph 构建包含三个节点的流程：`sample` 获取最近 15 分钟的遥测切片，`walk` 查询图中的相邻对象，`hypothesize` 提出带遥测引用的根因候选并排序。

5. **证据评分。** 按以下公式给每项假设评分：分数 = 新近程度 * 具体程度 * 图路径长度的倒数 * 引用数量。返回排名前三的假设。

6. **Slack 简报。** 发送一条 Slack 消息附件，其中包含假设、图路径可视化（由服务端渲染的子图图片），以及至多一项补救操作的审批按钮。

7. **补救审批闸门。** 将缩容、回滚和删除等破坏性工具放在第二台 MCP 服务器上，并要求提供审批令牌。只有人工批准 Slack 卡片后，智能体才能调用这些工具。

8. **审计日志。** 以仅追加的 JSONL 格式记录日志。每条候选命令都要注明是否考虑过、是否执行过，以及批准人。日志每天归档到 S3。

9. **合成故障场景集。** 构建 20 个场景，例如 OOMKill 连锁故障、DNS 抖动、HPA 反复伸缩、PVC 写满、资源争用（noisy neighbor）、异常边车、错误的 ConfigMap 发布、证书轮换和镜像拉取退避。最后按根因准确率和提出假设所需时间为智能体评分。

## 实际运行

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付成果

`outputs/skill-devops-agent.md` 是本课的交付物。给定 K8s 集群和告警源后，智能体会给出按可能性排序的根因假设，并提供一个由 Slack 审批控制的补救流程。

| 权重 | 评估项 | 衡量方式 |
|:-:|---|---|
| 25 | 场景集上的根因分析（RCA）准确率 | 在 20 起合成故障中，根因判断正确率达到 ≥80% |
| 20 | 安全性 | 审计日志证明，没有任何破坏性操作在缺少 Slack 批准时执行 |
| 20 | 提出假设所需时间 | 从告警触发到发出 Slack 简报，p50 小于 5 分钟 |
| 20 | 可解释性 | 每个假设都附带图路径和遥测引用 |
| 15 | 集成完整度 | PagerDuty、Slack、ArgoCD 和 Prometheus 的端到端流程能够正常运行 |
| **100** | | |

## 练习

1. 让你的智能体处理 AWS DevOps Agent 演示过的 3 起故障，发布并排对比结果，并说明两者的判断在哪些地方出现分歧。

2. 增加“险些执行”审计：凡是智能体*考虑过*且在未经审批时会造成破坏的命令，都要单独标记。统计一周内此类命令的比例。

3. 将生成假设的模型从 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B，测量根因分析准确率的变化和每起故障的成本。

4. 增加因果过滤器，将相关的遥测尖峰与真正的根因区分开。使用这 20 个场景的标签训练一个小型分类器。

5. 增加回滚演练：先在预发布集群中，用与生产环境相同的清单执行一次 ArgoCD 回滚。Slack 审批按钮出现之前，必须已经在真实集群上验证回滚方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱（K8s knowledge graph） | “集群图” | 节点是 K8s 对象和遥测序列；边表示归属、调度和观测关系 |
| 默认只读（Read-only-by-default） | “限定范围的 RBAC” | 智能体的服务账号只有 get、list 和 describe 权限；破坏性操作位于审批闸门后的独立服务器中 |
| 审计日志（Audit log） | “考虑过与实际执行” | 以仅追加方式记录每条候选命令、是否执行以及由谁批准 |
| 假设排序（Hypothesis ranking） | “证据分数” | 新近程度 × 具体程度 × 图路径长度的倒数 × 引用数量 |
| Slack 审批卡片（Slack approval card） | “人工参与闸门” | 带补救按钮的交互式 Slack 消息；人工点击批准后，智能体才能继续 |
| 遥测引用（Telemetry citation） | “证据指针” | 支持某项判断的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | “解决时间” | 从告警触发到 SLO 恢复所经过的实际时间 |

## 延伸阅读

- [AWS DevOps Agent 正式发布](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年的代表性参考实现
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品实现参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图谱方案
- [Metoro AI SRE](https://metoro.io) — 以 SLO 为先的生产实践思路
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态数据源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 智能体编排框架参考
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 受审批闸门控制的补救操作
