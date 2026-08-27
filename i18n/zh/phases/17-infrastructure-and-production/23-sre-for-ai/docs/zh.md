# AI 时代的 SRE —— 多代理事件响应、Runbook 与预测性检测

> AI SRE 的核心做法，是让 LLM 通过 RAG 读取基础设施数据，比如日志、runbook 和服务拓扑，从而自动化调查、文档整理和协同流程。到 2026 年，主流架构模式已经演化成多代理编排：日志代理、指标代理、runbook 代理等专职代理由一个 supervisor 协调；AI 负责提出假设和查询建议，人类负责批准真正的判断动作。Datadog Bits AI 和 Azure SRE Agent 已经把这套思路产品化。Runbook 也在进化：NeuBird Hawkeye 采用对抗式评估，同一事件由两个模型独立分析，一致时提高置信度，不一致时升级给人类；运营记忆会跨团队成员变动持续保留。自动修复仍然非常克制：AI 提建议，人类做批准。完全自治的动作只适用于很窄的一组操作，比如重启 pod 或回滚明确的 deploy，而且必须有严密护栏。凡是卖你“设完就不用管”的，都在过度承诺。另一个前沿方向是事故前预测。MIT 的研究报告称，用历史日志、GPU 温度和 API 错误模式训练的 LLM，能在故障发生前 10-15 分钟预测出 89% 的停机事件。行业预测则认为，到 2026 年底，95% 的企业级 LLM 系统都会具备自动 failover 能力。

**Type:** 学习
**Languages:** Python（标准库，玩具级多智能体事件分诊模拟器）
**Prerequisites:** 阶段 17 · 13（可观测性）、阶段 17 · 24（混沌工程）
**Time:** 约 60 分钟

## 学习目标

- 画出多代理 AI SRE 架构：supervisor + specialized agents（logs、metrics、runbooks）+ human approval gate。
- 解释为什么 auto-remediation 只能是窄动作集（restart pod、revert deploy），而不能扩展到大范围架构改造。
- 说清 NeuBird Hawkeye 的对抗式评估模式：两个模型一致 = 信心更高；不一致 = 升级处理。
- 记住 MIT 的 89% 提前预测结果，同时理解关键约束：只有预测、没有后续动作，本质上只是更花哨的 dashboard。

## 问题

值班工程师凌晨 3 点收到告警：“checkout 错误率过高。” 他打开 Datadog、Loki、三份 runbook 和部署日志。30 分钟后才发现，根因其实是 vLLM 因 KV cache 突增而 OOM。重启 pod 之后，错误率恢复正常。

到了 2026 年，这场调查前 20 分钟里的很多动作都已经可以自动化。日志按服务聚类、关联到最近 deploy、匹配历史 runbook，这些都属于 RAG + tool use 的典型工作。一个受监督的 agent 可以在人工真正打开 Datadog 之前，就先做出第一轮分诊并给出假设。

但“自动调查”和“自动修复”是两件完全不同的事。重启 pod，通常安全。按策略扩容 GPU 池，也可能安全。重新设计服务架构？绝对不行。AI SRE 的纪律，本质上就是把那条窄边界画清楚。

## 概念

### 多代理架构

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

supervisor 会把事件拆成多个子查询。专职代理拥有各自的工具访问权限，比如日志搜索、PromQL、文档检索。supervisor 负责汇总这些结果，形成“假设 + 证据”并提交给人类。人类可以批准、否决，或者要求 agent 换一个方向继续调查。

### 自动修复的边界

**安全的窄动作**：重启 pod、回滚某个明确 deploy、在预批准范围内扩容资源池、打开某个预批准 feature flag。

**不安全的宽动作**：改服务拓扑、调资源限制、部署新代码、改 IAM、改数据库。

随着 AI SRE 的成熟，安全动作集会慢慢扩大，但那条边界始终是真实存在的。把它说成“全自动运维已经成熟”是在误导人。

### 对抗式评估（NeuBird Hawkeye）

两个模型独立分析同一个 incident。如果它们对根因判断一致，系统就可以给出更高置信度；如果两者不一致，就把两种假设同时呈现给人类，由人来裁决。这种模式简单，但对过滤幻觉式根因很有效。

### 运营记忆

传统 SRE 的一个隐性杀手是团队流动。很多 tribal knowledge 都随着人员离开而消失。AI SRE 会把 runbook 和 post-mortem 存进 vector DB，代理在每次新 incident 上线时都可以检索这些历史。当新工程师加入团队时，AI 理论上已经拥有完整上下文。

### 事故前预测

MIT 在 2025 年的研究里报告：一个用历史日志、GPU 温度和 API 错误模式训练的 LLM，在测试集上能提前 10-15 分钟预测出 89% 的故障。

但要做现实检验：只有预测、没有 actuation，本质上只是 dashboard。真正的运营问题不是“能不能预测”，而是“预测到了以后怎么办”。是预先 drain 流量？拉 pager？自动扩容？这些都必须落回具体策略。

### 2026 年的产品版图

- **Datadog Bits AI**：Datadog 内部的托管式 SRE copilot。
- **Azure SRE Agent**：Azure 原生方案。
- **NeuBird Hawkeye**：对抗式评估 + 运营记忆。
- **PagerDuty AIOps**：更侧重 triage 和 deduplication。
- **Incident.io Autopilot**：更偏 incident commander 与协同流程。

### 将 Runbook 作为代码管理

runbook 正在从 Confluence 页面演变成结构化、版本化的 markdown 文档，至少会拆成 symptom、hypothesis、verify、act 这样的固定部分。结构化 runbook 能显著提升 RAG 检索质量。很多 AI SRE rollout 的第一步，并不是先上 agent，而是先把零散 runbook 结构化。

### 你应该记住的数字

- MIT 提前检测：89% 的故障，10-15 分钟提前量。
- 多代理分诊结构：supervisor +（logs、metrics、runbooks）+ human。
- 安全 auto-remediation 集合：restart pod、revert deploy、在边界内 scale。
- 对抗式评估：两个模型独立分析；一致 = 更高信心。

```figure
i4-incident-agents
```

## 用起来

`code/main.py` 会模拟一次多代理分诊：log agent 找到错误信号，metric agent 找到 CPU 尖峰，runbook agent 把问题匹配到已知故障。supervisor 则负责给不同假设排序。

## 交付物

这一课会产出 `outputs/skill-ai-sre-plan.md`。输入当前 on-call 模式、事件量和团队成熟度之后，它会帮你设计一条 AI SRE 落地路径。

## 练习

1. 运行 `code/main.py`。如果 log agent 和 metric agent 结论相互矛盾，supervisor 应该怎么处理？
2. 为你的服务定义三项“安全”的 auto-remediation 动作，并分别说明理由。
3. 写一个结构化 runbook 模板：包括哪些部分、哪些字段必填、哪些验证命令必须出现？
4. 如果 predictive detection 在 12 分钟前触发，你的策略是什么：打 pager、提前 drain，还是两者都做？
5. 论证一个只有 3 人的团队在 2026 年是否应该采用 AI SRE，还是应该等一等。把成熟度、事件量和风险一起考虑进去。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| AI SRE | “值班 agent” | 由 LLM 支持的事件调查与协同系统 |
| Supervisor agent | “总控编排器” | 负责把 incident 拆成多个子查询的顶层代理 |
| Specialized agent | “领域代理” | 拥有日志、指标、runbook 等工具访问权限的专职代理 |
| Auto-remediation | “AI 自己修了” | 一组预批准的窄动作，不是大范围架构改造 |
| Operational memory | “向量化 runbook” | 用于 RAG 的 post-mortem + runbook 记忆层 |
| Adversarial eval | “双模型校验” | 两份独立分析，一致时提高信心 |
| NeuBird Hawkeye | “那个对抗式产品” | 采用 adversarial-eval + memory 模式的产品 |
| Bits AI | “Datadog 的 SRE agent” | Datadog 托管式 AI SRE 产品 |
| Pre-incident prediction | “提前发现” | 在故障前 10-15 分钟发出预测信号 |

## 延伸阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
