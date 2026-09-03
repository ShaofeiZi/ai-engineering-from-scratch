---
name: managed-platform-picker
description: 根据工作负载、SLA 和合规要求选择一个托管 LLM 平台（Bedrock、Azure OpenAI、Vertex AI）以及一个用于冗余的备用平台，然后产出 FinOps 可观测性计划。
version: 1.0.0
phase: 17
lesson: 01
tags: [bedrock, azure-openai, vertex-ai, ptu, finops, managed-platforms]
---

给定工作负载画像（所需模型、月 token 量、P50/P99 的 TTFT SLA、合规约束、现有云足迹），产出平台推荐方案。

产出：

1. 主平台。命名平台及其覆盖的具体模型，并根据利用率判断按量调用还是 Provisioned Throughput Units (PTUs) / Provisioned Throughput 更合适。引用盈亏平衡计算（PTU 在约 40-60% 持续利用率时盈亏平衡）。
2. 备用平台。命名满足双提供商最低要求的回退方案。论证配对理由——冗余必须覆盖模型重叠（Bedrock 上的 Claude + Azure OpenAI 上的 GPT 是常见配对）和区域重叠。
3. FinOps 可观测性。指定第一天就要启用的内容：Bedrock Application Inference Profiles、Azure scopes + PTU 预留作为成本对象、Vertex project-per-team + BigQuery Billing Export。命名归因维度——按用户、按任务、按租户。
4. SLA 检查。将目标 TTFT P99 与已发布基准对比（Azure OpenAI PTU ≈ 50 ms P50；Bedrock 按量调用 ≈ 75 ms P50）。如果 SLA 比按量调用所能达到的更严格，则要求使用 PTU。
5. 合规检查。根据需要验证 BAA、SOC 2 Type II、HIPAA、欧盟数据驻留。注意三者均满足基线，但留存策略和滥用监控退出机制不同。
6. 迁移路径。命名团队本周可执行的一个可逆步骤（例如通过 AI 网关抽象提供商部署；设置归因标头）和一个长期步骤（PTU 承诺；跨区域故障转移）。

硬性拒绝：
- 推荐单一平台而不指定备用方案。拒绝并坚持双提供商最低要求。
- 在没有利用率预估的情况下选择 PTU。拒绝并要求提供持续利用率数据。
- 当归因被列为需求时忽略 Bedrock Application Inference Profiles——它们是最干净的原生接口。

拒绝规则：
- 如果工作负载要求 Claude、Gemini 和 GPT 全部作为 P0，指出三平台现实（通过网关后端 Bedrock + Vertex + Azure OpenAI），而不是假装一个平台能同时服务三者。
- 如果 SLA 为 TTFT P99 < 100 ms 且预期预算无法支撑 PTU，拒绝承诺该 SLA——解释按量调用的方差上限。
- 如果客户要求"使用最便宜的提供商"，拒绝——价格是多维的（token 费率 + 专用容量 + 归因开销 + 锁定成本）。

输出：一页决策，包含主平台、备用平台、PTU 还是按量调用、可观测性清单、SLA/合规验证和两个迁移步骤。以一个能捕捉偏离计划的指标结尾（持续利用率、PTU 浪费或归因覆盖率）。
