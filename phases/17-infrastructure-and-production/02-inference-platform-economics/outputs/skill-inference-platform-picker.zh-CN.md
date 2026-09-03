---
name: inference-platform-picker
description: 根据工作负载、SLA、预算和运维约束选择推理平台（Fireworks、Together、Baseten、Modal、Replicate、Anyscale 或定制芯片）。将按 token、按分钟和按预测计费进行归一化对比。
version: 1.0.0
phase: 17
lesson: 02
tags: [inference, fireworks, together, baseten, modal, replicate, anyscale, economics]
---

给定工作负载画像（模型、每日 token 量、持续利用率、TTFT SLA、突发系数、合规要求、Python 还是混合技术栈），产出平台推荐方案。

产出：

1. 主平台。命名平台及具体定价层级（Serverless vs 专用 vs 批量）。用匹配的工作负载特征论证——例如"选择 Fireworks Serverless，因为 SLA 要求 TTFT < 500 ms 且流量具有突发性。"
2. 有效成本。将所选定价模型归一化为 $/M 输出 token。与至少两个替代方案对比。指出按分钟计费何时优于按 token 计费（持续利用率超过约 30% 时）或反之。
3. 冷启动计划。对于 Serverless 选择（Fireworks、Modal、Replicate），说明预期冷启动延迟和缓解措施（预热、min_workers=1、热迁移）。对于专用选择（Baseten、Anyscale），跳过此部分但注明权衡。
4. 备选方案。命名第二平台及切换的明确条件（例如"如果签署需要 HIPAA + 专用 GPU 的企业合同，则迁移到 Baseten"）。
5. 网关层。推荐是否在平台前部署 AI 网关（LiteLLM、Portkey、Kong AI Gateway）以将产品与提供商更迭隔离。默认：是，除非规模低于 500 RPS。

硬性拒绝：
- 未归一化即对比按 token 与按分钟计费。拒绝并坚持使用有效 $/M token。
- 因"最快"而选择 Fireworks，却未根据已发布基准验证 TTFT SLA。
- 对任何非延迟受限的工作负载推荐定制芯片（Groq、Cerebras、SambaNova）。其定价溢价仅在交互式 SLA 下才合理。

拒绝规则：
- 如果工作负载需要受监管框架（SOC 2 Type II、HIPAA）而客户选择了 Modal 或 Replicate，拒绝——两者的企业级覆盖面不如 Baseten 或 Anyscale。建议 Baseten。
- 如果预期流量低于每日 100k token，拒绝推荐按分钟计费（Baseten、Modal、Anyscale）。经济性不成立——默认选择市场平台（OpenRouter、DeepInfra）或托管超大规模云。
- 如果客户想要"最便宜的"，拒绝——指出多维成本函数（token 费率 + 冷启动 + 归因 + 网关 + DX）。

输出：一页推荐，命名主平台、有效成本、冷启动计划、备选方案、网关策略。以一个能揭示选择失误的指标结尾（冷启动 P99、按 token 费率或利用率漂移）。
