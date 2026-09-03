---
name: multi-region-router
description: 设计多区域 LLM 路由方案，涵盖 KV 缓存局部性、驻留边界、灾备清单及季度故障切换演练。
version: 1.0.0
phase: 17
lesson: 11
tags: [multi-region, kv-cache, routing, dr, bedrock-cri, vllm-router, llm-d, gorgo]
---

给定在范围内的区域、驻留边界、预期前缀缓存多样性及 TTFT SLA，制定一份多区域路由与灾备方案。

产出内容：

1. 路由器选型。选择缓存感知路由器（vLLM Router、llm-d router），描述 KV 事件通道。声明前缀哈希算法（例如 512-token 滚动哈希）及平局打破规则（最小队列深度）。
2. 路由策略。区域优先还是全局（GORGO 风格）最小化 prefill + RTT？用提示词长度分布来论证——长提示词（>8K tokens）受益于跨区域路由；短提示词则不然。
3. 驻留分区。在任何优化之前：出于法律原因（GDPR、HIPAA），哪些请求绑定到哪些区域。即使 TTFT 能改善，也禁止跨驻留路由。
4. 商业 CRI 层。建议是否启用 Bedrock Cross-Region Inference 或 GKE Multi-Cluster Gateway 作为可用性层。明确声明此层不是 TTFT 优化。
5. 灾备清单。至少三文件（HF 仓库 + 引擎配置 + 部署清单）。验证 tokenizer、量化配置、RoPE、聊天模板、LoRA 适配器是否均已包含。声明存储方式（S3 跨区域复制、多区域 GCS）。
6. 故障切换演练。季度频率。由谁执行、测量什么（RTO、RPO、缓存预热时间）。目标：30 分钟 RTO，对标 2024 年 JPMorgan 的真实演练。

硬性拒绝：
- 为了路由优化而忽视驻留要求。拒绝——GDPR 违规优先于 TTFT 收益。
- 声称 Bedrock CRI "解决"了跨区域路由。拒绝——CRI 是可用性层，不是 TTFT。
- 仅备份权重。拒绝——引用 32% 灾备失败统计数据，要求提供三文件清单。

拒绝规则：
- 如果范围内只有一个区域，拒绝该方案——单区域有不同的故障模式（Phase 17 · 03 已覆盖）。
- 如果驻留与 TTFT SLA 不兼容（例如欧盟驻留要求每个请求在冷前缀上做 prefill，且 P99 TTFT < 100 ms 处理 8K 提示词），拒绝承诺该 SLA 并升级产品需求。

输出：一页方案，命名路由器、路由策略、驻留分区、CRI 层姿态、灾备清单、季度演练负责人。以单一告警指标结尾：跨区域前缀缓存命中率降至方案规定阈值以下。
