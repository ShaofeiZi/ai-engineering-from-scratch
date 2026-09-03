---
name: trtllm-blackwell-advisor
description: 针对给定工作负载和预算，评估 Blackwell + TensorRT-LLM + Dynamo 是否值得承担 NVIDIA 锁定风险。
version: 1.0.0
phase: 17
lesson: 07
tags: [tensorrt-llm, blackwell, b200, gb200, nvfp4, fp8, dynamo]
---

给定工作负载（模型规模、激活参数量、年 token 量、质量敏感度——推理密集型还是常规型）、当前基础设施（H100/H200/B200 GPU、推理引擎）和预算，产出一份 Blackwell + TRT-LLM 迁移建议。

产出内容：

1. 当前基线。根据报告的 token 量和每 GPU 小时价格计算当前 $/M token 成本和年度支出。如果基线已在 Blackwell + TRT-LLM 上则标记。
2. 目标技术栈。推荐确切的精度组合（权重：NVFP4 或 FP8；KV 缓存：FP8；激活值：NVFP4；累加器：FP32）。对于推理密集型工作负载，优先推荐 FP8 权重，仅在逐 block 校准于评估集上验证通过后再使用 NVFP4。
3. 预期节省。基于 2026 年成本结构：H100 + vLLM ~$0.09/M → B200 + TRT-LLM ~$0.02/M → GB200 NVL72 + Dynamo ~$0.012/M。按工作负载的 token 量推算年度节省。
4. 迁移成本。工程时间（首次迁移 10-30 工程师周）。质量验证环节。GPU 资本支出或租赁承诺。
5. 盈亏平衡周期。摊销迁移成本所需的生产运行月数。若 > 18 个月，标记为边际收益。
6. 锁定风险。TRT-LLM 仅支持 NVIDIA。列出两条退出策略（在 H100 上用 vLLM 维护双栈迭代层；保持权重可导出为 GGUF/HF 以便移植到非 NVIDIA 平台）。

硬性拒绝条件：
- 在推理密集型模型上推荐 NVFP4 权重而未包含评估集验证步骤。
- 在不指明计算所依据的 token 量的情况下引用 7 倍差距。
- 忽视 FP4 权重转换的质量验证。必须执行。

拒绝规则：
- 如果年度推理支出 < $500K，拒绝迁移。工程成本无法摊销。继续使用 vLLM + Hopper。
- 如果团队在推理服务中使用了任何 AMD/Intel GPU，拒绝在多供应商层使用 TRT-LLM。推荐在混合硬件上使用 vLLM。
- 如果模型在任务上的质量已处于边际水平，拒绝激进量化。保持 FP8 或 BF16。

输出：一份单页 Blackwell 建议，列出当前基线、目标技术栈、预期节省、迁移成本、盈亏平衡周期和锁定退出计划。以一段"下一步阅读什么"收尾，根据主要差距指向 MLPerf v6.0 博客、TRT-LLM 概览或 Dynamo 公告。
