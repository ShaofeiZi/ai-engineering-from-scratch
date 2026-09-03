---
name: red-team-stack
description: 为给定部署推荐红队工具栈及配置。
version: 1.0.0
phase: 18
lesson: 16
tags: [llama-guard, garak, pyrit, red-team-tooling, mlcommons-hazards]
---

给定一份部署描述，推荐红队工具栈与回归节奏。

产出：

1. 分类器部署位置。推荐 Llama Guard（3-8B、3-1B-INT4 或 4-12B）部署在输入端、输出端还是两端。对于边缘部署，优先选择 3-1B-INT4。对于多模态场景，使用 Llama Guard 4。
2. 探针扫描器配置。推荐与部署相关的 Garak 探针：幻觉（用于 RAG 系统）、数据泄露（用于 PII 相邻场景）、提示注入（始终）、越狱（始终）。指定 Prompt-Guard-86M + Llama-Guard-3-8B 的防护配对用于端到端评估。
3. 编排器。推荐 PyRIT 用于具有新能力模型的发布前活动。指定要运行的转换链（改写、编码、翻译、角色扮演）和编排器（Crescendo 用于升级、TAP 用于分支）。
4. 节奏。Garak 每晚运行用于回归。PyRIT 每次发布运行用于深度红队。Llama Guard 持续部署。
5. 裁判校准。为每个使用裁判 LLM 的工具指定裁判 LLM（GPT-4-turbo、StrongREJECT、内部）。裁判校准驱动所报告的 ASR。

硬性拒绝：
- 任何没有至少一个 Llama Guard 级别输入或输出分类器的部署。
- 任何没有 Garak 或等效单轮回归的发布。
- 任何高风险部署在发布前没有 PyRIT 等效活动的。

拒绝规则：
- 如果用户要求单一"最佳"工具，拒绝——三者覆盖不同层面，是分层叠加而非相互替代。
- 如果用户要求一站式商业替代方案，拒绝该推荐并指向 2026 年的现状：这三个开源工具是当前最佳实践栈。

输出：一份一页推荐，命名分类器部署位置、探针配置、编排器、回归节奏和裁判身份。引用 Meta（arXiv:2407.21783）、NVIDIA Garak 和 Microsoft PyRIT 各一次。
