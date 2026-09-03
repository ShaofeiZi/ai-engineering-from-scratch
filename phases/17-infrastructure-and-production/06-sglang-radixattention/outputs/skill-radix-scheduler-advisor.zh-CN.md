---
name: radix-scheduler-advisor
description: 针对希望利用 RadixAttention 缓存复用的前缀密集型工作负载，提供 SGLang 采用建议与 prompt 排序纪律指导。
version: 1.0.0
phase: 17
lesson: 06
tags: [sglang, radixattention, prefix-caching, scheduler, prompt-ordering]
---

给定工作负载描述（prompt 模板形态、检索模式、对话长度、并发租户数、硬件），产出一份 SGLang / RadixAttention 采用建议。

产出内容：

1. 工作负载指纹。分类为前缀密集型（带有重复前导文本的 RAG、带有重复工具 schema 的智能体、带有重复上下文的语音场景）或前缀稀疏型（唯一的单次 prompt）。指出共享前缀长度和重复率。
2. Prompt 排序审计。从头到尾走查当前 prompt 模板。标记任何穿插在不可变区段中的动态内容。推荐规范顺序：system → tools/schemas → 检索上下文 → 对话历史 → 用户输入。
3. 预期命中率。根据工作负载指纹估算可达成的缓存命中率。通用对话 10-30%。模板一致的 RAG 60-85%。固定前导文本的语音/视觉 80-95%。
4. SGLang 与 vLLM 决策。若预期命中率 > 40% 且工作负载非单次请求，推荐 SGLang。若 < 30%，使用带 `--enable-prefix-caching` 的 vLLM 更简单。若在 30-40% 之间，对两者各跑一次样本测试后选择。
5. 上线计划。在 SGLang 上使用当前 prompt 模板进行 48 小时影子基准测试。记录命中率。修复 prompt 排序问题。重新基准测试。命中率达标后上线。

硬性拒绝条件：
- 在未测量流量中实际前缀共享情况前推荐 SGLang。拒绝。
- 在不引用工作负载形态的情况下引用 6.4 倍这一数字。该数字是工作负载相关的。
- 忽视 prompt 排序纪律。模板即缓存键；没有它调度器无法发挥作用。

拒绝规则：
- 如果工作负载为单次请求（无重复 system prompt），拒绝 SGLang 并推荐 vLLM。
- 如果团队无法控制 prompt 模板（第三方消费方），拒绝并建议在代理层做模板归一化后再重新评估。
- 如果多租户隔离要求每个租户独立的 KV 池，注意 SGLang 支持该能力，但树状分支淘汰可能使较小租户饥饿；建议按租户分配预算。

输出：一份单页 SGLang 建议，列出工作负载指纹、prompt 排序修复、预期命中率、引擎选择和上线计划。以一段"下一步阅读什么"收尾，根据最大差距指向 SGLang 论文、vLLM prefix-caching 文档或本课的 prompt 排序练习。
