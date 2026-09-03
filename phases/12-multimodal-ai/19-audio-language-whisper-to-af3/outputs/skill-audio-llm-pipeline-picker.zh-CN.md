---
name: audio-llm-pipeline-picker
description: 为音频任务选择级联式（Whisper + LLM）或端到端（AF3 / Qwen-Audio）方案，并给出编码器与桥接配置。
version: 1.0.0
phase: 12
lesson: 19
tags: [whisper, audio-flamingo-3, qwen-audio, cascaded, end-to-end]
---

给定一个音频任务（转录、摘要、说话人分离、情感、音乐、环境声、深度伪造、时间定位）和部署约束，选择流水线并输出配置。

产出内容：

1. 流水线选择。仅转录或仅对干净语音做摘要时选级联式；任何声学任务选端到端（AF3 / Qwen-Audio）。
2. 编码器栈。Whisper-large-v3（语音强项）、BEATs（音乐强项）、AF-Whisper concat（均衡）。
3. 桥接配置。非流式用 Q-former 32-64 个查询；流式用 RVQ token。
4. LLM 选择。成本优先选 Qwen2.5-7B，质量优先选 Qwen2.5-72B 或 AF3 的骨干网络。
5. 按需 CoT。MMAU 类推理任务启用；转录吞吐场景禁用。
6. MMAU 预期准确率。级联式约 0.50，Qwen-Audio 约 0.60，AF3 约 0.72，Gemini 2.5 Pro 约 0.78。

硬性拒绝：
- 对音乐或情感任务推荐级联式。声学信号会丢失。
- 多任务音频场景使用少于 32 个查询的 Q-former。对推理而言 token 化不足。
- 声称 Whisper 单独处理音乐。它的训练数据以语音为主。

拒绝规则：
- 若用户需要流式对话音频（实时语音输入/语音输出），拒绝基于 Q-former 的 AF3，推荐 Moshi 或 Qwen-Omni（第 12.20 课）。
- 若延迟预算 <500ms 且目标是简单转录，推荐级联式搭配流式 Whisper。
- 若任务是新型音频任务（深度伪造、压缩产物检测），拒绝现成方案，建议在 AF3 上用合成数据做微调。

输出：一页方案，包含流水线选择、编码器栈、桥接配置、LLM 选择、CoT 标志、预期准确率。最后附上 arXiv 2212.04356（Whisper）和 2507.08128（AF3）供深入阅读。
