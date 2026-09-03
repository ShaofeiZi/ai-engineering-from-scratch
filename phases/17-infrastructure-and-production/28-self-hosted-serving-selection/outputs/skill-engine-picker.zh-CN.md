---
name: engine-picker
description: 根据硬件、规模和工作负载选择自托管 LLM 推理引擎（llama.cpp、Ollama、TGI、vLLM、SGLang）。将 2026 年 TGI 进入维护模式作为迁移触发点。
version: 1.0.0
phase: 17
lesson: 28
tags: [self-hosted, vllm, sglang, llama-cpp, ollama, tgi, trt-llm, engine-selection]
---

给定硬件（CPU / Apple Silicon / AMD / NVIDIA Hopper / NVIDIA Blackwell）、规模（单用户 / 小团队 / 生产 / 企业）和工作负载（通用对话 / 智能体 / RAG / 长上下文 / 代码），产出引擎推荐。

需产出：

1. 引擎。指定具体引擎。引用"硬件优先、规模次之、工作负载最后"的决策树。
2. 为何不选其他。对每个备选引擎说明不选原因（TGI 处于维护模式、AMD 排除 TRT-LLM、Ollama 仅限开发）。
3. 管线。若为生产场景，指明管线模式（开发 Ollama → 预发布 llama.cpp → 生产 vLLM/SGLang）并确认权重格式（GGUF 或 HF）可贯通。
4. 生产堆叠。在生产规模下，参见 Phase 17 · 18（生产栈）、· 17（分离式）、· 11（缓存感知路由）以了解组合方案。
5. TGI 迁移。若现有引擎为 TGI，须指定迁移计划与时间表——不紧迫，但应在 6 个月内启动。
6. 硬件陷阱。指出两个硬性约束：仅 CPU → llama.cpp；AMD → 无 TRT-LLM。

硬性拒绝：
- 在 2026 年将新项目默认选用 TGI。拒绝——处于维护模式。
- 在并发用户 >1 的共享生产环境中使用 Ollama。拒绝——吞吐量差距。
- 在未确认仅限 NVIDIA 的情况下推荐 TRT-LLM。拒绝——AMD / 非 NVIDIA 为硬性阻断。

拒绝规则：
- 若硬件混合（部分 AMD、部分 NVIDIA），须按集群逐一决策引擎；不得强制使用单一引擎。
- 若工作负载在生产规模下为"未知/通用"，默认选用 vLLM 并计划在积累 3 个月流量数据后重新评估。
- 若团队需要"每 GPU 最快"但无 Blackwell 可用且坚持仅用 Hopper，予以确认——TRT-LLM 或 vLLM 均可接受。

输出：一页推荐，包含引擎、被排除的备选、管线、生产堆叠、TGI 迁移姿态。末尾附单一季度审阅：当工作负载形态发生实质性变化时重新评估引擎选择。
