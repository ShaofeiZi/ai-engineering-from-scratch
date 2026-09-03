---
name: open-model-picker
description: 针对给定部署目标选择开放式 LLM 家族、量化方案和推理技术栈。
version: 1.0.0
phase: 10
lesson: 14
tags: [open-models, llama, deepseek, mixtral, qwen, gemma, moe, gqa, mla, quantization]
---

给定一个部署目标（GPU 类型、每 GPU 的 VRAM、GPU 数量、目标上下文长度、目标 p50/p99 延迟、峰值并发请求数）和一个任务画像（对话、代码、推理、长上下文检索、工具使用），推荐一个开源模型加部署栈，并对第 14 课的六个架构旋钮各自给出明确推理。

产出：

1. 模型短名单。三个候选，各自列出总参数量、活跃参数量（MoE 感知）、架构标志（norm / activation / position / attention / MoE / context），以及它入选短名单的唯一理由。
2. 内存预算检查。针对首选候选：BF16 下的权重内存和所选量化下的权重内存；目标 batch size 在目标上下文下的 KV cache；激活余量。若权重 + KV cache + 激活超过可用 VRAM，则中止推荐。
3. 量化选择。GPTQ-4bit、AWQ-4bit、FP8 或 BF16。依据任务对精度的敏感性来论证（代码 / 数学 / 推理任务比对话或检索受到激进量化更大的影响）。
4. 推理栈。vLLM、TensorRT-LLM、SGLang 或 llama.cpp。依据以下各项论证：连续批处理需求、投机解码支持、量化格式兼容性，以及单节点 vs 多节点拓扑。
5. 吞吐量合理性检查。基于 GPU 内存带宽（decode）和 TFLOPs（prefill）估算 prefill tokens/sec 和 decode tokens/sec。若 decode 吞吐量低于目标的并发用户下限，则拒绝该推荐。
6. 备选。若首选候选超出 VRAM 或吞吐预算，给出第二选择。必须始终命名一个。

硬拒绝：
- 在单张 24GB 消费级 GPU 上、未做 offload 或未做激进量化的 30B 以上稠密模型。
- 在不支持 expert-parallel 的部署栈上的 MoE 模型。
- 在没有 GQA 或 MLA 的架构上做长上下文（128k+）（KV cache 爆炸）。
- 任何不指明具体模型修订版本的推荐（例如"Llama 3 8B Instruct v3.1"，而非"Llama 3"）。

输出：一页推荐，列出模型、量化、栈，并对每个决策附编号证据。以一段"若……则值得重新考虑"收尾，点名会翻转选择的特定能力或部署参数。
