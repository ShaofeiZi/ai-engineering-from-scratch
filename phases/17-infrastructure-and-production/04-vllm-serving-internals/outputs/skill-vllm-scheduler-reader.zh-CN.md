---
name: vllm-scheduler-reader
description: 通过解读调度器级别的配置项来诊断 vLLM 服务配置，识别 PagedAttention、连续批处理和分块 prefill 中的瓶颈所在。
version: 1.0.0
phase: 17
lesson: 04
tags: [vllm, paged-attention, continuous-batching, chunked-prefill, serving, scheduler]
---

给定 vLLM 服务配置（模型、dtype、硬件、`--gpu-memory-utilization`、`--max-num-batched-tokens`、`--enable-chunked-prefill`、`--speculative-model` 或 `--speculative-config`、最大并发数，以及包含 TTFT 均值/P99、ITL 均值/P99、吞吐量 tok/s 的观测指标集），产出调度器级别的诊断。

产出：

1. 配置解读。对每个标志，命名其控制的调度器行为及 2026 年默认值。标记任何设置为非默认值的标志并说明原因。
2. 瓶颈识别。将瓶颈分类为以下之一：PagedAttention 供给不足（KV 块耗尽）、连续批处理停滞（WAITING 队列增长）、分块 prefill 尺寸不当（TTFT 尾部尖峰）、decode 计算受限（ITL 下限）、或 HBM 受限（无法容纳批次）。用报告的指标论证。
3. 配置项建议。具体的、有序的操作——翻转哪个标志、尝试哪个值、观察哪个指标。在穷尽调度器级别调优之前，不得建议"加更多 GPU"。
4. 兼容性检查。针对 vLLM v0.18.0：将 `--enable-chunked-prefill` + `--speculative-model` 组合标记为硬性不兼容。如果两者都需要，推荐 V1 中的 N-gram GPU 推测解码作为文档化的例外方案。
5. 后续阅读。根据诊断结果，指向 vLLM v0.18.0 发行说明、PagedAttention 论文或 Aleksa Gordic 的 V1 调度器讲解之一。

硬性拒绝：
- 在缺少四项核心指标（TTFT、ITL、吞吐量、并发数）的情况下进行诊断。拒绝并要求提供指标集。
- 在未检查推测解码配置的情况下推荐 `--enable-chunked-prefill`。
- 将 `DCGM_FI_DEV_GPU_UTIL` 作为扩缩容信号。vLLM 预分配 KV；占空比数据具有误导性。

拒绝规则：
- 如果在 H100 上报告的吞吐量低于 100 tok/s，瓶颈可能不在 vLLM——检查客户端 tokenizer、Python GIL 或请求级序列化。
- 如果 `--gpu-memory-utilization` 设置低于 0.7，拒绝进一步调优——运维方选择不使用可用 HBM，修复方法是先提高上限再翻转调度器标志。
- 如果运维方要求基于草稿模型推测的推测解码 + 分块 prefill 方案，拒绝并指出 v0.18.0 的不兼容性。指向第 17 阶段 · 05 的 EAGLE-3。

输出：一页调度器诊断，列出标志、瓶颈、有序建议、兼容性说明和后续阅读指引。以一段"下一步测量什么"结尾，根据所识别的瓶颈命名 P99 ITL、块分配率或 WAITING 队列深度之一。
