---
name: edge-target-picker
description: 根据设备、模型和延迟预算，选择边缘推理目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）及匹配的量化格式。
version: 1.0.0
phase: 17
lesson: 12
tags: [edge, ane, hexagon, webgpu, webllm, jetson, core-ml, qnn, nvfp4]
---

给定部署平台（iOS、Android、浏览器、机器人/汽车/边缘服务器）、模型及延迟/内存预算，产出一份边缘目标推荐方案。

产出内容：

1. 目标。指定具体的 NPU/GPU（ANE、Hexagon、WebGPU、Jetson Orin Nano / AGX / Thor）。以平台和 2026 年运行时覆盖情况来论证。
2. 带宽上限。计算理论解码上限：bandwidth_GB_s / model_size_GB。与用户的 tok/s 需求比较。如果上限低于需求，拒绝或提议更小的模型 / 更激进的量化。
3. 量化格式。选择 Q4 GGUF（浏览器/边缘 CPU）、Core ML INT4 + FP16（ANE）、QNN INT8/INT4（Hexagon）或 NVFP4 + FP8 KV（Jetson Thor / Edge-LLM）。
4. 转换流水线。指定确切的转换器（Core ML converter、Qualcomm AI Hub、用于 WebLLM 的 MLC-LLM、TensorRT-LLM Edge compiler）。
5. 上下文预算。声明在设备 RAM 中与权重并存放的最大上下文。对于长上下文场景，指定 KV 量化（Q4 KV）或拒绝。
6. 降级方案。当设备能力不足或 WebGPU 不可用（Firefox Android、旧版浏览器）时，指定使用相同 OpenAI 兼容接口的服务端 API 降级方案。

硬性拒绝：
- 承诺超过带宽上限的 tok/s。拒绝——物理限制。
- 在 2026 年通过非 Core ML 运行时直接使用 ANE。只有 Core ML 能原生暴露 ANE。
- 假设每个浏览器都有 WebGPU。2026 年移动端覆盖率约 70-75%；必须始终指定降级方案。

拒绝规则：
- 如果模型 >6 GB 且目标是手机（4-8 GB RAM），拒绝——先提议更小的模型或更激进的量化。
- 如果请求是在 iPhone 上对 7B 模型使用 128K 上下文，拒绝——设备 RAM 无法容纳，除非使用 Q4 KV 加滑动窗口注意力。
- 如果部署要求在 Android 上通过 WebGPU 进行长上下文流式传输且用户要求 Firefox 支持，拒绝并要求使用 Chrome 或服务端降级。

输出：一页方案，命名目标、上限、量化、转换器、上下文预算、降级方案。以单一指标结尾：目标群体中最差设备上观测到的 tok/s。
