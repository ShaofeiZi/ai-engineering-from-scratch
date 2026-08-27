# 边缘推理：Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘侧最核心的约束不是算力，而是内存带宽。移动端 DRAM 通常只有 50-90 GB/s；数据中心 HBM3 则能达到 2-3 TB/s，两者相差 30-50 倍。解码阶段本质上受内存限制，因此这个差距具有决定性意义。到 2026 年，边缘推理已经分化成四条路线。Apple M4/A18 Neural Engine 在统一内存架构下峰值可达 38 TOPS，不需要 CPU↔NPU 拷贝。Qualcomm Snapdragon X Elite / 8 Gen 4 的 Hexagon 可达 45 TOPS。WebGPU + WebLLM 在 M3 Max 上运行 Llama 3.1 8B（Q4）大约能到 41 tok/s，大约是原生性能的 70-80%；GitHub 17.6k stars，提供 OpenAI-compatible API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）可以放下 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b 时约为 40 tok/s；Jetson T4000（JetPack 7.1）性能约是 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4 和 chunked prefill，并在 2026 年 CES 上由 Bosch、ThunderSoft、MediaTek 展示。

**Type:** 学习
**Languages:** Python（标准库，玩具级带宽受限解码模拟器）
**Prerequisites:** 第 17 阶段 · 04（服务引擎内部原理），第 17 阶段 · 09（生产量化）
**Time:** 约 60 分钟

## 学习目标

- 解释为什么移动端 LLM 推理受内存带宽限制，而算力反而是次要因素。
- 列出四类边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并分别匹配其适用场景。
- 说出 2026 年 WebGPU 的覆盖缺口（Firefox Android 仍在追赶）以及 Safari iOS 26 的落地情况。
- 为不同目标选择合适的量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题

客户想要一个设备端聊天机器人：语音优先、默认保护隐私、离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 能跑到约 55 tok/s，没问题；在 iPhone 16 Pro 上，同一个模型只有 3 tok/s，就不行了；在搭载 Snapdragon 8 Gen 3 的中端 Android 手机上，大约 7 tok/s；在浏览器里通过 WebGPU 跑在 Chrome Android v121+ 上，则根据设备不同在 4-8 tok/s 之间波动。

这种吞吐差异不是“移植没做好”，而是带宽差距、量化格式，以及用户态能否访问 NPU 三个因素叠加的结果。2026 年的边缘推理不是一个问题，而是四个问题，对应四种解法。

## 概念

### 带宽才是真正的上限

解码时，每生成一个 token 都要读取整套权重。一个 7B 的 Q4 模型大约是 3.5 GB。若带宽是 50 GB/s，读取 3.5 GB 需要 70 ms，对应理论上限大约只有 14 tok/s。若带宽提升到 90 GB/s（高端移动 DRAM），上限也不过升到约 25 tok/s。低于这个数字时，再多算力也救不了。

数据中心的 HBM3 如果能到 3 TB/s，读同样的 3.5 GB 只要 1.2 ms，上限就是 830 tok/s。模型相同，权重相同，差别只在内存子系统。

### Apple Neural Engine（M4 / A18）

- 峰值可达 38 TOPS。统一内存架构下，CPU 和 ANE 共享同一内存池，没有额外拷贝开销。
- 可通过 Core ML + `.mlmodel` 编译模型访问，也可以经由 PyTorch 走 Metal Performance Shaders（MPS）。
- Llama.cpp Metal backend 用的是 MPS，而不是直接调用 ANE；若要原生使用 ANE，需要先做 Core ML 转换。
- 对 2026 年的 iOS 应用而言，最实用的路径是 Core ML + INT4 weights + FP16 activations。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 峰值可达 45 TOPS。它与 CPU、GPU 集成在同一 SoC 内，但属于独立内存域。
- QNN（Qualcomm Neural Network）SDK 与 AI Hub 提供了从 PyTorch/ONNX 转换的能力。
- Chat templates、Llama 3.2、Phi-3 都已经在 AI Hub 中作为一等产物提供。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件生态明显落后于 Apple 和 Qualcomm；OpenVINO 在进步，但仍偏小众。
- 更适合 Windows ARM copilot 应用，也适合在 AMD/Intel 桌面端做本地优先场景。

### WebGPU + WebLLM

- 通过 WebGPU compute shaders 在浏览器里直接运行模型，无需安装。
- Llama 3.1 8B Q4 在 M3 Max 上约为 41 tok/s，大约相当于同一后端原生性能的 70-80%。
- WebLLM 在 GitHub 上有 17.6k stars；提供 OpenAI-compatible JS API；采用 Apache 2.0 许可。
- 2026 年的覆盖情况是：Chrome Android v121+、Safari iOS 26 GA，而 Firefox Android 仍在追赶。整体移动端覆盖约 70-75%。

### NVIDIA Jetson 家族

- Orin Nano Super（8GB）：可以容纳 Llama 3.2 3B、Phi-3，并获得不错的 tok/s。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 时约为 40 tok/s。
- Thor / T4000（JetPack 7.1）：性能约为 AGX Orin 的 2 倍，并支持 EAGLE-3 与 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 speculative decoding、NVFP4 weights、chunked prefill，也就是把数据中心那套优化移植到了边缘侧。

### 各目标的量化选择

| 目标 | 格式 | 说明 |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译后的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 受内存带宽限制 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路线 |

### 边缘侧的长上下文陷阱

Llama 3.1 的 128K context 是数据中心特性。放到一台只有 8 GB RAM 的手机上，4 GB 模型 + 2 GB KV cache（对应 32K tokens）+ 操作系统开销，结果就是 OOM。除非接受激进的 KV 量化（Q4 KV），否则边缘部署通常把上下文控制在 4K-8K。

### 语音才是杀手级应用

语音代理对延迟极其敏感，要求 first token < 500 ms。本地推理可以把网络延迟完全消掉。再配合语音转文本能力（Whisper Turbo 的一些变体已经可以在边缘端运行），边缘推理就能形成生产可用的语音闭环。

### 你需要记住的数字

- Apple M4 / A18 ANE: 38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max: 在 Llama 3.1 8B Q4 上约 41 tok/s。
- AGX Orin: 通过 vLLM 跑 gpt-oss-20b 时约 40 tok/s。
- 数据中心与边缘之间的带宽差距：30-50 倍。
- WebGPU 移动端覆盖率：约 70-75%（Firefox Android 仍然落后）。

```figure
edge-bandwidth-pipe
```

## 用起来

`code/main.py` 会基于带宽约束的数学模型，计算不同边缘目标上的理论解码吞吐上限。它还会与观测到的 benchmark 做对比，指出真正的瓶颈是带宽而不是算力。

## 产出

这一课会产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/browser/Jetson）、模型，以及延迟和内存预算，它会选出对应的量化格式与转换管线。

## 练习

1. 运行 `code/main.py`。对一个运行在 Snapdragon 8 Gen 3（约 77 GB/s 带宽）上的 7B Q4 模型，计算其解码上限。再和实测的 6-8 tok/s 对比，判断运行时是否高效。
2. Android 上的 WebGPU 需要 Chrome v121+。请为更老的浏览器设计一个回退方案，即走同一套 OpenAI-compatible API 的服务端推理。
3. 你的 iOS 应用需要 4K-context streaming。哪种模型与格式组合，能让你在 iPhone 16 上把活动内存控制在 4 GB 以下？
4. Jetson AGX Orin 能以 40 tok/s 运行 gpt-oss-20b，而 Jetson Nano 只能放下 3B 模型。如果你的产品同时面向这两类设备，如何统一推理栈？
5. 论证“WebLLM 在 2026 年是否已达到生产可用”。请引用覆盖率、性能，以及 Firefox Android 的缺口。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| ANE | “Apple 神经网络引擎” | M 系列和 A 系列上的设备端 NPU；统一内存架构 |
| Hexagon | “Qualcomm NPU” | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | “浏览器 GPU” | W3C 标准化的浏览器 GPU API；Chrome/Safari 在 2026 年已支持 |
| WebLLM | “浏览器 LLM 运行时” | MLC-LLM 项目；Apache 2.0；OpenAI-compatible JS |
| Jetson | “NVIDIA 边缘平台” | Orin Nano / AGX / Thor / T4000 家族 |
| TRT Edge-LLM | “边缘版 TensorRT” | 2026 年的 TensorRT-LLM 边缘版；支持 EAGLE-3 + NVFP4 |
| Unified memory | “共享内存池” | CPU 与 NPU 看到的是同一块 RAM；无拷贝开销 |
| Bandwidth-bound | “受内存带宽限制” | 解码受限于每秒读取权重的字节数 |
| Core ML | “Apple 转换框架” | Apple 的 ANE-native 模型转换框架 |
| QNN | “Qualcomm 技术栈” | Qualcomm 神经网络 SDK |

## 延伸阅读

- [端侧 LLM 现状总览 2026](https://v-chandra.github.io/on-device-llms/) — 边缘模型格局与基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘版发布说明。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计与基准。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE-native 转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 为 Hexagon 预转换的模型。
