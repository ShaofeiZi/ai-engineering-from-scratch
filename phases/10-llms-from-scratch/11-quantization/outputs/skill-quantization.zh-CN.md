---
name: skill-quantization
description: 根据硬件、质量和延迟约束，为 LLM 部署选择合适的量化策略
version: 1.0.0
phase: 10
lesson: 11
tags: [quantization, inference, deployment, optimization, fp8, int4, int8, gptq, awq, gguf]
---

# 量化决策框架

在部署语言模型时，使用该框架选择正确的数字格式、量化方法和质量验证策略。

## 输入要求

提供以下信息：
- **模型**（名称、参数量、原始精度）
- **目标硬件**（GPU 型号/VRAM、CPU、Apple Silicon、边缘设备）
- **延迟目标**（tokens/秒、首 token 时间）
- **质量下限**（可接受的最大困惑度上升、基准偏差）
- **服务模式**（批大小、最大上下文长度、并发用户数）

## 快速选择

| 你的情况 | 格式 | 方法 | 预期质量损失 |
|---------------|--------|--------|----------------------|
| H100 GPU，追求最大吞吐 | FP8 E4M3 | 原生 H100 转换 | < 0.1% |
| A100/A10，需要 2 倍吞吐 | INT8 | LLM.int8() 或 SmoothQuant | < 0.5% |
| 单张 24GB GPU，70B 模型 | INT4 | AWQ 或 GPTQ | 1-3% |
| MacBook / Apple Silicon | INT4 GGUF | 经由 llama.cpp 的 Q4_K_M | 1-2% |
| 移动 / 边缘设备 | INT4 或 INT3 | QAT + 设备专属 | 2-5% |
| 追求最大压缩，可接受一些损失 | INT2 | QuIP# 或 AQLM | 5-15% |
| 训练（混合精度） | BF16 + FP32 累加 | 框架原生支持 | 0% |

## 按组件选择精度

并非所有张量都应一视同仁。

| 组件 | 安全下限 | 推荐 | 避免 |
|-----------|-------------|-------------|-------|
| FFN 权重 | INT4 | INT4 (AWQ/GPTQ) | 不带 QAT 的 INT2 |
| 注意力权重 | INT4 | INT8 或 FP8 | INT2 |
| 嵌入层 | INT8 | FP16（保留原始） | INT4 |
| 输出头 | INT8 | FP16（保留原始） | INT4 |
| KV 缓存 | FP8 | FP8 或 INT8 | 长上下文下的 INT4 |
| 注意力 logits | FP16 | FP16 或 BF16 | INT8 |
| 激活值（推理） | INT8 | FP8 或 INT8 | INT4 |

## 方法对比

### GPTQ
- **何时用：** GPU 推理，你想要 Hugging Face 兼容的模型
- **校准数据：** 128 个样本，每个 2048 token
- **耗时：** 在 A100 上对 70B 约 30-60 分钟
- **工具：** `auto-gptq`、`exllama`、`exllamav2`
- **优势：** 经过充分测试，Hugging Face 上模型库庞大
- **劣势：** 应用比 AWQ 慢，在某些模型上质量略低于 AWQ

### AWQ
- **何时用：** GPU 推理，你追求每比特最佳质量
- **校准数据：** 128 个样本
- **耗时：** 在 A100 上对 70B 约 15-30 分钟
- **工具：** `autoawq`、`vLLM`（原生支持）
- **优势：** 最佳 INT4 质量，应用速度快，vLLM 集成
- **劣势：** 模型库比 GPTQ 小

### GGUF
- **何时用：** CPU 推理、Apple Silicon、llama.cpp 生态
- **变体：** Q2_K、Q3_K_S/M/L、Q4_K_S/M、Q5_K_S/M、Q6_K、Q8_0、F16
- **推荐默认：** Q4_K_M（质量/大小平衡最佳）
- **工具：** `llama.cpp`、`ollama`、`LM Studio`
- **优势：** 自包含文件、混合精度、生态庞大
- **劣势：** 对 GPU 非最优（专为 CPU/Metal 设计）

### SmoothQuant
- **何时用：** GPU 上的 INT8，同时对权重和激活做量化
- **核心思想：** 通过逐通道缩放，将量化难度从激活迁移到权重
- **工具：** `smoothquant`、`TensorRT-LLM`
- **优势：** 实现 W8A8（权重和激活均为 INT8），约 2 倍加速
- **劣势：** 仅限 INT8，无法扩展到 INT4

## 质量验证协议

量化之后、部署之前进行验证：

1. **困惑度测试。** 在 WikiText-2 或你的领域语料上计算。偏差 < 0.5 为优秀，0.5-1.0 为良好，> 2.0 为有问题。

2. **基准扫描。** 运行 MMLU（通用）、GSM8K（数学）、HumanEval（代码）。数学和代码对精度损失最敏感。

3. **输出对比。** 从原始模型和量化模型各生成 100 条响应。用 LLM 作为评判计算胜率。目标：量化模型在 > 90% 的提示上胜出或持平。

4. **延迟测量。** 在批大小为 1 和你的目标批大小下测量 tokens/秒。验证加速是否足以抵消质量代价。

5. **长上下文测试。** 如果服务长上下文（> 4K token），在你的最大上下文长度下测试。KV 缓存的量化误差随序列长度累积。

## 内存预算计算器

```
Weight memory (GB) = parameters (B) * bits / 8 / 1.073741824
KV cache per token (MB) = 2 * num_layers * d_model * bits / 8 / 1048576
KV cache for context (GB) = kv_per_token * max_context_length / 1024
Activation memory (GB) ~ 1-4 GB (relatively constant, depends on batch size)
Total = weight_memory + kv_cache + activation_memory + overhead (10-20%)
```

以 Llama 3 70B 在 INT4、32K 上下文为例：
- 权重：70B * 4 / 8 / 1.07 = 32.6 GB
- KV 缓存 (FP16)：2 * 80 * 8192 * 16 / 8 / 1e9 * 32768 = ~40 GB
- KV 缓存 (FP8)：~20 GB
- FP8 KV 下总计：~55 GB（可装入单张 80GB A100）

## 常见错误

| 错误 | 为何失败 | 修复 |
|---------|-------------|-----|
| 将嵌入层量化为 INT4 | 第一层会把误差放大到整个模型 | 嵌入层保留 FP16 或 INT8 |
| 对 INT4 使用逐张量缩放 | 一行离群值会破坏所有行的精度 | 使用逐通道或逐组缩放 |
| 不校准 GPTQ/AWQ | 没有代表性数据，缩放因子就是错的 | 使用来自你领域的 128 个样本 |
| 所有层用相同位宽 | 首尾层更敏感 | 混合精度：首尾层用更高位宽 |
| 在超长上下文下量化 KV 缓存 | 误差随序列长度二次累积 | KV 缓存用 FP8，而非 INT4 |
| 跳过质量验证 | 某些模型量化效果差（尤其在边界处） | 始终运行困惑度 + 任务评估 |

## 部署配方

### 配方 1：vLLM 搭配 AWQ（GPU 服务器）
```
pip install vllm autoawq
vllm serve model-awq --quantization awq --dtype half --max-model-len 8192
```

### 配方 2：llama.cpp 搭配 GGUF（MacBook）
```
./llama-server -m model.Q4_K_M.gguf -c 4096 -ngl 99
```

### 配方 3：TensorRT-LLM 搭配 FP8（H100）
```
trtllm-build --model_dir model --output_dir engine --dtype float16 --use_fp8
```
