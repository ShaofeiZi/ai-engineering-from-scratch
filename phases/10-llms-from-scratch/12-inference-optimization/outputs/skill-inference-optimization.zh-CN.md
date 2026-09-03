---
name: skill-inference-optimization
description: 诊断并优化 LLM 推理服务的吞吐量、延迟和成本
version: 1.0.0
phase: 10
lesson: 12
tags: [inference, kv-cache, batching, speculative-decoding, vllm, optimization]
---

# LLM 推理优化模式

两个阶段：预填充（计算受限、并行）和解码（内存受限、串行）。
每项优化都针对其中一个或两个阶段。

```
Request -> Prefill (process prompt) -> Decode (generate tokens) -> Response
              |                            |
         Compute-bound               Memory-bound
         Optimize: fusion,           Optimize: batching,
         prefix caching              quantization, speculation
```

## 决策框架

### 步骤 1：识别你的瓶颈

为你的工作负载测量 ops:byte 比率：

| ops:byte | 受限类型 | 优化什么 |
|----------|-------|-----------------|
| < 50 | 内存 | 量化 KV 缓存，增大批大小 |
| 50-200 | 过渡区 | 两者都重要，从批处理入手 |
| > 200 | 计算 | 算子融合、张量并行、FP8 |

### 步骤 2：选择你的引擎

- **默认**：vLLM（最广的模型支持、PagedAttention、OpenAI 兼容 API）
- **多轮 / 结构化输出**：SGLang（RadixAttention 前缀缓存、受限解码）
- **NVIDIA 最大吞吐**：TensorRT-LLM（算子融合、H100 上的 FP8）

### 步骤 3：按顺序应用优化

1. **KV 缓存** —— 始终开启，无副作用
2. **连续批处理** —— 始终开启，无副作用（vLLM/SGLang 默认开启）
3. **前缀缓存** —— 如果有共享系统提示则开启（大多数聊天机器人都有）
4. **量化** —— KV 缓存 INT8/FP8 可减少 2-4 倍内存，质量损失极小
5. **投机解码** —— 当延迟比吞吐更重要时加入
6. **张量并行** —— 当模型无法装入单卡时跨 GPU 切分

## KV 缓存内存公式

```
per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param
total = per_token * sequence_length * num_concurrent_users
```

常见模型的快速参考（BF16）：

| 模型 | 每 token | 100 用户 @ 4K |
|-------|-----------|----------------|
| Llama 3 8B | 32 KB | 12.5 GB |
| Llama 3 70B | 320 KB | 125 GB |
| Llama 3 405B | 504 KB | 197 GB |

## 投机解码清单

- 草稿模型应比目标小 5-10 倍（例如用 8B 为 70B 起草）
- 接受率 > 70% 才有有意义的加速
- 在可预测文本上效果最好（代码、结构化输出、自然语言）
- 在创意/重度采样任务上效果最差（低温有帮助）
- 对大多数工作负载 EAGLE > draft-target > n-gram

## 常见错误

- 在 batch=1 下运行解码（内存受限，计算上 GPU 95% 闲置）
- 分配连续的 KV 缓存块（使用 PagedAttention，近乎零浪费）
- 当 80% 请求共享同一系统提示时忽略前缀缓存
- 为模型权重过度配置 GPU 内存，KV 缓存无处安放
- 只测吞吐不测延迟（10 秒 TTFT 下的高吞吐毫无用处）
- 在高温下使用投机解码（接受率降至 50% 以下）

## 监控清单

- 首 token 时间 (TTFT)：预填充延迟，交互使用目标 < 500ms
- token 间延迟 (ITL)：解码速度，流式目标 < 50ms
- 吞吐（tokens/秒）：跨所有并发用户的总量
- KV 缓存利用率：已分配缓存的使用百分比
- 批利用率：每次迭代中填充的批槽位百分比
- 队列深度：等待批槽位的请求数
