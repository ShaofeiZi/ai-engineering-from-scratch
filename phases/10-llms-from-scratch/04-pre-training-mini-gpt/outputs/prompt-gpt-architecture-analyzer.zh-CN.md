---
name: prompt-gpt-architecture-analyzer
description: 分析任意 GPT 风格 Transformer 模型的架构选择
version: 1.0.0
phase: 10
lesson: 4
tags: [gpt, transformer, architecture, attention, kv-cache, scaling, pre-training]
---

# GPT 架构分析器

在从技术报告、模型卡或训练日志评估一个 GPT 风格模型时,使用此框架拆解架构并识别设计权衡。

## 分析流程

### 1. 参数分配拆解

计算每个组件的精确参数量:

- **Token 嵌入**:vocab_size x embed_dim
- **位置嵌入**:max_seq_len x embed_dim
- **每个 block 的注意力**:4 x embed_dim x embed_dim(Q、K、V、输出投影)
- **每个 block 的 FFN**:2 x embed_dim x ff_dim + embed_dim + ff_dim(两个线性层加偏置)
- **每个 block 的 LayerNorm**:4 x embed_dim(两个归一化,各含 scale + bias)
- **最终 LayerNorm**:2 x embed_dim
- **输出头**:vocab_size x embed_dim(若与 token 嵌入共享权重则为 0)

若任一组件超过总参数的 40%,则标记。嵌入矩阵在小模型中占主导。注意力和 FFN 在大模型中占主导。

### 2. 注意力设计分析

评估注意力配置:

- **头维度**:embed_dim / num_heads。标准值为 64(GPT-2)或 128(Llama 3)。低于 32 会限制每个头的表达能力。高于 128 会浪费算力且收益甚微。
- **每层头数**:头越多 = 注意力模式越多样,但 KV cache 占用更多内存。
- **分组查询注意力 (GQA)**:模型是否在多个 Q 头之间共享 K/V 头?Llama 3 使用 GQA,32 个 Q 头对应 8 个 KV 头。这将 KV cache 减少 4 倍。
- **上下文长度**:最大位置嵌入数。RoPE 允许在训练长度之外外推。绝对位置嵌入则不能。

### 3. 内存预算

在模型最大上下文长度下推理:

- **权重 (FP16)**:total_params x 2 字节
- **KV Cache (FP16)**:2 x num_layers x num_kv_heads x head_dim x max_seq_len x 2 字节
- **激活**:batch_size x seq_len x embed_dim x 2 字节 x num_layers(近似)

若 KV cache 超过权重内存则标记。这发生在长上下文模型(128K+)上,表明模型在解码阶段受内存限制。

### 4. 计算画像

- **Prefill 每 token FLOPS**:约 2 x total_params(每个参数一次矩阵乘,前向)
- **Decode 每 token FLOPS**:与 prefill 相同但作用在单个 token 上
- **Prefill 瓶颈**:受算力限制(GPU TFLOPS)
- **Decode 瓶颈**:受内存限制(GPU 内存带宽)
- **算术强度**:每访问一字节内存的 FLOPS。低于 100 = 受内存限制。

### 5. 缩放决策

对照已知缩放定律评估:

- **Chinchilla 最优**:对于给定算力预算 C,最优模型大小 N 和 token 数 D 满足 N ~ D(大致等比例缩放)。7B 模型约需 140B tokens。
- **Llama 3 过训练**:Meta 在 15T tokens 上训练 Llama 3 8B(Chinchilla 最优的 100 倍)。用更多数据过训练小模型可带来更好的每 token 推理成本。
- **宽度与深度**:在相同参数量下,更深(更多层)的模型通常比更宽(更大 embed_dim)的模型具有更好的样本效率。

## 危险信号

- **FFN 比例不是 4 倍**:标准是 ff_dim = 4 x embed_dim。Llama 使用 8/3 x embed_dim 加 SwiGLU。偏离应有正当理由。
- **未共享权重**:输出头应与 token 嵌入共享权重,除非 vocab_size 相对 embed_dim 非常大。
- **13B 以上无 GQA**:超过 13B 的模型若没有分组查询注意力,将产生过大的 KV cache。
- **长上下文无 RoPE**:绝对位置嵌入无法在训练长度之外外推。面向 32K+ 上下文的模型应使用旋转嵌入。
- **学习率相对于模型大小过高**:更大的模型需要更低的峰值学习率。GPT-2 Small 使用 6e-4。Llama 3 405B 使用 8e-5。

## 输出格式

1. **参数表**:逐组件的参数量及百分比
2. **内存预算**:最大上下文长度下的权重、KV cache 和激活内存
3. **计算画像**:A100/H100 的 prefill 和 decode 吞吐量估算
4. **设计评估**:模型做得对的地方和非标准之处
5. **缩放结论**:模型规模是否与其训练数据相匹配
