# 梯度检查点与激活重计算

> 反向传播会保留每个中间激活值。对于 70B 参数、128K 上下文的模型，每个进程需要保存 3 TB 激活值。检查点技术以内存换 FLOP：不再保存，而是在需要时重新计算。真正的问题是应丢弃哪些片段，答案并不是“全部”。

**Type:** 构建
**Languages:** Python（使用 numpy，可选 torch）
**Prerequisites:** 阶段 10 第 04 课（预训练 Mini-GPT）、阶段 10 第 05 课（规模化与分布式训练）
**Time:** 约 70 分钟

## 问题

训练 Transformer 时，每一层都要保存反向传播所需的所有操作输入：注意力输入、Q/K/V 投影、Softmax 输出、前馈网络输入、归一化输出和残差流。对于隐藏维度为 `d`、序列长度为 `L`、批大小为 `B` 的层，每层大约需要保存 `12 * B * L * d` 个浮点数。

当 `d=8192, L=8192, B=1` 时，每层激活值在 BF16 下占 800 MB。64 层模型需要 51 GB 激活值——这还没有乘以微批次大小，没有加入每个头的注意力 Softmax 中间值（`L^2`），也没有考虑张量并行的局部副本。

账单有两面：BF16 权重加优化器状态可能勉强装进 80GB，激活值却会把你推过上限。梯度检查点（又称激活重计算）是标准解决方案。丢弃大多数激活值，在反向传播时重新执行前向过程来恢复它们。代价是更多 FLOP，收益是内存随检查点分段数相对于总层数的比例下降。

朴素实现检查点，每步大约会多消耗 33% 的前向传播 FLOP。若依据 Korthikanti 等人提出的“智能选择”妥善实现选择性检查点，可以在不到 5% 的 FLOP 开销下节省 5 倍内存。再考虑 FP8 矩阵乘法、FSDP 卸载和专家并行 MoE，这一点尤其重要：内存和浪费的计算，两者都承受不起。

## 概念

### 反向传播究竟需要什么

`output = layer(input)`。反向传播需要 `grad_input` 与 `grad_params`。为了计算它们，它还需要：

- `input`（线性层通过 `grad_params = input.T @ grad_output` 计算参数梯度）
- 一些激活函数导数所需的中间值（ReLU/GELU/Softmax 的导数取决于激活值）

前向传播会自动把这些值保存在自动微分图中。每次调用 `tensor.retain_grad()`，以及每个需要其输入的操作，都会保留一个引用。

### 朴素完整检查点

把网络划分为 `N` 个片段。前向传播时，只保存每个片段的*输入*。反向传播需要中间值时，重新运行该片段的前向传播来重新生成，再执行求导。

例如，把 32 层 Transformer 拆成 32 个每段 1 层的片段。

- 内存：保存 32 份层输入（较小），而不是 32 × 每层激活量（巨大）。
- 额外计算：每个片段多执行一次前向传播，即总前向 FLOP 增加约 33%（因为反向传播为前向的 2 倍，完整步骤从 1 + 2 = 3 个单位变为 1 + 1 + 2 = 4 个单位）。

这就是 Chen 等人 2016 年提出的原始方案：每隔 `sqrt(L)` 层设置一个检查点，以平衡内存与计算。L=64 时，就是 8 个检查点。

### 选择性检查点（Korthikanti，2022）

不同激活值的存储成本并不相同。注意力 Softmax 输出为 `B*L*L*heads`，随序列长度呈*二次方*增长；前馈网络隐藏激活为 `B*L*4d`，只呈线性增长。序列很长时，Softmax 占据主导。

选择性检查点会保留存储成本低的激活值（线性投影、残差），只重新计算存储成本高的部分（注意力）。这样只需少量重新计算 FLOP，就能节省 O(L²) 内存。

Megatron-Core 把它实现为“选择性”激活重计算。2024 年之后的大多数前沿训练任务都会使用它。

### 卸载

除重新计算外，还可以在前向与反向传播之间把激活值发送到 CPU 内存。它需要 PCIe 带宽；只有空闲带宽足以抵消重新物化成本时才有收益。混合策略很常见：一部分层使用检查点，另一部分层使用卸载。

FSDP2 把卸载作为一等选项提供。当 GPU 受内存限制，而 CPU-GPU 传输仍有余量时，卸载特别有效。

### 重计算成本模型

每 `k` 层使用一次朴素检查点、网络共 `L` 层时，每步 FLOP 为：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性检查点只重新计算注意力内核，而不是完整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量为 `A`。对于 `L` 层，总激活内存为 `L * A`。

完整检查点（片段大小为 1）：只保存 `L * input_volume`（对标准 Transformer 约为 `L * 1/10 A`），节省约 `9 * L * A * 1/10`。

每隔 `k` 层设置检查点：保存 `L/k * A`，再加当前活跃片段中 `k-1` 层的激活量。

当 `k = sqrt(L)` 时，内存和重计算成本都随 `sqrt(L)` 增长——对于成本均匀的层，这是最优权衡。

### 不应使用检查点的情况

- 流水线阶段中已经在途的最内层，因为它们无论如何都必须完成。
- 如果首层与末层主导该阶段的计算，则不应对它们使用检查点（Transformer 中很少出现）。
- 已经使用 FlashAttention 的注意力内核——Flash Attention 本身会快速重算 Softmax，因此额外的层级检查点收益很小。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包裹一个片段。PyTorch 只保存 `input`，反向传播时重新计算其他所有内容。

2. **基于装饰器：** 把各层标记为可设置检查点；训练器在配置阶段决定包裹哪些片段。

3. **手工显式重计算：** 自行编写反向传播，调用重复前向过程的自定义 `recompute_forward`。

三种方式产生相同的函数结果，包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 重计算时必须全收集或重新散布检查点输入；需要计入通信成本。
- **流水线并行：** 典型模式是为每个流水线阶段的前向传播设置检查点，使反向顺序执行的微批次可以复用激活内存。
- **FP8 重计算：** 重计算期间更新的 amax 历史必须与原始前向传播一致，否则 FP8 缩放会漂移。多数框架会保存缩放状态快照。

```figure
activation-recompute
```

## 动手构建

### 第 1 步：带分段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第 2 步：需要全部激活值的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第 3 步：每 k 层设置检查点后的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第 4 步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第 5 步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第 6 步：最优片段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第 7 步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 学以致用

- **torch.utils.checkpoint：** `from torch.utils.checkpoint import checkpoint`——PyTorch 的标准包装器。它包裹一个函数，只保存输入，并在反向传播时重算。
- **Megatron-Core 激活重计算：** 支持 `selective`、`full` 与 `block` 模式，是 2024 年以后前沿训练的标准配置。
- **FSDP2 卸载：** FSDP2 中的 `module.to_empty(device="cpu")` 与 `offload_policy` 会把激活值分片到 CPU，而不是重新计算。
- **DeepSpeed ZeRO-Offload：** 把优化器状态与激活值卸载到 CPU，与检查点技术互补。

## 交付成果

本课会生成 `outputs/prompt-activation-recompute-policy.md`——一个接收模型配置（层数、隐藏维度、序列长度、批大小）与可用 GPU 内存，再输出逐层重计算策略（无/选择性/完整/卸载）的提示词。

## 练习

1. 验证正确性。比较 `model_forward` + `model_backward`（保留全部激活值）与 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须在机器精度范围内完全一致。

2. 将片段大小 `k` 从 1 扫描到 `L`，绘制 FLOP 开销与内存曲线，并找出曲线拐点。

3. 实现选择性检查点：保存注意力模块输入，但不保存其中间值。对于序列长度为 8192 的 32 层模型，测量它相对于完整逐层检查点的 FLOP 开销。

4. 添加卸载。把片段输入保存到模拟“CPU 缓冲区”（独立列表），以字节数/时间衡量“PCIe 带宽”，并找出卸载与重计算之间的盈亏平衡点。

5. 使用和不使用 `torch.utils.checkpoint` 分别对真实 PyTorch Transformer 进行基准测试。通过 `torch.cuda.max_memory_allocated` 测量内存和单步耗时。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|----------------------|
| 梯度检查点 | “重新执行前向来节省内存” | 只保存片段输入；反向传播时重新计算中间值，以获得计算梯度所需的张量 |
| 激活重计算 | “就是检查点” | 高性能计算领域对同一种技术的称呼 |
| 片段大小（k） | “每个检查点包含多少层” | 中间值被丢弃并一同重新物化的层数 |
| 选择性检查点 | “Korthikanti 的技巧” | 只重新计算存储成本高的激活值（注意力 Softmax），保留成本低的激活值 |
| 完整检查点 | “朴素版本” | 重新计算每个片段中每一层的中间值 |
| 块级检查点 | “粗粒度” | 对完整 Transformer 块设置检查点；粒度最大 |
| FLOP 开销 | “计算税” | 每步额外 FLOP = 重计算 FLOP /（前向 + 反向 FLOP）；朴素方案为 33%，选择性方案为 5% |
| 激活卸载 | “发送到 CPU” | 在前向到反向之间把激活值移至 CPU 内存；重计算的替代方案 |
| sqrt-L 规则 | “经典最优值” | 对成本均匀的层，最佳检查点间距为 sqrt(L) 层 |
| 注意力 Softmax 体积 | “O(L²) 问题” | L² × 头数 × 批大小个浮点数；在长上下文中主导激活内存 |

## 延伸阅读

- [Chen 等，2016——“以次线性内存成本训练深度网络”](https://arxiv.org/abs/1604.06174)——正式提出梯度检查点的原始论文
- [Korthikanti 等，2022——“减少大型 Transformer 模型中的激活重计算”](https://arxiv.org/abs/2205.05198)——选择性激活重计算与形式化成本分析
- [Pudipeddi 等，2020——“使用新型执行算法以恒定内存训练大型神经网络”](https://arxiv.org/abs/2002.05645)——通过反向模式重新物化实现恒定内存的替代方案
- [Ren 等，2021——“ZeRO-Offload：让十亿级模型训练大众化”](https://arxiv.org/abs/2101.06840)——大规模激活卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html)——标准 API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html)——选择性、完整与块级模式
