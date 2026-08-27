# KV 缓存、Flash Attention 与推理优化

> 训练可以并行，受 FLOP 限制；推理只能串行，受内存限制。瓶颈不同，优化手段也不同。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 02（自注意力）、阶段 7 · 05（完整 Transformer）、阶段 7 · 07（GPT）
**Time:** 约 75 分钟

## 问题

朴素自回归解码器需要执行 `O(N²)` 的工作才能生成 `N` 个词元：每一步都会重新计算整个前缀上的注意力。对于 4K 词元的回答，这意味着 1600 万次注意力运算，其中大部分都是重复的。每个前缀词元的隐藏状态计算一次后便已确定——新词元只需用自己的查询，与此前所有词元已经缓存的键和值进行计算。

除此之外，注意力本身还会搬运大量数据。标准注意力会物化 N×N 分数矩阵、N×d softmax 输出和 N×d 最终输出——需要对 HBM 进行太多次读写。当 N≥2K 时，注意力会先受到内存瓶颈限制，而不是 FLOP 限制。经典注意力内核只能发挥现代 GPU 四分之一到十分之一的能力。

Dao 等人提出的两项优化，把前沿推理从“缓慢”推向“快速”：

1. **KV 缓存。** 存储每个前缀词元的 K 与 V 向量。每个新词元只需用一个查询访问缓存的键。每个生成步骤的推理复杂度从 `O(N²)` 降至 `O(N)`。
2. **Flash Attention。** 对注意力计算分块，使完整的 N×N 矩阵永远不进入 HBM。softmax 与矩阵乘法全部在 SRAM 中完成。在 A100 上墙钟速度提高 2～4 倍；使用 FP8 时，在 H100 上提高 5～10 倍。

到 2026 年，二者已经无处不在。每套生产推理技术栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都以它们为前提，每个前沿模型也都会启用 Flash Attention。

## 概念

![KV 缓存增长与 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存的数学

每个解码器层、每个词元、每个头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于包含 32 层、32 个头、d_head=128 且使用 fp16 的 7B 模型：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 层、d_head=128、8 个 KV 头的 GQA）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这 10 GB 正是 Llama 3 70B 在 128K 上下文、批量大小为 1 时，单是 KV 缓存就要占据 40 GB A100 大部分显存的原因。

**GQA 的优势就在 KV 缓存。** 使用 64 个头的 MHA 需要 32 GB，MLA 则能进一步压缩。

拖动各个维度，观察缓存大小如何变化。增大序列长度或批量大小，就会看到它多快超出单张 GPU 的容量：

```figure
kv-cache-sizer
```

### Flash Attention——分块技巧

标准注意力：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次往返 HBM。在 H100 上，HBM 带宽为 3 TB/s，SRAM 为 30 TB/s。与把一切保留在片上相比，每次 HBM 往返都会造成 10 倍减速。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个图块只往返 HBM 一次。总内存占用从 `O(N²)` 降为 `O(N)`。反向传播会重新计算一部分前向传播结果，而不是将其存储下来，从而进一步节省内存。

**数值技巧。** 流式 softmax 在各个图块之间维护 `(max, sum)`，因此最终归一化完全精确。它不是近似——除 fp16 非结合性导致的差异外，Flash Attention 与标准注意力产生逐比特相同的输出。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速比 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 分块 SRAM 内核 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性、因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步、FP8 | H100 上 1.5～2×（约 740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 五阶段流水线、软件 exp2 | 推理优先（发布时仅支持前向传播） |

Flash 4 发布时仅支持前向传播。训练仍使用 Flash 3，Flash 4 对 GQA 和变长序列的支持预计在 2026 年中提供。

### 推测解码——另一项延迟优化

便宜模型提出 N 个词元，大模型并行验证全部 N 个。如果验证接受其中 k 个，就相当于只用一次大模型前向传播完成 k 次生成。在代码和普通文本上，典型的 k 为 3～5。

2026 年的默认方案：
- **EAGLE 2 / Medusa。** 与验证器共享隐藏状态的集成草稿头，在质量无损的情况下加速 2～3 倍。
- **使用草稿模型的推测解码。** 在消费级硬件上加速 2～4 倍。
- **前瞻解码。** 使用 Jacobi 迭代，不需要草稿模型。用途较窄，但无需额外模型。

### 连续批处理

经典批量推理必须等待最慢的序列结束，才能开始新一批任务；短回答提前结束后，GPU 就会被浪费。

连续批处理（最初在 Orca 中交付，如今已用于 vLLM、TensorRT-LLM、SGLang）会在旧请求结束后立即把新请求换入批次，无须等待整批排空。对典型聊天负载，可将吞吐量提高 5～10 倍。

### PagedAttention——把 KV 缓存当作虚拟内存

这是 vLLM 的核心功能。KV 缓存以 16 词元为单位分块分配，由页表把逻辑位置映射到物理块。它支持并行样本间共享 KV（束搜索、并行采样）、热切换前缀以进行提示缓存，并整理内存碎片。相比朴素的连续分配，吞吐量可提高 4 倍。

```figure
flash-attention-memory
```

## 动手构建

见 `code/main.py`。我们将实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个使用 KV 缓存、复杂度为 `O(N)` 的解码器。
3. 一个模拟 Flash Attention 流式最大值算法的分块 softmax。

### 第 1 步：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

方法很简单：在逐层、逐头列表中不断追加每个词元的 K、V 向量。

### 第 2 步：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

它与一次性计算 `softmax(qK) V` 得到逐比特相同的输出，但任意时刻的工作集只是一个 `tile × d_head` 块，而不是完整的 `N × d_head`。

### 第 3 步：在生成 100 个词元时比较朴素解码与缓存解码

计算注意力操作次数。朴素方案：`O(N²)` = 5050；缓存方案：`O(N)` = 100。代码会打印二者。

## 学以致用

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

生产环境使用 vLLM：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求前缀缓存是 2026 年的一项巨大收益——相同的系统提示、少样本示例或长上下文文档可以在不同调用间复用 KV。对于重复使用工具提示的智能体负载，前缀缓存通常能带来 5 倍吞吐量提升。

## 交付成果

见 `outputs/skill-inference-optimizer.md`。该技能会为新的推理部署选择注意力实现、KV 缓存策略、量化和推测解码方案。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素解码器与缓存解码器产生相同输出，并留意操作次数差异。
2. **中等。** 实现前缀缓存：给定提示 P 与多个补全结果，只对 P 执行一次前向传播来填充 KV 缓存，再为每个补全分支。测量相对于每次都重新编码 P 的加速比。
3. **困难。** 实现玩具版 PagedAttention：用固定的 16 词元块存储 KV 缓存，并维护空闲列表。当序列结束时，把它的块归还池中。模拟 1000 个长度不一的聊天补全，比较与连续分配相比的内存碎片。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| KV 缓存 | “让解码变快的技巧” | 保存每个前缀词元的 K 与 V；新查询直接关注缓存值，不再重新计算。 |
| HBM | “GPU 主内存” | 高带宽内存；H100 为 80 GB，B200 为 192 GB，带宽约 3 TB/s。 |
| SRAM | “片上内存” | 每个流式多处理器上的高速内存；H100 每个 SM 约 256 KB，带宽约 30 TB/s。 |
| Flash Attention | “分块注意力内核” | 无须在 HBM 中物化 N×N 矩阵即可计算注意力。 |
| 连续批处理 | “无等待批处理” | 已完成序列退出、新序列立即加入，无须排空整个批次。 |
| PagedAttention | “vLLM 的招牌功能” | 使用页表以定长块分配 KV 缓存，消除碎片。 |
| 前缀缓存 | “复用长提示” | 跨请求缓存共享前缀的 KV；能大幅降低智能体成本。 |
| 推测解码 | “草拟 + 验证” | 便宜的草稿模型提出词元，大模型在一次前向传播中验证 k 个。 |

## 延伸阅读

- [Dao 等（2022），FlashAttention：具有 IO 感知能力的快速、内存高效精确注意力](https://arxiv.org/abs/2205.14135)——Flash 1。
- [Dao（2023），FlashAttention-2：通过更好的并行性与工作划分实现更快注意力](https://arxiv.org/abs/2307.08691)——Flash 2。
- [Shah 等（2024），FlashAttention-3：利用异步与低精度实现快速准确注意力](https://arxiv.org/abs/2407.08608)——Flash 3。
- [FlashAttention-4 发布说明（Dao-AILab，2026）](https://github.com/Dao-AILab/flash-attention)——Blackwell 五阶段流水线与软件 exp2 技巧；请阅读代码库 README，了解本课提到的发布初期仅支持前向传播这一限制。
- [Kwon 等（2023），使用 PagedAttention 高效管理大语言模型服务内存](https://arxiv.org/abs/2309.06180)——vLLM 论文。
- [Leviathan 等（2023），通过推测解码实现 Transformer 快速推理](https://arxiv.org/abs/2211.17192)——推测解码。
- [Li 等（2024），EAGLE：推测采样需要重新思考特征不确定性](https://arxiv.org/abs/2401.15077)——本课提到的集成草稿方案 EAGLE-1/2。
- [Cai 等（2024），Medusa：使用多个解码头的简单大语言模型推理加速框架](https://arxiv.org/abs/2401.10774)——与 EAGLE 并列介绍的 Medusa 方法。
- [vLLM 文档——PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html)——深入讲解 16 词元分块与页表设计的权威资料。
