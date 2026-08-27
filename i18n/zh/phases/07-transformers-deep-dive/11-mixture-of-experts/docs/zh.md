# 混合专家模型（MoE）

> 一个稠密 70B Transformer 会为每个词元激活全部参数。一个 671B MoE 每个词元只激活 37B 参数，却在所有基准上胜过前者。稀疏性是这个十年中最重要的规模化思想。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 7 · 05（完整 Transformer）、阶段 7 · 07（GPT）
**Time:** 约 45 分钟

## 问题

稠密 Transformer 的推理 FLOPs 等于其参数量（前向传播时再乘以 2）。放大稠密模型后，每个词元都必须承担全部计算成本。到 2024 年，前沿模型撞上了算力墙：想让模型明显变聪明，每个词元所需的 FLOPs 就要指数增长。

混合专家模型切断了这种关联。把每个 FFN 替换为 `E` 个独立专家，再用一个路由器为每个词元选择 `k` 个专家。总参数量 = `E × FFN_size`，每个词元的激活参数量 = `k × FFN_size`。2026 年的典型配置为 `E=256`、`k=8`。存储随 `E` 增长，计算则随 `k` 增长。

2026 年的前沿模型几乎全部采用 MoE：DeepSeek-V3（总参数 671B / 激活参数 37B）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，排名前十的开源模型全部是 MoE。

## 概念

![MoE 层：路由器为每个词元从 E 个专家中选择 k 个](../assets/moe.svg)

### 替换 FFN

稠密 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是独立的 FFN（通常为 SwiGLU），路由器则是单个线性层。每个词元会选择自己的 `k` 个专家，并获得这些专家输出的门控混合。

### 负载均衡问题

如果路由器把 90% 的词元都送给专家 3，其他专家就得不到训练。人们尝试过三种修复方法：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。增加一个与专家使用率方差成正比的惩罚项。有效，但会引入额外超参数和第二路梯度信号。
2. **专家容量 + 丢弃词元**（早期 Switch）。每个专家最多处理 `C × N/E` 个词元；溢出词元跳过这一层。会损害质量。
3. **无辅助损失的均衡**（DeepSeek-V3）。增加一个学习式逐专家偏置，用于移动路由器的 top-k 选择。偏置在训练损失之外更新，不会给主目标增加惩罚。这是 2024 年的重大突破。

DeepSeek-V3 的方法是：每个训练步骤结束后，检查各专家的使用量高于还是低于目标，再以 `±γ` 微调偏置。选择时使用 `scores + bias`，门控概率则仍使用未修改的原始 `scores`。这样就把路由与表达解耦开来。

### 共享专家

DeepSeek-V2/V3 还会把专家分成*共享专家*与*路由专家*。每个词元都会经过所有共享专家，路由专家则通过 top-k 选择。共享专家负责捕捉通用知识，路由专家负责专门化。V3 使用 1 个共享专家，以及 256 个路由专家中的前 8 个。

### 细粒度专家

经典 MoE（GShard、Switch）中，每个专家都与完整 FFN 一样宽。`E` 较小（8～64），`k` 也较小（1～2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）中，每个专家更窄（完整 FFN 大小的八分之一）。`E` 很大（256+），`k` 也较大（8+）。总参数量相同，但组合数量的增长快得多。`C(256, 8) = 400 trillion`，即每个词元有 400 万亿种可能的“专家组合”。质量提高，延迟保持不变。

### 成本构成

每个词元、每一层：

| 配置 | 每词元激活参数量 | 总参数量 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | 约 39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | 约 32B | 1T |

DeepSeek-V3 几乎在每项基准上都胜过 Llama 3 70B（稠密），而且**每个词元使用的有效 FLOPs 更少**。更多参数意味着更多知识，更多激活 FLOPs 意味着每词元计算更多。MoE 将二者解耦。

### 代价：内存

无论哪些专家被激活，所有专家都必须驻留在 GPU 上。一个 671B 模型以 fp16 存储权重时需要约 1.3 TB 显存。部署前沿 MoE 需要专家并行——把专家分片到不同 GPU，并通过网络路由词元。延迟主要由全对全通信决定，而不是矩阵乘法。

```figure
expert-routing
```

## 动手构建

见 `code/main.py`。其中仅使用标准库实现了一个紧凑的 MoE 层，包括：

- `n_experts=8` 个近似 SwiGLU 的专家（为便于说明，每个专家使用一个线性层）
- top-k=2 路由
- 经 softmax 归一化的门控权重
- 通过逐专家偏置实现无辅助损失均衡

### 第 1 步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，但不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置可以纠正负载失衡，却不会左右模型的预测。

### 第 2 步：让 100 个词元通过路由器

记录每个专家被激活的次数。没有偏置时，使用分布会倾斜。加入偏置更新循环（过度使用的专家调整 `-γ`，使用不足的专家调整 `+γ`）后，经过几轮迭代，使用率会趋向均匀分布。

### 第 3 步：比较参数量

打印一个 MoE 配置对应的“稠密等价规模”。采用类似 DeepSeek-V3 的配置：256 个路由专家 + 1 个共享专家，激活 8 个，d_model=7168。总参数量令人咋舌，激活参数量却只有稠密 Llama 3 70B 的约七分之一。

## 学以致用

通过 Hugging Face 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理中，vLLM 原生支持 MoE 路由，SGLang 拥有最快的专家并行路径。二者都会自动处理 top-k 选择与专家并行。

**适合选择 MoE 的情况：**
- 希望用更低的每词元推理成本获得前沿质量。
- 拥有足够显存和专家并行基础设施。
- 工作负载以生成词元为主（聊天、代码），而不是以上下文为主（长文档）。

**不适合选择 MoE 的情况：**
- 边缘端部署——无论激活 FLOPs 多少，都必须承担完整存储成本。
- 延迟敏感的单用户服务——专家路由会增加开销。
- 小模型（小于 7B）——只有激活参数达到一定计算阈值（约 6B）后，MoE 的质量优势才会出现。

## 交付成果

见 `outputs/skill-moe-configurator.md`。该技能会根据参数预算、训练词元与部署目标，为新的 MoE 选择 E、k 与共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`，观察无辅助损失偏置更新如何在 50 次迭代中均衡各专家的使用率。
2. **中等。** 把学习式路由器替换为基于哈希的路由器（确定性、无学习），比较质量与负载均衡。学习式路由器为何更好？
3. **困难。** 实现 GRPO 风格的“与 rollout 一致的路由”（DeepSeek-V3.2 技巧）：记录推理时激活了哪些专家，并在梯度计算中强制采用相同路由。在玩具策略梯度设置上测量效果。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 专家 | “众多 FFN 中的一个” | 独立的前馈网络；其参数专门服务于 FFN 计算中的一个稀疏子集。 |
| 路由器 | “门” | 为每个词元与每个专家打分的小型线性层，并执行 top-k 选择。 |
| Top-k 路由 | “每词元激活 k 个专家” | 每个词元的 FFN 计算恰好经过 k 个专家，并按门控权重组合。 |
| 辅助损失 | “负载均衡惩罚” | 用于惩罚专家使用率倾斜的额外损失项。 |
| 无辅助损失 | “DeepSeek-V3 的技巧” | 只通过路由器选择上的逐专家偏置进行均衡，不增加额外梯度。 |
| 共享专家 | “始终开启” | 每个词元都会经过的额外专家，用于捕捉通用知识。 |
| 专家并行 | “按专家分片” | 把不同专家分布到不同 GPU，并通过网络路由词元。 |
| 稀疏性 | “激活参数 < 总参数” | 比例为 `k × expert_size / (E × expert_size)`；DeepSeek-V3 为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer 等（2017），超大规模神经网络：稀疏门控混合专家层](https://arxiv.org/abs/1701.06538)——这一思想的起点。
- [Fedus、Zoph、Shazeer（2022），Switch Transformer：通过简单高效的稀疏性扩展到万亿参数模型](https://arxiv.org/abs/2101.03961)——经典 MoE 模型 Switch。
- [Jiang 等（2024），Mixtral of Experts](https://arxiv.org/abs/2401.04088)——Mixtral 8×7B。
- [DeepSeek-AI（2024），DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)——MLA + 无辅助损失 MoE + MTP。
- [Wang 等（2024），混合专家模型的无辅助损失负载均衡策略](https://arxiv.org/abs/2408.15664)——基于偏置的均衡论文。
- [Dai 等（2024），DeepSeekMoE：迈向混合专家语言模型中的极致专家化](https://arxiv.org/abs/2401.06066)——本课路由器使用的细粒度 + 共享专家划分。
- [Kim 等（2022），DeepSpeed-MoE：推进混合专家模型的推理与训练](https://arxiv.org/abs/2201.05596)——最初的共享专家论文。
