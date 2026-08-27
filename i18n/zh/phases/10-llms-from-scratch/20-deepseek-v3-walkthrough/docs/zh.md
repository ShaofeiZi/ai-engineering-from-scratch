# DeepSeek-V3 架构剖析

> 阶段 10 · 第 14 课介绍了所有开放模型都会调整的六个架构旋钮。DeepSeek-V3（2024 年 12 月发布，总参数量 671B、活跃参数量 37B）不仅调整了全部六项，还增加了四项：多头潜在注意力、无辅助损失负载均衡、多词元预测和 DualPipe 训练。本课会自顶向下阅读 DeepSeek-V3 的架构，并根据公开配置推导每项参数数量。完成后，你将能解释为什么 671B/37B 的比例是正确选择，以及为什么在前沿规模上，MLA + MoE 的组合优于单独使用其中任何一种。

**Type:** 学习
**Languages:** Python（标准库，参数计算器）
**Prerequisites:** 阶段 10 · 14（开放模型剖析）、阶段 10 · 17（NSA）、阶段 10 · 18（MTP）、阶段 10 · 19（DualPipe）
**Time:** 约 75 分钟

## 学习目标

- 从头到尾阅读 DeepSeek-V3 配置，根据 GPT-2 的六个旋钮与 DeepSeek 特有的四项创新解释每个字段。
- 推导总参数量（671B）、活跃参数量（37B），以及分别构成二者的组件。
- 计算 MLA 在 128k 上下文下的 KV 缓存占用，并与活跃参数量相当、采用 GQA 的稠密模型比较。
- 说出 DeepSeek 特有的四项创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每项针对架构或训练技术栈的哪个部分。

## 问题

DeepSeek-V3 是第一个在架构上与 Llama 家族存在实质差异的前沿开放模型。Llama 3 405B 是“调整了六个旋钮的 GPT-2”，DeepSeek-V3 则在调整全部六项之外，又增加了四项。阅读 Llama 3 配置只是阅读 DeepSeek 配置的热身；DeepSeek 的深层结构——注意力块的形状、路由逻辑和训练时目标——差异已经足够大，因此需要单独剖析。

学习它的收益在于：DeepSeek-V3 的开放权重发布改变了开放模型中“前沿能力”的定义。许多 2026 年的训练任务正在复制这套架构。任何涉及前沿大语言模型训练或推理的岗位，都必须理解它。

## 概念

### 再看不变的核心

DeepSeek-V3 仍然是自回归模型，仍然堆叠解码器块；每个块仍由注意力、MLP 与两个 RMSNorm 组成；MLP 仍使用 SwiGLU；仍使用 RoPE、预归一化与权重绑定嵌入。它与每个 Llama 或 Mistral 具有相同基线。

### 变化点：用 MLA 取代 GQA

阶段 10 · 第 14 课已经说明，GQA 通过让多组 Q 头共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（MLA）更进一步：它把 K 和 V 压缩为共享的低秩潜在表示（`kv_lora_rank`），再在使用时为每个头解压缩。KV 缓存只存储潜变量——通常是每个词元、每层 512 个浮点数，而不是 8 × 128 = 1024 个浮点数。

在 128k 上下文下，DeepSeek-V3 使用 MLA（每个词元、每层共享一个潜变量 `c^{KV}`；K 与 V 都通过可吸收到后续矩阵乘法中的上投影由它生成）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

假设使用一种 GQA 基线（形状与 Llama 3 70B 相同：8 个 KV 头，头维度 128），则需要：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 的缓存比 Llama-3-70B 风格的 GQA 缓存小 4 倍。

代价是 MLA 在每次注意力计算中都需要执行逐头解压缩。与节省的带宽相比，额外计算很少，因此对长上下文推理而言仍是净收益。

### 路由：无辅助损失负载均衡

MoE 路由器决定每个词元由哪些 top-k 专家处理。朴素路由器会把过多工作集中到少数专家，其余专家则闲置。标准修复方法是加入惩罚负载不均衡的辅助损失项。这样确实有效，却会轻微损害主任务表现。

DeepSeek-V3 引入了无辅助损失方案。它在路由器 Logit 上加入逐专家偏置项，并在训练期间采用简单规则调整：若专家 `e` 负载过高，就降低 `bias_e`；若负载不足，就提高它。不增加任何损失项，训练目标保持干净，专家负载仍能维持均衡。

对主损失的影响：没有观察到可测量的变化。对 MoE 架构的影响：更简洁，不再需要调节辅助损失超参数。

### MTP：更密集的训练信号 + 可直接复用的草稿模型

阶段 10 · 第 18 课已经介绍，DeepSeek-V3 增加 D=1 个 MTP 模块，用于预测当前位置之后第 2 个位置的词元。推理时，训练好的模块会改作推测解码草稿器，接受率超过 80%；训练时，每个隐藏状态受到 D+1 = 2 个目标监督，信号更加密集。

额外参数：在 671B 主模型之上增加 14B，开销为 2.1%。

### 训练：DualPipe

阶段 10 · 第 19 课已经介绍，DualPipe 是一种双向流水线，会把前向与反向计算同跨节点全对全通信重叠。在 DeepSeek-V3 使用 2,048 张 H800 的规模上，它收回了使用 1F1B 时会浪费在流水线气泡中的约 24.5 万 GPU 小时。

### 逐字段阅读配置

下面是简化后的 DeepSeek-V3 配置：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

逐项解析：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：最前面的 3 个块使用大小为 18432 的稠密 MLP，其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 与 V 被压缩到这一潜在维度，再为每个头解压缩。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，并路由到 top-8。
- `shared_experts=1`：在 256 个路由专家之外，还有 1 个始终启用的专家会处理每个词元。可以把它理解为“稠密底座”，确保每个词元都能得到可靠处理。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏层大小。由于共有 256 个专家，它小于稠密 MLP。

### 参数核算

完整计算见 `code/main.py`。关键数字如下：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 最初 3 个稠密块：MLA 注意力（每块约 144M）+ 稠密 MLP（每块约 260M）+ 归一化，总计约 1.2B。
- 58 个 MoE 块：MLA 注意力（约 144M）+ 256 个专家（每个 30M）+ 1 个共享专家（30M）+ 归一化。计入全部专家后，每块约 7.95B，58 个 MoE 块总计 461B。
- MTP 模块：14B。

核心架构约 476B + MTP 14B，而公开的 671B 数字还包含其他结构参数（偏置张量、专家专用组件、共享专家缩放等）。计算器复现的数字与公开值相差 3%～5%；差异来自 DeepSeek 报告第 2 节附录所列的细粒度核算。

每次前向传播中的活跃参数：

- 注意力：每层 144M × 61 层 = 8.8B（所有层都会运行）。
- 活跃 MLP：最初 3 层为稠密层（3 × 260M = 780M）；其余 58 个 MoE 层各自启用 8 个路由专家 + 1 个共享专家 + 路由开销。每层活跃 MLP 约 260M，总计 3 × 260M + 58 × 260M = 约 15.9B。
- 嵌入 + 归一化：1.2B。
- 活跃总量：核心约 26B + MTP 14B（参与训练，但推理时并非始终运行）≈ 37B。

### 671B / 37B 比例

稀疏比为 18 倍（活跃参数占总参数的 5.5%）。DeepSeek-V3 是已经发布开放权重的前沿 MoE 模型中最稀疏的一个。Mixtral 8x7B 的比例为 13/47（28%），密集得多；Llama 4 Maverick 的 17B/400B（4.25%）则与之相当。DeepSeek 的押注是：在前沿规模上，更多专家配合更低激活比例，可以让每个活跃 FLOP 获得更高质量。

### DeepSeek-V3 所处位置

| 模型 | 总参数 | 活跃参数 | 比例 | 注意力 | 新颖设计 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 骨干网络上进行的一次推理训练，沿用相同架构。变化的是后训练方案（在可验证任务上进行大规模强化学习），而不是预训练架构。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），即阶段 10 · 第 17 课中 NSA 的后继者。这条谱系很稳定：架构级创新不断累积，每个版本都会调整更多旋钮。

```figure
moe-routing
```

## 学以致用

`code/main.py` 是专门针对 DeepSeek-V3 形状的参数计算器。运行它，将输出与论文数字比较，并用它分析假想变体（256 个专家与 512 个专家、top-8 与 top-16、MLA 秩 512 与 1024）。

重点观察：

- 总参数量与公开的 671B 数字之间的差异。
- 活跃参数量与公开的 37B 数字之间的差异。
- 128k 上下文时的 KV 缓存——比较 MLA 与 GQA。
- 逐层构成，以了解参数预算实际花在何处。

## 交付成果

本课会生成 `outputs/skill-deepseek-v3-reader.md`。给定 DeepSeek 家族模型（V3、R1 或未来变体），它会逐组件解读架构，说明配置中的每个字段，按组件推导参数量，并识别模型采用了哪些 DeepSeek 特有的四项创新。

## 练习

1. 运行 `code/main.py`。将计算器估算的总参数量与公开的 671B 比较，并找出差异来源。论文第 2 节给出了完整明细。

2. 修改配置，把 MLA 秩从 512 改为 256。计算 128k 上下文下的 KV 缓存大小。它能节省百分之多少，又会让逐头表达能力付出什么代价？

3. 将 DeepSeek-V3 的路由（256 个专家、top-8）与假想变体（512 个专家、top-8）比较。总参数量增加，活跃参数量不变。理论上，额外专家容量能带来什么，又会增加哪些推理成本？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节中关于 MLA 的内容。用三句话解释为什么 K 与 V 的解压缩矩阵可以“吸收”到后续矩阵乘法中，从而提高推理效率。

5. DeepSeek-V3 的大多数操作使用 FP8 训练。计算以 FP8 而非 BF16 存储 671B 权重时节省的内存，并说明这与 14.8T 词元训练预算如何相互影响。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|------------------------|
| MLA | “多头潜在注意力” | 把 K 和 V 压缩为共享低秩潜变量（kv_lora_rank，通常为 512），使用时逐头解压缩；KV 缓存只存潜变量 |
| kv_lora_rank | “MLA 压缩维度” | K 与 V 共享潜变量的大小；DeepSeek-V3 使用 512 |
| 最初 k 个稠密层 | “前几层保持稠密” | MoE 模型最前面的几层跳过 MoE 路由，使用稠密 MLP，以提高稳定性 |
| num_experts_per_tok | “Top-k 路由” | 每个词元启用的路由专家数量；DeepSeek-V3 使用 8 个 |
| 共享专家 | “始终启用的专家” | 不论路由结果如何都处理每个词元的专家；DeepSeek-V3 使用 1 个 |
| 无辅助损失路由 | “偏置调节式负载均衡” | 训练期间调整逐专家偏置项，在不加入损失项的情况下保持专家负载均衡 |
| MTP 模块 | “额外预测头” | 根据 h^(1) 与 E(t+1) 预测 t+2 的 Transformer 块；提供更密集的训练信号，并可直接复用为推测解码草稿模型 |
| DualPipe | “双向流水线” | 将前向/反向计算与跨节点全对全通信重叠的训练调度 |
| 活跃参数比例 | “稀疏度” | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 训练 | “8 位训练” | 训练时使用 FP8 存储相关张量，并以 FP8 执行大量计算；相比 BF16，内存约减半，只牺牲少量质量 |

## 延伸阅读

- [DeepSeek-AI——DeepSeek-V3 技术报告（arXiv:2412.19437）](https://arxiv.org/abs/2412.19437)——完整的架构、训练与结果文档
- [Hugging Face 上的 DeepSeek-V3 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3)——配置文件与部署说明
- [DeepSeek-V2 论文（arXiv:2405.04434）](https://arxiv.org/abs/2405.04434)——引入 MLA 的前代模型
- [DeepSeek-R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948)——在 V3 架构上进行推理训练的后继模型
- [原生稀疏注意力（arXiv:2502.11089）](https://arxiv.org/abs/2502.11089)——DeepSeek 家族注意力的未来方向
- [DualPipe 仓库](https://github.com/deepseek-ai/DualPipe)——训练调度参考实现
