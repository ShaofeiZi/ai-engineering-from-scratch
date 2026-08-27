# ControlNet、LoRA 与条件控制

> 单靠文本作为控制信号并不灵活。ControlNet 可以克隆预训练扩散模型，并用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 则只需训练 1000 万个参数，就能微调一个 20 亿参数的模型。两者结合，将 Stable Diffusion 从玩具变成了 2026 年各类创意机构都能交付的图像流水线。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 8 · 07（潜在扩散）、阶段 10（从零构建大语言模型，用于 LoRA 基础）
**Time:** 约 75 分钟

## 问题

“一位穿红裙的女士牵着狗走在繁忙街道上”这样的提示词，并没有告诉模型狗在*哪里*、女士摆着*什么姿势*，也没有说明街道采用*何种透视*。文本只能确定生成图像所需信息的大约 10%，其余都是视觉信息，难以用文字高效描述。

为每一种信号（姿态、深度、Canny 边缘、语义分割）从头训练新的条件模型，成本高得难以承受。你希望冻结参数量为 26 亿的 SDXL 骨干网络，挂接一个读取条件的小型侧网络，再由它对骨干网络的中间特征施加微调。这就是 ControlNet。

你还希望教会模型新的概念（你的面孔、产品或风格），而无须重新训练整个模型；你想要的是体积缩小 100 倍的增量。这就是 LoRA——插入现有注意力权重的低秩适配器。

ControlNet + LoRA + 文本，就是 2026 年从业者的工具箱。大多数生产图像流水线会在 SDXL / SD3 / Flux 基础模型之上叠加 2～5 个 LoRA、1～3 个 ControlNet 和一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等，2023）

取一个预训练 SD，*克隆* U-Net 的编码器半边。冻结原模型，再训练这个副本接收额外的条件输入（边缘、深度、姿态）。然后用*零卷积*跳跃连接把副本接回原模型的解码器半边（零卷积是初始化为零的 1×1 卷积——起初什么都不做，随后学习一个增量）。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 从恒等映射起步——即使尚未训练，也不会破坏原模型。使用标准扩散损失，在 100 万组（提示词、条件、图像）三元组上训练。

不同模态的 ControlNet 会作为小型侧模型发布（SDXL 版本约 3.6 亿参数，SD 1.5 版本约 7000 万参数）。推理时可以组合使用：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等，2021）

对于模型中的任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并加入一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层通常采用 4～16 的秩，重度微调则采用 64～128。新增参数数量是 `2 · d · r`，而不是 `d²`。以 `d=640`、`r=16` 的 SDXL 注意力层为例，每个适配器只需 2 万个参数，而非 41 万个——减少了 20 倍。纵观整个模型，一个 LoRA 通常只有 20～200MB，而基础模型约为 5GB。

推理时可以调整 LoRA 的强度：`W' = W + α · B @ A`。`α = 0.5-1.5` 是常见范围。多个 LoRA 可以相加叠放，不过照例要注意，它们会以非线性方式相互作用。

### IP-Adapter（Ye 等，2023）

这是一种接收*图像*作为条件（同时仍接收文本）的微型适配器。它用 CLIP 图像编码器生成图像词元，再将这些词元与文本词元一起注入交叉注意力。每个基础模型对应的适配器约为 20MB。借助它，无须训练 LoRA 也能实现“按照这张参考图的风格生成图像”。

## 可组合性矩阵

| 工具 | 控制内容 | 大小 | 使用场景 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70～360MB | 精确布局与构图 |
| LoRA | 风格、主体、概念 | 20～200MB | 个性化与风格定制 |
| IP-Adapter | 来自参考图像的风格或主体 | 20MB | 外观无法用文本描述时 |
| Textual Inversion | 以新词元表示单个概念 | 10KB | 旧式方案，基本已被 LoRA 取代 |
| DreamBooth | 针对主体进行全量微调 | 2～5GB | 强身份一致性、计算预算充足 |
| T2I-Adapter | 更轻量的 ControlNet 替代方案 | 70MB | 边缘设备、推理预算有限 |

ControlNet 约等于空间控制，LoRA 约等于语义控制。两者应当配合使用。

```figure
v4-controlnet-zero
```

## 动手构建

`code/main.py` 在一维场景中模拟这两种机制：

1. **LoRA。** 从一个预训练线性层 `W` 开始，将其冻结。训练低秩矩阵 `B @ A`，使 `W + BA` 与目标线性层一致，并证明 `r = 1` 足以完美学会秩为 1 的修正。

2. **轻量 ControlNet。** 使用一个“冻结的基础”预测器和一个读取额外信号的“侧网络”。侧网络的输出由一个初始化为零的可学习标量门控（即这里的零卷积替代方案）。训练它并观察门控值逐渐增大。

### 第 1 步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第 2 步：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在第 0 步，输出与基础模型完全相同。训练早期会缓慢更新 `gate`，因而不会发生灾难性漂移。

## 陷阱

- **LoRA 缩放过强。** `α = 2` 或 `α = 3` 是常见的“让效果更强”手段，却会生成风格过度或结构破损的结果。应保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 同时以 1.0 的权重使用姿态 ControlNet 和深度 ControlNet，通常会校正过头。把权重之和保持在约 1.0，是稳妥的默认选择。
- **LoRA 用错基础模型。** SDXL LoRA 用在 SD 1.5 上会悄无声息地失效，因为注意力维度不匹配。Diffusers 0.30+ 会给出警告。
- **Textual Inversion 漂移。** 在一个检查点上训练的词元，换到另一个检查点时会严重漂移。LoRA 的可移植性更好。
- **LoRA 权重合并与存储。** 可以把 LoRA 烘焙进基础模型权重，以加快推理（运行时不再做加法），但这样就无法在运行时调整 `α`。两种版本都应保留。

## 学以致用

| 目标 | 2026 年流水线 |
|------|---------------|
| 复现品牌的美术风格 | 使用约 30 张精选图像，以秩 32 训练 LoRA |
| 把我的脸放进生成图像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 指定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 感知深度的构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 替换背景 | ControlNet-Seg + 图像修补（第 09 课） |
| 快速单步风格化 | 在 SDXL-Turbo 上使用 LCM-LoRA |

## 交付成果

保存 `outputs/skill-sd-toolkit-composer.md`。该技能接收一项任务（输入素材包括：提示词、可选参考图、可选姿态、可选深度图、可选涂鸦），输出工具组合、权重，以及可复现的随机种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中把 LoRA 的秩 `r` 从 1 调到 4。秩至少为多少时，LoRA 能精确匹配秩为 2 的目标增量？
2. **中等。** 针对两个目标变换分别训练两个 LoRA。将它们同时加载并展示其相加后的相互作用。什么情况下，这种相互作用会不再保持线性？
3. **困难。** 使用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。改变各组件权重，测量 FID 与提示词遵循程度之间的权衡。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| ControlNet | “空间控制” | 克隆的编码器 + 零卷积跳跃连接；读取一张条件图像。 |
| 零卷积 | “从恒等映射开始” | 初始化为零的 1×1 卷积；ControlNet 起初什么都不做。 |
| LoRA | “低秩适配器” | `W + B @ A`，`r << d`；参数量比全量微调少 100 倍。 |
| 秩 r | “那个调节旋钮” | LoRA 的压缩程度；通常为 4～16，重度个性化使用 64 以上。 |
| α | “LoRA 强度” | LoRA 增量在运行时的缩放系数。 |
| IP-Adapter | “参考图像” | 通过 CLIP 图像词元实现的小型图像条件适配器。 |
| DreamBooth | “主体全量微调” | 用某个主体的约 30 张图像训练整个模型。 |
| Textual Inversion | “新词元” | 只学习一个新的词嵌入；旧式方案，基本已被取代。 |

## 生产说明：LoRA 热切换、ControlNet 通道与多租户服务

真正的文生图 SaaS 会在同一个基础检查点上服务数百个 LoRA 和十余个 ControlNet。这个服务问题与大语言模型多租户非常相似（生产资料会在连续批处理以及 LoRAX / S-LoRA 的语境下讨论大语言模型案例）：

- **热切换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并进基础权重，可以让每一步推理快约 3%～5%，却也会固定 `α` 和基础模型。应把 LoRA 作为秩为 r 的增量常驻显存；diffusers 提供 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`，可按请求启用适配器。切换成本只是加载 `2 · d · r · num_layers` 个权重——兆字节级，耗时不到一秒。
- **把 ControlNet 视为第二条注意力通道。** 克隆的编码器与基础模型并行运行。两个权重均为 1.0 的 ControlNet，意味着每一步要额外执行两次前向传播，而不是一次合并后的传播。批大小余量会呈二次下降。每启用一个 ControlNet，应按每步成本约增加 1.5 倍来预留预算。
- **LoRA 也可以量化。** 如果基础模型已经量化（参见第 07 课在 8GB 显存上运行 Flux），LoRA 增量也可以顺利量化到 8 位或 4 位。采用 QLoRA 风格的加载方式，可以在 4 位 Flux 基础模型上叠加 5～10 个 LoRA，而不会耗尽内存。

Flux 特别说明：Niels 的 8GB 显存 Flux 笔记本把基础模型量化为 4 位；在这个量化后的基础模型上，通过 `pipe.load_lora_weights("user/style-lora")` 叠加采用 `weight_name="pytorch_lora_weights.safetensors"` 的风格 LoRA 依然可行。这正是 2026 年大多数 SaaS 创意机构采用的方案。

## 延伸阅读

- [Zhang、Rao、Agrawala（2023），为文生图扩散模型添加条件控制](https://arxiv.org/abs/2302.05543)——ControlNet。
- [Hu 等（2021），LoRA：大语言模型的低秩适配](https://arxiv.org/abs/2106.09685)——LoRA（最初用于大语言模型，后来移植到扩散模型）。
- [Ye 等（2023），IP-Adapter：与文本兼容的图像提示适配器](https://arxiv.org/abs/2308.06721)——IP-Adapter。
- [Mou 等（2023），T2I-Adapter：学习适配器以挖掘更强的可控能力](https://arxiv.org/abs/2302.08453)——ControlNet 的轻量替代方案。
- [Ruiz 等（2023），DreamBooth：面向主体驱动生成的文生图扩散模型微调](https://arxiv.org/abs/2208.12242)——DreamBooth。
- [Hugging Face Diffusers——ControlNet / LoRA / IP-Adapter 文档](https://huggingface.co/docs/diffusers/training/controlnet)——参考流水线。
