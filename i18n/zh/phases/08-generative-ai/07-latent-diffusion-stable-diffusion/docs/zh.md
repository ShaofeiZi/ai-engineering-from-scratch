# 潜在扩散与 Stable Diffusion

> 在 512×512 图像上做像素空间扩散，计算量堪称暴行。Rombach 等人（2022）意识到，生成图像并不需要用到全部 78.6 万个维度——只需保留足以表达语义结构的信息，再用单独的解码器还原其余细节。把扩散过程放进 VAE 的潜在空间即可。这个想法就是 Stable Diffusion 的核心。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 8 · 02（VAE）、阶段 8 · 06（DDPM）、阶段 7 · 09（ViT）
**Time:** 约 75 分钟

## 问题

在 512² 分辨率的像素空间中做扩散，意味着 U-Net 要处理形状为 `[B, 3, 512, 512]` 的张量。对于参数量为 5 亿的 U-Net，每个采样步骤约需 100 GFLOPS；五十步就是每张图像 5 TFLOPS。若用十亿张图像训练，计算费用将高得离谱。

这些 FLOP 大多耗在让感知上并不重要的细节穿过网络，例如本可由有损 VAE 压缩掉的高频纹理。Rombach 的思路是：只训练一次 VAE（*第一阶段*），将其冻结，然后完全在 4 通道、64×64 的潜在空间中运行扩散（*第二阶段*）。仍然使用同样的 U-Net，像素数却只有 1/16，达到相近质量所需的 FLOP 约少 64 倍。

这就是 Stable Diffusion 的基本配方。SD 1.x / 2.x 使用参数量为 8.6 亿的 U-Net 处理 `64×64×4` 潜变量；SDXL 使用参数量为 26 亿的 U-Net 处理 `128×128×4` 潜变量；SD3 则以结合流匹配的扩散 Transformer（DiT）取代 U-Net。Flux.1-dev（Black Forest Labs，2024）采用参数量为 120 亿的 DiT-MMDiT。它们都建立在同一种两阶段底座之上。

## 概念

![潜在扩散：VAE 压缩 + 在潜在空间中扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **第一阶段——VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩方式是：在每个空间轴上进行 8 倍下采样，再调整通道数，使潜变量的总大小约为像素数的 1/16。损失 = 重建损失（L1 + LPIPS 感知损失）+ KL；KL 的权重较小，以免强迫 `z` 过度接近高斯分布，因为我们不需要从 `z` 中进行精确采样。训练通常还会加入对抗损失，使解码后的图像更加清晰。

2. **第二阶段——在 `z` 上扩散。** 把 `z = E(x_real)` 当作数据，训练 U-Net（或 DiT）对 `z_t` 去噪。推理时，通过扩散采样得到 `z_0`，再计算 `x = D(z_0)`。

**文本条件。** 还需要两个组件：一个冻结的文本编码器（SD 1.x 使用 CLIP-L，SD 2/XL 使用 CLIP-L+OpenCLIP-G，SD3 与 Flux 使用 T5-XXL）；以及交叉注意力注入机制——每个 U-Net 块都接收 `[Q = image features, K = V = text tokens]` 并将其混合。文本只能通过这些词元影响图像。

**损失函数与第 06 课完全相同。** 仍然使用同一种 DDPM / 流匹配噪声 MSE，只是替换了数据所在的空间。

## 架构变体

| 模型 | 年份 | 骨干网络 | 潜变量形状 | 文本编码器 | 参数量 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 个词元） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + 精炼器 | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏模型 | 128×128×4 | 相同 | 1～4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | 蒸馏版 MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B，1～4 步 |

总体趋势是：用 DiT（在潜在图块上运行的 Transformer）取代 U-Net；扩大文本编码器规模（T5 的提示词遵循能力优于 CLIP）；增加潜变量通道数（从 4 增至 16，为细节留出更多容量）。

```figure
noise-schedule
```

## 动手构建

`code/main.py` 在第 06 课的 DDPM 之上叠加了一个玩具一维“VAE”（为便于演示，编码器与解码器采用恒等映射；真正的 VAE 会使用卷积网络），并加入带无分类器引导的类别条件。它表明，无论对原始一维值还是编码后的值运行扩散，都可以使用同一种扩散损失——这正是关键洞见。

### 第 1 步：编码器/解码器

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真正的 VAE 拥有训练得到的权重。出于教学目的，这个线性映射已足以说明扩散可以在 `z` 上运行，而无须关心原始数据空间。

### 第 2 步：在 `z` 空间中扩散

使用与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样得到 `z_0` 后，再用 `D(z_0)` 解码。

### 第 3 步：无分类器引导

训练时，以 10% 的概率丢弃类别标签（替换为空词元）。推理时，同时计算 `ε_cond` 与 `ε_uncond`，然后执行：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` 表示不使用引导（多样性最高），`w = 3` 是默认值，`w = 7+` 则会出现饱和或过度锐化。

### 第 4 步：文本条件（仅讲概念，不写代码）

用冻结文本编码器的输出替代类别标签，再通过交叉注意力将文本嵌入送入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类别条件扩散模型与 Stable Diffusion 之间唯一实质性的差异。

## 陷阱

- **VAE 缩放不匹配。** SD 1.x 的 VAE 在编码后会应用一个缩放常数（`scaling_factor ≈ 0.18215`）。若忘记这一步，U-Net 将在方差严重错误的潜变量上训练。每个检查点都附带这个常数。
- **文本编码器悄悄用错。** SD3 需要支持不少于 128 个词元的 T5-XXL；回退为仅使用 CLIP 会损失信息。务必检查 `use_t5=True`，否则提示词保真度会骤降。
- **混用潜在空间。** SDXL、SD3 与 Flux 使用不同的 VAE。在 SDXL 潜变量上训练的 LoRA 无法用于 SD3。Hugging Face diffusers 0.30+ 会拒绝加载不匹配的检查点。
- **CFG 过高。** `w > 10` 会生成饱和、油腻的图像，并以牺牲多样性为代价过度贴合提示词。合适的范围是 `w = 3-7`。
- **负向提示词泄漏。** 空的负向提示词会成为空词元；填写了内容的负向提示词则成为 `ε_uncond`。两者并不相同，有些流水线却会悄悄默认使用空词元。

## 学以致用

2026 年的生产技术栈：

| 目标 | 推荐骨干网络 |
|--------|----------------------|
| 狭窄领域、有配对数据、从头训练模型 | 微调 SDXL（LoRA / 全量）——交付最快 |
| 开放领域文生图、开放权重 | Flux.1-dev（12B，Apache / 非商用）或 SD3.5-Large |
| 最快推理、开放权重 | Flux.1-schnell（1～4 步，Apache）或 SDXL-Lightning |
| 最佳提示词遵循能力、托管服务 | GPT-Image / DALL-E 3（依然如此）、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）——原生接收图像 + 文本 |
| 研究与基线 | SD 1.5——虽然古老，但研究充分 |

## 交付成果

保存 `outputs/skill-sd-prompter.md`。该技能接收文本提示词与目标风格，并输出：模型与检查点、CFG 强度、采样器、负向提示词、分辨率、可选的 ControlNet/IP-Adapter 组合，以及逐步 QA 检查清单。

## 练习

1. **简单。** 运行 `code/main.py`，依次使用引导强度 `w ∈ {0, 1, 3, 7, 15}`，记录各类别的样本均值。在多大的 `w` 下，类别均值会越过真实数据的均值并进一步分离？
2. **中等。** 将玩具线性编码器替换为带重建损失的 tanh-MLP 编码器/解码器对，在新的潜变量上重新训练扩散。样本质量是否发生变化？
3. **困难。** 使用 diffusers 配置真正的 Stable Diffusion 推理：加载 `sdxl-base`，以 CFG=7 运行 30 个 Euler 步骤并计时。然后切换到 `sdxl-turbo`，使用 4 步且 CFG=0。对同一主题比较生成质量——描述发生了什么变化，并解释原因。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 第一阶段 | “VAE” | 训练得到的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段 | “U-Net” | 在潜在空间上运行的扩散模型。 |
| CFG | “引导强度” | `(1+w)·ε_cond - w·ε_uncond`；用于调节条件信号的强度。 |
| 空词元 | “空提示词嵌入” | 用作 `ε_uncond` 的无条件嵌入。 |
| 交叉注意力 | “文本如何进入模型” | 每个 U-Net 块都以文本词元作为 K 和 V 进行注意力计算。 |
| DiT | “扩散 Transformer” | 用在潜在图块上运行的 Transformer 取代 U-Net；扩展性更好。 |
| MMDiT | “多模态 DiT” | SD3 的架构：文本流与图像流执行联合注意力。 |
| VAE 缩放因子 | “魔法数字” | 将潜变量除以约 5.4，使扩散在单位方差空间中运行。 |

## 生产说明：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是回答“我只有消费级 GPU，能把它交付上线吗？”的标准方案。诀窍是把生产推理资料中常见的三个旋钮用于扩散 DiT：

1. **错峰加载。** Flux 有三组无需同时驻留显存的网络：T5-XXL 文本编码器（fp32 下约 10 GB）、CLIP-L（很小）、12B MMDiT，以及 VAE。先编码提示词，*删除*编码器；再加载 DiT、执行去噪，随后*删除* DiT；最后加载 VAE 并解码。8GB 消费级 GPU 每次只能容纳一个阶段。
2. **通过 bitsandbytes 进行 4 位量化。** 在 T5 编码器和 DiT 上都设置 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。这样可将内存占用缩减 8 倍；根据 Aritra 的基准测试（已在笔记本中链接），对文生图而言，质量下降几乎无法察觉。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 会随着前向传播推进，自动在 CPU 与 GPU 之间换入换出模块。这会增加 10%～20% 的延迟，却能让流水线真正运行起来。

内存账可以这样算：量化后的 `10 GB T5 / 8 = 1.25 GB`，以及量化 DiT 的 `12 B params × 0.5 bytes = ~6 GB`，再加上激活值。用 stas00 的术语说，这是 TP=1 推理的极端形态——不使用模型并行，并将量化做到最大。生产环境会在 H100 上运行 TP=2 或 TP=4；对于单台开发者笔记本，这就是可行方案。

## 延伸阅读

- [Rombach 等（2022），使用潜在扩散模型合成高分辨率图像](https://arxiv.org/abs/2112.10752)——Stable Diffusion。
- [Podell 等（2023），SDXL：改进用于高分辨率图像合成的潜在扩散模型](https://arxiv.org/abs/2307.01952)——SDXL。
- [Peebles 与 Xie（2023），使用 Transformer 扩展扩散模型（DiT）](https://arxiv.org/abs/2212.09748)——DiT。
- [Esser 等（2024），扩展用于高分辨率图像合成的整流流 Transformer](https://arxiv.org/abs/2403.03206)——SD3、MMDiT。
- [Ho 与 Salimans（2022），无分类器扩散引导](https://arxiv.org/abs/2207.12598)——CFG。
- [Black Forest Labs（2024），Flux.1](https://blackforestlabs.ai/announcing-black-forest-labs/)——Flux.1 系列。
- [Hugging Face Diffusers 文档](https://huggingface.co/docs/diffusers/index)——上述各检查点的参考实现。
