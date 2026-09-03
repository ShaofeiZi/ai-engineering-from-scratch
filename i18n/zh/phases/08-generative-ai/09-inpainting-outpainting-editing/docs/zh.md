# 图像修补、扩图与图像编辑

> 文生图用于创造新事物，图像修补则用于修复已有内容。在生产环境中，70% 的可计费图像工作都是编辑——替换背景、移除标志、扩展画布、重新生成手部。图像修补正是扩散模型真正发挥价值的地方。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 8 · 07（潜在扩散）、阶段 8 · 08（ControlNet 与 LoRA）
**Time:** 约 75 分钟

## 问题

客户发来一张近乎完美的产品照片，但背景里有一块分散注意力的招牌。你希望擦掉招牌，同时让其他所有像素保持完全一致。不能从头运行文生图，因为结果会出现不同的颜色、光照和产品角度。你只想重新生成*蒙版覆盖的区域*，而且新内容必须与周围环境协调。

这就是图像修补。它还有几种变体：

- **图像修补。** 重新生成蒙版内部，保留外部像素。
- **扩图。** 重新生成蒙版外部（或画布以外）的区域，保留内部内容。
- **图像编辑。** 重新生成整张图像，但保持与原图的语义或结构一致性（SDEdit、InstructPix2Pix）。

2026 年的每套扩散流水线都提供图像修补模式：Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们遵循相同的原理。

## 概念

![图像修补：利用保留上下文的重新注入进行蒙版感知去噪](../../../../../../phases/08-generative-ai/09-inpainting-outpainting-editing/assets/inpainting.svg)

### 朴素方法（以及它为什么不对）

带着蒙版运行标准文生图。在每个采样步骤，把带噪潜变量的非蒙版区域替换为经过正向扩散的干净图像。这种方法能用……但效果很差。由于模型不知道蒙版区域中原本有什么，边界伪影会向内渗透。

### 正确的图像修补模型

训练一个经过修改的 U-Net，使其接收 9 个输入通道，而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外通道由 VAE 编码后的源图像副本以及单通道蒙版组成。训练时，随机遮盖图像中的区域，让模型只对蒙版区域去噪，同时把未遮盖区域作为干净的条件信号。推理时，模型能够“看到”蒙版周围的内容，从而生成连贯的补全结果。

SD-Inpaint、SDXL-Inpaint 和 Flux-Fill 都采用这种 9 通道输入（或类似形式）。Diffusers 对应的流水线是 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等，2022）——免训练编辑

先向源图像加噪至某个中间时刻 `t`，再在新提示词的条件下，从 `t` 沿反向链运行至 0。无须重新训练。起始 `t` 的选择决定了保真度与创作自由度之间的权衡：

- `t/T = 0.3` → 与原图几乎完全相同，只做轻微风格变化
- `t/T = 0.6` → 中等程度的编辑，保留粗略结构
- `t/T = 0.9` → 从接近纯噪声的状态开始生成，只保留极少的源图信息

### InstructPix2Pix（Brooks 等，2023）

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时，同时以输入图像和文本指令（“改成日落时分”“加一条龙”）作为条件。它使用两个 CFG 强度：图像强度与文本强度。

### RePaint（Lugmayr 等，2022）

保留标准的无条件扩散模型。在反向过程的每一步进行重采样——偶尔跳回噪声更多的状态，再重新生成。这能避免边界伪影，适合没有专用图像修补模型的情况。

```figure
inpaint-mask-reinject
```

## 动手构建

`code/main.py` 在五维数据上实现一个玩具一维图像修补方案。我们用五维混合数据训练 DDPM，每个样本由来自两个簇之一的 5 个浮点数组成。推理时，“遮盖”5 个维度中的 2 个，在每一步注入其余 3 个未遮盖维度经过正向加噪后的版本，并且只重新生成被遮盖的维度。

### 第 1 步：五维 DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 第 2 步：在全部 5 个维度上训练去噪器

使用标准 DDPM。网络接收五维带噪输入，并输出五维噪声预测。

### 第 3 步：推理时执行蒙版感知的反向过程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，在玩具一维数据上足以奏效。真正的图像修补会使用 9 通道输入，因为纹理连贯性更为重要。

### 第 4 步：扩图

扩图就是把蒙版反转的图像修补：遮盖新增的（原本不存在的）画布，用原图填充其余区域。训练目标完全相同。

## 陷阱

- **接缝。** 朴素方法会留下明显边界，因为梯度信息无法跨过蒙版传播。解决方法是把蒙版向外膨胀 8～16 个像素，或使用专用图像修补模型。
- **蒙版泄漏。** 如果条件图像的非蒙版区域质量较低或带有噪声，它会污染蒙版内部的生成结果。应先去噪，或稍作模糊。
- **CFG 会与蒙版大小相互影响。** 对小蒙版使用高 CFG，会产生过饱和的补丁。小范围编辑应降低 CFG。
- **SDEdit 的保真度断崖。** 从 `t/T = 0.5` 提高到 `t/T = 0.6`，就可能丢失主体身份。应扫描不同取值并保存检查点。
- **提示词不匹配。** 提示词应描述*整张*图像，而不只是新增内容。应写“一只猫坐在椅子上”，而不是“一只猫”。

## 学以致用

| 任务 | 流水线 |
|------|----------|
| 移除物体、小蒙版 | SD-Inpaint 或 Flux-Fill，使用标准提示词 |
| 替换天空 | SD-Inpaint + “日落时的蓝色天空” |
| 扩展画布 | SDXL 扩图模式（8px 羽化）或带扩图蒙版的 Flux-Fill |
| 重新生成手部/面部 | SD-Inpaint + 重新描述主体的提示词 + ControlNet-Openpose |
| 改变单个区域的风格 | 在蒙版区域以 `t/T=0.5` 运行 SDEdit |
| “改成日落时分” | InstructPix2Pix 或 Flux-Kontext |
| 替换背景 | SAM 蒙版 → SD-Inpaint |
| 超高保真度 | 最棘手的场景使用 Flux-Fill 或托管版 GPT-Image |

SAM（Meta 于 2023 年推出的 Segment Anything）+ 扩散修补，是 2026 年的背景移除流水线。SAM 2（2024）还可处理视频。

## 交付成果

保存 `outputs/skill-editing-pipeline.md`。该技能接收原始图像、编辑描述和可选蒙版（或 SAM 提示），并输出：蒙版生成方法、基础模型、CFG 强度（图像 + 文本）、SDEdit-t 或图像修补模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，把被遮盖维度的比例从 0.2 调到 0.8。遮盖比例达到多少时，图像修补质量（被遮盖维度中的残差）会与无条件生成相当？
2. **中等。** 实现 RePaint：每逢第 10 个反向步骤，就跳回 5 步（添加噪声），然后重新去噪。测量这种做法能否减少蒙版边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers，在 20 项面部重新生成任务上比较 SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill。分别评估姿态遵循程度与身份保留程度。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 图像修补 | “填洞” | 重新生成蒙版内部，保留外部像素。 |
| 扩图 | “扩展画布” | 重新生成画布以外的区域，保留内部内容。 |
| 9 通道 U-Net | “正规的图像修补模型” | 以 `noisy \| encoded-source \| mask` 作为输入的 U-Net。 |
| SDEdit | “带噪声强度的图生图” | 加噪至时刻 `t`，再使用新提示词去噪。 |
| InstructPix2Pix | “纯文本编辑” | 在（图像、指令、输出）三元组上微调的扩散模型。 |
| RePaint | “无须重新训练” | 在反向过程中定期重新加噪，以减少接缝。 |
| SAM | “Segment Anything” | 通过点击或方框生成蒙版；与图像修补搭配使用。 |
| Flux-Kontext | “结合上下文编辑” | 接收参考图像 + 编辑指令的 Flux 变体。 |

## 生产说明：编辑流水线对延迟敏感

编辑图像的用户期待低于 5 秒的往返时间。在 L4 上以 1024² 分辨率运行 30 步 SDXL-Inpaint 需要 3～4 秒，此外 SAM 生成蒙版约需 200 ms，VAE 编码与解码合计约需 500 ms。用生产系统的表述来说，这种场景受 TTFT 而非吞吐量制约——批大小为 1、并发较低，必须压缩每个阶段的耗时：

- **SAM-H 才是慢的那个。** SAM-H 在 1024² 下约需 200 ms；SAM-ViT-B 只需约 40 ms，质量损失很小。SAM 2（视频）会引入时间维度的额外开销，不要将它用于单图编辑。
- **能跳过编码就跳过。** `pipe.image_processor.preprocess(img)` 会把图像编码为潜变量。如果你还保留着上一次生成的潜变量（迭代式编辑界面通常如此），可通过 `latents=...` 直接传入，从而省去一次 VAE 编码。
- **蒙版膨胀也会影响吞吐量。** 小蒙版意味着 U-Net 的大部分前向传播都被浪费了（反正非蒙版像素会被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何都会运行完整 U-Net；只有正规的 9 通道图像修补变体才能利用蒙版计算。
- **Flux-Kontext 是 2025 年的答案。** 对 `(source_image, instruction)` 执行一次前向传播——无须单独的蒙版，也无须扫描 SDEdit 噪声强度。在 H100 上，约 1.5 秒就能交付一次编辑。这里的架构启示是：合并各个阶段。

## 延伸阅读

- [Lugmayr 等（2022），RePaint：使用去噪扩散概率模型进行图像修补](https://arxiv.org/abs/2201.09865)——免训练图像修补。
- [Meng 等（2022），SDEdit：使用随机微分方程进行引导式图像合成与编辑](https://arxiv.org/abs/2108.01073)——SDEdit。
- [Brooks、Holynski、Efros（2023），InstructPix2Pix](https://arxiv.org/abs/2211.09800)——文本指令编辑。
- [Kirillov 等（2023），Segment Anything](https://arxiv.org/abs/2304.02643)——SAM，蒙版来源。
- [Ravi 等（2024），SAM 2：在图像与视频中分割一切](https://arxiv.org/abs/2408.00714)——视频 SAM。
- [Hertz 等（2022），通过交叉注意力控制进行 Prompt-to-Prompt 图像编辑](https://arxiv.org/abs/2208.01626)——注意力层级编辑。
- [Black Forest Labs（2024），Flux.1-Fill 与 Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/)——2024 年工具。
