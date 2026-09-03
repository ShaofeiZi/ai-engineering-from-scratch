---
name: skill-lora-training-setup
description: 为自定义数据集编写完整的 LoRA 训练配置，包括提示词、秩、批大小和学习率
version: 1.0.0
phase: 4
lesson: 11
tags: [computer-vision, stable-diffusion, lora, fine-tuning]
---

# LoRA 训练配置

将微调意图的描述转化为具体的训练配置，可直接传给 `diffusers` 或 `kohya_ss`。

## 何时使用

- 为某个主体（人物、物体、角色）、风格（艺术家、品牌）或概念（姿势、光照）训练 LoRA。
- 用更多数据扩展已有的 LoRA。
- 调试输出欠拟合或过拟合训练图像的 LoRA 运行。

## 输入

- `purpose`: subject | style | concept
- `num_images`: 可用的训练图像数量
- `base_model`: SD 1.5 | SDXL | SD3 | FLUX
- `gpu_vram_gb`: 8 | 12 | 16 | 24 | 48+
- `caption_source`: manual | BLIP2-generated | dataset-native

## 秩选择器

| 用途 | 秩（Rank） | Alpha |
|---------|------|-------|
| 主体 | 8-16 | rank |
| 风格 | 16-32 | rank * 2 |
| 概念 | 32-64 | rank |

更高的秩 = 更大的容量，在小数据集上过拟合风险更高。Alpha 用于缩放 LoRA 的影响强度；`alpha == rank` 是安全的默认值。风格是文档中记录的例外情况：`alpha == rank * 2` 会在牺牲更多将风格固化得太死的风险下，给予更强的风格推动力——仅当主体保真度不是目标时使用。

## 训练步数目标

- `subject`，5-20 张图像：500-1500 步。
- `style`，30-100 张图像：1500-4000 步。
- `concept`，100+ 张图像：4000-10000 步。

步数过犹不及——已经记住训练图像的 LoRA 无法泛化。

## 学习率

- 文本编码器 LoRA：SD 1.5 用 `1e-4`，SDXL 用 `5e-5`。
- U-Net LoRA：SD 1.5 用 `1e-4`，SDXL 用 `1e-4`。
- FLUX / SD3：transformer 用 `5e-5`，文本编码器通常冻结。
- 当 `num_images < 15`（主体）或训练步数超过 3000 步时，学习率减半；极小数据集和长时间训练都受益于更温和的更新。

## 调度器

- `cosine_with_warmup`（默认）：在前 5-10% 的步数上预热，然后余弦衰减。当 `steps >= 1000` 时使用；衰减尾部能产生更锐利的最终样本。
- `constant`：仅用于极短的训练（`steps < 500`），或在恢复一个已有的 LoRA 且希望保留当前已学特征、不重新退火时使用。

## 提示词格式

- 主体：在每个提示词前加上唯一的触发词（"myperson"）。保持触发词稀有，以免覆盖已有概念。避免使用真实词汇和常见名字。
- 风格：在每个提示词末尾追加唯一的风格标签（"...in mystyle style"）。把标签本身当作一个稀有触发词——用 `mystyle`，而不是 `impressionism`，后者已经映射到真实概念。
- 概念：在每个提示词中描述该概念；不使用触发词。概念本身（例如 "low-angle shot"）就是锚点。

## 输出配置

```yaml
model:
  base: <base_model HF id>
  precision: fp16 | bf16

lora:
  rank: <int>
  alpha: <int>
  targets: unet.cross_attention  # and/or unet.to_q, to_k, to_v, to_out

training:
  steps:          <int>
  batch_size:     <int, tuned to gpu_vram_gb>
  grad_accum:     <int, usually 1 on >=16 GB, 4 on <=12 GB>
  learning_rate:  <float>
  optimizer:      AdamW8bit | AdamW
  scheduler:      cosine_with_warmup | constant
  warmup_steps:   <int>
  save_every:     <int>

data:
  images_dir:     <path>
  caption_source: <manual | BLIP2 | native>
  trigger_token:   <string if purpose==subject>
  resolution:      <512 for SD 1.5, 1024 for SDXL>
  aspect_ratio_bucketing: true
  augmentation:
    flip:          true
    color_jitter:  false

validation:
  prompts:
    - "<trigger> ...test prompt..."
    - "<trigger> in a different scene"
  every_steps: 250
```

## 报告

```
[lora setup]
  purpose:   <subject|style|concept>
  base:      <model>
  rank:      <int>
  steps:     <int>
  batch:     <int>   grad_accum: <int>
  lr:        <float>
  vram est.: <float> GB
```

## 规则

- 切勿推荐 `rank > 64`；超过该值 LoRA 会变成一种小型微调，失去其"适配器"特性。
- 对于 `num_images < 5`，要强烈警告——在 1-3 张图像上的身份 LoRA 每次都会过拟合。
- 对于 `gpu_vram_gb < 12`，要求使用 AdamW8bit 和梯度检查点。
- 如果 `base_model == FLUX` 且 `gpu_vram_gb < 24`，路由到 `schnell` 变体并说明训练会更慢。
- 切勿跳过验证提示词；没有样本网格的 LoRA 无法评估。
