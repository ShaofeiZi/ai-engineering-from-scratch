---
name: skill-dcgan-scaffold
description: 根据z_dim、image_size和num_channels编写完整的DCGAN脚手架,包括训练循环和样本保存器
version: 1.0.0
phase: 4
lesson: 9
tags: [computer-vision, gan, dcgan, scaffolding]
---

# DCGAN 脚手架

给定三个参数,生成一个可运行的 DCGAN 项目骨架,其架构尺寸针对目标图像分辨率进行了正确设置。

## 何时使用

- 在小数据集上开始新的生成式实验时。
- 用一个可运行的最小示例教授 DCGAN 基础知识时。
- 原型化条件 GAN 时(标签注入在同一脚手架中完成)。

## 输入

- `image_size`:32、64、128 中的一个(必须是 2 的幂)。
- `num_channels`:1(灰度)或 3(RGB)。
- `z_dim`:通常为 64 或 128。
- `with_spectral_norm`:yes | no;默认 yes。

## 架构尺寸

G 中转置卷积块的数量和 D 中跨步卷积块的数量取决于 `image_size`:

| image_size | G 块数 | D 块数 |
|------------|----------|----------|
| 32         | 4        | 4        |
| 64         | 5        | 5        |
| 128        | 6        | 6        |

每个额外的块将空间维度加倍(G)或减半(D)。特征数量从 32 开始,并按 `feat_base * 2^block_index` 进行缩放。

## 输出文件

- `model.py` — Generator + Discriminator 类
- `train.py` — 训练循环、损失、优化器设置
- `sample.py` — 样本网格保存器
- `config.json` — 超参数
- `README.md` — 10 行快速入门

## 报告

```
[scaffold]
  image_size:       <int>
  num_channels:     <int>
  z_dim:            <int>
  spectral_norm:    yes | no

[arch]
  G blocks:         <N>, channels: [list]
  D blocks:         <N>, channels: [list]
  G params (est):   <N>
  D params (est):   <N>

[training defaults]
  optimizer:   Adam(lr=2e-4, betas=(0.5, 0.999))
  batch_size:  64
  epochs:      50
  sample_every: 1 epoch

[files written]
  - model.py
  - train.py
  - sample.py
  - config.json
  - README.md
```

## 规则

- 始终在 G 的输出上使用 `nn.Tanh()`,并在训练期间将数据缩放到 [-1, 1]。
- 始终在 D 中使用 `LeakyReLU(0.2)`。
- 当 `with_spectral_norm == yes` 时,用 `spectral_norm()` 包装 D 中的每个卷积,并从 D 中移除 BatchNorm。在 G 中保留 BatchNorm。
- 绝不要为 image_size > 128 生成脚手架 — DCGAN 在此之上会变得不稳定;引导用户使用 StyleGAN 或扩散模型。
