# 自动编码器与变分自动编码器（VAE）

> 普通自动编码器先压缩再重建。它会记忆，却不能生成。加入一个技巧——迫使编码看起来像高斯分布——就能得到采样器。正是 `z = μ + σ·ε` 这一重参数化技巧，让你在 2026 年使用的每个潜在扩散与流匹配图像模型都在输入端配有 VAE。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 3 · 02（反向传播）、阶段 3 · 07（CNN）、阶段 8 · 01（分类体系）
**Time:** 约 75 分钟

## 问题

把一个包含 784 个像素的 MNIST 数字压缩为 16 个数的编码，再将其重建。普通自动编码器可以轻松取得很低的重建 MSE，但编码空间却凹凸不平、杂乱无章。从编码空间中随机取一点并解码，只会得到噪声。它没有采样器，只是披着生成模型外衣的压缩模型。

你真正想要的是：（a）编码空间构成一个干净、平滑、可以采样的分布——例如各向同性高斯 `N(0, I)`；（b）解码任何样本都能产生合理数字；（c）编码器与解码器仍能有效压缩。三个目标，一套架构，一个损失函数。

Kingma 2013 年提出的 VAE 通过以下方式解决问题：训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`；使用 KL 惩罚把该分布拉向先验 `N(0, I)`；解码前先采样 `z`，样本来自 `q(z|x)`。推理时丢弃编码器，直接采样 `z ~ N(0, I)` 并解码。正是 KL 惩罚迫使编码空间具有结构。

到 2026 年，VAE 很少独立交付——在原始图像质量上，扩散模型已经超过它——但每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）都优先使用 VAE 作为编码器。学会 VAE，就学会了每条图像流水线中看不见的第一层。

## 概念

![自动编码器与 VAE：重参数化技巧](../../../../../../phases/08-generative-ai/02-autoencoders-vae/assets/vae.svg)

**自动编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间没有结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 与 `log σ²(x)`。二者定义 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样不可微。把样本改写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。这样一来，`z` 就是 `(μ, σ)` 与无参数噪声组成的确定性函数，梯度可以流经 `μ` 与 `σ`。

**损失。** 证据下界（ELBO），由两项组成：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项推动 `x̂` 接近 `x`，KL 项推动 `q(z|x)` 接近先验。二者需要权衡。较小的 β（<1）会产生更清晰的样本，但编码空间不那么接近高斯分布；较大的 β（>1）会让编码空间更规整，样本却更模糊。β-VAE（Higgins，2017）让这个旋钮广为人知，也开启了解耦表示研究。

**采样。** 推理时，从 `z ~ N(0, I)` 中采样，再通过解码器前向传播。只需一次前向传播，不像扩散那样迭代采样。

```figure
vae-latent-grid
```

## 动手构建

`code/main.py` 不依赖 NumPy 或 PyTorch，实现了一个微型 VAE。输入是从含两个分量的八维高斯混合分布中抽取的合成数据。编码器和解码器都是单隐藏层 MLP。我们会实现 tanh 激活、前向传播、损失与手写反向传播。它不适用于生产，仅用于教学。

### 第 1 步：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

输出 `log σ²` 而不是 `σ`，使网络输出不受约束（对 σ 使用 softplus 是个陷阱——当 σ ≈ 0 时梯度会消失）。

### 第 2 步：重参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第 3 步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个分布都是高斯分布，KL 可以精确地闭式计算，不要进行数值积分。2026 年仍有人交付使用蒙特卡洛估算 KL 的代码——无缘无故慢了 3 倍。

### 第 4 步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型，只需五行代码。

## 陷阱

- **后验坍塌。** KL 项过于强烈地推动 `q(z|x) → N(0, I)`，导致 `z` 不再携带关于 `x` 的信息。修复方法：β 退火（从 β=0 开始，逐渐升至 1）、自由比特，或对不活跃维度跳过 KL。
- **样本模糊。** 高斯解码器似然隐含 MSE 重建；对于 L2 而言，贝叶斯最优结果是均值，而多个合理数字的均值就是模糊数字。修复方法：使用离散解码器（VQ-VAE、NVAE），或只把 VAE 当作编码器，再在潜变量上叠加扩散模型（Stable Diffusion 正是如此）。
- **β 过早设得太大。** 会导致上述后验坍塌。应从 β≈0.01 开始并逐步提高。
- **潜变量维度过小。** MNIST 使用 16 维，ImageNet 256² 使用 256 维，ImageNet 1024² 使用 2048 维。Stable Diffusion 的 VAE 把 512×512×3 压缩为 64×64×4（空间面积缩小 32 倍，通道也缩小 32 倍）。

## 学以致用

2026 年的 VAE 技术栈：

| 场景 | 选择 |
|-----------|------|
| 用于扩散的图像潜变量编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频潜变量编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜变量 | Sora 的时空图块、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜变量（供 Transformer 建模） | VQ-VAE、RVQ（ResidualVQ） |
| 用于生成的连续潜变量 | 普通 VAE，再在该潜在空间中加入条件流/扩散模型 |

潜在扩散模型就是在编码器与解码器之间放入扩散模型的 VAE。VAE 负责粗粒度压缩，扩散模型承担主要工作。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen Transformer）也采用相同模式。

## 交付成果

保存 `outputs/skill-vae-trainer.md`。

该技能接收数据集特征、目标潜变量维度与下游用途（重建、采样或作为潜在扩散输入），并输出：架构选择（普通/β/VQ/RVQ）、β 调度、潜变量维度、解码器似然（高斯或分类），以及评估方案（重建 MSE、逐维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 将 `β` 在 `code/main.py` 中依次改为 `0.01`、`0.1`、`1.0`、`5.0`，记录最终重建 MSE 与 KL。哪一个 β 在你的合成数据上处于帕累托最优？
2. **中等。** 用伯努利似然（交叉熵损失）替换高斯解码器似然，在同一合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展成微型 VQ-VAE：用大小为 K=32 的码本中的最近邻查找替换连续 `z`。比较重建 MSE，并报告实际使用了多少个码本条目（码本坍塌确实存在）。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 自动编码器 | 编码—解码网络 | `x → z → x̂`，学习 MSE；不具备生成能力。 |
| VAE | 带采样器的自动编码器 | 编码器输出一个分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；`q = p(z\|x)` 时取等号。 |
| 重参数化 | `z = μ + σ·ε` | 把随机节点改写为确定性运算 + 纯噪声，使梯度可以通过采样传播。 |
| 先验 | `p(z)` | 潜变量的目标分布，通常为 `N(0, I)`。 |
| 后验坍塌 | “KL 项胜出” | 编码器忽略 `x` 并输出先验；解码器只能自行臆造。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。β 越高，表示越解耦，样本也越模糊。 |
| VQ-VAE | 离散潜变量 | 用码本中的最近向量替换连续 `z`，从而支持 Transformer 建模。 |

## 生产说明：扩散服务中的 VAE

在 Stable Diffusion / Flux / SD3 流水线中，每次请求都会调用 VAE 两次——进行图生图/局部重绘时先编码一次，最后再解码一次。在 1024² 分辨率下，解码器通常是整条流水线中激活内存峰值最大的单次操作，因为它要把 `128×128×16` 的潜变量上采样回 `1024×1024×3`。这带来两个实际结论：

- **切片或分块解码。** `diffusers` 提供 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块以少量接缝瑕疵为代价，使内存变为 `O(tile²)`，而不是 `O(H·W)`。在消费级 GPU 上生成 1024² 以上图像时必不可少。
- **解码器使用 bf16，最终缩放使用 fp32 数值。** SD 1.x VAE 最初以 fp32 发布，在 1024² 以上直接转换成 fp16 会*悄然产生 NaN*。SDXL 提供 `madebyollin/sdxl-vae-fp16-fix`——应始终优先使用修复版，或改用 bf16。

## 延伸阅读

- [Kingma 与 Welling（2013），自动编码变分贝叶斯](https://arxiv.org/abs/1312.6114)——VAE 论文。
- [Higgins 等（2017），β-VAE：通过受约束变分框架学习基础视觉概念](https://openreview.net/forum?id=Sy2fzU9gl)——解耦表示 β-VAE。
- [van den Oord 等（2017），神经离散表示学习](https://arxiv.org/abs/1711.00937)——VQ-VAE。
- [Vahdat 与 Kautz（2021），NVAE：深层层次化变分自动编码器](https://arxiv.org/abs/2007.03898)——顶尖图像 VAE。
- [Rombach 等（2022），使用潜在扩散模型合成高分辨率图像](https://arxiv.org/abs/2112.10752)——Stable Diffusion；VAE 作为编码器。
- [Défossez 等（2022），高保真神经音频压缩](https://arxiv.org/abs/2210.13438)——Encodec，音频 VAE 标准。
