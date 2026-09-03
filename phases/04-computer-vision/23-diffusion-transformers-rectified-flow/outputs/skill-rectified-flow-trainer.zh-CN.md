---
name: skill-rectified-flow-trainer
description: 编写一个完整的整流流（rectified flow）训练循环，包含 AdaLN DiT 与 Euler 采样
version: 1.0.0
phase: 4
lesson: 23
tags: [diffusion, rectified-flow, DiT, training]
---

# 整流流训练器

提供一个干净、最小化的训练循环，能够基于任意图像张量数据集，成功训练一个采用整流流的小型 DiT。

## 何时使用

- 在小规模上复现 SD3 / FLUX 的训练目标。
- 在相同数据上对整流流与 DDPM 进行基准对比。
- 为非标准领域（医学、卫星）构建自定义整流流模型。

## 输入

- `model`：一个 `nn.Module`，接受 `(x, t)` 并返回预测的速度。
- `dataset`：模型所属领域中干净图像的可迭代对象。
- `optimizer`：AdamW，参数为 `lr=1e-4`、`weight_decay=0.01`、`betas=(0.9, 0.99)`。
- `scheduler`：带预热的余弦调度，默认预热 1000 步。

## 训练步骤

```python
def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)                     # uniform in [0, 1]
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon
    target_v = epsilon - x0                              # velocity target
    pred_v = model(x_t, t)
    loss = F.mse_loss(pred_v, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## 采样（Euler）

```python
@torch.no_grad()
def sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

## 提示

- 使用 `torch.rand` 生成均匀分布的 `t`；采用 logit-normal 或 SD3 风格的加权 `t` 采样略有帮助，但入门时并非必需。
- 模型权重的 EMA 是标准做法；维护衰减系数为 0.9999 的 `ema_model`。
- 针对条件模型的 classifier-free guidance：在训练时以 10% 的概率将条件替换为空/null 嵌入；在推理时以约 3-5 的 `v_uncond + w * (v_cond - v_uncond)` 混合 `w`。
- 对于 LLM 风格的训练（FLUX、SD3），整个循环在 VAE 的潜在空间中运行；上文中的干净 `x0` 实际上是 `VAE.encode(image)`。
- 在 32x32 玩具数据集上的典型收敛：2000-5000 步。在真实潜在 SD3 训练上：数十万步。

## 报告

```
[rectified flow training]
  steps:        <int>
  final loss:   <float>
  ema decay:    <float>
  vae?:         yes | no
  cfg dropout:  <fraction>

[sampling]
  default steps: 20
  schnell / turbo target: 4
  full quality reference: 50+ (for comparison only)
```

## 规则

- 切勿在 RGB `uint8` 数据上以图像空间速度目标训练整流流；应先归一化到零均值、单位方差。
- 始终按时间步分桶记录训练损失；如果早期时间步（接近 0）的损失高于晚期时间步（接近 1），则速度参数化很可能接线有误。
- 不要在同一训练循环中将整流流速度目标与 DDPM 噪声目标混用；只选其一。
- 在 Ampere 及以上 GPU 上使用 bfloat16 进行训练；在整流流中，float16 有时因速度量级较大而产生 NaN 梯度。
