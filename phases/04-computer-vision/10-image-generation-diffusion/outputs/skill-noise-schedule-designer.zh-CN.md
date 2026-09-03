---
name: skill-noise-schedule-designer
description: 在给定 T 和目标损坏程度的情况下，生成线性、余弦或 S 形 beta 调度，并附带 SNR 图
version: 1.0.0
phase: 4
lesson: 10
tags: [computer-vision, diffusion, noise-schedule, training]
---

# 噪声调度设计器

beta 调度控制在每个扩散步骤中保留多少信号。糟糕的调度会在每一个下游决策处限制训练效率和样本质量。

## 何时使用

- 开始一个新的扩散训练运行并选择 T 和 beta 时。
- 调试产生模糊样本（调度过于激进）或无法学习结构（调度过于温和）的扩散模型时。
- 比较不同论文中报告的不同调度设计时。

## 输入

- `T`：时间步数，通常为 100-1000。
- `type`：linear | cosine | sigmoid。
- `target_alpha_bar_final`：在 t=T 时保留的信号比例，默认为 0.001（99.9% 被损坏）。
- 可选 `image_resolution` —— 较大的图像受益于损坏更慢的调度（余弦调度或偏移调度）。

## 调度公式

### 线性
```
beta_t = beta_start + (beta_end - beta_start) * (t - 1) / (T - 1)
```
默认值：beta_start=1e-4，beta_end=0.02（DDPM 论文）。

### 余弦（Nichol & Dhariwal, 2021）
```
alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi/2)
beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
```
s = 0.008。使信号保留更久；在低步数下表现更好。

### S 形
```
alpha_bar_t = 1 / (1 + exp(k * (t/T - 0.5)))
```
k = 6 到 12。良好的折中选择；某些 SDXL 变体使用它。

## 步骤

1. 按公式计算 betas。
2. 预计算 `alphas`、`alphas_cumprod`、`sqrt_alphas_cumprod`、`sqrt_one_minus_alphas_cumprod`。
3. 计算 SNR_t = alpha_bar_t / (1 - alpha_bar_t)；生成随时间变化的 SNR 摘要。
4. 验证 `alphas_cumprod[T-1]` 在 `target_alpha_bar_final` 的 10% 范围内；否则调整 beta_end（线性）、s（余弦）或 k（S 形）并重试。
5. 报告三个检查点：
   - `t=T*0.25` —— 早期损坏
   - `t=T*0.5` —— 中期
   - `t=T*0.75` —— 接近最终

## 报告

```
[schedule]
  type:   <name>
  T:      <int>
  beta_start: <float>   beta_end: <float>

[signal retention]
  t=0.25T:  alpha_bar=<X>  SNR=<X>
  t=0.5T:   alpha_bar=<X>  SNR=<X>
  t=0.75T:  alpha_bar=<X>  SNR=<X>
  t=T:      alpha_bar=<X>  SNR=<X>

[warnings]
  - <if alpha_bar collapses before 0.75T>
  - <if beta_end produces NaN in log-SNR>
```

## 规则

- 永远不要输出任何 `alpha_bar_t <= 0` 的调度；将低于 1e-5 的值截断并发出警告。
- 对于低步数采样（< 30 步），余弦调度是默认推荐。
- 当 `quality_target == research` 时，线性调度是默认选择 —— DDPM 基线使用线性调度报告。
- 当 `image_resolution > 256` 时，建议偏移调度（Chen, 2023），以在高分辨率下保留更多信号。
