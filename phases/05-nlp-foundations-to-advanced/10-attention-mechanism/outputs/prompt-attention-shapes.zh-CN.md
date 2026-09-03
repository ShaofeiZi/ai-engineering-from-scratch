---
name: attention-shapes
description: 调试注意力实现中的形状错误。
phase: 5
lesson: 10
---

给定一个有问题的注意力实现，你需要找出其中的形状不匹配问题。输出：

1. 哪个矩阵的形状有误，指出该张量的名称。
2. 它的形状应该是什么，依据 `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)` 推导得出。
3. 一行修复方案：转置、重塑或投影。
4. 一个用于捕捉回归的测试。通常断言 `output.shape == (batch, T_dec, d_h)` 且 `weights.shape == (batch, T_dec, T_enc)` 且 `weights.sum(dim=-1)` 接近 1。

拒绝推荐会静默广播的修复方案。隐藏广播的 bug 日后会以静默的精度下降形式暴露出来。

对于 Bahdanau 的混淆，坚持解码器输入为 `s_{t-1}`（步前状态）。对于 Luong，则为 `s_t`（步后状态）。点积注意力中最常见的初次错误是查询/键维度不匹配——需要明确指出。
