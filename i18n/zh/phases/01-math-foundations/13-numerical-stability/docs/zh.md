# 数值稳定性

> 浮点数是一种会泄漏细节的抽象。它会在训练过程中反噬你，而且往往毫无征兆。

**Type:** 构建
**Language:** Python
**Prerequisites:** 第 1 阶段，第 01–04 课
**Time:** 约 2 小时

## 学习目标

- 使用减去最大值的技巧，实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的上溢、下溢和灾难性消去
- 使用中心有限差分，将解析梯度与数值梯度进行验证
- 解释训练为何更偏好 bfloat16 而不是 float16，以及损失缩放如何防止梯度下溢

## 问题

模型训练了三个小时，损失突然变成 NaN。你加入打印语句，发现第 9,000 步的 logits 还正常，第 9,001 步却变成了 `inf`；到了第 9,002 步，每个梯度都是 `nan`，训练彻底失败。

还有一种情况：模型顺利完成训练，但准确率比论文低 2%。你逐项检查，架构、超参数和数据都一致。真正的问题是论文使用 float32，而你使用 float16，却没有正确缩放。累积的舍入误差悄无声息地吞掉了准确率。

又或者，你从零实现交叉熵损失。较小的 logits 没问题，logits 超过 100 后却返回 `inf`。原因是 softmax 上溢了：`exp(100)` 超过 float32 能表示的范围。所有机器学习框架都会用一个两行技巧处理它，但你此前不知道这个技巧存在。

数值稳定性不是纯理论问题。它决定了一次训练能够成功，还是会悄无声息地失败。你最终遇到的许多严重机器学习缺陷，都会追溯到浮点数。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准，以浮点格式存储实数。一个浮点数由三部分构成：符号位、指数和尾数（有效数字）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度，也就是能够保留多少位有效数字；指数决定范围，也就是能够表示多大或多小的数。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 大约提供 7 位十进制有效数字。它能区分 1.0000001 和 1.0000002，却无法区分 1.00000001 和 1.00000002。超过 7 位后，剩下的都只是舍入噪声。

float16 只有大约 3 位有效数字，能表示的最大数是 65,504。对于 logits、梯度和激活值经常超过这一范围的机器学习任务而言，这个上限小得令人不安。

bfloat16 是 Google 针对 float16 范围问题提出的格式。它和 float32 一样使用 8 位指数，因此范围相同，可达 3.4e38；但只有 7 位尾数，精度甚至低于 float16。训练神经网络时，范围通常比精度更重要，所以 bfloat16 往往更合适。

### 为什么 0.1 + 0.2 != 0.3

二进制浮点数无法精确表示 0.1。在二进制中，它是无限循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 会把它截断为 23 位尾数，实际存储值约为 0.100000001490116。类似地，0.2 实际存储为约 0.200000002980232，二者之和是 0.300000004470348，而不是精确的 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这会影响机器学习，因为：

1. `if loss < threshold` 这样的损失比较可能给出错误答案
2. 累积大量小数值时，例如在数千个步骤中累积梯度更新，结果会逐渐偏离真实总和
3. 如果使用 `==` 比较浮点数，校验和与可复现性测试会失败

解决方法是：绝不要用 `==` 比较浮点数，应使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性消去

两个非常接近的浮点数相减时，有效数字会互相抵消，原本位于末尾的舍入噪声会被提升为结果的主要数字。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

一次减法就产生了 19% 的相对误差。在机器学习中，以下情况都会触发这一问题：

- 使用 `E[x^2] - E[x]^2` 计算均值很大的数据的方差
- 对两个非常接近的对数概率做减法
- 用过小的 epsilon 计算有限差分梯度

解决方法是重排公式，避免让两个很大且非常接近的数相减。计算方差时使用 Welford 算法，或者先把数据中心化；处理对数概率时，始终在对数空间中计算。

### 上溢与下溢

计算结果大到无法表示时会发生上溢；结果太小、比最小正数更接近零时则会发生下溢。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 是机器学习中最常见的上溢来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 会在另一端出问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

机器学习中的 softmax、sigmoid 和概率计算都会用到 `exp()`；交叉熵、对数似然和 KL 散度都会用到 `log()`。如果没有正确技巧，`log(exp(x))` 这类组合就是雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上十分危险。如果某个 `x_i` 很大，`exp(x_i)` 会上溢；如果所有 `x_i` 都很负，每个 `exp(x_i)` 都会下溢到零，随后 `log(0)` 就会得到 `-inf`。

解决方法是在取指数之前减去最大值：

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

它之所以有效，是因为减去 `max(x)` 后，最大的指数项是 `exp(0) = 1`，不可能上溢；同时至少有一项为 1，因此总和至少为 1，而 `log(1) = 0`，也不会下溢成 `-inf`。

证明如下：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)`，即可消除上溢。

这一技巧在机器学习中随处可见：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- Gaussian 混合模型
- 变分推断

### Softmax 为什么必须减去最大值

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不使用该技巧时，logits [100, 101, 102] 会引发上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用该技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

所得概率完全相同，但计算过程安全。这不是性能优化，而是保证正确性的必要条件。

### NaN 与 Inf：检测和预防

`nan`（Not a Number）和 `inf`（无穷大）会像病毒一样沿计算传播。梯度更新中出现一个 `nan`，权重就会变成 `nan`，此后的每个输出也会变成 `nan`，训练会在一步内彻底失败。

`inf` 的常见来源：
- 对很大的正数执行 `exp()`
- 除以零：`1.0 / 0.0`
- `float32` 累加过程上溢

`nan` 的常见来源：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数执行 `sqrt()`
- 对负数执行 `log()`
- 任何包含已有 `nan` 的算术运算

检测方法：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 限制 `exp()` 的输入：`exp(clamp(x, -80, 80))`
2. 给分母添加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内添加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 使用梯度裁剪防止权重爆炸
6. 调试期间，在每次前向传播后检查 `nan`/`inf`

### 数值梯度检查

通过反向传播得到的解析梯度可能包含缺陷。数值梯度检查使用有限差分计算梯度，从而验证解析实现。

中心差分公式为：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

该公式具有 O(h^2) 精度，远好于只有 O(h) 精度的前向差分 `(f(x+h) - f(x)) / h`。

h 的选择很重要：太大，近似会不准确；太小，灾难性消去会破坏结果。通常使用 `h = 1e-5` 到 `1e-7`。

检查时，计算解析梯度与数值梯度的相对差异：

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验判断：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可以接受，通常正确
- relative_error > 1e-3：存在问题
- relative_error > 1：梯度完全错误

实现新网络层或损失函数时，始终应检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 的专用硬件（Tensor Core）执行 float16 矩阵乘法时，比 float32 快 2–8 倍。混合精度训练会利用这一特性：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题在于，梯度通常很小，可能只有 1e-8 或更小。Float16 会把低于约 6e-8 的值下溢为零，模型于是停止学习，因为所有梯度更新都成了零。

解决方案是损失缩放：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放会自动调整缩放因子。先从较大的值（65536）开始；如果梯度上溢为 `inf`，就将其减半；如果连续 N 步没有上溢，就将其加倍。

### bfloat16 与 float16：训练为何更偏好 bfloat16

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高（10 位尾数，而不是 7 位），但范围有限，最大值约为 65,504。bfloat16 精度较低，却拥有与 float32 相同的范围，最大值约为 3.4e38。

对于神经网络训练：

- 训练波动期间，激活值和 logits 经常超过 65,504。float16 会上溢，bfloat16 则能容纳这些值。
- float16 需要损失缩放；bfloat16 的范围能够覆盖梯度数量级，因此通常无需缩放。
- bfloat16 只是截断 float32：丢弃尾数最低的 16 位。转换十分简单，而且指数完全保留。

数值范围有界、精度更重要的推理任务更偏好 float16；数值范围更重要的训练则更偏好 bfloat16。TPU 和现代 NVIDIA GPU（A100、H100）原生支持 bfloat16，原因就在这里。

### 梯度裁剪

梯度经过很多层后呈指数增长，就会发生梯度爆炸。这在 RNN、深层网络和 Transformer 中很常见。一个过大的梯度，就可能在一步内破坏全部权重。

有两种裁剪方式：

**按值裁剪：**独立限制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

这种方式简单，但可能改变梯度向量的方向。

**按范数裁剪：**缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

它会保留梯度方向，这也是 `torch.nn.utils.clip_grad_norm_()` 的工作方式，通常应优先选择。

典型值包括：Transformer 使用 `max_norm=1.0`，强化学习使用 `max_norm=0.5`，较简单网络使用 `max_norm=5.0`。

梯度裁剪不是临时补丁，而是一种安全机制。没有它，一批异常样本就可能产生巨大梯度，毁掉数周训练成果。

### 归一化层也是数值稳定器

Batch normalization、layer normalization 和 RMS normalization 通常被介绍为帮助训练收敛的正则化手段，但它们也是数值稳定器。

没有归一化时，激活值可能随网络层数指数增长或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化会在每一层重新中心化并缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为 1e-5）可防止所有激活值相同时出现除零；可学习参数 `gamma` 和 `beta` 则允许网络恢复所需的任意尺度。

这能让数值在整个网络中保持安全范围，既避免前向传播上溢，也避免反向传播梯度爆炸。

### 常见机器学习数值缺陷

**缺陷：训练几个 epoch 后，损失变为 NaN。**
原因：logits 变得过大，softmax 上溢；或者学习率过高，权重发散。
修复：使用稳定 softmax（减去最大值）、降低学习率并添加梯度裁剪。

**缺陷：损失一直停在 log(num_classes)。**
原因：模型输出接近均匀概率，通常意味着梯度消失或模型根本没有学习。
修复：检查数据标签、验证损失函数，并检查是否出现失活 ReLU。

**缺陷：验证准确率比预期低 1%–3%。**
原因：使用混合精度却没有正确进行损失缩放，小梯度下溢后悄悄变成了零。
修复：启用动态损失缩放，或改用 bfloat16。

**缺陷：某些网络层的梯度范数为 0.0。**
原因：ReLU 神经元全部失活（输入均为负数），或者 float16 下溢。
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，并检查权重初始化。

**缺陷：模型在不同 GPU 上得到不同结果。**
原因：浮点累加顺序不确定。不同 GPU 硬件上的并行归约会以不同顺序求和，而浮点加法不满足结合律。
修复：接受 1e-6 量级的小差异，或者设置 `torch.use_deterministic_algorithms(True)` 并承担性能损失。

**缺陷：损失计算中的 `exp()` 返回 `inf`。**
原因：没有使用减去最大值技巧，直接把原始 logits 传给了 `exp()`。
修复：使用内部实现了 log-sum-exp 的 `torch.nn.functional.log_softmax()`。

**缺陷：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示小于 6e-8 的梯度或大于 65,504 的激活值。
修复：使用带损失缩放的混合精度训练（AMP），或者改用 bfloat16。

```figure
logsumexp-stability
```

## 动手构建

### 第 1 步：演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第 2 步：实现朴素与稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 第 3 步：实现稳定 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 第 4 步：实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第 5 步：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 实际使用

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

包含全部边界情况的完整实现位于 `code/numerical.py`。

## 交付成果

本课会产出：
- `code/numerical.py`，包含稳定 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和其他数值问题

这些稳定实现会在第 3 阶段构建训练循环时再次出现，也会在第 4 阶段实现注意力机制时使用。

## 练习

1. **灾难性消去。**在 float32 中，使用朴素公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的方差；再使用 Welford 在线算法计算。将两种结果与真实方差 0.6667 比较。

2. **寻找精度极限。**在 Python 中找出最小的正 float32 值 `x`，使 `1.0 + x == 1.0`；这就是机器 epsilon。验证它与 `numpy.finfo(numpy.float32).eps` 一致。

3. **Log-sum-exp 边界情况。**使用以下输入测试 `logsumexp_stable`：（a）所有值相等；（b）一个值远大于其他值；（c）所有值都非常负（-1000）。验证稳定版本能够在朴素版本失败时仍给出正确结果。

4. **检查神经网络层的梯度。**实现一个线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证一个 3x2 权重矩阵上的实现是否正确。

5. **损失缩放实验。**模拟 float16 训练：生成范围为 [1e-9, 1e-3] 的随机梯度，转换成 float16，并统计变为零的比例；然后应用损失缩放（乘以 1024），转换成 float16，再缩放回来，重新统计零值比例。

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| IEEE 754 | “浮点标准” | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准；所有现代 CPU 和 GPU 都实现它 |
| Machine epsilon | “精度极限” | 在特定浮点格式中，满足 1.0 + e != 1.0 的最小值 e；float32 约为 1.19e-7 |
| Catastrophic cancellation | “减法造成的精度损失” | 两个非常接近的浮点数相减时，有效数字消去，舍入噪声主导结果 |
| Overflow | “数字太大” | 结果超过最大可表示值并变为 inf；exp(89) 在 float32 中会上溢 |
| Underflow | “数字太小” | 结果比最小可表示正数更接近零并变为 0.0；exp(-104) 在 float32 中会下溢 |
| 对数求和指数技巧 | “先减去最大值” | 提取 exp(max(x)) 后计算 log(sum(exp(x)))，从而防止上溢与下溢；用于 softmax、交叉熵和对数概率计算 |
| Stable softmax | “不会爆炸的 softmax” | 取指数前减去 max(logits)，数值结果相同，但不可能上溢 |
| Gradient checking | “验证反向传播” | 把反向传播得到的解析梯度与有限差分得到的数值梯度比较，以捕获实现缺陷 |
| Mixed precision | “float16 前向、float32 反向” | 对速度敏感的运算使用低精度浮点数，对数值敏感的运算使用高精度浮点数；通常可加速 2–3 倍 |
| Loss scaling | “防止梯度下溢” | 反向传播前用较大常数乘以损失，使梯度落入 float16 可表示范围，再在更新权重前除以相同常数 |
| bfloat16 | “Brain 浮点数” | Google 的 16 位格式，拥有 8 位指数和 7 位尾数；范围与 float32 相同，但精度更低 |
| Gradient clipping | “限制梯度范数” | 缩放梯度向量，使其范数不超过阈值；防止梯度爆炸破坏权重 |
| NaN | “不是一个数” | 未定义运算（0/0、inf-inf、sqrt(-1)）产生的特殊浮点值，会传播到后续所有算术运算 |
| Inf | “无穷大” | 上溢或除零产生的特殊浮点值，与其他值组合时可能产生 NaN（inf - inf、inf * 0） |
| Numerical gradient | “暴力求导” | 计算 f(x+h) 与 f(x-h)，再除以 2h 来近似导数；速度慢，但适合验证 |

## 延伸阅读

- [每位计算机科学家都应该了解的浮点运算知识（Goldberg，1991）](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)——权威、密集但完整的参考资料
- [混合精度训练（Micikevicius 等，2018）](https://arxiv.org/abs/1710.03740)——首次为 float16 训练引入损失缩放的 NVIDIA 论文
- [AMP：自动混合精度（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html)——PyTorch 混合精度实践指南
- [bfloat16 格式（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16)——Google 为 TPU 选择该格式的原因
- [Kahan 求和（Wikipedia）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)——降低浮点求和舍入误差的算法
