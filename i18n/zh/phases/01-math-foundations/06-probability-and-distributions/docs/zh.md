# 概率与概率分布

> 概率是 AI 用来表达不确定性的语言。

**Type:** 学习
**Language:** Python
**Prerequisites:** 第 1 阶段，第 01–04 课
**Time:** 约 75 分钟

## 学习目标

- 从零实现 Bernoulli、categorical、Poisson、uniform 和 normal 分布的 PMF 与 PDF
- 计算期望值和方差，并用中心极限定理解释高斯分布为何无处不在
- 使用数值稳定技巧（减去最大 logit）构建 softmax 和 log-softmax 函数
- 根据 logits 计算交叉熵损失，并理解它与负对数似然的联系

## 问题

分类器输出 `[0.03, 0.91, 0.06]`；语言模型从 50,000 个候选词中选择下一个词；扩散模型通过从学习到的分布中采样来生成图像。这些都是概率在实际发挥作用。

模型做出的每次预测都是一个概率分布。每个损失函数都在衡量预测分布与真实分布之间的距离。每个训练步骤都在调整参数，使一个分布变得更像另一个分布。如果不懂概率，你将无法读懂机器学习论文、调试模型，也无法理解训练损失为何会变成 NaN。

## 核心概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果组成的集合，事件则是样本空间的一个子集。概率把事件映射到 0 和 1 之间的数值。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了整个概率体系：
1. 对任意事件 A，都有 P(A) >= 0
2. P(S) = 1（总会发生某个结果）
3. 当 A 和 B 不可能同时发生时，P(A or B) = P(A) + P(B)

其他一切概念，包括 Bayes 定理、期望和各种分布，都可以由这三条规则推导出来。

### 条件概率与独立性

P(A|B) 表示已知 B 发生时 A 发生的概率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

如果知道一个事件是否发生，并不会提供有关另一个事件的任何信息，那么两个事件相互独立：

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

多次抛硬币相互独立；不放回地抽取扑克牌则不独立。

### 概率质量函数与概率密度函数

离散随机变量具有概率质量函数（PMF）。每个结果都有一个可以直接读取的具体概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量具有概率密度函数（PDF）。单个点上的密度并不是概率；只有对某个区间上的密度积分，才能得到概率。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这一区别在机器学习中很重要。分类器输出是 PMF（离散选择），VAE 的潜在空间则使用 PDF（连续变量）。

### 常见分布

**Bernoulli 分布：**一次试验，两个结果。用于建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical 分布：**一次试验，k 个结果。用于建模多分类（softmax 输出）。

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform 分布：**所有结果出现的概率相同，常用于随机初始化。

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal（Gaussian）分布：**也就是钟形曲线，由均值（mu）和方差（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson 分布：**描述固定区间内稀有事件出现次数的分布，用于建模事件发生率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望值与方差

期望值是对各结果按概率加权后的平均值。

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差衡量数据围绕均值的离散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在机器学习中，期望值会以损失函数的形式出现，即数据分布上的平均损失。方差反映模型稳定性；梯度方差很高意味着训练噪声很大。

### 联合分布与边缘分布

联合分布 P(X, Y) 同时描述两个随机变量。

联合 PMF 示例（X = 天气，Y = 是否带伞）：

| | Y=0（没带伞） | Y=1（带伞） | 边缘概率 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（下雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘概率 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过对另一个变量的所有可能值求和得到：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表的行合计与列合计就是边缘概率。

### 正态分布为何无处不在

中心极限定理指出：许多独立随机变量的和（或平均值）会趋近正态分布，无论原始变量服从什么分布。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这解释了以下现象：
- 测量误差近似服从正态分布，因为它由许多微小且独立的误差源叠加而成
- 神经网络权重初始化会使用正态分布
- SGD 中的梯度噪声近似服从正态分布，因为它是许多样本梯度之和
- 在均值和方差给定时，正态分布是最大熵分布

### 对数概率

直接使用概率会造成数值问题。许多很小的概率相乘，很快就会下溢到零。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

对数概率可以解决这个问题，因为乘法会变成加法。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则如下：
- log(a * b) = log(a) + log(b)
- 对数概率始终 <= 0（因为 0 < P <= 1）
- 数值越负，发生的可能性越小
- 交叉熵损失就是正确类别概率的负对数

### Softmax 将分数转换为概率分布

神经网络输出原始分数（logits）。Softmax 会把它们转换成有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax 技巧：在取指数之前减去最大的 logit，以防数值上溢。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 把 softmax 与对数运算结合起来，以提高数值稳定性。PyTorch 在内部计算交叉熵损失时会使用它。

### 采样

采样是指从一个分布中随机抽取数值。在机器学习中：
- Dropout 会随机选择需要置零的神经元
- 数据增强会采样随机变换
- 语言模型会从预测分布中采样下一个 token
- 扩散模型会采样噪声并逐步去噪

从任意分布中采样，需要使用逆变换采样、拒绝采样或重参数化技巧（VAE 使用）等方法。

```figure
gaussian-pdf
```

## 动手构建

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 与 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：演示中心极限定理

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含全部可视化的完整实现位于 `code/probability.py`。

## 实际使用

借助 NumPy 和 SciPy，上述所有操作都可以用一行函数调用完成：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从零实现了这些运算，现在也知道库函数调用在底层做了什么。

## 练习

1. 为指数分布实现逆变换采样。采样 10,000 个值，将直方图与真实 PDF 比较以验证结果。

2. 为两枚加权骰子构建联合分布表，计算边缘分布，并检查两枚骰子是否独立。

3. 一个五分类分类器输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别索引为 3。计算其交叉熵损失，再使用 PyTorch 的 `nn.CrossEntropyLoss` 验证答案。

4. 编写一个函数，输入一列对数概率，返回最可能的序列、总对数概率和等价的原始概率。使用一句包含 50 个词、每个词概率均为 0.01 的句子进行测试。

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| Sample space | “所有可能性” | 一次实验全部可能结果组成的集合 S |
| PMF | “概率函数” | 给出每个离散结果准确概率的函数，所有概率之和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数；在一个区间上对它积分才能得到概率 |
| Conditional probability | “在某件事发生条件下的概率” | P(A\|B) = P(A and B) / P(B)，是 Bayesian 思维和 Bayes 定理的基础 |
| Independence | “二者互不影响” | P(A and B) = P(A) * P(B)；知道一个事件是否发生，不会提供另一个事件的信息 |
| Expected value | “平均值” | 对所有结果按概率加权后求和；损失函数就是一种期望值 |
| Variance | “分散程度” | 相对于均值的平方偏差的期望；方差高意味着估计噪声大、不稳定 |
| Normal distribution | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))，由于中心极限定理而无处不在 |
| Central Limit Theorem | “平均值会变成正态分布” | 无论原始分布如何，大量独立样本的均值都会趋近正态分布 |
| Joint distribution | “两个变量一起看” | P(X, Y) 描述 X 与 Y 每种结果组合的概率 |
| Marginal distribution | “把另一个变量求和消掉” | P(X) = sum_y P(X, Y)，从联合分布中恢复单个变量的分布 |
| Log probability | “概率的对数” | log P(x)，把乘法变成加法，防止长序列计算出现数值下溢 |
| Softmax | “把分数变成概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))，把实数 logits 映射为有效概率分布 |
| Cross-entropy | “损失函数” | -sum(p_true * log(p_predicted))，衡量两个分布的差异；越低越好 |
| Logits | “模型原始输出” | softmax 之前尚未归一化的分数，名称源自 logistic 函数 |
| Sampling | “抽取随机值” | 按照概率分布生成数值，也是模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown：中心极限定理究竟是什么？](https://www.youtube.com/watch?v=zeJD6dqJ5lo)——解释平均值为何趋近正态分布的可视化证明
- [Stanford CS229 概率复习资料](https://cs229.stanford.edu/section/cs229-prob.pdf)——涵盖本课全部内容及更多主题的精炼参考
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)——为什么数值稳定性很重要，以及如何实现它
