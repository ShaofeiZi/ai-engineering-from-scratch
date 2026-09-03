---
name: skill-probability-reasoning
description: 为给定的 ML 问题选择合适的概率分布
version: 1.0.0
phase: 1
lesson: 6
tags: [probability, distributions, modeling]
---

# 概率分布选择

如何选择正确的分布,当模型数据,设计损失函数,或设置优先级.

## 决策检查清单

1. 结果是分离式 (类别,数量) 或连续式 (测量,分数)?
2. 结果是否有限 (例如, [0, 1]) 或无限?
3. 几种可能的结果?两个?
4. 数据是否对称或偏差?
5. 事件是独立的还是相互关联的?
6. 你正在模拟一个速度,一个数量,一个比例,或者一个测量?

## 分布决策树

```
Is the variable discrete?
  Yes --> Only 2 outcomes? --> Bernoulli (p)
     |    k outcomes, one trial? --> Categorical (p1...pk)
     |    k outcomes, n trials? --> Multinomial (n, p1...pk)
     |    Count of successes in n trials? --> Binomial (n, p)
     |    Count of events per interval? --> Poisson (lambda)
     |    Count of trials until first success? --> Geometric (p)
     |    Count of trials until r successes? --> Negative Binomial (r, p)
  No --> Symmetric, bell-shaped? --> Normal (mu, sigma)
     |   Positive values, right-skewed? --> Log-normal or Exponential
     |   Bounded in [0, 1]? --> Beta (alpha, beta)
     |   Positive values, flexible shape? --> Gamma (alpha, beta)
     |   Time between events? --> Exponential (lambda)
     |   Heavy tails needed? --> Student's t (nu) or Cauchy
     |   Multivariate, bell-shaped? --> Multivariate Normal
     |   On a simplex (sums to 1)? --> Dirichlet (alpha)
```

## 绘制现实世界 ML 分配的场景

| 情景 | 分配 | 参数 |
|---|---|---|
| 双式分类输出 | 贝尔诺利 | p = sigmoid(logit) |
| 多类分类输出 | 类别 | p = softmax(logits) |
| 语言模型中的代币预测 | 类别对词汇 | 软max 的 p |
| 像素强度 (正常化) | 测试的时间 | 取决于图像统计 |
| 文件中的字数 | 鱼类 | lambda = avg 字数 |
| 用户请求之间的时间 | 增量 | lambda = request 税率 |
| 测量错误 | 正常 | mu = 0根据数据的sigma |
| 重量初始化 | 常规或统一 | 凯明/萨维埃规则 |
| VAE 隐藏空间前 | 标准正常 | mu = 0, sigma = 1 |
| 比例的贝叶斯式先例 | 贝塔 | 根据信仰的"alpha,beta" |
| 类型权重的贝耶斯式先例 | 子 | 向量 |
| 逆转目标中的噪音 | 正常 | mu = 0据估计 |
| 异常强度回归 | 学生的 t | 自由度较低 |
| 时间/寿命建模 | 威布尔或加玛 | 形状和规模 |
| 文件的主题分类 (LDA) | 子 | alpha < 1 稀有的 |

## 当分发不当时

- 使用正常时,数据具有硬底界限 (例如价格,距离).正常将非零概率分配给负值.使用日记正常或Gamma.
- 通过使用波森的变量与平均差异时. mean = variance. 如果 variance > mean通过负二项来表示
- 采用伯诺利为多类问题.伯诺利是严格的二进制. k > 2.
- 假设在观测相关时独立.时间序列,空间数据和集成数据违反独立.使用自行降低或等级模型.

## 常见的错误

- 混 PDF 具有概率的值. PDF 概率来自集成 PDF 在一个间隔内.
- 忘记软max输出是类型概率,而不是独立的伯诺利概率.
- 信息的先例可以减少差异性,如果选择得很好,则不会偏见结果.
- 记录概率是概率的.记录探测器总是负 (或零).它们不总和为1.

## 快速参考:分布特性

| 分配 | 支持 | 平均值 | 变化 | 关键的财产 |
|---|---|---|---|---|
| Bernoulli(p) | {0, 1} | 其他 | p(1-p) | 简单的离散 |
| Binomial(n, p) | 没有什么. | np 其他 | np(1-p) | 伯诺利的数量 |
| Poisson(lam) | {0, 1, 2, ...} |  |  | Mean = variance |
| Normal(mu, s^2) | 其他类型 |  | 子 | 给定的平均/变量最大值 |
| Exponential(lam) | 其他类型 | 子 | 子 | 没有记忆 |
| Beta(a, b) | [0, 1] | a/(a+b) | 其他类型的产品 | 结合到二元 |
| Gamma(a, b) | 其他类型 | 其他 | 子 | 鱼的结合 |
| Dirichlet(alpha) | 简单 | alpha_i/sum | (见公式) | 连接到类别 |
