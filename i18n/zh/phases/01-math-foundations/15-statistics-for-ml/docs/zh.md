# 机器学习统计学

> 统计学帮助你判断模型是真的有效，还是仅仅运气好。

**Type:** 构建
**Language:** Python
**Prerequisites:** 第 1 阶段，第 06 课（Probability and Distributions）和第 07 课（Bayes' Theorem）
**Time:** 约 2 小时

## 学习目标

- 从零计算描述性统计量、Pearson/Spearman 相关系数和协方差矩阵
- 执行假设检验（t 检验、卡方检验），并正确解释 p 值与置信区间
- 使用 bootstrap 重采样，在不做分布假设的情况下为任意指标构造置信区间
- 使用效应量区分统计显著性与实际显著性

## 问题

你训练了两个模型。模型 A 在测试集上得分 0.87，模型 B 得分 0.89，于是你部署了模型 B。三周后，生产指标反而比以前更差。发生了什么？

模型 B 并没有真正胜过模型 A。0.02 的差异只是噪声：可能测试集太小，也可能方差太高，或者两者兼有。你把披着改进外衣的随机波动发布到了生产环境。

这种情况经常发生：Kaggle 排行榜大幅洗牌，论文无法复现，A/B 测试只凭几百个样本就宣布赢家。根本原因总是相同——有人跳过了统计分析。

统计学为你提供区分信号与噪声的工具。它会告诉你差异是否真实、应该有多大把握，以及需要多少数据才能信任结果。每条机器学习流水线、每次模型比较和每项实验都需要统计学；没有它，你只是在猜测。

## 核心概念

### 描述性统计：概括数据

建模之前，必须先了解数据长什么样。描述性统计会用少数几个数字压缩数据集，概括其分布形态。

**集中趋势度量**回答“数据的中心在哪里？”

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

均值是平衡点，中位数是把数据一分为二的位置。当二者明显不同，说明分布存在偏斜。收入分布的均值远高于中位数，因为亿万富翁造成右偏；训练期间的损失分布则常常均值低于中位数，因为容易样本会造成左偏。

**离散程度度量**回答“数据分散得有多开？”

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数**会把排序后的数据划分为 100 个相等部分。第 25 百分位数（Q1）表示 25% 的数值低于该点；第 50 百分位数就是中位数；第 75 百分位数是 Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

机器学习会使用百分位数分析推理延迟、预测置信度分布和误差分布。一个平均误差很低、P99 误差却极高的模型，可能完全不适合安全关键应用。

**样本统计量与总体统计量。**根据样本计算方差时，应除以 (n-1)，而不是 n，这称为 Bessel 校正。它补偿了样本均值不等于真实总体均值这一事实。使用 n 作分母，会系统性低估真实方差；使用 (n-1) 则能得到无偏估计。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实践中，如果 n 很大，例如包含数千个样本，二者差异可以忽略；如果 n 很小，例如只有几十个样本，这个差异就很重要。

### 相关性：变量如何共同变化

相关性衡量两个变量之间线性关系的强度和方向。

**Pearson 相关系数**衡量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 相关假设关系是线性的，且两个变量大致服从正态分布。它对异常值很敏感，一个极端点就可能把 r 从 0.1 拉到 0.9。

**Spearman 秩相关**衡量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**如何选择：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：**相关性不代表因果关系。冰淇淋销量与溺水死亡人数相关，是因为二者都会在夏季增加。模型准确率与参数数量可能相关，但增加参数不会自动提高准确率，过拟合就是反例。

### 协方差矩阵

两个变量之间的协方差衡量它们如何共同变化：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于 d 个特征，协方差矩阵 C 是一个 d x d 矩阵，其中 C[i][j] = Cov(feature_i, feature_j)，对角元素 C[i][i] 是各特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与 PCA 的联系。**PCA 会对协方差矩阵进行特征分解。特征向量是主成分，也就是方差最大的方向；特征值表示每个主成分捕获多少方差。第 10 课已经介绍了这一点，现在你也能理解为何应该分解协方差矩阵：它编码了数据中所有成对线性关系。

**与相关性的联系。**相关矩阵就是标准化变量的协方差矩阵，每个变量都除以自身标准差。相关系数会将协方差归一化到 [-1, 1]。

### 假设检验

假设检验是一套在不确定条件下做决策的框架。你先提出一个主张，收集数据，再判断这些数据是否与该主张一致。

**基本设定：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p 值**是在 H0 为真时，观察到当前数据或更极端数据的概率。它不是 H0 为真的概率，这是统计学中最常见的误解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**置信区间**给出参数可能取值的范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间宽度反映估计精度。区间很宽意味着不确定性高，区间很窄意味着估计精确——但如果数据有偏，精确并不等同于准确。

### t 检验

t 检验用于比较均值，有多种形式。

**单样本 t 检验：**总体均值是否不同于某个假设值？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**双样本 t 检验（独立样本）：**两个组的均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**配对 t 检验：**测量结果成对出现时使用，例如两个模型在相同数据划分上的评估结果。

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在机器学习中，配对 t 检验很常见：让两个模型都在相同的 10 个交叉验证 fold 上运行，再逐对比较得分。

### 卡方检验

卡方检验用于检查观测频数是否符合期望频数，适合类别数据。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### 机器学习模型的 A/B 测试

机器学习中的 A/B 测试与网页 A/B 测试并不完全相同。模型比较有一些特有挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**具体流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性与实际显著性

一个结果可能在统计上显著，却在实践中毫无意义。只要数据足够多，即便微不足道的差异也会达到统计显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量**用于量化差异有多大，并且不受样本量影响：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

始终同时报告 p 值和效应量。p 值告诉你差异是否真实，效应量告诉你差异是否重要。

### 多重比较问题

检验许多假设时，总有一些会纯粹因为随机性而“显著”。如果以 alpha = 0.05 检验 20 个假设，即使没有任何真实效应，也预期会出现 1 个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni 校正：**用检验数量除以 alpha。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在机器学习中，当你比较多个指标、测试许多超参数配置，或在多个数据集上评估时，都必须考虑这一问题。

### Bootstrap 方法

Bootstrap 通过有放回地重采样数据，估计某个统计量的抽样分布，不需要对底层分布做任何假设。

**算法步骤：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap 置信区间（百分位法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Bootstrap 为什么对机器学习很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**使用 Bootstrap 比较模型：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这种方法比配对 t 检验更稳健，因为它不做任何分布假设。

### 参数检验与非参数检验

**参数检验**会假设数据服从特定分布，通常是正态分布：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验**不做分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用非参数检验：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

机器学习实验通常只有很小的 n，例如 5 或 10 个交叉验证 fold，因此 Wilcoxon 符号秩检验等非参数方法往往比 t 检验更合适。

### 中心极限定理的实践意义

中心极限定理指出，无论总体服从何种分布，随着 n 增大，样本均值的分布都会趋近正态分布。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**这为什么对机器学习很重要：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**中心极限定理不意味着：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### 机器学习论文中的常见统计错误

1. **在训练集上测试。**这必然造成过拟合。始终留出模型训练期间从未见过的数据。

2. **不报告置信区间。**只报告单个准确率数值、不说明不确定性，会让结果无法复现和验证。

3. **忽略多重比较。**测试 50 个配置后只报告最好的一个，却不进行校正，会抬高假阳性率。

4. **混淆统计显著性与实际显著性。**如果准确率只提高 0.01%，即使 p 值为 0.001，也不一定有实际意义。

5. **在类别不平衡数据上使用准确率。**如果数据集 99% 都是负类，那么 99% 准确率可能意味着模型什么都没学到。应使用 precision、recall、F1 或 AUC。

6. **挑选指标。**只报告模型胜出的指标。诚实的评估应报告所有相关指标。

7. **训练集与测试集之间发生信息泄漏。**例如先归一化再划分，或者用未来数据预测过去。

8. **测试集过小且不估计方差。**只用 100 个样本评估就声称提高了 2%，得到的只是噪声，不是信号。

9. **在数据不独立时假设独立。**例如同一患者的多张医学图像、同一文档中的多个句子；同一组内的观测彼此相关。

10. **P-hacking。**不断尝试不同检验、子集或排除标准，直到得到 p < 0.05。这样的结果只是搜索过程的产物。

## 动手构建

你将实现：

1. **从零实现描述性统计**（均值、中位数、众数、标准差、百分位数和 IQR）
2. **相关性函数**（Pearson、Spearman 和协方差矩阵）
3. **假设检验**（单样本 t 检验、双样本 t 检验、卡方检验）
4. **Bootstrap 置信区间**（适用于任意统计量，无需分布假设）
5. **A/B 测试模拟器**（生成数据、执行检验、检查第一类和第二类错误）
6. **统计显著性与实际显著性演示**（展示样本量很大时，一切都会变得“显著”）

全部从零实现，只使用 `math` 和 `random`，不使用 numpy 或 scipy。

```figure
f3-bootstrap-resample
```

## 关键术语

| 术语 | 定义 |
|---|---|
| Mean | 数值之和除以数量，对异常值敏感 |
| Median | 排序后的中间值，对异常值稳健 |
| Standard deviation | 方差的平方根，使用原始数据单位衡量离散程度 |
| Percentile | 有指定比例的数据低于该值 |
| IQR | 四分位距，Q3 减 Q1，表示中间 50% 数据的跨度 |
| Pearson correlation | 衡量两个变量之间的线性关联，范围为 [-1, 1] |
| Spearman correlation | 使用秩衡量单调关联 |
| Covariance matrix | 所有特征两两协方差组成的矩阵 |
| Null hypothesis | 默认的无效应或无差异假设 |
| p-value | 在零假设成立时，观察到当前数据或更极端数据的概率 |
| Confidence interval | 在给定置信水平下，参数可能取值的范围 |
| t-test | 使用 t 分布检验均值是否显著不同 |
| Chi-squared test | 检验观测频数是否显著偏离期望频数 |
| Effect size | 不受样本量影响的差异幅度，常用 Cohen's d |
| Bonferroni correction | 用检验数量除显著性阈值，以控制假阳性 |
| Bootstrap | 通过有放回重采样估计抽样分布 |
| Type I error | 假阳性，在 H0 为真时错误地拒绝 H0 |
| Type II error | 假阴性，在 H0 为假时未能拒绝 H0 |
| Statistical power | 正确拒绝错误 H0 的概率，等于 1 减第二类错误率 |
| Central limit theorem | 随样本量增加，样本均值会趋近正态分布 |
| Parametric test | 假设数据服从特定分布，通常是正态分布 |
| Non-parametric test | 不做分布假设，通常基于秩或符号进行检验 |
