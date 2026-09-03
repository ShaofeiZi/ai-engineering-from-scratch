---
name: skill-statistical-testing
description: 选择合适的统计检验方法来比较 ML 模型并评估实验
version: 1.0.0
phase: 1
lesson: 15
tags: [statistics, hypothesis-testing, model-comparison]
---

# 统计测试 ML

如何选择正确的测试,比较模型,运行A/B实验或验证结果.

## 决策检查清单

1. 你比较的是什么?
2. 两个组,或者多个组?
3. 观察是否对 (相同的测试组,相同的折叠) 或独立?
4. 数据是否正常分布? n < 30 没有明显的正常,使用非参数.
5. 数据是连续的,排序的,还是断层的?
6. 你正在进行多少测试?

## 决策树

```text
Comparing means?
  Two groups?
    Paired (same data splits)? --> Paired t-test (or Wilcoxon signed-rank if non-normal)
    Independent? --> Welch's t-test (or Mann-Whitney U if non-normal)
  Multiple groups?
    Paired? --> Repeated measures ANOVA (or Friedman test)
    Independent? --> One-way ANOVA (or Kruskal-Wallis)

Comparing proportions?
  Two groups? --> Chi-squared test or Fisher's exact test (small n)
  Multiple groups? --> Chi-squared test

Comparing distributions?
  Is one distribution a reference? --> Kolmogorov-Smirnov test
  Are both empirical? --> Two-sample KS test

Measuring association?
  Both continuous, roughly normal? --> Pearson correlation
  Ordinal or non-normal? --> Spearman rank correlation
  Categorical x Categorical? --> Chi-squared test of independence

Running many tests?
  Apply Bonferroni correction: alpha_adjusted = alpha / number_of_tests
  Or use Holm-Bonferroni (less conservative, still controls family-wise error)
```

## 每次测试的使用时间

| 测试 | 数据类型 | 假设 | ML 使用情况 |
|---|---|---|---|
| 配对t测试 | 连续,配对 | 常见的差异 | 进行相同的k折分的2种模型的比较 |
| 威尔科森签名级别 | 连续/序列,配对 | 没有 (非参数) | 较量2个模型,小的 k (5-10折) |
| 威尔奇的T测试 | 连续,独立 | 基本上是正常的. | 两个独立的数据集的模型进行比较 |
| 曼·怀特尼 U | 连续/定制,独立 | 没有 | 进行延迟分布的比较 |
| ANOVA | Continuous, 3+ groups | 正常,等差 | 进行多个模型架构的比较 |
| Kruskal-Wallis | Continuous/ordinal, 3+ groups | 没有 | 进行多个模型的比较,非正常的指标 |
| 方形 | 类别计数 | 预期 count >= 5 | 类分布,混矩阵的比较 |
| 菲舍尔的确切 | 类别计数 | 小样品 | 罕见事件的比较 |
| KS 测试 | 持续 | 没有 | 检查预测是否遵循预期分布 |
| 启动带 CI | 任何统计数据 | 没有 | 对于 AUC, F1任何指标 |
| McNemar测试 | 双组 | 没有 | 同样测试组的两个分类器进行比较 |

## 模型比较配方

1. 定义指标和意义水平 (alpha = 0.05) 在进行实验之前.
2. 运行两种模型在相同的k倍交叉验证分区 (k = 5 或10).
3. 收集对成分数: (a_1, b_1), (a_2, b_2), ..., (a_k, b_k).
4. 计算差异: d_i = b_i - 没有.
5. 运行对对测试 (Wilcoxon为 k <= 10, paired t-test 对于 k > 10 或正常差异).
6. 报告:p值,平均差异,95%的保证间隔,效果大小 (Cohen的d).
7. 如果 p < alpha AND 效果大小是有意义的,差异是真实的,值得采取行动的.

## 常见的错误

- 如果两个模型都在同一测试折叠上进行评估,则必须使用一个对测试.独立测试会丢弃对测,从而失去统计能力.
- 报告 p < 0.05 没有效果大小. 统计上显著的0.1%的精度提高是不值得部署的. 总是计算科恩的d或原始平均差异.
- 测试组的模型与不同测试组的模型进行比较. MUST 两种模型的测试组都是一样的.
- 没有Bonferroni纠正的20次比较报告, alpha = 0.05假阳性是偶然的.
- 在99%多数类上,一个微不足道的分类器达到99%. F1精度回收 AUC它们是马修斯的相关系数.
- 通过通过跨验证折叠来处理独立样本.它们共享训练数据,这违反了独立假设. 修改的重新样本的t测试解释了这一点.

## 快速参考:效果尺寸解释

| 科恩的D | 解释 |
|---|---|
| 0.2 | 影响小 |
| 0.5 | 平均效果 |
| 0.8 | 大效果 |
| > 1.0 | 非常大的效果 |

| 报告什么 | 为什么? |
|---|---|
| 值 | 这种区别是真的吗? |
| 信心间隔 | 什么样的差异? |
| 效果大小 (Cohen d) | 这种区别有意义吗? |
| 样品大小 (n或k折叠) | 我们能信任结果吗? |
