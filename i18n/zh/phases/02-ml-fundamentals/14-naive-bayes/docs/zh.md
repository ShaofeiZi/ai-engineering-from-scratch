# 朴素贝叶斯

> “朴素”的假设明明不成立，算法却依然有效。这正是它的精妙之处。

**Type:** 构建
**Language:** Python
**Prerequisites:** 阶段 2，第 01–07 课（分类、贝叶斯定理）
**Time:** 约 75 分钟

## 学习目标

- 从零实现带拉普拉斯平滑的多项式朴素贝叶斯，用于文本分类
- 解释朴素独立性假设在数学上为何不成立，以及它在实践中为何仍能给出正确的类别排序
- 比较多项式、伯努利和高斯朴素贝叶斯，并针对给定的特征类型选择合适变体
- 在高维稀疏数据上比较朴素贝叶斯与逻辑回归，并解释其中体现的偏差—方差权衡

## 问题

你需要对文本进行分类：判断邮件是否为垃圾邮件，判断客户评价是正面还是负面，或者把支持工单分到不同类别。数据中包含数千个特征，每个单词对应一个特征，而训练数据却很有限。

大多数分类器在这里都会陷入困境。逻辑回归需要足够多的样本，才能可靠估计数千个权重。决策树每次只能选择一个单词进行分裂，很容易严重过拟合。在 10,000 维空间中，KNN 也失去了意义，因为每个点到其他点的距离几乎都一样。

朴素贝叶斯可以应对这种情况。它作出了一个数学上错误的假设：给定类别后，每个特征都与其他特征相互独立。然而在文本分类中，它仍然能够胜过许多看似更“聪明”的模型，训练集较小时尤其如此。它只需单次遍历数据即可完成训练，能够扩展到数百万个特征，还能输出概率估计，尽管受独立性假设影响，这些概率往往没有得到良好校准。

理解错误假设为何能够带来良好预测，会揭示机器学习的一条基本规律：最好的模型并不是理论上最正确的模型，而是针对你的数据拥有最佳偏差—方差权衡的模型。

## 核心概念

### 贝叶斯定理（快速回顾）

贝叶斯定理可以反转条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们希望求出 `P(class | features)`，也就是已知文档中的单词后，该文档属于某个类别的概率。可以通过以下三项计算：
- `P(features | class)`：在该类别文档中看到这些单词的似然
- `P(class)`：该类别的先验概率，例如垃圾邮件总体上有多常见
- `P(features)`：证据项；它对所有类别都相同，因此比较类别时可以忽略

`P(class | features)` 最高的类别获胜。

### 朴素独立性假设

精确计算 `P(features | class)`，需要估计所有特征共同出现的联合概率。如果词表中有 10,000 个单词，就必须估计覆盖 2^10,000 种可能组合的分布，这是不可能做到的。

朴素假设认为：给定类别后，每个特征都条件独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

这样一来，就不必估计一个无法处理的联合分布，而只需估计 n 个简单的单特征分布。每个分布只需要一次计数。

这个假设显然是错误的。在任何文档中，“machine”和“learning”都不可能彼此独立。不过，分类器并不需要得到准确的概率估计，它只需要给出正确的排序，也就是找到概率最高的类别。独立性假设会引入系统性误差，但这些误差对所有类别的影响往往相似，所以类别排序仍然可能正确。

### 它为何依然有效

主要有三个原因：

1. **排序比校准更重要。** 分类只要求排在第一位的类别正确。即使模型给出 P(spam) = 0.99999，而真实概率其实是 0.7，它仍然正确选择了垃圾邮件。我们不需要概率本身完全准确，只需要选对胜者。

2. **高偏差、低方差。** 独立性假设是一种很强的先验，会对模型施加严格约束，从而防止过拟合。训练数据有限时，一个略有偏差但表现稳定的模型，会胜过一个理论正确却极不稳定的模型。这正是偏差—方差权衡的实际体现。

3. **特征冗余会相互抵消。** 相关特征提供了重复证据。分类器的确会把证据重复计算，但也会为正确类别重复计算。如果“machine”和“learning”总是一起出现，它们都会为“tech”类别提供证据。朴素贝叶斯把证据算了两次，但这两次都支持正确类别。

还有第四个更实际的原因：朴素贝叶斯速度极快。训练只需单次遍历数据并统计频率，预测则只是一次矩阵乘法。它可以在数秒内用一百万篇文档完成训练。这种速度意味着，与较慢的模型相比，你能更快迭代、尝试更多特征组合并开展更多实验。

### 分步理解数学过程

下面通过一个具体例子完整推导。假设有两个类别：垃圾邮件和非垃圾邮件。词表中只有三个单词：“free”“money”“meeting”。

训练数据如下：
- 垃圾邮件中，“free”出现 80 次，“money”出现 60 次，“meeting”出现 10 次，共 150 个单词
- 非垃圾邮件中，“free”出现 5 次，“money”出现 10 次，“meeting”出现 100 次，共 115 个单词
- 40% 的邮件是垃圾邮件，60% 是非垃圾邮件

采用拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含：“free”2 次、“money”1 次、“meeting”0 次。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以很大优势胜出。“free”出现两次，是支持垃圾邮件的强证据。注意，“meeting”没有出现，因此它对两个对数和的贡献都是零（0 * log(P)）。在多项式朴素贝叶斯中，未出现的单词没有影响；明确建模单词缺失的是伯努利朴素贝叶斯。

### 三种变体

朴素贝叶斯有三种主要形式，区别在于它们对 `P(feature | class)` 的建模方式。

#### 多项式朴素贝叶斯

把每个特征建模为计数。最适合特征为词频或 TF-IDF 值的文本数据。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

其中的 `alpha` 是拉普拉斯平滑参数，稍后会详细解释。它是文本分类中最常用的朴素贝叶斯变体。

#### 高斯朴素贝叶斯

把每个特征建模为正态分布，最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别都为每个特征拥有独立的均值和方差。当同一类别内的特征确实近似服从钟形曲线时，这种方法效果很好。

#### 伯努利朴素贝叶斯

把每个特征建模为二元状态，即出现或未出现。最适合短文本或二元特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与多项式变体不同，伯努利朴素贝叶斯会明确惩罚某个单词的缺失。如果“free”通常会出现在垃圾邮件中，但当前邮件中没有它，伯努利朴素贝叶斯会把这一点计作反对垃圾邮件的证据。

### 如何选择变体

| 变体 | 特征类型 | 最适合 | 示例 |
|---------|-------------|----------|---------|
| 多项式 | 计数或频率 | 文本分类、词袋模型 | 邮件垃圾分类、主题分类 |
| 高斯 | 连续值 | 特征近似正态分布的表格数据 | 鸢尾花分类、传感器数据 |
| 伯努利 | 二元值（0/1） | 短文本、二元特征向量 | 短信垃圾分类、出现/未出现特征 |

### 拉普拉斯平滑

如果测试数据中出现了某个单词，但在某个类别的训练数据中从未出现过，会发生什么？

不使用平滑时，`P(word | class) = 0/N = 0`。整个连乘积中只要出现一个零，`P(class | features) = 0`，其他证据无论多么有力都无法挽救结果。一个未见过的单词就会摧毁整次预测。

拉普拉斯平滑为每个特征计数都加上一个较小值 `alpha`，通常取 1：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当 alpha=1 时，每个单词至少都会获得一个很小的概率。这样一来，即使测试邮件中出现“discombobulate”，垃圾邮件概率也不会直接归零。从贝叶斯角度看，这种平滑等价于在单词分布上施加均匀 Dirichlet 先验。

alpha 越大，平滑越强，分布也越均匀；alpha 越小，模型就越相信数据。Alpha 是一个需要调节的超参数。

alpha 的影响如下：

| Alpha | 效果 | 适用场景 |
|-------|--------|-------------|
| 0.001 | 几乎不平滑，高度相信数据 | 训练集非常大，预计不会出现未知特征 |
| 0.1 | 轻度平滑 | 训练集较大 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 强平滑，使分布趋于平坦 | 训练集很小，预计会出现大量未知特征 |

### 在对数空间中计算

把数百个都小于 1 的概率相乘，会导致浮点数下溢。即使真实结果是一个很小的正数，浮点表示中的乘积也可能变成零。

解决方案是在对数空间中计算。与其把概率相乘，不如把它们的对数相加：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这样，预测就变成了点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

归根结底就是矩阵乘法。这也是朴素贝叶斯预测如此之快的原因：它执行的操作与单层线性模型相同。

### 朴素贝叶斯与逻辑回归

两者都是可用于文本的线性分类器，区别在于它们建模的对象不同。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式（建模 P(X\|Y)） | 判别式（建模 P(Y\|X)） |
| 训练 | 统计频率 | 优化损失函数 |
| 小数据 | 更好（强先验有所帮助） | 更差（数据不足以估计权重） |
| 大数据 | 更差（错误假设形成限制） | 更好（决策边界更灵活） |
| 特征 | 假设相互独立 | 能处理相关性 |
| 速度 | 单次遍历，速度极快 | 迭代优化 |
| 校准 | 概率较差 | 概率更好 |

经验法则是：先使用朴素贝叶斯。如果数据足够多，而且朴素贝叶斯的性能不再提升，再切换到逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实践中，为避免浮点数下溢，我们会在对数空间中计算。也就是说，不再把大量微小概率相乘，而是把它们的对数相加：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 动手构建

`code/naive_bayes.py` 中的代码会从零实现 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现包含以下步骤：

1. **fit(X, y)**：针对每个类别，统计各特征出现的频率；加入拉普拉斯平滑；计算对数概率；保存类别先验，也就是类别频率的对数。

2. **predict_log_proba(X)**：针对每个样本和所有类别，计算 log P(class) 与各项 log P(feature_i | class) 之和。这就是一次矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键之处在于：拟合完成后，预测只需要执行矩阵乘法再加上一个偏置项。这就是朴素贝叶斯速度如此之快的原因。

### GaussianNB

对于连续特征，我们会针对每个类别、每个特征分别估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测时，先针对每个特征计算高斯概率密度函数，再跨特征相乘；实际实现会在对数空间中把它们相加。

### 演示：文本分类

代码会生成合成词袋数据，模拟两个类别，即技术文章和体育文章。每个类别有不同的词频分布，MultinomialNB 根据单词计数对它们进行分类。

合成数据的生成方式如下：我们创建 200 个“单词”，也就是 200 个特征列。编号 0–39 的单词在技术文章中频率较高，在体育文章中频率较低；编号 80–119 的单词在体育文章中频率较高，在技术文章中频率较低；编号 40–79 的单词在两类文章中都有中等频率。这个场景很接近真实情况：一部分词是强类别指标，其余词则是噪声。

### 演示：连续特征

代码会生成类似鸢尾花的数据，其中有 3 个类别、4 个特征，并形成高斯簇。GaussianNB 使用每个类别的均值和方差进行分类。各类别拥有不同的中心，也就是均值向量，以及不同的分散程度，也就是方差，以此模拟现实中不同类别测量值存在系统性差异的情况。

代码还会演示：
- **平滑强度比较：** 使用不同 alpha 训练 MultinomialNB，观察平滑强度对准确率的影响。
- **训练规模实验：** 随着训练数据从 20 个样本增长到 1600 个样本，观察朴素贝叶斯准确率如何变化。即使样本很少，它也能达到不错的准确率，这正是其主要优势。
- **混淆矩阵：** 计算每个类别的精确率、召回率和 F1 分数，展示朴素贝叶斯会在哪里犯错。

### 预测速度

朴素贝叶斯的预测就是一次矩阵乘法。对于包含 n 个样本、d 个特征和 k 个类别的问题：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k)，复杂度为 O(n * d * k)
- GaussianNB：对 n * k 个组合分别计算一次覆盖 d 个特征的高斯概率密度，复杂度为 O(n * d * k)

两者对每个维度的复杂度都是线性的。相比之下，KNN 需要计算到所有训练点的距离，使用 RBF 核的 SVM 则需要针对所有支持向量计算核函数。在预测阶段，朴素贝叶斯可以快上几个数量级。

## 实际应用

使用 sklearn 时，两种变体都只需几行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

使用 sklearn 进行文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码会在同一份数据上比较从零实现与 sklearn 实现，以验证结果是否正确。

### 将 TF-IDF 与朴素贝叶斯结合

原始单词计数会让每个单词的每次出现都拥有相同权重。但“the”“is”之类常见词在每个类别中都会频繁出现，几乎不携带信息。TF-IDF（Term Frequency - Inverse Document Frequency，词频—逆文档频率）会降低常见词的权重，提高罕见且具有判别力的单词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值都是非负数，因此可以与 MultinomialNB 配合使用。TF-IDF + MultinomialNB 是文本分类中最强的基线方案之一。在训练样本少于 10,000 条的数据集上，它经常能够击败更复杂的模型。

### 用 BernoulliNB 处理短文本

处理推文、短信、聊天消息等短文本时，BernoulliNB 可能优于 MultinomialNB。短文本中的单词计数很低，因此 MultinomialNB 依赖的频率信息噪声较大。BernoulliNB 只关心单词是否出现，这种信息对短文本更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 中的 `binary=True` 会把所有计数转换成 0/1。即使不使用它，BernoulliNB 仍能运行，但模型看到的将是它原本并非为之设计的计数值。

### 校准朴素贝叶斯概率

朴素贝叶斯输出的概率往往没有得到良好校准。它给出 P(spam) = 0.95 时，真实概率可能只有 0.7。如果你需要可靠的概率估计，例如设置阈值或与其他模型组合，可以使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

它通过交叉验证，在朴素贝叶斯的原始分数之上拟合逻辑回归。得到的概率会更接近真实类别频率。

### 常见陷阱

1. **负特征值。** MultinomialNB 要求特征非负。如果数据中存在负值，例如使用特定设置的 TF-IDF 或标准化后的特征，应改用 GaussianNB，或者把所有特征平移到正数范围。

2. **零方差特征。** GaussianNB 的计算中需要除以方差。如果某个类别下的某项特征方差为零，也就是所有值完全相同，概率计算就会失败。示例代码为所有方差加入了一个很小的平滑项 1e-9，以防止这种情况。

3. **类别不均衡。** 如果 99% 的邮件都不是垃圾邮件，先验 P(not-spam) = 0.99 就会强到压过似然提供的证据。可以手动设置类别先验，或者使用 sklearn 的 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放，因为它直接处理计数；GaussianNB 同样不需要缩放，因为它会估计每项特征自己的统计量。相比对特征尺度敏感的逻辑回归和 SVM，这是朴素贝叶斯的一项优势。

## 交付成果

本课会产出：
- `outputs/skill-naive-bayes-chooser.md`——用于选择合适朴素贝叶斯变体的决策技能
- `code/naive_bayes.py`——从零实现的 MultinomialNB 与 GaussianNB，以及与 sklearn 的比较

### 朴素贝叶斯何时会失败

当独立性假设导致类别排序错误，而不只是概率数值不准时，朴素贝叶斯就会失败。常见情形包括：

1. **强特征交互。** 如果类别取决于两个特征的组合，而单独任何一个特征都不起作用，例如类似 XOR 的模式，朴素贝叶斯会完全漏掉这种关系。单个特征都无法提供证据，而朴素贝叶斯不能以非线性方式组合它们。

2. **提供相反证据的高度相关特征。** 如果特征 A 指向“垃圾邮件”，特征 B 指向“非垃圾邮件”，但 A 和 B 又完全相关，也就是在现实中它们总是一致，朴素贝叶斯就会看到实际上并不存在的矛盾证据。

3. **训练集非常大。** 数据足够多时，逻辑回归等判别式模型能够学到真实的决策边界，并超越朴素贝叶斯。此前在小数据场景中有所帮助的独立性假设，此时反而会限制模型。

实践中，这些失败模式在文本分类里并不常见。文本特征数量多、单个特征的信号弱，而且独立性假设造成的误差往往会相互抵消。对于只有少量强相关特征的表格数据，应优先考虑逻辑回归或树模型。

## 练习

1. **平滑实验。** 使用 0.01、0.1、1.0、10.0 和 100.0 作为 alpha，在文本数据上训练 MultinomialNB。绘制准确率随 alpha 变化的曲线。性能在哪里达到峰值？alpha 过高为何会损害性能？

2. **特征独立性检验。** 选取一个真实文本数据集，再选两个明显相关的单词，例如“machine”和“learning”。计算 P(word1 | class) * P(word2 | class)，并与 P(word1 AND word2 | class) 比较。独立性假设错得有多严重？它是否影响了分类准确率？

3. **实现伯努利变体。** 扩展示例代码，加入 BernoulliNB 类。把词袋表示转换成二元值，即出现/未出现，再比较它与 MultinomialNB 在文本数据上的准确率。伯努利变体在什么时候胜出？

4. **朴素贝叶斯与逻辑回归。** 在文本数据上分别训练两种模型。从 100 个训练样本开始，逐步增加到 10,000 个。绘制两种模型的准确率随训练集大小变化的曲线。逻辑回归在什么时候超过朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建完整的垃圾邮件分类器：对原始邮件文本分词、建立词表、创建词袋特征、训练 MultinomialNB，并使用精确率和召回率评估，而不只是准确率。为什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯 | “简单的概率分类器” | 应用贝叶斯定理，并假设给定类别后各特征条件独立的分类器 |
| 条件独立 | “特征互不影响” | P(A, B \| C) = P(A \| C) * P(B \| C)——已知 C 后，知道 B 不会为 A 提供任何新信息 |
| 拉普拉斯平滑 | “加一平滑” | 为每个特征加入一个小计数，防止零概率主导整个预测 |
| 先验 | “看到数据前的判断” | P(class)——观察任何特征前，每个类别的概率 |
| 似然 | “数据与假设有多吻合” | P(features \| class)——已知类别时，观察到这些特征的概率 |
| 后验 | “看到数据后的判断” | P(class \| features)——观察特征后更新得到的类别概率 |
| 生成式模型 | “对数据如何产生建模” | 学习 P(X \| Y) 和 P(Y)，再使用贝叶斯定理得到 P(Y \| X) 的模型 |
| 判别式模型 | “对决策边界建模” | 不对 X 的产生过程建模，而是直接学习 P(Y \| X) 的模型 |
| 对数概率 | “避免下溢” | 使用 log P 而非 P 计算，防止许多小数相乘后在浮点表示中变成零 |

## 延伸阅读

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html)——包含三种变体及其数学细节
- [McCallum 与 Nigam：《A Comparison of Event Models for Naive Bayes Text Classification》（1998）](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)——多项式与伯努利变体在文本任务上的经典比较
- [Rennie 等：《Tackling the Poor Assumptions of Naive Bayes Text Classifiers》（2003）](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)——改进朴素贝叶斯文本分类器的方法
- [Ng 与 Jordan：《On Discriminative vs. Generative Classifiers》（2001）](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)——证明在较少数据下，朴素贝叶斯比逻辑回归收敛得更快
