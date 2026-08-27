# 范数与距离

> 距离函数定义了“相似”的含义。选错了，后续所有环节都会出问题。

**Type:** 构建
**Language:** Python
**Prerequisites:** 第 1 阶段，第 01 课（Linear Algebra Intuition）和第 02 课（Vectors, Matrices & Operations）
**Time:** 约 90 分钟

## 学习目标

- 从零实现 L1、L2、余弦、Mahalanobis、Jaccard 和编辑距离函数
- 为给定的机器学习任务选择合适的距离度量，并解释其他选择为何不适用
- 将 L1 和 L2 范数与 LASSO、Ridge 正则化及其几何约束区域联系起来
- 演示同一个数据集在不同度量下如何产生不同的最近邻

## 问题

你有两个向量。它们可能是词嵌入、用户画像，也可能是像素数组。你需要回答：它们有多接近？

答案完全取决于你选择的距离函数。同一对数据点，在一种度量下可能是最近邻，在另一种度量下却相距很远。KNN 分类器、推荐引擎、向量数据库、聚类算法和损失函数都依赖这一选择。选错距离，模型就会优化错误的目标。

不存在适用于所有场景的最佳距离。L2 适合空间数据，余弦相似度主导 NLP，Jaccard 处理集合，编辑距离处理字符串，Mahalanobis 会考虑相关性，Wasserstein 则衡量概率质量的搬运成本。每种度量都编码了对“相似”含义的不同假设。

本课会从零构建主要距离函数，说明各自适用的场景，并演示同一份数据如何因所选度量不同而产生完全不同的最近邻。

## 核心概念

### 范数：衡量向量大小

范数衡量向量的“大小”。两个向量之间的每种距离，都可以写成二者之差的范数：d(a, b) = ||a - b||。因此，理解范数就是理解距离。

### L1 范数（Manhattan 距离）

L1 范数对所有分量的绝对值求和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它之所以称为 Manhattan 距离，是因为它衡量在城市网格中只能沿坐标轴行走时的路程，不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

适合使用 L1 的场景：
- 高维稀疏数据（文本特征、one-hot 编码）
- 希望对异常值保持稳健时，因为单个巨大差异不会主导结果
- 特征选择问题，因为 L1 正则化会促进稀疏性

与 L1 正则化（Lasso）的联系：在损失函数中加入 ||w||_1，会惩罚权重绝对值之和。这会把较小权重推到恰好为零，从而自动完成特征选择。L1 惩罚在权重空间中形成菱形约束区域，菱形的顶角位于坐标轴上，对应某些权重为零。

与损失函数的联系：平均绝对误差（MAE）是预测值与目标值之间 L1 距离的平均值。它对所有误差施加线性惩罚，因此比 MSE 更能抵抗异常值。

### L2 范数（Euclidean 距离）

L2 范数就是直线距离，即各分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是几何课上学过的距离公式，即 n 维空间中的勾股定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

适合使用 L2 的场景：
- 中低维连续数据
- 各特征尺度可比时
- 物理距离（空间数据、传感器读数）
- 像素级图像相似度

与 L2 正则化（Ridge）的联系：在损失函数中加入 ||w||_2^2，会惩罚较大权重。与 L1 不同，它不会把权重推到零，而是按比例让所有权重缩小并趋近于零。L2 惩罚形成圆形约束区域，坐标轴上没有顶角，因此权重会变小，却很少恰好为零。

与损失函数的联系：均方误差（MSE）是 L2 距离平方的平均值。平方操作会对大误差施加比小误差更强的惩罚。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：通用范数族

L1 和 L2 都是 Lp 范数的特殊情况：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

p 取不同值时，会产生形状不同的“单位球”，即距离原点为 1 的所有点组成的集合：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity 范数（Chebyshev 距离）

当 p 趋近无穷大时，Lp 范数会收敛到绝对值最大的分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离只由差异最大的那个维度决定，其他维度全部被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

适合使用 L-infinity 的场景：
- 任意单一维度上的最坏偏差很重要时
- 棋盘移动，例如国际象棋中的王按照 L-infinity 距离移动，任意方向一步的成本都为 1
- 制造公差，因为每个维度都必须处于规格范围内

### 余弦相似度与余弦距离

余弦相似度衡量两个向量之间的夹角，而忽略它们的长度。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的范围从 -1（方向相反）到 +1（方向相同），互相垂直的向量余弦相似度为 0。

余弦距离把相似度转换为距离：cosine_distance = 1 - cosine_similarity。其范围从 0（方向完全相同）到 2（方向完全相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

余弦相似度为何主导 NLP 和嵌入场景：在文本中，文档长度不应影响相似度。一篇关于猫的文档，即使长度是另一篇猫主题文档的两倍，二者仍应“相似”。余弦相似度忽略长度，只关注方向。词语分布相同、长度不同的两个文档会指向同一方向，因此余弦相似度为 1.0。

适合使用余弦相似度的场景：
- 文本相似度（TF-IDF 向量、词嵌入、句子嵌入）
- 向量长度是噪声、方向才是信号的领域
- 推荐系统（用户偏好向量）
- 嵌入搜索（向量数据库几乎总是使用余弦相似度或点积）

### 点积相似度与余弦相似度

两个向量的点积为：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是除以两个向量长度后归一化的点积。当两个向量都已经归一化为单位向量（长度为 1）时，点积与余弦相似度完全相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

二者的差异在于：点积包含长度信息。长度较大的向量会获得更高的点积分数。在某些检索系统中，你可能希望“热门”项目排名更高，此时向量长度会充当隐式质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践建议：
- 只关心方向相似度时，使用余弦相似度
- 长度携带有意义的信息时，使用点积
- Pinecone、Weaviate、Qdrant 等许多向量数据库允许你在二者之间选择
- 如果嵌入已经进行 L2 归一化，选择二者不会产生差异

### Mahalanobis 距离

Euclidean 距离平等对待所有维度。但当特征存在相关性或尺度不同时，L2 会给出误导性结果。

Mahalanobis 距离会考虑数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观来说，Mahalanobis 距离会先对数据去相关并归一化，也就是白化，然后在变换后的空间中计算 L2 距离。如果 S 是单位矩阵，即特征互不相关且方差均为 1，Mahalanobis 距离就退化为 Euclidean 距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

适合使用 Mahalanobis 距离的场景：
- 异常值检测（距均值 Mahalanobis 距离很大的点是异常值）
- 特征尺度不同且彼此相关的分类任务
- 有足够数据可靠估计协方差矩阵时
- 制造质量控制（多变量过程监控）

### Jaccard 相似度（用于集合）

Jaccard 相似度衡量两个集合的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

它的范围从 0（没有重叠）到 1（集合完全相同）。Jaccard 距离 = 1 - Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

适合使用 Jaccard 的场景：
- 比较标签、类别或特征集合
- 基于词是否出现，而非出现频率，比较文档相似度
- 近重复检测（使用 MinHash 近似 Jaccard）
- 比较二元特征向量（存在/缺失数据）
- 评估分割模型（Intersection over Union 就是 Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离统计把一个字符串变成另一个字符串所需的最少单字符操作次数。允许的操作是插入、删除和替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

可以使用动态规划计算编辑距离。构建一个矩阵，其中第 (i, j) 个元素表示字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

适合使用编辑距离的场景：
- 拼写检查与纠错
- DNA 序列比对（对不同操作设置权重）
- 模糊字符串匹配
- 对杂乱文本数据去重

### KL 散度（不是真正的距离，但常被当作距离使用）

KL 散度衡量一个概率分布与另一个概率分布之间的差异。第 09 课已介绍过它；这里再次讨论，是因为人们经常把它当作“距离”使用，尽管它并不满足距离定义。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质是：KL 散度不对称。

```
D_KL(P || Q) != D_KL(Q || P)
```

因此，它不满足距离度量的基本要求，也不满足三角不等式。它是一种 divergence，而不是 distance。

正向 KL（D_KL(P || Q)）倾向于“覆盖均值”：Q 会努力覆盖 P 的所有模式。反向 KL（D_KL(Q || P)）倾向于“寻找模式”：Q 会集中于 P 的某一个模式。

KL 散度的常见应用：
- VAE（ELBO 中的 KL 项把潜在分布推向先验）
- 知识蒸馏（学生模型尝试匹配教师模型的分布）
- RLHF（KL 惩罚让微调模型保持接近基础模型）
- 策略梯度方法（限制策略更新幅度）

### Wasserstein 距离（Earth Mover's Distance）

Wasserstein 距离衡量把一个概率分布转换为另一个分布所需的最小“功”。可以把一个分布想象成一堆土，另一个分布想象成一个坑：需要搬运多少土，又要搬多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它可以简化为两个累积分布函数绝对差的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Wasserstein 距离的重要性：
- 它是真正的度量，具有对称性并满足三角不等式
- 即使两个分布没有重叠，它仍能提供梯度，而 KL 散度会变成无穷大
- 正是这一性质让它成为 Wasserstein GAN（WGAN）的核心，并改善了原始 GAN 的训练不稳定问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

适合使用 Wasserstein 的场景：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能完全不重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 不同任务为何需要不同距离

| 任务 | 最佳距离 | 原因 |
|------|--------------|-----|
| 文本相似度 | 余弦距离 | 长度是噪声，方向表示含义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 | 稳健，不会放大少数罕见的大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据天然是集合，而不是向量 |
| 字符串匹配 | 编辑距离 | 运算与人类编辑直觉一致 |
| 异常值检测 | Mahalanobis | 考虑特征相关性和尺度 |
| 比较分布 | KL 散度 | 衡量使用 Q 代替 P 时损失的信息 |
| GAN 训练 | Wasserstein | 即使分布不重叠也能提供梯度 |
| 嵌入（向量数据库） | 余弦或点积 | 嵌入经过训练，会在方向中编码含义 |
| 推荐系统 | 点积 | 长度可以编码热门程度或置信度 |
| DNA 序列 | 加权编辑距离 | 不同核苷酸对的替换成本不同 |
| 制造质量控制 | L-infinity | 任一维度上的最坏偏差都很重要 |

### 与损失函数的联系

损失函数就是应用于预测值与目标值的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化会把权重的范数惩罚加入损失函数。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

L1 能产生稀疏解而 L2 不能，原因可以从二维权重空间中的约束区域看出。L1 约束是菱形，L2 约束是圆形。损失函数的等高线（椭圆）最容易在菱形顶角处与之相切，此时某个权重为零；而它会在圆形的光滑位置相切，此时两个权重都不为零。

### 最近邻搜索

每种距离函数都定义了一个最近邻搜索问题：给定查询点，在数据集中寻找距离最近的点。

对于包含 n 个点、每点 d 维的数据集，精确最近邻搜索的每次查询复杂度是 O(n * d)，大型数据集无法承受这一成本。

近似最近邻（ANN）算法牺牲少量准确率，换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World）是现代向量数据库中的主流算法。它构建多层图，每个节点连接到自己的近似最近邻。搜索从顶层开始，那里图较稀疏，适合长距离跳跃；随后逐层下降到底层，那里图更稠密，适合短距离精细搜索。

```figure
norm-unit-balls
```

## 动手构建

### 第 1 步：实现全部范数与距离函数

完整实现见 `code/distances.py`。每个函数都只使用基础 Python 数学从零构建。

### 第 2 步：相同数据，不同距离，不同邻居

`distances.py` 中的演示会创建一个数据集，选择一个查询点，然后展示距离度量变化时，最近邻如何随之变化。在 L1 下“最近”的点，在 L2 或余弦距离下可能并不是最近点。

### 第 3 步：嵌入相似度搜索

代码还包含一个模拟嵌入相似度搜索，分别使用余弦相似度和 L2 距离，为查询寻找最相似的“文档”，并展示两种度量可能产生不同排名。

## 实际使用

最常见的实际用途是在向量数据库中寻找相似项目。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 并搜索向量数据库时，底层做的正是这些工作。嵌入模型把文本映射成向量，向量数据库则使用 ANN 算法避免检查全部向量，同时计算查询向量与存储向量之间的余弦相似度（或点积）。

## 练习

1. 计算 (1, 2, 3) 与 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证任意两点都满足 L-inf <= L2 <= L1，并证明为何总是如此。

2. 构造两个余弦相似度很高（> 0.9）、但 L2 距离很大（> 10）的向量，并从几何角度解释发生了什么；再构造两个余弦相似度很低（< 0.3）、但 L2 距离很小（< 0.5）的向量。

3. 实现一个函数：输入数据集和查询点，分别返回 L1、L2、余弦和 Mahalanobis 距离下的最近邻。找到一个数据集，使四种度量对最近点的判断全部不同。

4. 使用 CDF 方法手工计算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离，再计算 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5] 之间的距离。哪一个更大？为什么？

5. 为近似 Jaccard 相似度实现 MinHash。生成 100 个随机集合，计算所有集合对的精确 Jaccard，再分别使用 50、100 和 200 个哈希函数比较 MinHash 近似，绘制近似误差。

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| Norm | “向量大小” | 把向量映射到非负标量的函数，满足三角不等式、绝对齐次性，并且只有零向量的范数为零 |
| L1 norm | “Manhattan 距离” | 分量绝对值之和；在优化中产生稀疏性，并对异常值稳健 |
| L2 norm | “Euclidean 距离” | 各分量平方和的平方根，也就是 Euclidean 空间中的直线距离 |
| Lp norm | “广义范数” | 各分量绝对值 p 次幂之和再开 p 次方根；L1 与 L2 是特殊情况 |
| L-infinity norm | “最大范数”或“Chebyshev 距离” | 最大分量绝对值，也是 p 趋近无穷大时 Lp 范数的极限 |
| Cosine similarity | “向量夹角” | 点积除以两个向量长度，范围为 -1 到 +1，忽略向量长度 |
| Cosine distance | “1 减余弦相似度” | 把余弦相似度转换为距离，范围从 0 到 2 |
| Dot product | “未归一化的余弦” | 分量乘积之和，等于余弦相似度乘以两个向量长度 |
| Mahalanobis distance | “考虑相关性的距离” | 使用数据协方差矩阵将空间白化（去相关并归一化）后计算的 L2 距离 |
| Jaccard similarity | “集合重叠” | 交集大小除以并集大小，适用于集合而非一般向量 |
| Edit distance | “Levenshtein 距离” | 把一个字符串变成另一个字符串所需的最少插入、删除和替换次数 |
| KL divergence | “分布之间的距离” | 不是真正的距离，因为它不对称；衡量使用 Q 编码 P 时增加的信息量 |
| Wasserstein distance | “Earth mover's distance” | 把概率质量从一个分布搬运到另一个分布所需的最小功，是真正的距离度量 |
| Approximate nearest neighbor | “ANN 搜索” | HNSW、LSH、IVF 等算法，以少量近似误差换取远快于精确搜索的速度 |
| HNSW | “向量数据库算法” | Hierarchical Navigable Small World 图，通过多层图快速进行近似最近邻搜索 |
| L1 regularization | “Lasso” | 把权重的 L1 范数加入损失，使权重趋于零并产生稀疏性 |
| L2 regularization | “Ridge”或“weight decay” | 把权重的 L2 范数平方加入损失，使权重趋近零，但不会产生稀疏性 |
| Elastic Net | “L1 + L2” | 结合 L1 与 L2 正则化，比单独使用任一种方法更善于处理相关特征组 |

## 延伸阅读

- [FAISS：高效相似度搜索库](https://github.com/facebookresearch/faiss)——Meta 面向十亿规模 ANN 搜索的库
- [Wasserstein GAN（Arjovsky 等，2017）](https://arxiv.org/abs/1701.07875)——把 Earth Mover's distance 引入 GAN 的论文
- [局部敏感哈希（Indyk 与 Motwani，1998）](https://dl.acm.org/doi/10.1145/276698.276876)——奠定 ANN 基础的算法
- [高效估计词表示（Mikolov 等，2013）](https://arxiv.org/abs/1301.3781)——Word2Vec 论文，余弦相似度由此成为嵌入的默认度量
- [sklearn.neighbors 文档](https://scikit-learn.org/stable/modules/neighbors.html)——scikit-learn 中距离度量和邻居算法的实践指南
