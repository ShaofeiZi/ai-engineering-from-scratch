# 无监督学习

> 没有标签，也没有老师，算法自行发现结构。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 1（范数与距离、概率与分布）、阶段 2 第 1–6 课
**Time:** 约 90 分钟

## 学习目标

- 从零实现 K-Means、DBSCAN 和 Gaussian Mixture Model，并比较它们的聚类行为
- 使用 silhouette score 评估聚类质量，并通过肘部法选择最佳 K
- 解释 DBSCAN 何时优于 K-Means，并判断哪种算法能处理非球形簇与异常值
- 使用聚类方法构建异常检测流水线，标记偏离正常模式的数据点

## 问题

此前每节机器学习课程都假设数据带有标签：“这是输入，这是正确输出。”真实世界中，标签十分昂贵。医院拥有数百万份患者记录，却没人逐一标注疾病类别；电商网站拥有数百万次用户会话，却没人手工标注客户群体；安全团队拥有海量网络日志，也没人标记每一个异常。

无监督学习无需别人告诉它应该寻找什么，就能发现模式。它会把相似数据分组、发现隐藏结构并暴露异常。如果监督学习像拿着答案学习课本，无监督学习就像凝视原始数据，直到模式自己浮现出来。

难点在于：没有标签，就无法直接判断“对”或“错”。因此需要其他工具，评估算法找到的结构是否有意义。

## 核心概念

### 聚类：把相似对象放在一起

聚类会把每个数据点分配到一个组（cluster），使同一组中的点彼此更相似，而与其他组中的点差异更大。核心问题始终是：“相似”究竟意味着什么？

```mermaid
flowchart LR
    A[Raw Data] --> B{Choose Method}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[Hierarchical]
    B --> F[GMM]
    C --> G[Flat, spherical clusters]
    D --> H[Arbitrary shapes, noise detection]
    E --> I[Tree of nested clusters]
    F --> J[Soft assignments, elliptical clusters]
```

### K-Means：主力算法

K-Means 会把数据准确划分成 K 个簇。每个簇拥有一个质心，也就是质量中心；每个点都归入距离最近的质心。

Lloyd 算法：

1. 随机选择 K 个点作为初始质心
2. 把每个数据点分配给距离最近的质心
3. 把每个质心重新计算为所属数据点的均值
4. 重复第 2–3 步，直到分配结果不再变化

目标函数 inertia 衡量每个点到所属质心的平方距离总和。K-Means 会最小化它，但只能找到局部最小值；不同初始化可能产生不同结果。

### 选择 K

有两种标准方法：

**肘部法：**分别对 K = 1, 2, 3, ..., n 运行 K-Means，绘制 inertia 随 K 变化的曲线，寻找增加簇数后 inertia 不再显著下降的“肘部”。

**Silhouette score：**对每个点，比较它与所属簇的相似程度（a），以及它与最近其他簇的相似程度（b）。Silhouette 系数为 (b - a) / max(a, b)，范围从 -1（分到了错误簇）到 +1（聚类良好）。对所有点求平均，即得到全局分数。

### DBSCAN：基于密度的聚类

K-Means 假设簇为球形，并要求预先选择 K。DBSCAN 不需要这两个假设，它会寻找由稀疏区域隔开的高密度区域。

两个参数：
- **eps：**邻域半径
- **min_samples：**构成高密度区域所需的最少点数

三类数据点：
- **核心点：**在 eps 距离内至少有 min_samples 个点
- **边界点：**位于某个核心点的 eps 范围内，但自身不是核心点
- **噪声点：**既不是核心点，也不是边界点，也就是异常值

DBSCAN 会把 eps 范围内相连的核心点归入同一个簇，边界点加入附近核心点所属的簇，噪声点则不属于任何簇。

优点：能够发现任意形状的簇、自动确定簇数、识别异常值。缺点：难以处理密度差异很大的多个簇。

### 层次聚类

层次聚类会构建由嵌套簇组成的树，也就是 dendrogram。

Agglomerative（自底向上）方法：
1. 让每个点各自成为一个簇
2. 合并距离最近的两个簇
3. 重复，直到只剩一个簇
4. 在需要的高度切断 dendrogram，得到 K 个簇

簇间“距离”可以有多种定义：
- **Single linkage：**两个簇中任意点对的最小距离
- **Complete linkage：**任意点对的最大距离
- **Average linkage：**所有点对距离的平均值
- **Ward 方法：**选择使簇内总方差增加最小的合并

### Gaussian Mixture Model（GMM）

K-Means 进行硬分配：每个点恰好属于一个簇。GMM 进行软分配：每个点都有属于各个簇的概率。

GMM 假设数据由 K 个 Gaussian 分布混合产生，每个分布都有自己的均值和协方差。Expectation-Maximization（EM）算法在两步之间交替：

- **E-step：**计算每个点属于每个 Gaussian 的概率
- **M-step：**更新每个 Gaussian 的均值、协方差和混合权重，使数据似然最大

GMM 可以建模椭圆形簇，而不仅是 K-Means 的球形簇，也能自然处理重叠簇。

### 如何选择算法

| 方法 | 最适合 | 应避免的情况 |
|--------|----------|------------|
| K-Means | 大型数据集、球形簇、已知 K | 形状不规则、存在异常值 |
| DBSCAN | K 未知、任意形状、异常检测 | 各簇密度不同、维度很高 |
| 层次聚类 | 小型数据集、需要 dendrogram、K 未知 | 大型数据集（内存 O(n^2)） |
| GMM | 簇重叠、需要软分配 | 超大型数据集、维度过多 |

### 使用聚类进行异常检测

聚类天然支持异常检测：
- **K-Means：**距离所有质心都很远的点是异常值
- **DBSCAN：**噪声点按定义就是异常值
- **GMM：**在所有 Gaussian 下概率都很低的点是异常值

```figure
kmeans-step
```

## 动手构建

### 第 1 步：从零实现 K-Means

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
```

### 第 2 步：肘部法与 silhouette score

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### 第 3 步：从零实现 DBSCAN

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### 第 4 步：Gaussian Mixture Model（EM 算法）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### 第 5 步：生成测试数据并运行全部算法

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
```

## 实际使用

使用 scikit-learn 时，这些算法都只需一行：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

从零实现能准确展示这些库的计算方式：K-Means 在分配样本与重算质心之间迭代；DBSCAN 从高密度种子扩展簇；GMM 在 expectation 与 maximization 之间交替。库版本还增加了数值稳定性、更智能的初始化（K-Means++）和 GPU 加速，但核心逻辑相同。

## 交付成果

本课会产出从零实现且可运行的 K-Means、DBSCAN 和 GMM。这些聚类代码可以作为更高级无监督方法的基础。

## 练习

1. 实现 K-Means++ 初始化：不再随机选择全部质心，而是随机选择第一个质心，后续每个质心按其到最近已有质心的平方距离成比例抽取。与随机初始化比较收敛速度。
2. 向代码中加入层次凝聚聚类，实现 Ward 联接法，并以嵌套的合并列表生成树状图；在不同高度截断树状图，再与 K-Means 结果比较。
3. 构建简单异常检测流水线：在同一数据上运行 DBSCAN 与 GMM，标记两个方法都认定为异常的点，也就是 DBSCAN 中的噪声、GMM 中的低概率点。测量重叠程度，并讨论二者何时会得出不同结论。

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| Clustering | “把相似对象分组” | 按指定距离度量划分数据，使组内相似度高于组间相似度 |
| Centroid | “簇中心” | 分配给某个簇的全部点的均值，K-Means 用它代表该簇 |
| Inertia | “簇有多紧密” | 每个点到所属质心的平方距离总和；越低越紧密 |
| Silhouette score | “簇分得有多开” | 对每个点计算 (b - a) / max(a, b)，a 为平均簇内距离，b 为最近其他簇的平均距离 |
| Core point | “高密度区域中的点” | DBSCAN 中 eps 距离内至少有 min_samples 个邻居的点 |
| EM algorithm | “软 K-Means” | Expectation-Maximization：反复计算成员概率（E-step），再更新分布参数（M-step） |
| Dendrogram | “簇组成的树” | 展示层次聚类中簇以何种顺序、在哪个距离合并的树状图 |
| Anomaly | “异常值” | 不符合预期模式的数据点；在 DBSCAN 中被标记为噪声，在 GMM 中概率很低 |

## 延伸阅读

- [Stanford CS229——无监督学习](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)——Andrew Ng 关于聚类与 EM 的讲义
- [scikit-learn 聚类指南](https://scikit-learn.org/stable/modules/clustering.html)——使用可视化示例比较各种聚类算法的实践资料
- [DBSCAN 原始论文（Ester 等，1996）](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)——提出基于密度聚类的论文
