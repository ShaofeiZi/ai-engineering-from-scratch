import math
import random


def l2_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def l1_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def cosine_distance(a, b):
    dot_val = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot_val / (norm_a * norm_b)


def minkowski_distance(a, b, p=2):
    if p == float("inf"):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return sum(abs(ai - bi) ** p for ai, bi in zip(a, b)) ** (1 / p)


def standardize(X):
    n = len(X)
    d = len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    stds = [
        max(
            1e-10,
            (sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n) ** 0.5,
        )
        for j in range(d)
    ]
    X_scaled = [
        [(X[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)
    ]
    return X_scaled, means, stds


def apply_standardize(X, means, stds):
    return [[(x[j] - means[j]) / stds[j] for j in range(len(x))] for x in X]


class KNN:
    def __init__(self, k=5, distance_fn=l2_distance, weighted=False,
                 task="classification"):
        self.k = k
        self.distance_fn = distance_fn
        self.weighted = weighted
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = list(X)
        self.y_train = list(y)

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            d = self.distance_fn(x, self.X_train[i])
            distances.append((d, self.y_train[i]))
        distances.sort(key=lambda pair: pair[0])
        neighbors = distances[: self.k]

        if self.task == "classification":
            return self._classify(neighbors)
        return self._regress(neighbors)

    def _classify(self, neighbors):
        if self.weighted:
            votes = {}
            for dist, label in neighbors:
                w = 1.0 / (dist + 1e-10)
                votes[label] = votes.get(label, 0) + w
        else:
            votes = {}
            for _, label in neighbors:
                votes[label] = votes.get(label, 0) + 1
        return max(votes, key=votes.get)

    def _regress(self, neighbors):
        if self.weighted:
            w_sum = 0.0
            val_sum = 0.0
            for dist, val in neighbors:
                w = 1.0 / (dist + 1e-10)
                val_sum += w * val
                w_sum += w
            return val_sum / w_sum if w_sum > 0 else 0.0
        return sum(val for _, val in neighbors) / len(neighbors)

    def predict_with_neighbors(self, x):
        distances = []
        for i in range(len(self.X_train)):
            d = self.distance_fn(x, self.X_train[i])
            distances.append((d, i, self.y_train[i]))
        distances.sort(key=lambda t: t[0])
        neighbors = distances[: self.k]
        prediction = self._predict_one(x)
        return prediction, neighbors


class KDNode:
    def __init__(self, point, index, axis, left=None, right=None):
        self.point = point
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    def __init__(self, X):
        self.dim = len(X[0])
        indexed = [(X[i], i) for i in range(len(X))]
        self.root = self._build(indexed, depth=0)

    def _build(self, points, depth):
        if not points:
            return None
        axis = depth % self.dim
        points.sort(key=lambda p: p[0][axis])
        mid = len(points) // 2
        return KDNode(
            point=points[mid][0],
            index=points[mid][1],
            axis=axis,
            left=self._build(points[:mid], depth + 1),
            right=self._build(points[mid + 1 :], depth + 1),
        )

    def query(self, point, k=1):
        best = []
        self._search(self.root, point, k, best)
        best.sort(key=lambda x: x[0])
        return best

    def _search(self, node, point, k, best):
        if node is None:
            return

        dist = l2_distance(point, node.point)

        if len(best) < k:
            best.append((dist, node.index, node.point))
            best.sort(key=lambda x: x[0])
        elif dist < best[-1][0]:
            best[-1] = (dist, node.index, node.point)
            best.sort(key=lambda x: x[0])

        axis = node.axis
        diff = point[axis] - node.point[axis]

        if diff <= 0:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        self._search(first, point, k, best)

        if len(best) < k or abs(diff) < best[-1][0]:
            self._search(second, point, k, best)


def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def mse(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def generate_classification_data(n_samples=200, n_classes=3, seed=42):
    random.seed(seed)
    X = []
    y = []
    centers = [
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
    ]
    for _ in range(n_samples):
        c = random.randint(0, n_classes - 1)
        x1 = centers[c][0] + random.gauss(0, 0.5)
        x2 = centers[c][1] + random.gauss(0, 0.5)
        X.append([x1, x2])
        y.append(c)
    return X, y


def generate_regression_data(n_samples=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x = random.uniform(-3, 3)
        target = math.sin(x) + random.gauss(0, 0.15)
        X.append([x])
        y.append(target)
    return X, y


def generate_high_dim_data(n_samples=500, n_dims=2, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        point = [random.uniform(0, 1) for _ in range(n_dims)]
        label = 1 if sum(point[:2]) > 1.0 else 0
        X.append(point)
        y.append(label)
    return X, y


def train_test_split(X, y, test_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        [X[i] for i in train_idx],
        [y[i] for i in train_idx],
        [X[i] for i in test_idx],
        [y[i] for i in test_idx],
    )


def demo_basic_knn():
    print("=" * 65)
    print("KNN 分类：基础入门")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集：{len(X)} 个样本，2 个特征，3 个类别")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()

    k_values = [1, 3, 5, 7, 11, 15, 25, 50]
    print(f"  {'K':>6s}  {'训练准确率':>10s}  {'测试准确率':>10s}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}")

    for k in k_values:
        knn = KNN(k=k, task="classification")
        knn.fit(X_train, y_train)
        train_acc = accuracy(y_train, knn.predict(X_train))
        test_acc = accuracy(y_test, knn.predict(X_test))
        print(f"  {k:>6d}  {train_acc:>10.4f}  {test_acc:>10.4f}")

    print()
    print("  K=1：训练准确率完美（记忆化），测试准确率较低。")
    print("  增大 K 可使决策边界更平滑。")
    print()


def demo_distance_metrics():
    print("=" * 65)
    print("距离度量：相同数据，不同邻居")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)
    X_scaled, means, stds = standardize(X)
    X_train, y_train, X_test, y_test = train_test_split(X_scaled, y)

    metrics = [
        ("L2 (Euclidean)", l2_distance),
        ("L1 (Manhattan)", l1_distance),
        ("Cosine", cosine_distance),
    ]

    k = 5
    print(f"  K = {k}，特征已标准化")
    print()
    print(f"  {'度量':<20s}  {'测试准确率':>14s}")
    print(f"  {'-' * 20}  {'-' * 14}")

    for name, dist_fn in metrics:
        knn = KNN(k=k, distance_fn=dist_fn, task="classification")
        knn.fit(X_train, y_train)
        test_acc = accuracy(y_test, knn.predict(X_test))
        print(f"  {name:<20s}  {test_acc:>14.4f}")

    print()

    query = X_test[0]
    print(f"  查询点：[{query[0]:.3f}, {query[1]:.3f}]")
    print(f"  真实标签：{y_test[0]}")
    print()

    for name, dist_fn in metrics:
        knn = KNN(k=k, distance_fn=dist_fn, task="classification")
        knn.fit(X_train, y_train)
        pred, neighbors = knn.predict_with_neighbors(query)
        print(f"  {name}：预测 = {pred}")
        for dist, idx, label in neighbors:
            print(f"    邻居索引={idx}，标签={label}，距离={dist:.4f}")
        print()


def demo_weighted_knn():
    print("=" * 65)
    print("加权 KNN 与不加权 KNN")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)
    X_scaled, _, _ = standardize(X)
    X_train, y_train, X_test, y_test = train_test_split(X_scaled, y)

    k_values = [3, 7, 15, 25]
    print(f"  {'K':>6s}  {'不加权':>12s}  {'加权':>12s}  {'差值':>8s}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}  {'-' * 8}")

    for k in k_values:
        knn_uw = KNN(k=k, weighted=False, task="classification")
        knn_w = KNN(k=k, weighted=True, task="classification")
        knn_uw.fit(X_train, y_train)
        knn_w.fit(X_train, y_train)
        acc_uw = accuracy(y_test, knn_uw.predict(X_test))
        acc_w = accuracy(y_test, knn_w.predict(X_test))
        diff = acc_w - acc_uw
        print(f"  {k:>6d}  {acc_uw:>12.4f}  {acc_w:>12.4f}  {diff:>+8.4f}")

    print()
    print("  加权 KNN 对较大的 K 值不那么敏感。")
    print("  远处的邻居贡献更小，因此增大 K 更安全。")
    print()


def demo_regression():
    print("=" * 65)
    print("KNN 回归：逼近 sin(x)")
    print("=" * 65)
    print()

    X, y = generate_regression_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    k_values = [1, 3, 5, 10, 20, 50]
    print(f"  目标：y = sin(x) + 噪声")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()
    print(f"  {'K':>6s}  {'不加权 MSE':>16s}  {'加权 MSE':>14s}")
    print(f"  {'-' * 6}  {'-' * 16}  {'-' * 14}")

    for k in k_values:
        knn_uw = KNN(k=k, task="regression", weighted=False)
        knn_w = KNN(k=k, task="regression", weighted=True)
        knn_uw.fit(X_train, y_train)
        knn_w.fit(X_train, y_train)
        mse_uw = mse(y_test, knn_uw.predict(X_test))
        mse_w = mse(y_test, knn_w.predict(X_test))
        print(f"  {k:>6d}  {mse_uw:>16.6f}  {mse_w:>14.6f}")

    print()
    print("  K=1 会过拟合（跟随噪声）。K 太大会欠拟合（过度平滑）。")
    print("  加权 KNN 在平滑预测的同时保留局部结构。")
    print()

    knn = KNN(k=5, task="regression", weighted=True)
    knn.fit(X_train, y_train)

    print("  预测示例（K=5，加权）：")
    print(f"  {'x':>8s}  {'真实 y':>8s}  {'预测 y':>8s}  {'误差':>8s}")
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for i in range(min(10, len(X_test))):
        pred = knn.predict([X_test[i]])[0]
        err = abs(y_test[i] - pred)
        print(f"  {X_test[i][0]:>8.3f}  {y_test[i]:>8.3f}  {pred:>8.3f}  {err:>8.3f}")
    print()


def demo_curse_of_dimensionality():
    print("=" * 65)
    print("维度灾难")
    print("=" * 65)
    print()

    dims = [2, 5, 10, 20, 50, 100]
    n_points = 200

    print("  第一部分：距离比值的收敛")
    print(f"  在 [0, 1]^d 中生成 {n_points} 个随机均匀分布的点")
    print()
    print(f"  {'维度':>12s}  {'最大/最小距离':>14s}  {'平均距离':>10s}  {'距离标准差':>10s}")
    print(f"  {'-' * 12}  {'-' * 14}  {'-' * 10}  {'-' * 10}")

    for d in dims:
        random.seed(42)
        points = [[random.uniform(0, 1) for _ in range(d)] for _ in range(n_points)]

        distances = []
        sample_size = min(500, n_points * (n_points - 1) // 2)
        for _ in range(sample_size):
            i = random.randint(0, n_points - 1)
            j = random.randint(0, n_points - 1)
            if i != j:
                distances.append(l2_distance(points[i], points[j]))

        if distances:
            max_d = max(distances)
            min_d = min(d_val for d_val in distances if d_val > 0)
            mean_d = sum(distances) / len(distances)
            std_d = (sum((d_val - mean_d) ** 2 for d_val in distances) / len(distances)) ** 0.5
            ratio = max_d / min_d if min_d > 0 else float("inf")
            print(f"  {d:>12d}  {ratio:>14.4f}  {mean_d:>10.4f}  {std_d:>10.4f}")

    print()
    print("  随着维度增长，最大/最小比值趋向于 1。")
    print("  所有点都变得同样遥远，“最近”失去了意义。")
    print()

    print("  第二部分：KNN 准确率随维度变化")
    print(f"  二分类：若 x[0] + x[1] > 1 则标签 = 1，否则为 0")
    print(f"  额外的维度都是纯噪声。")
    print()
    print(f"  {'维度':>12s}  {'K=5 准确率':>10s}  {'K=15 准确率':>10s}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}")

    for d in [2, 5, 10, 20, 50]:
        X, y = generate_high_dim_data(400, n_dims=d, seed=42)
        X_scaled, _, _ = standardize(X)
        X_train, y_train, X_test, y_test = train_test_split(X_scaled, y)

        knn5 = KNN(k=5, task="classification")
        knn15 = KNN(k=15, task="classification")
        knn5.fit(X_train, y_train)
        knn15.fit(X_train, y_train)
        acc5 = accuracy(y_test, knn5.predict(X_test))
        acc15 = accuracy(y_test, knn15.predict(X_test))
        print(f"  {d:>12d}  {acc5:>10.4f}  {acc15:>10.4f}")

    print()
    print("  随着噪声维度增加，准确率下降。")
    print("  信号（前 2 维）被噪声维度淹没。")
    print()


def demo_kdtree():
    print("=" * 65)
    print("KD-Tree：高效的最近邻搜索")
    print("=" * 65)
    print()

    random.seed(42)
    sizes = [100, 500, 1000, 5000]

    print(f"  二维数据，查找 5 个最近邻")
    print()
    print(f"  {'点数':>10s}  {'暴力搜索':>14s}  {'KD-tree':>14s}  {'加速比':>10s}")
    print(f"  {'-' * 10}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    for n in sizes:
        X = [[random.uniform(0, 10) for _ in range(2)] for _ in range(n)]
        query = [5.0, 5.0]
        k = 5

        import time

        n_queries = 100
        queries = [[random.uniform(0, 10) for _ in range(2)] for _ in range(n_queries)]

        start = time.time()
        for q in queries:
            dists = [(l2_distance(q, X[i]), i) for i in range(n)]
            dists.sort()
            _ = dists[:k]
        brute_time = time.time() - start

        tree = KDTree(X)

        start = time.time()
        for q in queries:
            _ = tree.query(q, k=k)
        kd_time = time.time() - start

        speedup = brute_time / kd_time if kd_time > 0 else float("inf")
        print(f"  {n:>10d}  {brute_time:>14.4f}s  {kd_time:>14.4f}s  {speedup:>10.1f}x")

    print()

    X = [[random.uniform(0, 10) for _ in range(2)] for _ in range(100)]
    tree = KDTree(X)
    query = [5.0, 5.0]

    brute = [(l2_distance(query, X[i]), i) for i in range(len(X))]
    brute.sort()
    brute_top5 = [(d, idx) for d, idx in brute[:5]]

    kd_top5 = [(d, idx) for d, idx, _ in tree.query(query, k=5)]

    print("  验证（100 个点，k=5）：")
    print(f"    暴力搜索：{[(round(d, 4), idx) for d, idx in brute_top5]}")
    print(f"    KD-tree： {[(round(d, 4), idx) for d, idx in kd_top5]}")
    match = set(idx for _, idx in brute_top5) == set(idx for _, idx in kd_top5)
    print(f"    结果是否一致：{match}")
    print()


def demo_scaling_importance():
    print("=" * 65)
    print("特征缩放：为什么它对 KNN 至关重要")
    print("=" * 65)
    print()

    random.seed(42)
    X = []
    y = []
    for _ in range(200):
        age = random.gauss(40, 15)
        salary = random.gauss(50000, 20000)
        label = 1 if age > 45 and salary < 40000 else 0
        X.append([age, salary])
        y.append(label)

    X_train, y_train, X_test, y_test = train_test_split(X, y)

    knn_raw = KNN(k=5, task="classification")
    knn_raw.fit(X_train, y_train)
    acc_raw = accuracy(y_test, knn_raw.predict(X_test))

    X_train_s, means, stds = standardize(X_train)
    X_test_s = apply_standardize(X_test, means, stds)

    knn_scaled = KNN(k=5, task="classification")
    knn_scaled.fit(X_train_s, y_train)
    acc_scaled = accuracy(y_test, knn_scaled.predict(X_test_s))

    print(f"  特征：年龄（范围约 10-70），收入（范围约 10k-90k）")
    print()
    print(f"  未缩放：准确率 = {acc_raw:.4f}")
    print(f"  缩放后：准确率 = {acc_scaled:.4f}")
    print()

    query = X_test[0]
    query_s = X_test_s[0]

    dists_raw = [(l2_distance(query, X_train[i]), i) for i in range(5)]
    dists_raw.sort()
    dists_scaled = [(l2_distance(query_s, X_train_s[i]), i) for i in range(5)]
    dists_scaled.sort()

    print(f"  第一个测试点的距离示例：")
    print(f"  未缩放：{[round(d, 1) for d, _ in dists_raw]}")
    print(f"  缩放后：{[round(d, 4) for d, _ in dists_scaled]}")
    print()
    print("  未缩放：收入占主导（数万对比数十）。")
    print("  缩放后：两个特征对距离的贡献相等。")
    print()


def demo_lazy_vs_eager():
    print("=" * 65)
    print("惰性学习与急切学习：耗时对比")
    print("=" * 65)
    print()

    import time

    random.seed(42)
    sizes = [100, 500, 1000, 5000]

    print(f"  {'N':>6s}  {'KNN 训练':>12s}  {'KNN 预测':>14s}  {'总计':>10s}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 14}  {'-' * 10}")

    for n in sizes:
        X = [[random.gauss(0, 1) for _ in range(5)] for _ in range(n)]
        y = [random.choice([0, 1]) for _ in range(n)]

        n_test = min(50, n // 5)
        X_test_local = [[random.gauss(0, 1) for _ in range(5)] for _ in range(n_test)]

        knn = KNN(k=5, task="classification")

        start = time.time()
        knn.fit(X, y)
        train_time = time.time() - start

        start = time.time()
        knn.predict(X_test_local)
        pred_time = time.time() - start

        total = train_time + pred_time
        print(f"  {n:>6d}  {train_time:>12.6f}s  {pred_time:>14.6f}s  {total:>10.6f}s")

    print()
    print("  KNN 训练复杂度为 O(1)：只需存储数据。")
    print("  KNN 预测复杂度为 O(n*d) 每次查询：要计算所有距离。")
    print("  对于急切学习器（神经网络），情况正好相反。")
    print()


def demo_minkowski_family():
    print("=" * 65)
    print("Minkowski 距离族")
    print("=" * 65)
    print()

    a = [1.0, 2.0, 3.0]
    b = [4.0, 0.0, 6.0]

    p_values = [1, 1.5, 2, 3, 5, 10, float("inf")]
    print(f"  a = {a}")
    print(f"  b = {b}")
    print()
    print(f"  {'p':>8s}  {'距离':>12s}  {'名称':>15s}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 15}")

    for p in p_values:
        d = minkowski_distance(a, b, p)
        if p == 1:
            name = "Manhattan (L1)"
        elif p == 2:
            name = "Euclidean (L2)"
        elif p == float("inf"):
            name = "Chebyshev (Linf)"
        else:
            name = f"Lp (p={p})"
        p_str = "inf" if p == float("inf") else str(p)
        print(f"  {p_str:>8s}  {d:>12.4f}  {name:>15s}")

    print()
    print("  随着 p 增大，距离由最大的分量差决定。")
    print("  始终满足 L-inf <= L2 <= L1。")
    print()


def demo_k_selection():
    print("=" * 65)
    print("选择 K：交叉验证方法")
    print("=" * 65)
    print()

    X, y = generate_classification_data(300, seed=42)

    n = len(X)
    random.seed(42)
    indices = list(range(n))
    random.shuffle(indices)

    n_folds = 5
    fold_size = n // n_folds

    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]

    print(f"  在 {n} 个样本上进行 {n_folds} 折交叉验证")
    print()
    print(f"  {'K':>6s}  {'平均准确率':>10s}  {'准确率标准差':>10s}  {'可视化':>20s}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 20}")

    best_k = 1
    best_mean = 0.0

    for k in k_values:
        fold_accs = []

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size
            val_idx = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]

            X_tr = [X[i] for i in train_idx]
            y_tr = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]

            knn = KNN(k=k, task="classification")
            knn.fit(X_tr, y_tr)
            acc_val = accuracy(y_val, knn.predict(X_val))
            fold_accs.append(acc_val)

        mean_acc = sum(fold_accs) / len(fold_accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs)) ** 0.5

        bar_len = int(mean_acc * 20)
        bar = "#" * bar_len

        if mean_acc > best_mean:
            best_mean = mean_acc
            best_k = k

        print(f"  {k:>6d}  {mean_acc:>10.4f}  {std_acc:>10.4f}  {bar}")

    print()
    print(f"  最佳 K = {best_k}，平均准确率 = {best_mean:.4f}")
    print()


def print_summary():
    print()
    print("=" * 65)
    print("总结")
    print("=" * 65)
    print()
    print("  1. KNN 是惰性学习：训练零成本，所有工作都在预测阶段完成。")
    print("  2. K 控制偏差-方差：K 过小会过拟合，K 太大会欠拟合。")
    print("  3. 距离度量的选择很重要。默认用 L2，文本用余弦。")
    print("  4. 务必缩放特征。未缩放的特征会扭曲距离。")
    print("  5. 加权 KNN 通过降低远处邻居的权重来减少对 K 的敏感度。")
    print("  6. 维度灾难：超过约 20-50 维后 KNN 性能下降。")
    print("  7. KD-tree 在低维下加速搜索。中等维度可用 Ball Tree。")
    print("  8. KNN 正是向量数据库和 RAG 检索背后的同一算法。")
    print()


if __name__ == "__main__":
    demo_basic_knn()
    demo_distance_metrics()
    demo_weighted_knn()
    demo_regression()
    demo_minkowski_family()
    demo_curse_of_dimensionality()
    demo_scaling_importance()
    demo_kdtree()
    demo_lazy_vs_eager()
    demo_k_selection()
    print_summary()
