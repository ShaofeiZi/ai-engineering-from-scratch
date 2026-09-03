import math
import random


def l1_norm(x):
    return sum(abs(xi) for xi in x)


def l2_norm(x):
    return math.sqrt(sum(xi ** 2 for xi in x))


def lp_norm(x, p):
    if p == float('inf'):
        return max(abs(xi) for xi in x)
    return sum(abs(xi) ** p for xi in x) ** (1 / p)


def linf_norm(x):
    return max(abs(xi) for xi in x)


def l1_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def l2_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def lp_distance(a, b, p):
    diff = [ai - bi for ai, bi in zip(a, b)]
    return lp_norm(diff, p)


def linf_distance(a, b):
    return max(abs(ai - bi) for ai, bi in zip(a, b))


def dot_product(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def cosine_similarity(a, b):
    dot = dot_product(a, b)
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_distance(a, b):
    return 1.0 - cosine_similarity(a, b)


def mahalanobis_distance(x, y, cov_matrix):
    n = len(x)
    diff = [xi - yi for xi, yi in zip(x, y)]

    inv_cov = invert_matrix(cov_matrix)

    temp = [0.0] * n
    for i in range(n):
        for j in range(n):
            temp[i] += diff[j] * inv_cov[j][i]

    result = sum(temp[i] * diff[i] for i in range(n))
    return math.sqrt(max(0, result))


def invert_matrix(matrix):
    n = len(matrix)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(augmented[row][col]) > abs(augmented[max_row][col]):
                max_row = row
        augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

        pivot = augmented[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("矩阵奇异或接近奇异")
        for j in range(2 * n):
            augmented[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                for j in range(2 * n):
                    augmented[row][j] -= factor * augmented[col][j]

    return [row[n:] for row in augmented]


def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def jaccard_distance(set_a, set_b):
    return 1.0 - jaccard_similarity(set_a, set_b)


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    return dp[m][n]


def kl_divergence(p, q):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * math.log(pi / qi)
    return total


def wasserstein_1d(p, q):
    assert len(p) == len(q), "分布必须具有相同的分箱数量"
    n = len(p)
    cdf_p = [0.0] * n
    cdf_q = [0.0] * n

    cdf_p[0] = p[0]
    cdf_q[0] = q[0]
    for i in range(1, n):
        cdf_p[i] = cdf_p[i - 1] + p[i]
        cdf_q[i] = cdf_q[i - 1] + q[i]

    return sum(abs(cdf_p[i] - cdf_q[i]) for i in range(n))


def compute_covariance(data):
    n = len(data)
    d = len(data[0])
    means = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
    centered = [[data[i][j] - means[j] for j in range(d)] for i in range(n)]
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / (n - 1)
    return cov


def normalize_vector(v):
    norm = l2_norm(v)
    if norm == 0:
        return v[:]
    return [vi / norm for vi in v]


def find_nearest_neighbor(query, dataset, distance_fn, **kwargs):
    best_idx = 0
    best_dist = float('inf')
    for i, point in enumerate(dataset):
        d = distance_fn(query, point, **kwargs)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist


def find_k_nearest(query, dataset, distance_fn, k=5, **kwargs):
    distances = []
    for i, point in enumerate(dataset):
        d = distance_fn(query, point, **kwargs)
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def demo_norms():
    print("=" * 65)
    print("范数：衡量向量的大小")
    print("=" * 65)

    vectors = [
        ("(3, 4)", [3, 4]),
        ("(1, 1, 1, 1)", [1, 1, 1, 1]),
        ("(5, 0, 0)", [5, 0, 0]),
        ("(1, 2, 3, 4, 5)", [1, 2, 3, 4, 5]),
    ]

    print(f"  {'向量':<20s} {'L1':>8s} {'L2':>8s} {'L3':>8s} {'L-inf':>8s}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for name, v in vectors:
        print(f"  {name:<20s} {l1_norm(v):>8.3f} {l2_norm(v):>8.3f} "
              f"{lp_norm(v, 3):>8.3f} {linf_norm(v):>8.3f}")

    print()
    print("  注意：L-inf <= L2 <= L1 始终成立。")
    print()


def demo_distances():
    print("=" * 65)
    print("两点之间的距离")
    print("=" * 65)

    a = [1, 2, 3]
    b = [4, 0, 6]

    print(f"  A = {a}")
    print(f"  B = {b}")
    print()
    print(f"  L1（曼哈顿距离）:   {l1_distance(a, b):.4f}")
    print(f"  L2（欧几里得距离）: {l2_distance(a, b):.4f}")
    print(f"  L3:               {lp_distance(a, b, 3):.4f}")
    print(f"  L-inf（切比雪夫距离）:{linf_distance(a, b):.4f}")
    print(f"  余弦距离:  {cosine_distance(a, b):.4f}")
    print(f"  余弦相似度:{cosine_similarity(a, b):.4f}")
    print(f"  点积:      {dot_product(a, b):.4f}")
    print()


def demo_cosine_vs_dot():
    print("=" * 65)
    print("余弦相似度 vs 点积")
    print("=" * 65)

    a = [1, 2, 3]
    b = [2, 4, 6]
    c = [3, 1, 0]

    print(f"  A = {a}")
    print(f"  B = {b}  （A 放大 2 倍）")
    print(f"  C = {c}  （方向不同）")
    print()
    print(f"  {'配对':<10s} {'余弦':>10s} {'点积':>10s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'A vs B':<10s} {cosine_similarity(a, b):>10.4f} {dot_product(a, b):>10.4f}")
    print(f"  {'A vs C':<10s} {cosine_similarity(a, c):>10.4f} {dot_product(a, c):>10.4f}")
    print(f"  {'B vs C':<10s} {cosine_similarity(b, c):>10.4f} {dot_product(b, c):>10.4f}")
    print()
    print("  余弦相似度认为 A 和 B 完全相同（方向一致）。")
    print("  点积认为 B 更相似，因为其模更大。")
    print()

    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    c_norm = normalize_vector(c)

    print("  L2 归一化后：")
    print(f"  {'配对':<10s} {'余弦':>10s} {'点积':>10s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'A vs B':<10s} {cosine_similarity(a_norm, b_norm):>10.4f} {dot_product(a_norm, b_norm):>10.4f}")
    print(f"  {'A vs C':<10s} {cosine_similarity(a_norm, c_norm):>10.4f} {dot_product(a_norm, c_norm):>10.4f}")
    print()
    print("  归一化后，余弦相似度与点积完全相同。")
    print()


def demo_mahalanobis():
    print("=" * 65)
    print("马氏距离")
    print("=" * 65)

    random.seed(42)
    n = 200
    data = []
    for _ in range(n):
        x = random.gauss(0, 3)
        y = 0.8 * x + random.gauss(0, 1)
        data.append([x, y])

    cov = compute_covariance(data)
    mean = [sum(d[0] for d in data) / n, sum(d[1] for d in data) / n]

    point_along = [mean[0] + 3, mean[1] + 0.8 * 3]
    point_perp = [mean[0] + 1, mean[1] - 3]

    l2_along = l2_distance(mean, point_along)
    l2_perp = l2_distance(mean, point_perp)
    mah_along = mahalanobis_distance(mean, point_along, cov)
    mah_perp = mahalanobis_distance(mean, point_perp, cov)

    print(f"  数据：{n} 个点，特征之间相关（r ~ 0.8）")
    print(f"  均值：({mean[0]:.2f}, {mean[1]:.2f})")
    print(f"  协方差：[[{cov[0][0]:.2f}, {cov[0][1]:.2f}], [{cov[1][0]:.2f}, {cov[1][1]:.2f}]]")
    print()
    print(f"  沿相关轴方向的点：  {[round(x, 2) for x in point_along]}")
    print(f"    到均值的 L2 距离：  {l2_along:.4f}")
    print(f"    马氏距离：         {mah_along:.4f}")
    print()
    print(f"  垂直于轴方向的点：   {[round(x, 2) for x in point_perp]}")
    print(f"    到均值的 L2 距离：  {l2_perp:.4f}")
    print(f"    马氏距离：         {mah_perp:.4f}")
    print()
    print("  L2 认为两个点到均值的距离相近。")
    print("  马氏距离则能正确指出，在数据的协方差结构下，")
    print("  垂直方向上的点更加异常。")
    print()


def demo_jaccard():
    print("=" * 65)
    print("Jaccard 相似度（集合）")
    print("=" * 65)

    pairs = [
        ({"cat", "dog", "fish"}, {"cat", "bird", "fish", "snake"}),
        ({"python", "java", "rust"}, {"python", "java", "rust"}),
        ({"a", "b", "c"}, {"d", "e", "f"}),
        ({"ml", "ai", "data"}, {"ml", "ai", "data", "ops", "cloud"}),
    ]

    for a, b in pairs:
        j = jaccard_similarity(a, b)
        print(f"  A = {sorted(a)}")
        print(f"  B = {sorted(b)}")
        print(f"  Jaccard 相似度: {j:.4f}")
        print(f"  Jaccard 距离:   {1 - j:.4f}")
        print()


def demo_edit_distance():
    print("=" * 65)
    print("编辑距离（Levenshtein）")
    print("=" * 65)

    pairs = [
        ("kitten", "sitting"),
        ("sunday", "saturday"),
        ("hello", "hello"),
        ("", "abc"),
        ("algorithm", "altruistic"),
        ("python", "pytorch"),
    ]

    for s1, s2 in pairs:
        d = edit_distance(s1, s2)
        print(f"  '{s1}' -> '{s2}'：  距离 = {d}")

    print()


def demo_kl_divergence():
    print("=" * 65)
    print("KL 散度（非对称）")
    print("=" * 65)

    p = [0.9, 0.1]
    q = [0.5, 0.5]

    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)

    print(f"  P = {p}")
    print(f"  Q = {q}")
    print(f"  KL(P || Q) = {kl_pq:.4f} nats")
    print(f"  KL(Q || P) = {kl_qp:.4f} nats")
    print(f"  差值: {abs(kl_pq - kl_qp):.4f}")
    print(f"  KL 散度不是距离度量。")
    print()

    p2 = [0.25, 0.25, 0.25, 0.25]
    q2 = [0.1, 0.1, 0.1, 0.7]

    print(f"  P = {p2}")
    print(f"  Q = {q2}")
    print(f"  KL(P || Q) = {kl_divergence(p2, q2):.4f} nats")
    print(f"  KL(Q || P) = {kl_divergence(q2, p2):.4f} nats")
    print()


def demo_wasserstein():
    print("=" * 65)
    print("Wasserstein 距离（推土机距离）")
    print("=" * 65)

    cases = [
        ("完全相同",
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]),
        ("右移 1",
         [0.5, 0.5, 0.0, 0.0],
         [0.0, 0.5, 0.5, 0.0]),
        ("右移 2",
         [0.5, 0.5, 0.0, 0.0],
         [0.0, 0.0, 0.5, 0.5]),
        ("两端相反",
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]),
        ("分散 vs 集中",
         [0.25, 0.25, 0.25, 0.25],
         [0.0, 0.0, 0.0, 1.0]),
    ]

    for name, p, q in cases:
        w = wasserstein_1d(p, q)
        kl = kl_divergence(p, q)
        kl_str = f"{kl:.4f}" if kl != float('inf') else "inf"
        print(f"  {name}")
        print(f"    P = {p}")
        print(f"    Q = {q}")
        print(f"    Wasserstein: {w:.4f}    KL: {kl_str}")
        print()

    print("  当分布不重叠时（此时 KL 会趋于无穷），")
    print("  Wasserstein 仍能给出有限且有意义的结果。")
    print()


def demo_different_neighbors():
    print("=" * 65)
    print("相同数据，不同度量，不同最近邻")
    print("=" * 65)

    random.seed(123)
    n_points = 8
    dim = 5

    dataset = []
    for i in range(n_points):
        if i < 3:
            point = [random.gauss(0, 1) for _ in range(dim)]
        elif i < 6:
            base = [random.gauss(0, 0.5) for _ in range(dim)]
            base[0] *= 5
            point = base
        else:
            point = [random.gauss(3, 0.3) for _ in range(dim)]
        dataset.append(point)

    query = [1.0, 0.5, -0.5, 1.0, 0.2]

    print(f"  查询点: {[round(x, 2) for x in query]}")
    print()
    print(f"  {'点':<8s} {'L1':>8s} {'L2':>8s} {'余弦':>8s} {'L-inf':>8s}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    results = {"L1": [], "L2": [], "Cosine": [], "L-inf": []}

    for i, point in enumerate(dataset):
        d_l1 = l1_distance(query, point)
        d_l2 = l2_distance(query, point)
        d_cos = cosine_distance(query, point)
        d_linf = linf_distance(query, point)

        results["L1"].append((i, d_l1))
        results["L2"].append((i, d_l2))
        results["Cosine"].append((i, d_cos))
        results["L-inf"].append((i, d_linf))

        print(f"  P{i:<6d} {d_l1:>8.3f} {d_l2:>8.3f} {d_cos:>8.4f} {d_linf:>8.3f}")

    print()
    print("  各度量下的最近邻：")
    for metric_name, dists in results.items():
        best = min(dists, key=lambda x: x[1])
        print(f"    {metric_name:<8s}: 点 {best[0]}（距离 = {best[1]:.4f}）")

    l1_best = min(results["L1"], key=lambda x: x[1])[0]
    l2_best = min(results["L2"], key=lambda x: x[1])[0]
    cos_best = min(results["Cosine"], key=lambda x: x[1])[0]
    linf_best = min(results["L-inf"], key=lambda x: x[1])[0]

    all_same = (l1_best == l2_best == cos_best == linf_best)
    if not all_same:
        print()
        print("  各度量对哪个点最近存在分歧。")
        print("  距离函数决定了相似性的定义。")
    print()


def demo_embedding_search():
    print("=" * 65)
    print("嵌入相似度搜索")
    print("=" * 65)

    random.seed(77)
    dim = 64

    documents = [
        "machine learning algorithms",
        "deep neural networks",
        "natural language processing",
        "computer vision models",
        "reinforcement learning agents",
        "database query optimization",
        "web server configuration",
        "network security protocols",
    ]

    embeddings = []
    for i, doc in enumerate(documents):
        base = [random.gauss(0, 1) for _ in range(dim)]
        if i < 5:
            for j in range(10):
                base[j] += 2.0
        else:
            for j in range(10, 20):
                base[j] += 2.0
        if i in [0, 1]:
            for j in range(20, 25):
                base[j] += 1.5
        embeddings.append(base)

    query_embedding = embeddings[0][:]
    noise = [random.gauss(0, 0.3) for _ in range(dim)]
    query_embedding = [q + n for q, n in zip(query_embedding, noise)]

    print(f"  查询: '{documents[0]}'（含噪声）")
    print(f"  嵌入维度: {dim}")
    print()

    cosine_scores = []
    l2_scores = []
    dot_scores = []

    for i in range(len(documents)):
        cos = cosine_similarity(query_embedding, embeddings[i])
        l2 = l2_distance(query_embedding, embeddings[i])
        dp = dot_product(query_embedding, embeddings[i])
        cosine_scores.append((i, cos))
        l2_scores.append((i, l2))
        dot_scores.append((i, dp))

    cosine_ranked = sorted(cosine_scores, key=lambda x: -x[1])
    l2_ranked = sorted(l2_scores, key=lambda x: x[1])
    dot_ranked = sorted(dot_scores, key=lambda x: -x[1])

    print(f"  {'排名':<6s} {'余弦':<35s} {'L2':<35s} {'点积':<35s}")
    print(f"  {'-' * 6} {'-' * 35} {'-' * 35} {'-' * 35}")
    for rank in range(len(documents)):
        ci, cs = cosine_ranked[rank]
        li, ls = l2_ranked[rank]
        di, ds = dot_ranked[rank]
        cos_str = f"{documents[ci][:25]:<25s} ({cs:.3f})"
        l2_str = f"{documents[li][:25]:<25s} ({ls:.1f})"
        dot_str = f"{documents[di][:25]:<25s} ({ds:.1f})"
        print(f"  {rank + 1:<6d} {cos_str:<35s} {l2_str:<35s} {dot_str:<35s}")

    print()
    print("  余弦相似度关注方向（主题相似性）。")
    print("  L2 距离对模长差异敏感。")
    print("  点积则同时融合方向与模长。")
    print()


def demo_knn_classification():
    print("=" * 65)
    print("KNN 分类：距离度量改变预测结果")
    print("=" * 65)

    random.seed(99)

    training_data = [
        ([1.0, 5.0], "A"),
        ([1.5, 4.5], "A"),
        ([2.0, 4.0], "A"),
        ([5.0, 1.0], "B"),
        ([4.5, 1.5], "B"),
        ([4.0, 2.0], "B"),
        ([3.0, 3.0], "C"),
        ([3.5, 2.5], "C"),
        ([2.5, 3.5], "C"),
    ]

    query = [2.8, 2.8]

    print(f"  查询: {query}")
    print(f"  训练集: {len(training_data)} 个点，3 个类别")
    print()

    k = 3
    for metric_name, dist_fn in [("L1", l1_distance), ("L2", l2_distance),
                                   ("Cosine", cosine_distance), ("L-inf", linf_distance)]:
        distances = []
        for point, label in training_data:
            d = dist_fn(query, point)
            distances.append((d, label, point))
        distances.sort(key=lambda x: x[0])

        neighbors = distances[:k]
        votes = {}
        for d, label, point in neighbors:
            votes[label] = votes.get(label, 0) + 1
        prediction = max(votes, key=votes.get)

        print(f"  度量: {metric_name}")
        for d, label, point in neighbors:
            print(f"    邻居: {point}  类别={label}  距离={d:.4f}")
        print(f"    预测（k={k}）: {prediction}")
        print()


def demo_regularization():
    print("=" * 65)
    print("L1 与 L2 正则化对权重的影响")
    print("=" * 65)

    random.seed(42)
    n_features = 10
    weights = [random.gauss(0, 2) for _ in range(n_features)]

    print(f"  原始权重: {[round(w, 3) for w in weights]}")
    print(f"  L1 范数: {l1_norm(weights):.4f}")
    print(f"  L2 范数: {l2_norm(weights):.4f}")
    print()

    lr = 0.1

    w_l1 = weights[:]
    for step in range(50):
        for i in range(n_features):
            grad = lr * (1 if w_l1[i] > 0 else (-1 if w_l1[i] < 0 else 0))
            w_l1[i] -= grad
            if abs(w_l1[i]) < lr:
                w_l1[i] = 0.0

    w_l2 = weights[:]
    for step in range(50):
        for i in range(n_features):
            grad = lr * 2 * w_l2[i]
            w_l2[i] -= grad

    print(f"  L1 正则化后（50 步）:")
    print(f"    权重: {[round(w, 3) for w in w_l1]}")
    print(f"    零值:   {sum(1 for w in w_l1 if w == 0.0)}/{n_features}")
    print(f"    L1 范数: {l1_norm(w_l1):.4f}")
    print()
    print(f"  L2 正则化后（50 步）:")
    print(f"    权重: {[round(w, 3) for w in w_l2]}")
    print(f"    零值:   {sum(1 for w in w_l2 if abs(w) < 1e-10)}/{n_features}")
    print(f"    L2 范数: {l2_norm(w_l2):.4f}")
    print()
    print("  L1 会将“较小”的权重精确地置零（稀疏性）。")
    print("  L2 会缩小所有权重，但都不会精确归零。")
    print()


def demo_norm_ordering():
    print("=" * 65)
    print("范数排序：L-inf <= L2 <= L1（始终成立）")
    print("=" * 65)

    random.seed(55)
    for trial in range(5):
        dim = random.randint(2, 10)
        a = [random.gauss(0, 5) for _ in range(dim)]
        b = [random.gauss(0, 5) for _ in range(dim)]

        d1 = l1_distance(a, b)
        d2 = l2_distance(a, b)
        dinf = linf_distance(a, b)

        holds = dinf <= d2 <= d1
        print(f"  维度={dim:>2d}  L1={d1:>8.3f}  L2={d2:>8.3f}  L-inf={dinf:>8.3f}  排序成立: {holds}")

    print()
    print("  对于任意 p1 < p2：||x||_p2 <= ||x||_p1")
    print("  p 值越大，越关注较少的（较大的）分量。")
    print()


if __name__ == "__main__":
    demo_norms()
    demo_distances()
    demo_cosine_vs_dot()
    demo_mahalanobis()
    demo_jaccard()
    demo_edit_distance()
    demo_kl_divergence()
    demo_wasserstein()
    demo_norm_ordering()
    demo_different_neighbors()
    demo_embedding_search()
    demo_knn_classification()
    demo_regularization()
