import math
import random


def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )


def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels)
        + (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity


def variance_reduction(parent_values, left_values, right_values):
    if len(left_values) == 0 or len(right_values) == 0:
        return 0.0
    n = len(parent_values)
    parent_var = _variance(parent_values)
    child_var = (
        (len(left_values) / n) * _variance(left_values)
        + (len(right_values) / n) * _variance(right_values)
    )
    return parent_var - child_var


def _variance(values):
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _mean(values):
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def majority_vote(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None, task="classification"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.task = task
        self.tree = None
        self.feature_importances_ = None
        self.n_features = 0
        self.n_samples = 0

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def _build(self, X, y, depth):
        if self.task == "classification":
            all_same = len(set(y)) == 1
        else:
            all_same = len(set(y)) == 1

        if all_same:
            return {"leaf": True, "value": y[0] if self.task == "classification" else _mean(y)}

        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)

        if len(y) < self.min_samples_split:
            return self._make_leaf(y)

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain <= 0:
            return self._make_leaf(y)

        left_X, left_y, right_X, right_y = self._split_data(
            X, y, best_feature, best_threshold
        )

        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return self._make_leaf(y)

        weight = len(y) / self.n_samples
        self.feature_importances_[best_feature] += weight * best_gain

        left_child = self._build(left_X, left_y, depth + 1)
        right_child = self._build(right_X, right_y, depth + 1)

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_child,
            "right": right_child,
        }

    def _make_leaf(self, y):
        if self.task == "classification":
            return {"leaf": True, "value": majority_vote(y)}
        else:
            return {"leaf": True, "value": _mean(y)}

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = -1.0

        if self.max_features is None:
            feature_indices = list(range(self.n_features))
        elif self.max_features == "sqrt":
            k = max(1, int(math.sqrt(self.n_features)))
            feature_indices = random.sample(range(self.n_features), k)
        elif isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features 作为整数时必须至少为 1")
            k = min(self.max_features, self.n_features)
            feature_indices = random.sample(range(self.n_features), k)
        else:
            feature_indices = list(range(self.n_features))

        for feature_idx in feature_indices:
            values = sorted(set(X[i][feature_idx] for i in range(len(X))))
            if len(values) <= 1:
                continue

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2.0
                left_y = [y[j] for j in range(len(X)) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                if self.task == "classification":
                    gain = information_gain(y, left_y, right_y, self.criterion)
                else:
                    gain = variance_reduction(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _split_data(self, X, y, feature, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree
        if node["leaf"]:
            print(f"{indent}预测: {node['value']}")
            return
        print(f"{indent}特征 {node['feature']} <= {node['threshold']:.4f}?")
        print(f"{indent}  是:")
        self.print_tree(node["left"], indent + "    ")
        print(f"{indent}  否:")
        self.print_tree(node["right"], indent + "    ")


class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,
                 min_samples_split=2, max_features="sqrt",
                 criterion="gini", task="classification"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.task = task
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n = len(X)
        for _ in range(self.n_trees):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
                task=self.task,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            if self.task == "classification":
                votes = {}
                for preds in all_preds:
                    v = preds[i]
                    votes[v] = votes.get(v, 0) + 1
                predictions.append(max(votes, key=votes.get))
            else:
                predictions.append(
                    sum(preds[i] for preds in all_preds) / len(all_preds)
                )
        return predictions

    def feature_importances(self):
        n_features = self.trees[0].n_features
        importances = [0.0] * n_features
        for tree in self.trees:
            for j in range(n_features):
                importances[j] += tree.feature_importances_[j]
        total = sum(importances)
        if total > 0:
            importances = [imp / total for imp in importances]
        return importances


def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def generate_classification_data(n_samples=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        noise = random.gauss(0, 0.3)
        if x1 ** 2 + x2 ** 2 + noise < 3:
            label = 0
        elif x1 + x2 + noise > 1:
            label = 1
        else:
            label = 2
        X.append([x1, x2])
        y.append(label)
    return X, y


def generate_regression_data(n_samples=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x = random.uniform(-3, 3)
        target = math.sin(x) * x + random.gauss(0, 0.2)
        X.append([x])
        y.append(target)
    return X, y


def train_test_split(X, y, test_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, y_train, X_test, y_test


def demo_split_criteria():
    print("=" * 65)
    print("划分标准：基尼 vs 熵")
    print("=" * 65)
    print()

    test_cases = [
        ("纯节点 [A,A,A,A]", ["A", "A", "A", "A"]),
        ("均衡 [A,A,B,B]", ["A", "A", "B", "B"]),
        ("不均衡 [A,A,A,B]", ["A", "A", "A", "B"]),
        ("三类 [A,A,B,C]", ["A", "A", "B", "C"]),
        ("均匀四类", ["A", "B", "C", "D"]),
    ]

    print(f"  {'分布':<30s} {'Gini':>8s} {'熵':>8s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8}")
    for name, labels in test_cases:
        g = gini_impurity(labels)
        e = entropy(labels)
        print(f"  {name:<30s} {g:>8.4f} {e:>8.4f}")

    print()
    print("  两种度量一致：纯节点为 0，均衡时最大。")
    print("  对于多分类，熵比基尼增长略快。")
    print()


def demo_information_gain():
    print("=" * 65)
    print("信息增益：选择最佳划分")
    print("=" * 65)
    print()

    parent = ["cat", "cat", "cat", "cat", "dog", "dog", "dog",
              "bird", "bird", "bird"]

    splits = [
        ("特征 A: [cat,cat,cat,dog] | [cat,dog,dog,bird,bird,bird]",
         ["cat", "cat", "cat", "dog"],
         ["cat", "dog", "dog", "bird", "bird", "bird"]),
        ("特征 B: [cat,cat,cat,cat] | [dog,dog,dog,bird,bird,bird]",
         ["cat", "cat", "cat", "cat"],
         ["dog", "dog", "dog", "bird", "bird", "bird"]),
        ("特征 C: [cat,cat,dog,bird] | [cat,cat,dog,dog,bird,bird]",
         ["cat", "cat", "dog", "bird"],
         ["cat", "cat", "dog", "dog", "bird", "bird"]),
    ]

    print(f"  父节点: {parent}")
    print(f"  父节点 Gini: {gini_impurity(parent):.4f}")
    print(f"  父节点熵: {entropy(parent):.4f}")
    print()

    print(f"  {'划分':<55s} {'IG(Gini)':>10s} {'IG(熵)':>12s}")
    print(f"  {'-' * 55} {'-' * 10} {'-' * 12}")

    for name, left, right in splits:
        ig_gini = information_gain(parent, left, right, "gini")
        ig_ent = information_gain(parent, left, right, "entropy")
        print(f"  {name:<55s} {ig_gini:>10.4f} {ig_ent:>12.4f}")

    print()
    print("  特征 B 完美区分了 cat。信息增益最高。")
    print()


def demo_decision_tree():
    print("=" * 65)
    print("决策树：分类")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集：{len(X)} 个样本，2 个特征，3 个类别")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()

    depths = [1, 2, 3, 5, 10, None]
    print(f"  {'最大深度':>10s}  {'训练准确率':>10s}  {'测试准确率':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for d in depths:
        tree = DecisionTree(max_depth=d, criterion="gini")
        tree.fit(X_train, y_train)
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        train_acc = accuracy(y_train, train_pred)
        test_acc = accuracy(y_test, test_pred)
        d_str = str(d) if d is not None else "None"
        print(f"  {d_str:>10s}  {train_acc:>10.4f}  {test_acc:>10.4f}")

    print()
    print("  浅树会欠拟合，深树会过拟合。")
    print("  最佳深度介于两者之间。")
    print()

    tree = DecisionTree(max_depth=3, criterion="gini")
    tree.fit(X_train, y_train)
    print("  树结构（max_depth=3）：")
    tree.print_tree()
    print()


def demo_random_forest():
    print("=" * 65)
    print("随机森林：集成的力量")
    print("=" * 65)
    print()

    random.seed(42)
    X, y = generate_classification_data(300, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集：{len(X)} 个样本，2 个特征，3 个类别")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()

    tree_counts = [1, 3, 5, 10, 25, 50, 100]
    print(f"  {'树数量':>8s}  {'训练准确率':>10s}  {'测试准确率':>10s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}")

    for n in tree_counts:
        rf = RandomForest(n_trees=n, max_depth=5, criterion="gini")
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)
        train_acc = accuracy(y_train, train_pred)
        test_acc = accuracy(y_test, test_pred)
        print(f"  {n:>8d}  {train_acc:>10.4f}  {test_acc:>10.4f}")

    print()
    print("  树越多，泛化能力越好，但收益会逐渐递减。")
    print("  测试准确率会趋于稳定，但不会下降。")
    print()


def demo_feature_importance():
    print("=" * 65)
    print("特征重要性")
    print("=" * 65)
    print()

    random.seed(42)
    n = 200
    X = []
    y = []
    for _ in range(n):
        important1 = random.uniform(-2, 2)
        important2 = random.uniform(-2, 2)
        noise1 = random.gauss(0, 1)
        noise2 = random.gauss(0, 1)
        label = 1 if important1 + important2 > 0 else 0
        X.append([important1, important2, noise1, noise2])
        y.append(label)

    feature_names = ["important_1", "important_2", "noise_1", "noise_2"]

    rf = RandomForest(n_trees=50, max_depth=5)
    rf.fit(X, y)
    importances = rf.feature_importances()

    print(f"  目标：feature_0 + feature_1 > 0 时为 1，否则为 0")
    print(f"  特征 2 和特征 3 是纯噪声。")
    print()
    print(f"  {'特征':<15s}  {'重要性':>12s}")
    print(f"  {'-' * 15}  {'-' * 12}")
    for name, imp in sorted(zip(feature_names, importances),
                            key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"  {name:<15s}  {imp:>12.4f}  {bar}")

    print()
    print("  随机森林正确识别出了哪些特征真正重要。")
    print()


def demo_regression_tree():
    print("=" * 65)
    print("回归树：分段常数近似")
    print("=" * 65)
    print()

    X, y = generate_regression_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    depths = [1, 2, 3, 5, 10]
    print(f"  目标：y = sin(x) * x + 噪声")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()

    print(f"  {'最大深度':>10s}  {'训练 MSE':>10s}  {'测试 MSE':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for d in depths:
        tree = DecisionTree(max_depth=d, task="regression")
        tree.fit(X_train, y_train)
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        train_mse = sum((a - b) ** 2 for a, b in zip(y_train, train_pred)) / len(y_train)
        test_mse = sum((a - b) ** 2 for a, b in zip(y_test, test_pred)) / len(y_test)
        print(f"  {d:>10d}  {train_mse:>10.4f}  {test_mse:>10.4f}")

    print()

    rf = RandomForest(n_trees=50, max_depth=5, task="regression")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mse = sum((a - b) ** 2 for a, b in zip(y_test, rf_pred)) / len(y_test)
    print(f"  随机森林（50 棵树，深度=5）测试 MSE：{rf_mse:.4f}")
    print()
    print("  随机森林对多个分段预测取平均，使输出更加平滑。")
    print()


def demo_gini_vs_entropy():
    print("=" * 65)
    print("基尼 vs 熵：二者会产生分歧吗？")
    print("=" * 65)
    print()

    random.seed(42)
    X, y = generate_classification_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    for depth in [3, 5, 10]:
        tree_gini = DecisionTree(max_depth=depth, criterion="gini")
        tree_entropy = DecisionTree(max_depth=depth, criterion="entropy")
        tree_gini.fit(X_train, y_train)
        tree_entropy.fit(X_train, y_train)

        acc_gini = accuracy(y_test, tree_gini.predict(X_test))
        acc_entropy = accuracy(y_test, tree_entropy.predict(X_test))

        print(f"  深度={depth:<4d}  Gini 准确率：{acc_gini:.4f}  "
              f"熵准确率：{acc_entropy:.4f}  "
              f"差异：{abs(acc_gini - acc_entropy):.4f}")

    print()
    print("  实践中，Gini 和熵生成的树几乎相同。")
    print("  Gini 略快一些，因为不需要计算对数。")
    print()


def demo_single_tree_vs_forest():
    print("=" * 65)
    print("单棵树 vs 随机森林：稳定性")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)

    print("  在略有不同的数据子集上训练 5 棵单独的树：")
    single_accs = []
    for trial in range(5):
        random.seed(trial * 10)
        indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
        X_sub = [X[i] for i in indices]
        y_sub = [y[i] for i in indices]
        X_tr, y_tr, X_te, y_te = train_test_split(X_sub, y_sub, seed=trial)
        tree = DecisionTree(max_depth=5)
        tree.fit(X_tr, y_tr)
        acc = accuracy(y_te, tree.predict(X_te))
        single_accs.append(acc)
        print(f"    试验 {trial + 1}：准确率 = {acc:.4f}")

    print()
    print("  在相同数据子集上训练 5 个随机森林：")
    forest_accs = []
    for trial in range(5):
        random.seed(trial * 10)
        indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
        X_sub = [X[i] for i in indices]
        y_sub = [y[i] for i in indices]
        X_tr, y_tr, X_te, y_te = train_test_split(X_sub, y_sub, seed=trial)
        rf = RandomForest(n_trees=30, max_depth=5)
        rf.fit(X_tr, y_tr)
        acc = accuracy(y_te, rf.predict(X_te))
        forest_accs.append(acc)
        print(f"    试验 {trial + 1}：准确率 = {acc:.4f}")

    single_std = (sum((a - sum(single_accs) / 5) ** 2 for a in single_accs) / 5) ** 0.5
    forest_std = (sum((a - sum(forest_accs) / 5) ** 2 for a in forest_accs) / 5) ** 0.5

    print()
    print(f"  单棵树：   均值 = {sum(single_accs)/5:.4f}，"
          f"标准差 = {single_std:.4f}")
    print(f"  随机森林：均值 = {sum(forest_accs)/5:.4f}，"
          f"标准差 = {forest_std:.4f}")
    print()
    print("  面对数据扰动，随机森林更加稳定（方差更低）。")
    print()


def print_summary():
    print()
    print("=" * 65)
    print("总结")
    print("=" * 65)
    print()
    print("  1. 决策树通过最大化信息增益来划分数据。")
    print("  2. Gini 不纯度和熵产生的划分几乎相同。")
    print("  3. 单棵树并不稳定，数据的微小变化就可能生成不同的树。")
    print("  4. 随机森林对多棵树取平均，获得稳定且强大的预测。")
    print("  5. Bagging 与特征随机化可以降低树之间的相关性。")
    print("  6. 特征重要性可以从不纯度下降中自然得到。")
    print("  7. 在表格数据上，树模型通常优于神经网络。")
    print()


if __name__ == "__main__":
    demo_split_criteria()
    demo_information_gain()
    demo_decision_tree()
    demo_gini_vs_entropy()
    demo_random_forest()
    demo_feature_importance()
    demo_regression_tree()
    demo_single_tree_vs_forest()
    print_summary()
