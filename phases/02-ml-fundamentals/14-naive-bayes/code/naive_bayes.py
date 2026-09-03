import numpy as np


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        if np.any(X < 0):
            raise ValueError("MultinomialNB 要求特征值为非负数")
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            total = counts.sum()
            self.feature_log_prob_[i] = np.log(counts / total)

        return self

    def predict_log_proba(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.means_ = None
        self.vars_ = None
        self.priors_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + self.var_smoothing
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self

    def _log_likelihood(self, X):
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            diff = X - self.means_[i]
            log_prob_features = (
                -0.5 * np.log(2 * np.pi * self.vars_[i])
                - 0.5 * (diff ** 2) / self.vars_[i]
            )
            log_proba[:, i] = log_prob_features.sum(axis=1) + np.log(self.priors_[i])

        return log_proba

    def predict(self, X):
        log_proba = self._log_likelihood(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def predict_proba(self, X):
        log_proba = self._log_likelihood(X)
        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def make_text_data(n_samples=1000, n_features=200, seed=42):
    rng = np.random.RandomState(seed)

    tech_words_weight = np.zeros(n_features)
    tech_words_weight[:40] = rng.uniform(3, 10, 40)
    tech_words_weight[40:80] = rng.uniform(0.5, 2, 40)
    tech_words_weight[80:] = rng.uniform(0.1, 1, 120)

    sports_words_weight = np.zeros(n_features)
    sports_words_weight[:40] = rng.uniform(0.1, 1, 40)
    sports_words_weight[40:80] = rng.uniform(0.5, 2, 40)
    sports_words_weight[80:120] = rng.uniform(3, 10, 40)
    sports_words_weight[120:] = rng.uniform(0.1, 1, 80)

    n_tech = n_samples // 2
    n_sports = n_samples - n_tech

    X_tech = rng.poisson(tech_words_weight, (n_tech, n_features)).astype(float)
    X_sports = rng.poisson(sports_words_weight, (n_sports, n_features)).astype(float)

    X = np.vstack([X_tech, X_sports])
    y = np.array([0] * n_tech + [1] * n_sports)

    shuffle_idx = rng.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]


def make_continuous_data(n_samples=300, seed=42):
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 3

    class_0 = rng.multivariate_normal(
        [5.0, 3.4, 1.4, 0.2],
        np.diag([0.12, 0.14, 0.03, 0.01]),
        n_per_class,
    )
    class_1 = rng.multivariate_normal(
        [5.9, 2.8, 4.3, 1.3],
        np.diag([0.27, 0.10, 0.22, 0.04]),
        n_per_class,
    )
    class_2 = rng.multivariate_normal(
        [6.6, 3.0, 5.6, 2.0],
        np.diag([0.40, 0.10, 0.30, 0.08]),
        n_per_class,
    )

    X = np.vstack([class_0, class_1, class_2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def print_separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_multinomial():
    print_separator("多项式朴素贝叶斯 -- 文本分类")

    X, y = make_text_data(n_samples=1200, n_features=200, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25, seed=42)

    print(f"训练样本数: {X_train.shape[0]}")
    print(f"测试样本数: {X_test.shape[0]}")
    print(f"特征数（词）: {X_train.shape[1]}")
    print(f"类别:          科技 (0), 体育 (1)")
    print()

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)

    train_acc = mnb.score(X_train, y_train)
    test_acc = mnb.score(X_test, y_test)
    print(f"从零实现的 MultinomialNB:")
    print(f"  训练准确率: {train_acc:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")

    proba = mnb.predict_proba(X_test[:5])
    print(f"\n预测概率（前 5 个样本）:")
    for i in range(5):
        print(f"  样本 {i}: P(科技)={proba[i, 0]:.4f}, P(体育)={proba[i, 1]:.4f} -> {'科技' if proba[i, 0] > proba[i, 1] else '体育'}")

    print(f"\n平滑参数 (alpha) 对比:")
    for alpha in [0.01, 0.1, 1.0, 5.0, 10.0]:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"  alpha={alpha:5.2f} -> 测试准确率: {acc:.4f}")


def demo_gaussian():
    print_separator("高斯朴素贝叶斯 -- 连续特征")

    X, y = make_continuous_data(n_samples=450, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25, seed=42)

    print(f"训练样本数: {X_train.shape[0]}")
    print(f"测试样本数: {X_test.shape[0]}")
    print(f"特征数:     {X_train.shape[1]}")
    print(f"类别:       0, 1, 2（类似 Iris）")
    print()

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    train_acc = gnb.score(X_train, y_train)
    test_acc = gnb.score(X_test, y_test)
    print(f"从零实现的 GaussianNB:")
    print(f"  训练准确率: {train_acc:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")

    print(f"\n学到的参数:")
    for i, c in enumerate(gnb.classes_):
        print(f"  类别 {c}:")
        print(f"    均值: {gnb.means_[i].round(3)}")
        print(f"    方差: {gnb.vars_[i].round(4)}")
        print(f"    先验: {gnb.priors_[i]:.3f}")

    proba = gnb.predict_proba(X_test[:5])
    print(f"\n预测概率（前 5 个样本）:")
    for i in range(5):
        pred = gnb.classes_[np.argmax(proba[i])]
        probs_str = ", ".join(f"P({c})={proba[i, j]:.4f}" for j, c in enumerate(gnb.classes_))
        print(f"  样本 {i}: {probs_str} -> 类别 {pred}")


def demo_comparison():
    print_separator("对比: 多项式 vs 高斯")

    print("任务 1: 文本数据（词袋计数）")
    X, y = make_text_data(n_samples=1000, seed=99)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=99)

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    mnb_acc = mnb.score(X_test, y_test)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_acc = gnb.score(X_test, y_test)

    print(f"  MultinomialNB: {mnb_acc:.4f}")
    print(f"  GaussianNB:    {gnb_acc:.4f}")
    print(f"  胜出: {'MultinomialNB' if mnb_acc >= gnb_acc else 'GaussianNB'}")

    print(f"\n任务 2: 连续特征（类似 Iris）")
    X, y = make_continuous_data(n_samples=450, seed=99)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=99)

    X_train_pos = X_train - X_train.min(axis=0) + 0.01
    X_test_pos = X_test - X_train.min(axis=0) + 0.01

    mnb2 = MultinomialNB(alpha=1.0)
    mnb2.fit(X_train_pos, y_train)
    mnb_acc2 = mnb2.score(X_test_pos, y_test)

    gnb2 = GaussianNB()
    gnb2.fit(X_train, y_train)
    gnb_acc2 = gnb2.score(X_test, y_test)

    print(f"  MultinomialNB: {mnb_acc2:.4f}（平移为正值）")
    print(f"  GaussianNB:    {gnb_acc2:.4f}")
    print(f"  胜出: {'MultinomialNB' if mnb_acc2 >= gnb_acc2 else 'GaussianNB'}")


def demo_training_size():
    print_separator("朴素贝叶斯 vs 训练集大小")

    X_full, y_full = make_text_data(n_samples=2000, n_features=200, seed=42)
    X_test_full = X_full[1600:]
    y_test_full = y_full[1600:]

    print(f"{'训练集大小':>12} {'准确率':>10}")
    print(f"{'-' * 24}")

    for n_train in [20, 50, 100, 200, 500, 1000, 1600]:
        X_train = X_full[:n_train]
        y_train = y_full[:n_train]

        mnb = MultinomialNB(alpha=1.0)
        mnb.fit(X_train, y_train)
        acc = mnb.score(X_test_full, y_test_full)
        print(f"{n_train:>12} {acc:>10.4f}")


def demo_confusion_matrix():
    print_separator("混淆矩阵与每类指标")

    X, y = make_text_data(n_samples=800, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)

    classes = np.unique(y_test)
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[int(true), int(pred)] += 1

    class_names = ["科技", "体育"]
    print("混淆矩阵:")
    print(f"{'':>12} {'预测科技':>12} {'预测体育':>12}")
    for i, name in enumerate(class_names):
        row = "".join(f"{cm[i, j]:>12}" for j in range(n_classes))
        print(f"{'真实' + name:>12}{row}")

    print(f"\n每类指标:")
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {name:>8}: 精确率={precision:.4f}, 召回率={recall:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    demo_multinomial()
    demo_gaussian()
    demo_comparison()
    demo_training_size()
    demo_confusion_matrix()
