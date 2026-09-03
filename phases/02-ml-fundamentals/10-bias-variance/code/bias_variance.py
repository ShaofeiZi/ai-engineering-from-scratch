import numpy as np
import warnings
warnings.filterwarnings("ignore")


def true_function(x):
    return np.sin(1.5 * x) + 0.5 * x


def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y


def fit_polynomial(x_train, y_train, degree, lam=0.0):
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    if lam > 0:
        penalty = lam * np.eye(X.shape[1])
        penalty[0, 0] = 0
        w = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    else:
        w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w


def predict_polynomial(x, w):
    degree = len(w) - 1
    X = np.column_stack([x ** d for d in range(degree + 1)])
    return X @ w


def bias_variance_decomposition(
    degrees,
    n_bootstrap=200,
    n_train=30,
    noise_std=0.5,
    n_test=100,
    lam=0.0,
):
    rng = np.random.RandomState(42)
    x_test = np.linspace(-2.5, 2.5, n_test)
    y_true = true_function(x_test)

    results = {}

    for degree in degrees:
        predictions = np.zeros((n_bootstrap, n_test))

        for b in range(n_bootstrap):
            x_train, y_train = generate_data(
                n_samples=n_train, noise_std=noise_std, seed=rng.randint(0, 100000)
            )
            w = fit_polynomial(x_train, y_train, degree, lam=lam)
            predictions[b] = predict_polynomial(x_test, w)

        mean_pred = predictions.mean(axis=0)
        bias_sq = np.mean((mean_pred - y_true) ** 2)
        variance = np.mean(predictions.var(axis=0))
        total_error = np.mean(np.mean((predictions - y_true) ** 2, axis=1)) + noise_std ** 2

        results[degree] = {
            "bias_sq": bias_sq,
            "variance": variance,
            "total_error": total_error,
            "noise": noise_std ** 2,
        }

    return results


def print_decomposition(results):
    print(f"{'阶数':>6}  {'偏差^2':>10}  {'方差':>10}  {'噪声':>10}  {'总误差':>10}  {'偏+方+噪':>10}")
    print("-" * 70)
    for degree, r in sorted(results.items()):
        bvn = r["bias_sq"] + r["variance"] + r["noise"]
        print(
            f"{degree:>6d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['noise']:>10.4f}  {r['total_error']:>10.4f}  {bvn:>10.4f}"
        )


def find_optimal(results):
    best_degree = min(results, key=lambda d: results[d]["total_error"])
    return best_degree


def demo_basic_decomposition():
    print("=" * 70)
    print("偏差-方差分解")
    print("真实函数: sin(1.5x) + 0.5x")
    print("噪声标准差: 0.5, 训练样本: 30, Bootstrap 轮数: 200")
    print("=" * 70)
    print()

    degrees = [1, 2, 3, 5, 7, 10, 15]
    results = bias_variance_decomposition(degrees)
    print_decomposition(results)

    best = find_optimal(results)
    print(f"\n最优阶数: {best}")
    print(f"  偏差^2:   {results[best]['bias_sq']:.4f}")
    print(f"  方差:     {results[best]['variance']:.4f}")
    print(f"  总误差:   {results[best]['total_error']:.4f}")


def demo_complexity_tradeoff():
    print()
    print("=" * 70)
    print("模型复杂度权衡")
    print("多项式阶数从1扫描到15")
    print("=" * 70)
    print()

    degrees = list(range(1, 16))
    results = bias_variance_decomposition(degrees)

    print(f"{'阶数':>6}  {'偏差^2':>10}  {'方差':>10}  {'总误差':>10}  {'主导项':>12}")
    print("-" * 60)
    for degree in degrees:
        r = results[degree]
        dominant = "偏差" if r["bias_sq"] > r["variance"] else "方差"
        print(
            f"{degree:>6d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['total_error']:>10.4f}  {dominant:>12}"
        )

    crossover = None
    for d in degrees[:-1]:
        if results[d]["bias_sq"] > results[d]["variance"]:
            if results[d + 1]["bias_sq"] <= results[d + 1]["variance"]:
                crossover = d + 1
                break

    if crossover:
        print(f"\n偏差-方差交叉点在阶数 {crossover}")
        print("低于此阶数: 偏差主导（欠拟合）")
        print("高于此阶数: 方差主导（过拟合）")


def demo_regularization_effect():
    print()
    print("=" * 70)
    print("正则化效果 (L2 / Ridge)")
    print("固定阶数=10, 扫描 lambda")
    print("=" * 70)
    print()

    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"{'Lambda':>10}  {'偏差^2':>10}  {'方差':>10}  {'总误差':>10}")
    print("-" * 50)

    for lam in lambdas:
        results = bias_variance_decomposition([10], lam=lam)
        r = results[10]
        print(f"{lam:>10.3f}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  {r['total_error']:>10.4f}")

    print()
    print("随着 lambda 增大:")
    print("  - 方差减小（模型约束更强）")
    print("  - 偏差增大（模型被迫更简单）")
    print("  - 最优 lambda 平衡二者")


def demo_data_size_effect():
    print()
    print("=" * 70)
    print("训练集大小效果")
    print("固定阶数=5, 变化 n_train")
    print("=" * 70)
    print()

    sizes = [10, 20, 50, 100, 200, 500]

    print(f"{'N_train':>8}  {'偏差^2':>10}  {'方差':>10}  {'总误差':>10}")
    print("-" * 50)

    for n in sizes:
        results = bias_variance_decomposition([5], n_train=n)
        r = results[5]
        print(f"{n:>8d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  {r['total_error']:>10.4f}")

    print()
    print("更多数据可以减小方差，但不影响偏差。")
    print("如果问题是高偏差，更多数据也无济于事。")


def demo_diagnosis():
    print()
    print("=" * 70)
    print("欠拟合 vs 过拟合诊断")
    print("=" * 70)
    print()

    rng = np.random.RandomState(42)
    x_train, y_train = generate_data(n_samples=30, seed=42)
    x_test, y_test = generate_data(n_samples=100, seed=99)

    cases = [
        (1, "线性 (阶数 1)"),
        (4, "多项式 (阶数 4)"),
        (15, "多项式 (阶数 15)"),
    ]

    for degree, name in cases:
        w = fit_polynomial(x_train, y_train, degree)
        train_pred = predict_polynomial(x_train, w)
        test_pred = predict_polynomial(x_test, w)

        train_mse = np.mean((train_pred - y_train) ** 2)
        test_mse = np.mean((test_pred - y_test) ** 2)
        gap = test_mse - train_mse

        if train_mse > 0.5 and test_mse > 0.5 and gap < train_mse * 0.5:
            diagnosis = "高偏差（欠拟合）"
        elif gap > train_mse * 2:
            diagnosis = "高方差（过拟合）"
        else:
            diagnosis = "拟合合理"

        print(f"{name}:")
        print(f"  训练MSE: {train_mse:.4f}")
        print(f"  测试MSE: {test_mse:.4f}")
        print(f"  差距:    {gap:.4f}")
        print(f"  诊断:    {diagnosis}")
        print()


def demo_learning_curves():
    print()
    print("=" * 70)
    print("学习曲线")
    print("训练误差与测试误差随训练集大小的变化")
    print("=" * 70)
    print()

    rng = np.random.RandomState(42)
    x_test = np.linspace(-2.5, 2.5, 200)
    y_test = true_function(x_test)

    sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]

    for degree, label in [(1, "阶数1 (高偏差)"), (5, "阶数5 (均衡)"), (12, "阶数12 (高方差)")]:
        print(f"  {label}:")
        print(f"  {'N_train':>8}  {'训练MSE':>10}  {'测试MSE':>10}  {'差距':>10}")
        print(f"  {'-' * 48}")

        for n in sizes:
            train_errors = []
            test_errors = []
            for seed in range(50):
                x_train, y_train = generate_data(n_samples=n, seed=rng.randint(0, 100000))
                try:
                    w = fit_polynomial(x_train, y_train, degree)
                    train_pred = predict_polynomial(x_train, w)
                    test_pred = predict_polynomial(x_test, w)
                    train_mse = np.mean((train_pred - y_train) ** 2)
                    test_mse = np.mean((test_pred - y_test) ** 2)
                    train_errors.append(train_mse)
                    test_errors.append(test_mse)
                except (np.linalg.LinAlgError, ValueError):
                    continue

            if train_errors:
                avg_train = np.mean(train_errors)
                avg_test = np.mean(test_errors)
                gap = avg_test - avg_train
                print(f"  {n:>8d}  {avg_train:>10.4f}  {avg_test:>10.4f}  {gap:>10.4f}")

        print()

    print("高偏差（阶数1）: 两条曲线都收敛到高误差，差距始终很小。")
    print("高方差（阶数12）: 训练误差保持低，测试误差保持高。")
    print("更多数据可以减小方差，但无法修复偏差。")


def demo_regularization_sweep():
    print()
    print("=" * 70)
    print("正则化扫描 (Ridge alpha vs 偏差/方差)")
    print("固定阶数=15, 扫描 alpha 从 0.001 到 100")
    print("=" * 70)
    print()

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    print(f"  {'Alpha':>10}  {'偏差^2':>10}  {'方差':>10}  {'总误差':>10}  {'主导项':>12}")
    print(f"  {'-' * 60}")

    best_alpha = None
    best_total = float("inf")

    for alpha in alphas:
        results = bias_variance_decomposition([15], lam=alpha, n_bootstrap=200)
        r = results[15]
        dominant = "偏差" if r["bias_sq"] > r["variance"] else "方差"
        print(
            f"  {alpha:>10.3f}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['total_error']:>10.4f}  {dominant:>12}"
        )
        if r["total_error"] < best_total:
            best_total = r["total_error"]
            best_alpha = alpha

    print()
    print(f"最优 alpha: {best_alpha}")
    print(f"  最优总误差: {best_total:.4f}")
    print()
    print("小 alpha: 方差主导（模型不受约束，拟合了噪声）")
    print("大 alpha: 偏差主导（模型过度约束，丢失信号）")
    print("最优 alpha 平衡二者，位于U型曲线底部。")


if __name__ == "__main__":
    demo_basic_decomposition()
    demo_complexity_tradeoff()
    demo_regularization_effect()
    demo_data_size_effect()
    demo_diagnosis()
    demo_learning_curves()
    demo_regularization_sweep()
    print("偏差-方差全部演示完成。")
