import math
import random

random.seed(42)

EFFECT_INTERPRETATION_LABELS = {
    "negligible": "可忽略",
    "small": "小",
    "medium": "中",
    "large": "大",
}


def mean(data):
    return sum(data) / len(data)


def median(data):
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def mode(data):
    counts = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    modes.sort()
    return modes[0]


def variance(data, sample=True):
    n = len(data)
    m = mean(data)
    total = sum((x - m) ** 2 for x in data)
    if sample and n > 1:
        return total / (n - 1)
    return total / n


def std_dev(data, sample=True):
    return math.sqrt(variance(data, sample))


def percentile(data, p):
    s = sorted(data)
    n = len(s)
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def iqr(data):
    return percentile(data, 75) - percentile(data, 25)


def covariance(x, y, sample=True):
    n = len(x)
    mx = mean(x)
    my = mean(y)
    total = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if sample and n > 1:
        return total / (n - 1)
    return total / n


def pearson_correlation(x, y):
    n = len(x)
    mx = mean(x)
    my = mean(y)
    sx = std_dev(x, sample=False)
    sy = std_dev(y, sample=False)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    return cov / (sx * sy)


def rank_data(data):
    indexed = sorted(enumerate(data), key=lambda pair: pair[1])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x, y):
    rx = rank_data(x)
    ry = rank_data(y)
    return pearson_correlation(rx, ry)


def covariance_matrix(data):
    d = len(data)
    n = len(data[0])
    means = [mean(data[i]) for i in range(d)]
    matrix = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            cov = sum(
                (data[i][k] - means[i]) * (data[j][k] - means[j])
                for k in range(n)
            ) / (n - 1)
            matrix[i][j] = cov
            matrix[j][i] = cov
    return matrix


def t_statistic_one_sample(data, mu_0):
    n = len(data)
    m = mean(data)
    s = std_dev(data, sample=True)
    return (m - mu_0) / (s / math.sqrt(n))


def t_statistic_two_sample(data1, data2):
    n1 = len(data1)
    n2 = len(data2)
    m1 = mean(data1)
    m2 = mean(data2)
    v1 = variance(data1, sample=True)
    v2 = variance(data2, sample=True)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0
    return (m1 - m2) / se


def welch_df(data1, data2):
    n1 = len(data1)
    n2 = len(data2)
    v1 = variance(data1, sample=True)
    v2 = variance(data2, sample=True)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if denom == 0:
        return n1 + n2 - 2
    return num / denom


def t_cdf_approx(t_val, df):
    x = df / (df + t_val * t_val)
    if t_val < 0:
        return 0.5 * _regularized_beta(x, df / 2, 0.5)
    return 1.0 - 0.5 * _regularized_beta(x, df / 2, 0.5)


def _regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        a * math.log(x) + b * math.log(1.0 - x) - _log_beta_function(a, b)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(x, a, b) / a
    return 1.0 - front * _beta_continued_fraction(1.0 - x, b, a) / b


def _beta_continued_fraction(x, a, b, max_iter=300, tol=1e-15):
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h


def _log_beta_function(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_function(a, b):
    return math.exp(_log_beta_function(a, b))


def p_value_two_sided(t_val, df):
    p_left = t_cdf_approx(abs(t_val), df)
    return 2.0 * (1.0 - p_left)


def one_sample_ttest(data, mu_0=0):
    n = len(data)
    t = t_statistic_one_sample(data, mu_0)
    df = n - 1
    p = p_value_two_sided(t, df)
    return {"t_statistic": t, "df": df, "p_value": p}


def two_sample_ttest(data1, data2):
    t = t_statistic_two_sample(data1, data2)
    df = welch_df(data1, data2)
    p = p_value_two_sided(t, df)
    return {"t_statistic": t, "df": df, "p_value": p}


def paired_ttest(data1, data2):
    diffs = [a - b for a, b in zip(data1, data2)]
    return one_sample_ttest(diffs, mu_0=0)


def chi_squared_test(observed, expected):
    chi2 = sum(
        (o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0
    )
    df = len(observed) - 1
    p = chi_squared_p_value(chi2, df)
    return {"chi2": chi2, "df": df, "p_value": p}


def chi_squared_p_value(chi2, df):
    if chi2 <= 0:
        return 1.0
    return 1.0 - _lower_incomplete_gamma_ratio(df / 2.0, chi2 / 2.0)


def _lower_incomplete_gamma_ratio(a, x):
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_continued_fraction(a, x)


def _gamma_series(a, x, max_iter=1000, tol=1e-16):
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(max_iter):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * tol:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(a, x, max_iter=1000, tol=1e-16):
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def bootstrap_statistic(data, stat_func, n_bootstrap=5000, ci=95):
    n = len(data)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        bootstrap_stats.append(stat_func(sample))
    bootstrap_stats.sort()
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    ci_lower = percentile(bootstrap_stats, lower_pct)
    ci_upper = percentile(bootstrap_stats, upper_pct)
    return {
        "estimate": stat_func(data),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "std_error": std_dev(bootstrap_stats, sample=True),
    }


def bootstrap_compare(data1, data2, stat_func, n_bootstrap=5000, ci=95):
    n1 = len(data1)
    n2 = len(data2)
    diffs = []
    for _ in range(n_bootstrap):
        s1 = [data1[random.randint(0, n1 - 1)] for _ in range(n1)]
        s2 = [data2[random.randint(0, n2 - 1)] for _ in range(n2)]
        diffs.append(stat_func(s2) - stat_func(s1))
    diffs.sort()
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    ci_lower = percentile(diffs, lower_pct)
    ci_upper = percentile(diffs, upper_pct)
    observed_diff = stat_func(data2) - stat_func(data1)
    significant = ci_lower > 0 or ci_upper < 0
    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "ci_level": ci,
    }


def cohens_d(data1, data2):
    m1 = mean(data1)
    m2 = mean(data2)
    n1 = len(data1)
    n2 = len(data2)
    v1 = variance(data1, sample=True)
    v2 = variance(data2, sample=True)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (m1 - m2) / pooled


def interpret_cohens_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def format_effect_interpretation(label):
    return EFFECT_INTERPRETATION_LABELS.get(label, label)


def bonferroni_correction(p_values, alpha=0.05):
    m = len(p_values)
    adjusted_alpha = alpha / m
    results = []
    for p in p_values:
        results.append({
            "original_p": p,
            "adjusted_alpha": adjusted_alpha,
            "significant": p < adjusted_alpha,
        })
    return results


def generate_normal(n, mu=0, sigma=1):
    samples = []
    for _ in range(n // 2 + 1):
        u1 = random.random()
        u2 = random.random()
        while u1 == 0:
            u1 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        samples.append(mu + sigma * z0)
        samples.append(mu + sigma * z1)
    return samples[:n]


def ab_test_simulator(
    n_per_group=100,
    true_effect=0.0,
    base_mean=50,
    base_std=10,
    alpha=0.05,
):
    group_a = generate_normal(n_per_group, base_mean, base_std)
    group_b = generate_normal(n_per_group, base_mean + true_effect, base_std)

    result = two_sample_ttest(group_a, group_b)
    d = cohens_d(group_a, group_b)
    boot = bootstrap_compare(group_a, group_b, mean, n_bootstrap=2000)

    return {
        "group_a_mean": mean(group_a),
        "group_b_mean": mean(group_b),
        "observed_diff": mean(group_b) - mean(group_a),
        "true_effect": true_effect,
        "t_test": result,
        "cohens_d": d,
        "effect_interpretation": interpret_cohens_d(d),
        "bootstrap": boot,
        "significant_ttest": result["p_value"] < alpha,
        "significant_bootstrap": boot["significant"],
    }


def run_multiple_ab_tests(
    n_tests=20,
    n_per_group=100,
    true_effect=0.0,
    alpha=0.05,
):
    p_values = []
    significant_count = 0
    for _ in range(n_tests):
        group_a = generate_normal(n_per_group, 50, 10)
        group_b = generate_normal(n_per_group, 50 + true_effect, 10)
        result = two_sample_ttest(group_a, group_b)
        p_values.append(result["p_value"])
        if result["p_value"] < alpha:
            significant_count += 1

    corrected = bonferroni_correction(p_values, alpha)
    corrected_significant = sum(1 for r in corrected if r["significant"])

    return {
        "n_tests": n_tests,
        "true_effect": true_effect,
        "false_positive_rate": significant_count / n_tests if true_effect == 0 else None,
        "uncorrected_significant": significant_count,
        "corrected_significant": corrected_significant,
        "p_values": p_values,
    }


def statistical_vs_practical_significance(small_n=30, large_n=100000, effect=0.1):
    small_a = generate_normal(small_n, 50, 10)
    small_b = generate_normal(small_n, 50 + effect, 10)
    small_result = two_sample_ttest(small_a, small_b)
    small_d = cohens_d(small_a, small_b)

    large_a = generate_normal(large_n, 50, 10)
    large_b = generate_normal(large_n, 50 + effect, 10)
    large_result = two_sample_ttest(large_a, large_b)
    large_d = cohens_d(large_a, large_b)

    return {
        "small_sample": {
            "n": small_n,
            "p_value": small_result["p_value"],
            "cohens_d": small_d,
            "significant": small_result["p_value"] < 0.05,
            "interpretation": interpret_cohens_d(small_d),
        },
        "large_sample": {
            "n": large_n,
            "p_value": large_result["p_value"],
            "cohens_d": large_d,
            "significant": large_result["p_value"] < 0.05,
            "interpretation": interpret_cohens_d(large_d),
        },
        "true_effect": effect,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("描述性统计")
    print("=" * 60)
    data = [23, 45, 12, 67, 34, 89, 21, 56, 43, 78, 31, 64, 19, 52, 41]
    print(f"数据: {data}")
    print(f"均值:     {mean(data):.2f}")
    print(f"中位数:   {median(data):.2f}")
    print(f"众数:     {mode(data)}")
    print(f"标准差:  {std_dev(data):.2f}")
    print(f"方差: {variance(data):.2f}")
    print(f"P25:      {percentile(data, 25):.2f}")
    print(f"P50:      {percentile(data, 50):.2f}")
    print(f"P75:      {percentile(data, 75):.2f}")
    print(f"IQR:      {iqr(data):.2f}")

    skewed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]
    print(f"\n偏态数据: {skewed}")
    print(f"均值:   {mean(skewed):.2f}  (被离群值拉偏)")
    print(f"中位数: {median(skewed):.2f}  (对离群值稳健)")

    print("\n" + "=" * 60)
    print("相关性")
    print("=" * 60)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_linear = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.1, 18.0, 20.2]
    print(f"线性关系:")
    print(f"  Pearson:  {pearson_correlation(x, y_linear):.4f}")
    print(f"  Spearman: {spearman_correlation(x, y_linear):.4f}")

    y_quadratic = [xi ** 2 for xi in x]
    print(f"二次关系 (y = x^2):")
    print(f"  Pearson:  {pearson_correlation(x, y_quadratic):.4f}  (非完美,关系是非线性的)")
    print(f"  Spearman: {spearman_correlation(x, y_quadratic):.4f}  (完美,关系是单调的)")

    y_none = [random.gauss(0, 1) for _ in x]
    print(f"无关系 (随机):")
    print(f"  Pearson:  {pearson_correlation(x, y_none):.4f}")
    print(f"  Spearman: {spearman_correlation(x, y_none):.4f}")

    print("\n" + "=" * 60)
    print("协方差矩阵")
    print("=" * 60)
    feature1 = [random.gauss(0, 1) for _ in range(100)]
    feature2 = [f + random.gauss(0, 0.5) for f in feature1]
    feature3 = [random.gauss(0, 1) for _ in range(100)]
    cov_mat = covariance_matrix([feature1, feature2, feature3])
    print("3 个特征的协方差矩阵:")
    for row in cov_mat:
        print(f"  [{row[0]:7.3f}  {row[1]:7.3f}  {row[2]:7.3f}]")
    print("特征 1 和特征 2 是相关的 (构造时即如此)。")
    print("特征 3 是独立的。")

    print("\n" + "=" * 60)
    print("假设检验: 单样本 t 检验")
    print("=" * 60)
    sample = generate_normal(50, mu=52, sigma=10)
    result = one_sample_ttest(sample, mu_0=50)
    print(f"检验总体均值是否 = 50 (真实均值 = 52)")
    print(f"  样本均值: {mean(sample):.2f}")
    print(f"  t 统计量: {result['t_statistic']:.4f}")
    print(f"  自由度:          {result['df']}")
    print(f"  p 值:     {result['p_value']:.4f}")
    print(f"  在 alpha=0.05 下是否显著: {result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("假设检验: 双样本 t 检验")
    print("=" * 60)
    model_a_scores = generate_normal(30, mu=0.85, sigma=0.05)
    model_b_scores = generate_normal(30, mu=0.88, sigma=0.05)
    result = two_sample_ttest(model_a_scores, model_b_scores)
    d = cohens_d(model_a_scores, model_b_scores)
    print(f"模型 A 均值: {mean(model_a_scores):.4f}")
    print(f"模型 B 均值: {mean(model_b_scores):.4f}")
    print(f"  t 统计量: {result['t_statistic']:.4f}")
    print(f"  p 值:     {result['p_value']:.4f}")
    effect_label = interpret_cohens_d(d)
    print(f"  Cohen's d:   {d:.4f} ({format_effect_interpretation(effect_label)})")
    print(f"  是否显著: {result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("配对 t 检验 (交叉验证)")
    print("=" * 60)
    cv_a = [0.82, 0.85, 0.81, 0.84, 0.83, 0.86, 0.80, 0.84, 0.82, 0.85]
    cv_b = [0.84, 0.87, 0.83, 0.86, 0.85, 0.88, 0.83, 0.86, 0.85, 0.87]
    result = paired_ttest(cv_a, cv_b)
    print(f"模型 A 折: {cv_a}")
    print(f"模型 B 折: {cv_b}")
    print(f"  均值差:   {mean([b - a for a, b in zip(cv_a, cv_b)]):.4f}")
    print(f"  t 统计量: {result['t_statistic']:.4f}")
    print(f"  p 值:     {result['p_value']:.4f}")
    print(f"  是否显著: {result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("卡方检验")
    print("=" * 60)
    observed = [120, 80, 95, 105]
    expected = [100, 100, 100, 100]
    result = chi_squared_test(observed, expected)
    print(f"观察值: {observed}")
    print(f"期望值: {expected}")
    print(f"  卡方值: {result['chi2']:.4f}")
    print(f"  自由度:          {result['df']}")
    print(f"  p 值:     {result['p_value']:.4f}")
    print(f"  是否显著: {result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("Bootstrap 置信区间")
    print("=" * 60)
    data = generate_normal(50, mu=100, sigma=15)
    boot_mean = bootstrap_statistic(data, mean, n_bootstrap=5000)
    boot_median = bootstrap_statistic(data, median, n_bootstrap=5000)
    print(f"样本量: 50, 真实均值: 100")
    print(f"Bootstrap 均值:   {boot_mean['estimate']:.2f}  "
          f"95% CI: [{boot_mean['ci_lower']:.2f}, {boot_mean['ci_upper']:.2f}]  "
          f"SE: {boot_mean['std_error']:.2f}")
    print(f"Bootstrap 中位数: {boot_median['estimate']:.2f}  "
          f"95% CI: [{boot_median['ci_lower']:.2f}, {boot_median['ci_upper']:.2f}]  "
          f"SE: {boot_median['std_error']:.2f}")

    print("\nBootstrap 模型比较:")
    scores_a = generate_normal(40, mu=0.85, sigma=0.04)
    scores_b = generate_normal(40, mu=0.88, sigma=0.04)
    comp = bootstrap_compare(scores_a, scores_b, mean, n_bootstrap=5000)
    print(f"  模型 A 均值: {mean(scores_a):.4f}")
    print(f"  模型 B 均值: {mean(scores_b):.4f}")
    print(f"  差值:         {comp['observed_diff']:.4f}")
    print(f"  95% CI:       [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]")
    print(f"  是否显著:  {comp['significant']} (CI 不包含 0)")

    print("\n" + "=" * 60)
    print("A/B 测试模拟器")
    print("=" * 60)
    print("\n测试 1: 无真实效应 (true_effect = 0)")
    ab1 = ab_test_simulator(n_per_group=200, true_effect=0.0)
    print(f"  A 组均值: {ab1['group_a_mean']:.2f}")
    print(f"  B 组均值: {ab1['group_b_mean']:.2f}")
    print(f"  观察到的差值: {ab1['observed_diff']:.2f}")
    print(f"  p 值: {ab1['t_test']['p_value']:.4f}")
    print(f"  是否显著 (t 检验): {ab1['significant_ttest']}")
    print(
        f"  Cohen's d: {ab1['cohens_d']:.4f} "
        f"({format_effect_interpretation(ab1['effect_interpretation'])})"
    )

    print("\n测试 2: 有真实效应 (true_effect = 5)")
    ab2 = ab_test_simulator(n_per_group=200, true_effect=5.0)
    print(f"  A 组均值: {ab2['group_a_mean']:.2f}")
    print(f"  B 组均值: {ab2['group_b_mean']:.2f}")
    print(f"  观察到的差值: {ab2['observed_diff']:.2f}")
    print(f"  p 值: {ab2['t_test']['p_value']:.4f}")
    print(f"  是否显著 (t 检验): {ab2['significant_ttest']}")
    print(
        f"  Cohen's d: {ab2['cohens_d']:.4f} "
        f"({format_effect_interpretation(ab2['effect_interpretation'])})"
    )

    print("\n" + "=" * 60)
    print("多重比较问题")
    print("=" * 60)
    print("\n20 次检验均无真实效应 (所有原假设为真):")
    multi = run_multiple_ab_tests(n_tests=20, true_effect=0.0)
    print(f"  显著的检验数 (未校正): {multi['uncorrected_significant']}/20")
    print(f"  显著的检验数 (Bonferroni):  {multi['corrected_significant']}/20")
    print(f"  alpha=0.05 时的期望假阳性数: ~1")
    print(f"  Bonferroni 校正后 alpha: {0.05/20:.4f}")

    print("\n" + "=" * 60)
    print("统计显著性 vs 实际显著性")
    print("=" * 60)
    result = statistical_vs_practical_significance(
        small_n=30, large_n=100000, effect=0.1
    )
    print(f"\n真实效应: {result['true_effect']} (微小)")
    print(f"\n小样本 (n={result['small_sample']['n']}):")
    print(f"  p 值:  {result['small_sample']['p_value']:.4f}")
    print(
        f"  Cohen's d: {result['small_sample']['cohens_d']:.4f} "
        f"({format_effect_interpretation(result['small_sample']['interpretation'])})"
    )
    print(f"  是否显著: {result['small_sample']['significant']}")
    print(f"\n大样本 (n={result['large_sample']['n']}):")
    print(f"  p 值:  {result['large_sample']['p_value']:.4f}")
    print(
        f"  Cohen's d: {result['large_sample']['cohens_d']:.4f} "
        f"({format_effect_interpretation(result['large_sample']['interpretation'])})"
    )
    print(f"  是否显著: {result['large_sample']['significant']}")
    print(f"\n启示: 大样本量会让可忽略的效应也变得'显著'。")
    print("始终要检查效应量,而不仅仅是 p 值。")

    print("\n" + "=" * 60)
    print("统计功效分析模拟")
    print("=" * 60)
    print("\n我们能多频繁地检测到真实效应 (true_effect=3)?")
    n_sims = 200
    detected = 0
    for _ in range(n_sims):
        a = generate_normal(50, 50, 10)
        b = generate_normal(50, 53, 10)
        res = two_sample_ttest(a, b)
        if res["p_value"] < 0.05:
            detected += 1
    print(f"  功效 (n=50, effect=3, std=10): {detected/n_sims:.2f}")
    print(f"  ({detected}/{n_sims} 次模拟检测到了效应)")

    detected_large = 0
    for _ in range(n_sims):
        a = generate_normal(200, 50, 10)
        b = generate_normal(200, 53, 10)
        res = two_sample_ttest(a, b)
        if res["p_value"] < 0.05:
            detected_large += 1
    print(f"  功效 (n=200, effect=3, std=10): {detected_large/n_sims:.2f}")
    print(f"  ({detected_large}/{n_sims} 次模拟检测到了效应)")
    print("  更大的样本量能赋予更强的检测真实效应的功效。")
