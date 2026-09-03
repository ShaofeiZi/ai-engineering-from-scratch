import math
import struct
import random


def softmax_naive(logits):
    # 朴素 softmax：直接计算 exp(z)，大 logit 会溢出
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def softmax_stable(logits):
    # 数值稳定版：减去最大值后再取 exp，避免溢出
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def logsumexp_naive(values):
    # 朴素 log-sum-exp：对大值会溢出
    return math.log(sum(math.exp(v) for v in values))


def logsumexp_stable(values):
    # 稳定版 log-sum-exp：减去最大值，结果加回
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))


def log_softmax_stable(logits):
    # 稳定版 log-softmax：避免先 softmax 再 log 导致的数值问题
    c = max(logits)
    lse = c + math.log(sum(math.exp(z - c) for z in logits))
    return [z - lse for z in logits]


def cross_entropy_naive(true_class, logits):
    # 朴素交叉熵：先 softmax 再取 log，概率为 0 时会得到 -inf
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])


def cross_entropy_stable(true_class, logits):
    # 稳定版交叉熵：直接用 log-softmax，避免 log(0)
    log_probs = log_softmax_stable(logits)
    return -log_probs[true_class]


def sigmoid_naive(x):
    # 朴素 sigmoid：x 为大负数时 exp(-x) 会溢出
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_stable(x):
    # 稳定版 sigmoid：根据 x 正负选择不同计算路径，避免溢出
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def binary_cross_entropy_naive(y_true, y_pred):
    # 朴素二元交叉熵：y_pred 为 0 或 1 时 log 会得到 -inf
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


def binary_cross_entropy_stable(y_true, logit):
    # 稳定版二元交叉熵：基于 logit 直接计算，避免 log(0)
    max_val = max(0.0, logit)
    return max_val + math.log(math.exp(-max_val) + math.exp(logit - max_val)) - y_true * logit


def numerical_gradient(f, x, h=1e-5):
    # 中心差分法计算数值梯度
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


def check_gradient(analytical, numerical, tolerance=1e-5):
    # 对比解析梯度与数值梯度，检查正确性
    all_ok = True
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  参数 {i}: 解析值={a:.8f} 数值={n:.8f} "
              f"相对误差={rel_error:.2e} [{status}]")
    return all_ok


def clip_by_value(gradients, max_val):
    # 按值裁剪：将每个梯度限制在 [-max_val, max_val] 范围内
    return [max(-max_val, min(max_val, g)) for g in gradients]


def clip_by_norm(gradients, max_norm):
    # 按范数裁剪：若整体范数超过阈值，等比缩放，保持方向不变
    total_norm = math.sqrt(sum(g ** 2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return list(gradients)


def check_tensor(name, values):
    # 检查张量中是否存在 NaN 或 Inf
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    n_nan = sum(1 for v in values if math.isnan(v))
    n_inf = sum(1 for v in values if math.isinf(v))
    if has_nan or has_inf:
        print(f"  警告 {name}: {n_nan} 个 NaN, {n_inf} 个 Inf（共 {len(values)} 个值）")
        return False
    print(f"  正常 {name}: 全部 {len(values)} 个值均为有限值")
    return True


def simulate_bfloat16(x):
    # 模拟 bfloat16 精度：截断 float32 的低 16 位尾数
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]


def simulate_float16(x):
    # 模拟 float16 精度
    try:
        packed = struct.pack('e', x)
        return struct.unpack('e', packed)[0]
    except (OverflowError, struct.error):
        return float('inf') if x > 0 else float('-inf')


def kahan_sum(values):
    # Kahan 求和算法：补偿求和过程中的舍入误差
    total = 0.0
    compensation = 0.0
    for v in values:
        y = v - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total


def welford_variance(values):
    # Welford 在线方差算法：避免大数相减导致的精度损失
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in values:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    if n < 2:
        return 0.0
    return m2 / n


def variance_naive(values):
    # 朴素方差：用 E[x^2] - E[x]^2 公式，大均值时会有精度问题
    n = len(values)
    mean_x = sum(values) / n
    mean_x2 = sum(v ** 2 for v in values) / n
    return mean_x2 - mean_x ** 2


def layer_norm(values, epsilon=1e-5, gamma=1.0, beta=0.0):
    # 层归一化：加 epsilon 防止除以零
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var + epsilon)
    return [(v - mean) / std * gamma + beta for v in values]


def demo_float_precision():
    print("=" * 60)
    print("演示 1：浮点数精度极限")
    print("=" * 60)

    print(f"\n  0.1 + 0.2 = {0.1 + 0.2}")
    print(f"  0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
    print(f"  与 0.3 的差值: {(0.1 + 0.2) - 0.3:.2e}")
    print(f"  math.isclose(0.1 + 0.2, 0.3): {math.isclose(0.1 + 0.2, 0.3)}")

    print(f"\n  Float32 最大值: ~{3.4028235e+38:.2e}")
    print(f"  Float32 最小正规范数: ~{1.175e-38:.2e}")
    print(f"  Float32 机器精度: ~{1.1920929e-07:.2e}")

    print(f"\n  1.0 + 1e-7 == 1.0?  {1.0 + 1e-7 == 1.0}")
    print(f"  1.0 + 1e-8 == 1.0?  {1.0 + 1e-8 == 1.0}")
    print(f"  （Python 中为 float64。在 float32 中，机器精度约为 1.19e-7）")

    total_naive = 0.0
    for _ in range(1_000_000):
        total_naive += 1e-7
    total_kahan = kahan_sum([1e-7] * 1_000_000)
    true_value = 1e-7 * 1_000_000

    print(f"\n  将 1e-7 累加一百万次:")
    print(f"  真实值:      {true_value}")
    print(f"  朴素求和:    {total_naive:.10f}  (误差: {abs(total_naive - true_value):.2e})")
    print(f"  Kahan 求和:  {total_kahan:.10f}  (误差: {abs(total_kahan - true_value):.2e})")
    print()


def demo_catastrophic_cancellation():
    print("=" * 60)
    print("演示 2：灾难性抵消")
    print("=" * 60)

    data = [1_000_000.0, 1_000_001.0, 1_000_002.0]
    true_var = 2.0 / 3.0

    var_naive = variance_naive(data)
    var_welford = welford_variance(data)

    print(f"\n  数据: {data}")
    print(f"  真实方差: {true_var:.10f}")
    print(f"  朴素法 (E[x^2] - E[x]^2): {var_naive:.10f}")
    print(f"  Welford (在线):            {var_welford:.10f}")
    print(f"  朴素法误差:   {abs(var_naive - true_var):.2e}")
    print(f"  Welford 误差: {abs(var_welford - true_var):.2e}")

    a = 1.0000001
    b = 1.0000000
    true_diff = 1e-7
    computed_diff = a - b
    rel_error = abs(computed_diff - true_diff) / true_diff * 100

    print(f"\n  相减两个极为接近的数:")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  真实 a - b = {true_diff}")
    print(f"  计算值:       {computed_diff}")
    print(f"  相对误差:     {rel_error:.1f}%")
    print()


def demo_overflow_underflow():
    print("=" * 60)
    print("演示 3：exp() 和 log() 的溢出与下溢")
    print("=" * 60)

    print("\n  exp() 溢出边界（Python 中为 float64）:")
    for x in [700, 709, 709.78, 710]:
        try:
            result = math.exp(x)
            print(f"  exp({x}) = {result:.4e}")
        except OverflowError:
            print(f"  exp({x}) = 溢出")

    print("\n  exp() 下溢（结果变为 0.0）:")
    for x in [-700, -745, -746]:
        result = math.exp(x)
        print(f"  exp({x}) = {result}")

    print("\n  log() 边界情况:")
    for x in [1.0, 1e-300, 1e-323, 0.0]:
        try:
            if x == 0.0:
                print(f"  log(0.0) = -inf  （数学上）")
                result = math.log(1e-323)
                print(f"  log(1e-323) = {result:.2f}  （最接近的可行值）")
            else:
                result = math.log(x)
                print(f"  log({x}) = {result:.4f}")
        except ValueError:
            print(f"  log({x}) = 定义域错误")

    print("\n  Float16 溢出边界:")
    for val in [65000.0, 65504.0, 65520.0, 70000.0]:
        f16 = simulate_float16(val)
        print(f"  float16({val}) = {f16}")
    print()


def demo_softmax_stability():
    print("=" * 60)
    print("演示 4：朴素 softmax 与稳定 softmax 对比")
    print("=" * 60)

    safe_logits = [2.0, 1.0, 0.1]
    print(f"\n  安全的 logits: {safe_logits}")
    naive_result = softmax_naive(safe_logits)
    stable_result = softmax_stable(safe_logits)
    print(f"  朴素:  {[f'{p:.6f}' for p in naive_result]}")
    print(f"  稳定: {[f'{p:.6f}' for p in stable_result]}")
    print(f"  是否一致: {all(abs(a - b) < 1e-10 for a, b in zip(naive_result, stable_result))}")

    moderate_logits = [100.0, 101.0, 102.0]
    print(f"\n  中等大小的 logits: {moderate_logits}")
    stable_result = softmax_stable(moderate_logits)
    print(f"  稳定: {[f'{p:.6f}' for p in stable_result]}")
    try:
        naive_result = softmax_naive(moderate_logits)
        print(f"  朴素:  {[f'{p:.6f}' for p in naive_result]}")
    except OverflowError:
        print("  朴素:  溢出（exp(100) 太大）")

    extreme_logits = [1000.0, 1001.0, 1002.0]
    print(f"\n  极端 logits: {extreme_logits}")
    stable_result = softmax_stable(extreme_logits)
    print(f"  稳定: {[f'{p:.6f}' for p in stable_result]}")
    print("  朴素:  会得到 [nan, nan, nan] 或溢出")

    negative_logits = [-1000.0, -999.0, -998.0]
    print(f"\n  极负 logits: {negative_logits}")
    stable_result = softmax_stable(negative_logits)
    print(f"  稳定: {[f'{p:.6f}' for p in stable_result]}")
    print("  朴素:  会得到 [0/0 = nan]（所有 exp() 下溢为 0）")
    print()


def demo_logsumexp():
    print("=" * 60)
    print("演示 5：Log-Sum-Exp 技巧")
    print("=" * 60)

    safe = [1.0, 2.0, 3.0]
    print(f"\n  安全的值: {safe}")
    print(f"  朴素:  {logsumexp_naive(safe):.10f}")
    print(f"  稳定: {logsumexp_stable(safe):.10f}")

    large = [500.0, 501.0, 502.0]
    print(f"\n  大值: {large}")
    print(f"  稳定: {logsumexp_stable(large):.10f}")
    try:
        naive = logsumexp_naive(large)
        print(f"  朴素:  {naive}")
    except OverflowError:
        print("  朴素:  溢出")

    very_negative = [-1000.0, -999.0, -998.0]
    print(f"\n  极负的值: {very_negative}")
    print(f"  稳定: {logsumexp_stable(very_negative):.10f}")

    equal = [5.0, 5.0, 5.0]
    print(f"\n  相同的值: {equal}")
    expected = 5.0 + math.log(3.0)
    print(f"  稳定:   {logsumexp_stable(equal):.10f}")
    print(f"  期望值: {expected:.10f} (= 5.0 + ln(3))")

    one_dominant = [100.0, 1.0, 1.0]
    print(f"\n  一个占主导的值: {one_dominant}")
    print(f"  稳定: {logsumexp_stable(one_dominant):.10f}")
    print(f"  ~100.0（被 exp(100) 主导）")
    print()


def demo_cross_entropy():
    print("=" * 60)
    print("演示 6：稳定的交叉熵损失")
    print("=" * 60)

    logits = [2.0, 5.0, 1.0]
    true_class = 1

    print(f"\n  Logits: {logits}, 真实类别: {true_class}")
    ce_naive = cross_entropy_naive(true_class, logits)
    ce_stable = cross_entropy_stable(true_class, logits)
    print(f"  朴素:  {ce_naive:.10f}")
    print(f"  稳定: {ce_stable:.10f}")
    print(f"  是否一致:  {abs(ce_naive - ce_stable) < 1e-10}")

    large_logits = [100.0, 105.0, 99.0]
    true_class = 1
    print(f"\n  大 logits: {large_logits}, 真实类别: {true_class}")
    ce_stable = cross_entropy_stable(true_class, large_logits)
    print(f"  稳定: {ce_stable:.10f}")
    try:
        ce_naive = cross_entropy_naive(true_class, large_logits)
        print(f"  朴素:  {ce_naive:.10f}")
    except (OverflowError, ValueError):
        print("  朴素:  溢出或 NaN")

    confident_logits = [0.0, 0.0, 50.0]
    true_class = 2
    ce = cross_entropy_stable(true_class, confident_logits)
    print(f"\n  极度自信的预测:")
    print(f"  Logits: {confident_logits}, 真实类别: {true_class}")
    print(f"  损失: {ce:.10f}  （接近零，模型正确且自信）")

    wrong_logits = [0.0, 0.0, 50.0]
    true_class = 0
    ce = cross_entropy_stable(true_class, wrong_logits)
    print(f"\n  严重错误的预测:")
    print(f"  Logits: {wrong_logits}, 真实类别: {true_class}")
    print(f"  损失: {ce:.4f}  （非常大，模型自信但错误）")
    print()


def demo_sigmoid_stability():
    print("=" * 60)
    print("演示 7：稳定的 sigmoid")
    print("=" * 60)

    test_values = [0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0, 500.0, -500.0, 710.0, -710.0]
    print(f"\n  {'x':>8s}  {'朴素':>14s}  {'稳定':>14s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}")
    for x in test_values:
        try:
            naive = sigmoid_naive(x)
            naive_str = f"{naive:.10f}"
        except OverflowError:
            naive_str = "溢出"
        stable = sigmoid_stable(x)
        print(f"  {x:>8.1f}  {naive_str:>14s}  {stable:.10f}")
    print()


def demo_gradient_checking():
    print("=" * 60)
    print("演示 8：梯度检查")
    print("=" * 60)

    print("\n  测试 1: f(x,y) = x^2 + 3xy + y^3")

    def f1(params):
        x, y = params
        return x ** 2 + 3 * x * y + y ** 3

    def f1_grad(params):
        x, y = params
        return [2 * x + 3 * y, 3 * x + 3 * y ** 2]

    point = [2.0, 1.0]
    analytical = f1_grad(point)
    numerical = numerical_gradient(f1, point)
    print(f"  点: {point}")
    check_gradient(analytical, numerical)

    print("\n  测试 2: f(x) = softmax 交叉熵")

    def f2(logits):
        return cross_entropy_stable(0, logits)

    logits = [2.0, 1.0, 0.5]
    probs = softmax_stable(logits)
    analytical_ce = [probs[i] - (1.0 if i == 0 else 0.0) for i in range(len(logits))]
    numerical_ce = numerical_gradient(f2, logits)
    print(f"  Logits: {logits}")
    check_gradient(analytical_ce, numerical_ce)

    print("\n  测试 3: 故意使用错误梯度（应当 FAIL）")

    def f3(params):
        x, y = params
        return x ** 2 + y ** 2

    wrong_grad = [1.0, 1.0]
    numerical_f3 = numerical_gradient(f3, [3.0, 4.0])
    print(f"  错误解析梯度: {wrong_grad}")
    print(f"  正确数值梯度: {[f'{g:.4f}' for g in numerical_f3]}")
    check_gradient(wrong_grad, numerical_f3)
    print()


def demo_nan_inf():
    print("=" * 60)
    print("演示 9：NaN 与 Inf 的检测和传播")
    print("=" * 60)

    print("\n  Inf 如何产生:")
    print(f"  1.0 / 0.0    = {float('inf')}")
    print("  exp(710)     = 溢出 -> inf")
    print(f"  1e308 * 10   = {1e308 * 10}")

    print("\n  NaN 如何产生:")
    print(f"  0.0 / 0.0        = {float('nan')}")
    print(f"  inf - inf        = {float('inf') - float('inf')}")
    print(f"  inf * 0          = {float('inf') * 0}")
    print(f"  nan + 1          = {float('nan') + 1}")
    print(f"  nan == nan       = {float('nan') == float('nan')}")
    print(f"  nan < 0          = {float('nan') < 0}")
    print(f"  nan > 0          = {float('nan') > 0}")

    print("\n  NaN 传播（一个 NaN 会破坏所有结果）:")
    values = [1.0, 2.0, float('nan'), 4.0, 5.0]
    print(f"  values = {values}")
    print(f"  sum    = {sum(values)}")
    print("  max    = nan（与 nan 的比较结果始终为 False）")
    print(f"  mean   = {sum(values) / len(values)}")

    print("\n  张量健康检查:")
    check_tensor("weights", [0.1, -0.3, 0.5, 0.2])
    check_tensor("logits_bad", [1.0, float('inf'), -2.0])
    check_tensor("grads_bad", [0.01, float('nan'), -0.03])
    check_tensor("activations", [0.0, 0.5, 1.0, 0.3])
    print()


def demo_gradient_clipping():
    print("=" * 60)
    print("演示 10：梯度裁剪")
    print("=" * 60)

    grads = [10.0, 20.0, 30.0]
    norm = math.sqrt(sum(g ** 2 for g in grads))

    print(f"\n  梯度: {grads}")
    print(f"  范数: {norm:.4f}")

    clipped_val = clip_by_value(grads, max_val=15.0)
    clipped_norm = clip_by_norm(grads, max_norm=5.0)

    print(f"\n  按值裁剪 (max=15.0): {clipped_val}")
    print(f"  按值裁剪会改变方向: "
          f"{[g/grads[0] for g in grads]} 与 {[g/clipped_val[0] for g in clipped_val]}")

    print(f"\n  按范数裁剪 (max=5.0): {[f'{g:.4f}' for g in clipped_norm]}")
    clipped_norm_val = math.sqrt(sum(g ** 2 for g in clipped_norm))
    print(f"  裁剪后的范数: {clipped_norm_val:.4f}")
    print(f"  方向保持不变: "
          f"{[round(g/grads[0], 4) for g in grads]} == "
          f"{[round(g/clipped_norm[0], 4) for g in clipped_norm]}")

    print("\n  梯度爆炸模拟:")
    grad_val = 1.0
    max_norm = 1.0
    for step in range(8):
        grad_val *= 3.5
        clipped = clip_by_norm([grad_val], max_norm)[0]
        print(f"  步骤 {step}: 原始梯度={grad_val:>12.2f}  裁剪后={clipped:>8.4f}")
    print()


def demo_mixed_precision():
    print("=" * 60)
    print("演示 11：混合精度与损失缩放")
    print("=" * 60)

    print("\n  bfloat16 与 float16 精度对比:")
    test_values = [1.0, 0.1, 3.14159, 100.0, 65504.0, 65536.0, 100000.0]
    print(f"  {'值':>12s}  {'float16':>12s}  {'bfloat16':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}")
    for v in test_values:
        f16 = simulate_float16(v)
        bf16 = simulate_bfloat16(v)
        f16_str = f"{f16:.4f}" if not math.isinf(f16) else "inf"
        bf16_str = f"{bf16:.4f}" if not math.isinf(bf16) else "inf"
        print(f"  {v:>12.4f}  {f16_str:>12s}  {bf16_str:>12s}")

    print("\n  损失缩放模拟:")
    random.seed(42)
    n_grads = 1000
    tiny_grads = [random.uniform(1e-9, 1e-5) for _ in range(n_grads)]

    zeros_without_scaling = sum(1 for g in tiny_grads if simulate_float16(g) == 0.0)

    scale = 1024.0
    scaled_grads = [g * scale for g in tiny_grads]
    zeros_with_scaling = sum(1 for g in scaled_grads if simulate_float16(g) == 0.0)

    scaled_back = [simulate_float16(g * scale) / scale for g in tiny_grads]
    zeros_after_roundtrip = sum(1 for g in scaled_back if g == 0.0)

    print(f"  {n_grads} 个梯度位于 [1e-9, 1e-5] 范围内")
    print(f"  未缩放时的零值数: {zeros_without_scaling}/{n_grads} "
          f"({zeros_without_scaling/n_grads*100:.1f}%)")
    print(f"  缩放后 (x{scale:.0f}) 的零值数: {zeros_with_scaling}/{n_grads} "
          f"({zeros_with_scaling/n_grads*100:.1f}%)")
    print(f"  缩放、转换再还原后的零值数: {zeros_after_roundtrip}/{n_grads} "
          f"({zeros_after_roundtrip/n_grads*100:.1f}%)")

    print("\n  动态损失缩放模拟:")
    scale_factor = 65536.0
    no_overflow_steps = 0
    growth_interval = 100

    print(f"  {'步骤':>6s}  {'缩放系数':>12s}  {'事件':s}")
    for step in range(500):
        grad = random.gauss(0, 1)
        scaled = grad * scale_factor
        if math.isinf(simulate_float16(scaled)):
            scale_factor /= 2
            no_overflow_steps = 0
            if step < 20 or step % 100 == 0:
                print(f"  {step:>6d}  {scale_factor:>12.0f}  溢出 -> 减半")
        else:
            no_overflow_steps += 1
            if no_overflow_steps >= growth_interval:
                scale_factor *= 2
                no_overflow_steps = 0
                if step < 100 or step % 100 == 0:
                    print(f"  {step:>6d}  {scale_factor:>12.0f}  稳定 -> 加倍")
    print(f"  最终缩放系数: {scale_factor:.0f}")
    print()


def demo_layer_norm():
    print("=" * 60)
    print("演示 12：用归一化提高数值稳定性")
    print("=" * 60)

    print("\n  不做归一化（数值逐层增大）:")
    values = [1.0, 0.5, -0.3, 0.8, -0.1]
    for layer in range(10):
        values = [max(0, v * 2.5 + 0.1) for v in values]
        max_val = max(abs(v) for v in values)
        if layer % 2 == 0:
            print(f"  层 {layer:>2d}: 最大值={max_val:>12.2f}  数值={[f'{v:.2f}' for v in values[:3]]}...")

    print("\n  使用层归一化（数值保持有界）:")
    values = [1.0, 0.5, -0.3, 0.8, -0.1]
    for layer in range(10):
        values = [max(0, v * 2.5 + 0.1) for v in values]
        values = layer_norm(values)
        max_val = max(abs(v) for v in values)
        if layer % 2 == 0:
            print(f"  层 {layer:>2d}: 最大值={max_val:>6.4f}  数值={[f'{v:.4f}' for v in values[:3]]}...")
    print()


def demo_common_bugs():
    print("=" * 60)
    print("演示 13：常见 ML 数值问题")
    print("=" * 60)

    print("\n  问题 1：错误但置信度高的预测产生 log(0)")
    logits = [100.0, -100.0, -100.0]
    probs = softmax_stable(logits)
    print(f"  Softmax: {[f'{p:.2e}' for p in probs]}")
    print(f"  如果真实类别为 1: log({probs[1]:.2e}) = ", end="")
    if probs[1] == 0.0:
        print("log(0) = -inf（程序崩溃）")
    else:
        print(f"{math.log(probs[1]):.2f}")
    print(f"  稳定版交叉熵可以处理此情况: {cross_entropy_stable(1, logits):.4f}")

    print("\n  问题 2：朴素 softmax 中 exp() 溢出")
    logits = [800.0, 801.0, 802.0]
    try:
        naive = softmax_naive(logits)
        print(f"  朴素 softmax: {naive}")
    except OverflowError:
        print("  朴素 softmax: OverflowError（exp(800) 太大）")
    stable = softmax_stable(logits)
    print(f"  稳定版 softmax: {[f'{p:.6f}' for p in stable]}")

    print("\n  问题 3：大均值数据的方差下溢")
    data = [1e8 + 1, 1e8 + 2, 1e8 + 3, 1e8 + 4, 1e8 + 5]
    var_naive = variance_naive(data)
    var_welford = welford_variance(data)
    true_var = 2.0
    print(f"  数据: [{data[0]:.0f}, ..., {data[-1]:.0f}]")
    print(f"  真实方差: {true_var}")
    print(f"  朴素法:   {var_naive:.6f}  （误差: {abs(var_naive - true_var):.2e}）")
    print(f"  Welford: {var_welford:.6f}  （误差: {abs(var_welford - true_var):.2e}）")

    print("\n  问题 4：训练循环中的浮点数比较")
    loss = 0.0
    for _ in range(10):
        loss += 0.1
    print(f"  执行 10 次 loss += 0.1 后: loss = {loss}")
    print(f"  loss == 1.0? {loss == 1.0}（错误）")
    print(f"  math.isclose(loss, 1.0)? {math.isclose(loss, 1.0)}（正确）")

    print("\n  问题 5：归一化中的 0/0 产生 NaN")
    values = [5.0, 5.0, 5.0, 5.0]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    print(f"  常量输入: {values}")
    print(f"  方差: {var}")
    print(f"  1/sqrt(var) = 1/sqrt(0) = ", end="")
    try:
        result = 1.0 / math.sqrt(var)
        print(f"{result}")
    except ZeroDivisionError:
        print("ZeroDivisionError")
    safe = 1.0 / math.sqrt(var + 1e-5)
    print(f"  1/sqrt(var + 1e-5) = {safe:.2f}（使用 epsilon 后安全）")
    print()


def demo_format_comparison():
    print("=" * 60)
    print("演示 14：浮点格式对比总结")
    print("=" * 60)

    print(f"""
  格式       位数  指数  尾数      约有效位  最大值          最适用场景
  -------    ----  ----  --------  --------  ----------      --------
  float64    64    11    52        15-16     1.8e308         CPU 训练、累加
  float32    32    8     23        7-8       3.4e38          默认训练
  float16    16    5     10        3-4       65,504          推理
  bfloat16   16    8     7         2-3       3.4e38          GPU/TPU 训练
  float8     8     4     3         1-2       240             仅前向传播 (H100+)
""")

    print("  精度测试（表示 pi）:")
    pi = math.pi
    f16_pi = simulate_float16(pi)
    bf16_pi = simulate_bfloat16(pi)
    print(f"  float64:  {pi}")
    print(f"  float16:  {f16_pi}  （误差: {abs(f16_pi - pi):.6f}）")
    print(f"  bfloat16: {bf16_pi}  （误差: {abs(bf16_pi - pi):.6f}）")

    print("\n  范围测试（大数值）:")
    for val in [100.0, 1000.0, 10000.0, 65504.0, 100000.0]:
        f16 = simulate_float16(val)
        bf16 = simulate_bfloat16(val)
        f16_ok = "ok" if not math.isinf(f16) else "INF"
        bf16_ok = "ok" if not math.isinf(bf16) else "INF"
        print(f"  {val:>10.0f}  float16={f16_ok:>4s}  bfloat16={bf16_ok:>4s}")
    print()


if __name__ == "__main__":
    demo_float_precision()
    demo_catastrophic_cancellation()
    demo_overflow_underflow()
    demo_softmax_stability()
    demo_logsumexp()
    demo_cross_entropy()
    demo_sigmoid_stability()
    demo_gradient_checking()
    demo_nan_inf()
    demo_gradient_clipping()
    demo_mixed_precision()
    demo_layer_norm()
    demo_common_bugs()
    demo_format_comparison()
    print("所有演示均已完成。")
