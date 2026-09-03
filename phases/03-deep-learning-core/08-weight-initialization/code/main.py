import math
import random


def zero_init(fan_in, fan_out):
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_act(x):
    return math.tanh(x)


def relu(x):
    return max(0.0, x)


def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    random.seed(42)
    layer_magnitudes = []

    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width

        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j] for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs

        magnitudes = []
        for sample in inputs:
            magnitudes.append(sum(abs(v) for v in sample) / width)
        mean_mag = sum(magnitudes) / len(magnitudes)
        layer_magnitudes.append(mean_mag)

    return layer_magnitudes


def magnitude_report(name, magnitudes):
    print(f"\n{name}:")
    for i, mag in enumerate(magnitudes):
        if i % 5 == 0 or i == len(magnitudes) - 1:
            if mag > 1e6:
                bar = "X" * 50 + " 已爆炸"
            elif mag < 1e-6:
                bar = "." + " 已消失"
            else:
                bar_len = min(50, max(1, int(mag * 10)))
                bar = "#" * bar_len
            print(f"  第 {i+1:3d} 层: {bar} ({mag:.6f})")


def symmetry_demo():
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    print("对称性演示（4 个神经元，零初始化）:")
    for i, out in enumerate(outputs):
        print(f"  神经元 {i}: 输出 = {out:.6f}")
    all_same = all(abs(outputs[i] - outputs[0]) < 1e-10 for i in range(len(outputs)))
    print(f"  全部相同: {all_same}")
    print(f"  有效参数: 1 (而非 {len(weights) * len(weights[0])})")


def variance_analysis():
    fan_in = 64
    n_trials = 10000

    configs = [
        ("随机 N(0,1)", 1.0),
        ("随机 N(0,0.01)", 0.01),
        ("Xavier 标准差", math.sqrt(2.0 / (fan_in + fan_in))),
        ("Kaiming 标准差", math.sqrt(2.0 / fan_in)),
    ]

    print("\n方差分析（fan_in=64，单层）:")
    print(f"  {'策略':<25} {'权重方差':>12} {'输出方差':>12} {'比率':>10}")
    print("  " + "-" * 60)

    for name, std in configs:
        random.seed(42)
        output_vars = []
        for _ in range(n_trials):
            inputs = [random.gauss(0, 1) for _ in range(fan_in)]
            weights = [random.gauss(0, std) for _ in range(fan_in)]
            z = sum(w * x for w, x in zip(weights, inputs))
            output_vars.append(z * z)

        mean_output_var = sum(output_vars) / len(output_vars)
        weight_var = std * std
        ratio = mean_output_var
        print(f"  {name:<25} {weight_var:>12.6f} {mean_output_var:>12.4f} {ratio:>10.4f}")


def run_experiment():
    configs = [
        ("零初始化 + Sigmoid", lambda fi, fo: zero_init(fi, fo), sigmoid),
        ("随机 N(0,1) + ReLU", lambda fi, fo: random_init(fi, fo, 1.0), relu),
        ("随机 N(0,0.01) + ReLU", lambda fi, fo: random_init(fi, fo, 0.01), relu),
        ("Xavier + Sigmoid", xavier_init, sigmoid),
        ("Xavier + Tanh", xavier_init, tanh_act),
        ("Kaiming + ReLU", kaiming_init, relu),
    ]

    print(f"\n{'策略':<30} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("-" * 80)

    all_results = {}
    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        all_results[name] = mags
        row = f"{name:<30}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'已爆炸':>10}"
            elif val < 1e-6:
                row += f" {'已消失':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)

    return all_results


def training_comparison():
    random.seed(42)
    data = []
    for _ in range(200):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))

    def train_with_init(init_name, init_scale, activation_fn, activation_deriv):
        random.seed(0)
        hidden_size = 8
        lr = 0.1

        if init_name == "xavier":
            std_w1 = math.sqrt(2.0 / (2 + hidden_size))
            std_w2 = math.sqrt(2.0 / (hidden_size + 1))
        elif init_name == "kaiming":
            std_w1 = math.sqrt(2.0 / 2)
            std_w2 = math.sqrt(2.0 / hidden_size)
        else:
            std_w1 = init_scale
            std_w2 = init_scale

        w1 = [[random.gauss(0, std_w1) for _ in range(2)] for _ in range(hidden_size)]
        b1 = [0.0] * hidden_size
        w2 = [random.gauss(0, std_w2) for _ in range(hidden_size)]
        b2 = 0.0

        losses = []
        for epoch in range(300):
            total_loss = 0
            correct = 0
            for x, target in data:
                z1 = []
                h = []
                for i in range(hidden_size):
                    z = w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                    z1.append(z)
                    h.append(activation_fn(z))

                z2 = sum(w2[i] * h[i] for i in range(hidden_size)) + b2
                out = sigmoid(z2)

                error = out - target
                d_out = error * out * (1 - out)

                for i in range(hidden_size):
                    d_h = d_out * w2[i] * activation_deriv(z1[i])
                    w2[i] -= lr * d_out * h[i]
                    for j in range(2):
                        w1[i][j] -= lr * d_h * x[j]
                    b1[i] -= lr * d_h
                b2 -= lr * d_out

                total_loss += (out - target) ** 2
                if (out >= 0.5) == (target >= 0.5):
                    correct += 1

            losses.append(total_loss / len(data))

        return losses

    def sigmoid_d(x):
        s = sigmoid(x)
        return s * (1 - s)

    def relu_d(x):
        return 1.0 if x > 0 else 0.0

    configs = [
        ("随机(0.01) + Sigmoid", "random", 0.01, sigmoid, sigmoid_d),
        ("随机(1.0) + Sigmoid", "random", 1.0, sigmoid, sigmoid_d),
        ("Xavier + Sigmoid", "xavier", 0, sigmoid, sigmoid_d),
        ("随机(0.01) + ReLU", "random", 0.01, relu, relu_d),
        ("随机(1.0) + ReLU", "random", 1.0, relu, relu_d),
        ("Kaiming + ReLU", "kaiming", 0, relu, relu_d),
    ]

    print("\n训练对比（300 轮，圆形数据集）:")
    print(f"  {'配置':<30} {'初始损失':>12} {'最终损失':>12} {'提升':>12}")
    print("  " + "-" * 66)

    for name, init_name, scale, act_fn, act_d_fn in configs:
        losses = train_with_init(init_name, scale, act_fn, act_d_fn)
        start = losses[0]
        end = losses[-1]
        improvement = (1 - end / start) * 100 if start > 0 else 0
        print(f"  {name:<30} {start:>12.6f} {end:>12.6f} {improvement:>11.1f}%")


if __name__ == "__main__":
    print("=" * 70)
    print("步骤 1：对称性问题 -- 零初始化")
    print("=" * 70)
    symmetry_demo()

    print("\n" + "=" * 70)
    print("步骤 2：方差分析")
    print("=" * 70)
    variance_analysis()

    print("\n" + "=" * 70)
    print("步骤 3：50 层前向传播实验")
    print("=" * 70)
    all_results = run_experiment()

    print("\n" + "=" * 70)
    print("步骤 4：逐层幅度报告")
    print("=" * 70)
    for name, mags in all_results.items():
        magnitude_report(name, mags)

    print("\n" + "=" * 70)
    print("步骤 5：训练对比")
    print("=" * 70)
    training_comparison()
