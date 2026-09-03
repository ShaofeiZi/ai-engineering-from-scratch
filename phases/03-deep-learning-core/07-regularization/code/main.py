import math
import random


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training:
            return list(x)
        self.mask = []
        output = []
        for val in x:
            if random.random() < self.p:
                self.mask.append(0)
                output.append(0.0)
            else:
                self.mask.append(1)
                output.append(val / (1 - self.p))
        return output

    def backward(self, grad_output):
        grads = []
        for g, m in zip(grad_output, self.mask):
            if m == 0:
                grads.append(0.0)
            else:
                grads.append(g / (1 - self.p))
        return grads


def l2_regularization(weights, lambda_reg):
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty


def l2_gradient(weights, lambda_reg):
    return [lambda_reg * w for w in weights]


class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.training = True
        self.num_features = num_features

    def forward(self, batch):
        batch_size = len(batch)
        if self.training:
            mean = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            for j in range(self.num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            mean = list(self.running_mean)
            var = list(self.running_var)

        self.x_hat = []
        output = []
        for sample in batch:
            normalized = []
            out_sample = []
            for j in range(self.num_features):
                x_h = (sample[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                normalized.append(x_h)
                out_sample.append(self.gamma[j] * x_h + self.beta[j])
            self.x_hat.append(normalized)
            output.append(out_sample)
        return output


class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        self.x_hat = []
        output = []
        for j in range(self.num_features):
            x_h = (x[j] - mean) / math.sqrt(var + self.eps)
            self.x_hat.append(x_h)
            output.append(self.gamma[j] * x_h + self.beta[j])
        return output


class RMSNorm:
    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        rms = math.sqrt(sum(xi * xi for xi in x) / len(x) + self.eps)
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * x[j] / rms)
        return output


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class RegularizedNetwork:
    def __init__(self, hidden_size=16, lr=0.05, dropout_p=0.0, weight_decay=0.0):
        random.seed(0)
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x, training=True):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        if self.dropout and training:
            self.dropout.training = True
            self.h = self.dropout.forward(self.h)
        elif self.dropout:
            self.dropout.training = False
            self.h = self.dropout.forward(self.h)

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        d_h_dropout = [d_out * self.w2[i] for i in range(self.hidden_size)]
        if self.dropout and self.dropout.mask is not None:
            d_h_dropout = [g * m / (1 - self.dropout.p) if m else 0.0
                           for g, m in zip(d_h_dropout, self.dropout.mask)]

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_h_dropout[i] * d_relu
            self.w2[i] -= self.lr * (d_out * self.h[i] + self.weight_decay * self.w2[i])
            for j in range(2):
                self.w1[i][j] -= self.lr * (d_h * self.x[j] + self.weight_decay * self.w1[i][j])
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def evaluate(self, data):
        correct = 0
        total_loss = 0.0
        for x, y in data:
            pred = self.forward(x, training=False)
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if (pred >= 0.5) == (y >= 0.5):
                correct += 1
        return total_loss / len(data), correct / len(data) * 100

    def train_model(self, train_data, test_data, epochs=300):
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in train_data:
                pred = self.forward(x, training=True)
                self.backward(y)
                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            train_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data) * 100
            test_loss, test_acc = self.evaluate(test_data)
            history.append((train_loss, train_acc, test_loss, test_acc))
            if epoch % 75 == 0 or epoch == epochs - 1:
                gap = train_acc - test_acc
                print(f"    轮次 {epoch:3d}: 训练准确率={train_acc:.1f}%, 测试准确率={test_acc:.1f}%, 差距={gap:.1f}%")
        return history


if __name__ == "__main__":
    print("=" * 60)
    print("步骤 1：Dropout 演示")
    print("=" * 60)
    drop = Dropout(p=0.5)
    test_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    random.seed(42)

    drop.training = True
    print(f"  输入:           {test_input}")
    for trial in range(3):
        output = drop.forward(test_input)
        active = sum(1 for v in output if v > 0)
        print(f"  训练第 {trial+1} 次:   {[f'{v:.1f}' for v in output]}  ({active}/{len(test_input)} 激活)")

    drop.training = False
    output = drop.forward(test_input)
    print(f"  推理:           {[f'{v:.1f}' for v in output]}")
    print(f"  训练均值: ~{sum(test_input)/len(test_input):.1f} (缩放因子 1/(1-p))")
    print(f"  推理均值:   {sum(output)/len(output):.1f} (无需缩放)")

    print("\n" + "=" * 60)
    print("步骤 2：L2 正则化")
    print("=" * 60)
    weights = [0.5, -1.2, 3.0, 0.1, -2.5]
    lambda_val = 0.01
    penalty = l2_regularization(weights, lambda_val)
    grads = l2_gradient(weights, lambda_val)
    print(f"  权重: {weights}")
    print(f"  λ: {lambda_val}")
    print(f"  L2 惩罚: {penalty:.6f}")
    print(f"  L2 梯度:   {[f'{g:.4f}' for g in grads]}")
    print(f"  最大权重 (3.0) 对应最大梯度 ({grads[2]:.4f})")

    print("\n" + "=" * 60)
    print("步骤 3：BatchNorm 对比 LayerNorm 对比 RMSNorm")
    print("=" * 60)
    random.seed(42)
    batch = [[random.gauss(5, 2) for _ in range(4)] for _ in range(8)]
    sample = batch[0]

    bn = BatchNorm(4)
    bn_out = bn.forward(batch)

    ln = LayerNorm(4)
    ln_out = ln.forward(sample)

    rn = RMSNorm(4)
    rn_out = rn.forward(sample)

    print(f"  原始样本: {[f'{v:.2f}' for v in sample]}")
    print(f"  BatchNorm:  {[f'{v:.2f}' for v in bn_out[0]]}")
    print(f"  LayerNorm:  {[f'{v:.2f}' for v in ln_out]}")
    print(f"  RMSNorm:    {[f'{v:.2f}' for v in rn_out]}")

    ln_mean = sum(ln_out) / len(ln_out)
    ln_std = math.sqrt(sum((v - ln_mean) ** 2 for v in ln_out) / len(ln_out))
    rn_mean = sum(rn_out) / len(rn_out)
    rn_rms = math.sqrt(sum(v * v for v in rn_out) / len(rn_out))
    print(f"\n  LayerNorm 输出: 均值={ln_mean:.4f}, 标准差={ln_std:.4f}")
    print(f"  RMSNorm 输出: 均值={rn_mean:.4f}, 均方根={rn_rms:.4f}")
    print(f"  LayerNorm 将均值中心化为 0。RMSNorm 仅归一化尺度。")

    print("\n" + "=" * 60)
    print("步骤 4：BatchNorm 训练模式与推理模式")
    print("=" * 60)
    bn2 = BatchNorm(4)
    bn2.training = True
    for step in range(10):
        batch = [[random.gauss(3 + step * 0.1, 1) for _ in range(4)] for _ in range(16)]
        bn2.forward(batch)

    print(f"  10 个批次后的运行均值: {[f'{v:.3f}' for v in bn2.running_mean]}")
    print(f"  10 个批次后的运行方差: {[f'{v:.3f}' for v in bn2.running_var]}")

    bn2.training = False
    test_sample = [[5.0, 5.0, 5.0, 5.0]]
    eval_out = bn2.forward(test_sample)
    print(f"  推理模式使用运行统计量，而非批次统计量")
    print(f"  输入 [5,5,5,5] -> {[f'{v:.3f}' for v in eval_out[0]]}")

    print("\n" + "=" * 60)
    print("步骤 5：有无正则化的训练对比")
    print("=" * 60)
    all_data = make_circle_data(n=300, seed=42)
    train_data = all_data[:150]
    test_data = all_data[150:]

    configs = [
        ("无正则化", 0.0, 0.0),
        ("Dropout p=0.3", 0.3, 0.0),
        ("权重衰减 0.01", 0.0, 0.01),
        ("Dropout + 权重衰减", 0.3, 0.01),
    ]

    results = {}
    for name, drop_p, wd in configs:
        print(f"\n--- {name} ---")
        net = RegularizedNetwork(hidden_size=16, lr=0.05, dropout_p=drop_p, weight_decay=wd)
        history = net.train_model(train_data, test_data, epochs=300)
        results[name] = history

    print("\n" + "=" * 60)
    print("最终对比")
    print("=" * 60)
    print(f"  {'配置':30s} {'训练准确率':>10s} {'测试准确率':>10s} {'差距':>8s}")
    print("  " + "-" * 60)
    for name, history in results.items():
        train_loss, train_acc, test_loss, test_acc = history[-1]
        gap = train_acc - test_acc
        print(f"  {name:30s} {train_acc:>9.1f}% {test_acc:>9.1f}% {gap:>7.1f}%")

    print("\n  关键洞察：正则化缩小了训练与测试之间的差距。")
    print("  带 Dropout + 权重衰减的模型泛化能力最强，")
    print("  即便其训练准确率更低。")
