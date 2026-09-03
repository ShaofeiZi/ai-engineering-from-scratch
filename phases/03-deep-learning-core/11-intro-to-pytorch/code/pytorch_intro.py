import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os
import time


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_mnist(path="./mnist_data"):
    os.makedirs(path, exist_ok=True)
    for f in MNIST_FILES:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            print(f"  正在下载 {f}...")
            urllib.request.urlretrieve(MNIST_BASE_URL + f, filepath)


def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images


def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class MNISTModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def load_data(data_path="./mnist_data"):
    download_mnist(data_path)
    train_images = load_images(os.path.join(data_path, "train-images-idx3-ubyte.gz"))
    train_labels = load_labels(os.path.join(data_path, "train-labels-idx1-ubyte.gz"))
    test_images = load_images(os.path.join(data_path, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_labels(os.path.join(data_path, "t10k-labels-idx1-ubyte.gz"))
    return train_images, train_labels, test_images, test_labels


def create_loaders(train_images, train_labels, test_images, test_labels, batch_size=64):
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    return train_loader, test_loader


def run_experiment(name, model, train_loader, test_loader, optimizer, device, epochs=10):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")
    print(f"  优化器:  {optimizer.__class__.__name__}")
    print(f"  设备:    {device}")
    print()

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"  轮次 {epoch+1:2d} | "
            f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
            f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"\n  用时: {elapsed:.1f}s (每轮 {elapsed/epochs:.1f}s)")
    print(f"  最终测试准确率: {test_acc:.4f}")
    return test_acc


def experiment_adam(train_loader, test_loader, device):
    model = MNISTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return run_experiment(
        "实验 1: Adam + Dropout",
        model, train_loader, test_loader, optimizer, device
    )


def experiment_sgd(train_loader, test_loader, device):
    model = MNISTModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return run_experiment(
        "实验 2: SGD + Momentum + Dropout",
        model, train_loader, test_loader, optimizer, device
    )


def experiment_batchnorm(train_loader, test_loader, device):
    model = MNISTModelWithBatchNorm().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return run_experiment(
        "实验 3: Adam + BatchNorm(无 Dropout)",
        model, train_loader, test_loader, optimizer, device
    )


def experiment_sgd_cosine(train_loader, test_loader, device, epochs=10):
    model = MNISTModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  实验 4: SGD + 余弦学习率调度")
    print(f"{'='*60}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")
    print(f"  优化器:  SGD (lr=0.05, momentum=0.9) + CosineAnnealing")
    print(f"  设备:    {device}")
    print()

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    test_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  轮次 {epoch+1:2d} | "
            f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
            f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f} | "
            f"学习率: {current_lr:.6f}"
        )
        scheduler.step()

    elapsed = time.time() - start_time
    print(f"\n  用时: {elapsed:.1f}s (每轮 {elapsed/epochs:.1f}s)")
    print(f"  最终测试准确率: {test_acc:.4f}")
    return test_acc


def show_model_info(model, name="模型"):
    print(f"\n  {name} 结构:")
    print(f"  {'-'*40}")
    total = 0
    for pname, param in model.named_parameters():
        print(f"    {pname:30s} {str(list(param.shape)):15s} ({param.numel():,} 个参数)")
        total += param.numel()
    print(f"  {'-'*40}")
    print(f"    合计: {total:,} 个参数")


def demo_tensor_basics():
    print(f"\n{'='*60}")
    print(f"  张量基础")
    print(f"{'='*60}")

    x = torch.randn(3, 4)
    print(f"\n  torch.randn(3, 4):")
    print(f"    shape={x.shape}, dtype={x.dtype}, device={x.device}")

    x_int = x.to(torch.int8)
    print(f"\n  .to(torch.int8):")
    print(f"    dtype={x_int.dtype}")

    y = x.view(2, 6)
    print(f"\n  .view(2, 6):")
    print(f"    shape={y.shape}")

    z = x.unsqueeze(0)
    print(f"\n  .unsqueeze(0):")
    print(f"    shape={z.shape}")


def demo_autograd():
    print(f"\n{'='*60}")
    print(f"  自动求导演示")
    print(f"{'='*60}")

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2 + 3 * x
    z = y.sum()
    z.backward()

    print(f"\n  x = [1.0, 2.0, 3.0]")
    print(f"  y = x^2 + 3x")
    print(f"  z = sum(y) = {z.item():.1f}")
    print(f"  dz/dx = 2x + 3 = {x.grad.tolist()}")

    w = torch.randn(3, requires_grad=True)
    for step in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f"\n  第 {step} 步: loss={loss.item():.4f}, grad={w.grad.tolist()}")
        with torch.no_grad():
            w -= 0.1 * w.grad
        w.grad.zero_()


if __name__ == "__main__":
    print("=" * 60)
    print("  PyTorch 入门 -- 第 3 阶段,第 11 课")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  PyTorch 版本: {torch.__version__}")
    print(f"  设备: {device}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    demo_tensor_basics()
    demo_autograd()

    print(f"\n{'='*60}")
    print(f"  正在加载 MNIST...")
    print(f"{'='*60}")

    train_images, train_labels, test_images, test_labels = load_data()
    print(f"  训练集: {train_images.shape[0]:,} 张图像")
    print(f"  测试集: {test_images.shape[0]:,} 张图像")
    print(f"  图像形状: {train_images.shape[1]} 个特征(28x28 展平后)")
    print(f"  类别: {train_labels.unique().tolist()}")

    train_loader, test_loader = create_loaders(
        train_images, train_labels, test_images, test_labels
    )

    model_preview = MNISTModel()
    show_model_info(model_preview, "MNISTModel(Dropout)")

    model_preview_bn = MNISTModelWithBatchNorm()
    show_model_info(model_preview_bn, "MNISTModel(BatchNorm)")

    acc_adam = experiment_adam(train_loader, test_loader, device)
    acc_sgd = experiment_sgd(train_loader, test_loader, device)
    acc_bn = experiment_batchnorm(train_loader, test_loader, device)
    acc_cosine = experiment_sgd_cosine(train_loader, test_loader, device)

    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    print(f"  Adam + Dropout:           {acc_adam:.4f}")
    print(f"  SGD + Momentum + Dropout: {acc_sgd:.4f}")
    print(f"  Adam + BatchNorm:         {acc_bn:.4f}")
    print(f"  SGD + 余弦调度:           {acc_cosine:.4f}")
    print()

    best_model = MNISTModel().to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        train_one_epoch(best_model, train_loader, criterion, optimizer, device)

    torch.save(best_model.state_dict(), "mnist_mlp.pt")
    print(f"  模型已保存到 mnist_mlp.pt")

    loaded_model = MNISTModel().to(device)
    loaded_model.load_state_dict(
        torch.load("mnist_mlp.pt", map_location=device, weights_only=True)
    )
    _, loaded_acc = evaluate(loaded_model, test_loader, criterion, device)
    print(f"  加载模型的测试准确率: {loaded_acc:.4f}")
