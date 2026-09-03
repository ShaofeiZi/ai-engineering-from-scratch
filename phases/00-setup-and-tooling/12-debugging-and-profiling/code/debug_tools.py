import sys
import time
import tracemalloc
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def debug_print(name, tensor):
    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")


class Timer:
    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"  [{self.name}] {self.elapsed:.4f}s")


def check_shapes(model, sample_input):
    print(f"  输入: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out).__name__
            print(f"    {name}: {in_shape} -> {out_shape}")
        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()


def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"  在第 {step} 步检测到 NaN loss")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"    {name} 中存在 NaN 梯度")
                if torch.isinf(param.grad).any():
                    print(f"    {name} 中存在 Inf 梯度")
        return True
    return False


def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"  模型设备: {model_device}")
    for i, t in enumerate(tensors):
        status = "一致" if t.device == model_device else "不匹配"
        print(f"    张量 {i}: {t.device} [{status}]")


def check_gradient_health(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_norm += grad_norm ** 2
            if grad_norm > 100:
                print(f"    警告: {name} 中存在过大的梯度: {grad_norm:.2f}")
            if grad_norm == 0:
                print(f"    警告: {name} 的梯度为零")
    total_norm = total_norm ** 0.5
    print(f"  梯度总范数: {total_norm:.4f}")
    return total_norm


def demo_print_debugging():
    print("\n--- 1. 张量的打印调试 ---")
    x = torch.randn(32, 784)
    debug_print("输入批次", x)

    w = torch.randn(784, 128)
    out = x @ w
    debug_print("矩阵乘法后", out)

    with_nan = out.clone()
    with_nan[0, 0] = float("nan")
    debug_print("注入 NaN 后", with_nan)


def demo_timing():
    print("\n--- 2. 代码段计时 ---")

    with Timer("1000x1000 矩阵乘法"):
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        _ = a @ b

    with Timer("5000x5000 矩阵乘法"):
        a = torch.randn(5000, 5000)
        b = torch.randn(5000, 5000)
        _ = a @ b


def demo_memory_tracking():
    print("\n--- 3. 内存追踪 (tracemalloc) ---")
    tracemalloc.start()

    if HAS_TORCH:
        data = [torch.randn(100, 100) for _ in range(100)]
        more_data = torch.randn(1000, 1000)
    else:
        data = [bytearray(4 * 100 * 100) for _ in range(100)]
        more_data = bytearray(4 * 1000 * 1000)

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    print("  占用内存最多的前 5 处分配:")
    for stat in top_stats[:5]:
        print(f"    {stat}")

    del data, more_data
    tracemalloc.stop()


def demo_shape_checking():
    print("\n--- 4. 模型形状检查 ---")

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    sample = torch.randn(4, 784)
    check_shapes(model, sample)


def demo_nan_detection():
    print("\n--- 5. NaN 检测 ---")

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    x = torch.randn(4, 784)
    target = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    print(f"  正常 loss: {loss.item():.4f}")
    nan_found = detect_nan(model, loss, step=0)
    print(f"  是否检测到 NaN: {nan_found}")

    fake_nan_loss = torch.tensor(float("nan"))
    print(f"  模拟的 NaN loss: {fake_nan_loss.item()}")
    nan_found = detect_nan(model, fake_nan_loss, step=99)
    print(f"  是否检测到 NaN: {nan_found}")


def demo_device_checking():
    print("\n--- 6. 设备检查 ---")

    model = nn.Linear(10, 5)
    t1 = torch.randn(4, 10)
    t2 = torch.randn(4, 10)

    check_devices(model, t1, t2)

    if torch.cuda.is_available():
        model_gpu = model.cuda()
        t_cpu = torch.randn(4, 10)
        t_gpu = torch.randn(4, 10).cuda()
        print("  混合设备情况下:")
        check_devices(model_gpu, t_cpu, t_gpu)


def demo_gradient_health():
    print("\n--- 7. 梯度健康检查 ---")

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    x = torch.randn(4, 784)
    target = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()

    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    check_gradient_health(model)


def demo_gpu_memory():
    print("\n--- 8. GPU 内存汇总 ---")

    if not torch.cuda.is_available():
        print("  无可用 GPU，跳过 GPU 内存演示。")
        print("  在 GPU 机器上，torch.cuda.memory_summary() 会显示:")
        print("    - 每种块大小的已分配内存")
        print("    - 缓存（预留）内存")
        print("    - 峰值内存使用量")
        return

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

    large_tensor = torch.randn(10000, 10000, device="cuda")
    print(f"  创建 10k x 10k 张量后:")
    print(f"    已分配: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    del large_tensor
    torch.cuda.empty_cache()
    print(f"  清理后:")
    print(f"    已分配: {torch.cuda.memory_allocated() / 1e6:.1f} MB")


def demo_logging():
    print("\n--- 9. 结构化日志 ---")

    logger.info("训练开始: lr=0.001, batch_size=32, epochs=10")
    logger.info("第 100 步: loss=2.3026, accuracy=0.10")
    logger.warning("检测到 loss 突增: 15.7（第 450 步）")
    logger.info("第 1000 步: loss=0.4512, accuracy=0.87")
    logger.info("训练完成: best_loss=0.3201")


def demo_conditional_breakpoint():
    print("\n--- 10. 条件断点模式 ---")
    print("  在实际代码中，可使用如下模式:")
    print()
    print("    for step in range(num_steps):")
    print("        loss = train_step(model, batch)")
    print("        if loss.item() > 10 or torch.isnan(loss):")
    print("            breakpoint()  # 进入 pdb")
    print()
    print("  进入 pdb 后常用的命令:")
    print("    p tensor.shape       # 打印形状")
    print("    p tensor.device      # 检查设备")
    print("    p tensor.grad        # 查看梯度")
    print("    p tensor.isnan().sum()  # 统计 NaN 个数")
    print("    c                    # 继续执行")
    print("    q                    # 退出调试器")


def main():
    print("=" * 60)
    print("  AI 调试与性能分析工具包")
    print("  阶段 0，课程 12")
    print("=" * 60)

    if not HAS_TORCH:
        print("\n未安装 PyTorch。安装命令:")
        print("  uv pip install torch")
        print("\n仅运行非 PyTorch 演示...\n")
        demo_memory_tracking()
        demo_logging()
        return 1

    demo_print_debugging()
    demo_timing()
    demo_memory_tracking()
    demo_shape_checking()
    demo_nan_detection()
    demo_device_checking()
    demo_gradient_health()
    demo_gpu_memory()
    demo_logging()
    demo_conditional_breakpoint()

    print("\n" + "=" * 60)
    print("  全部演示完成。")
    print("  下一步: 故意引入 bug，练习捕获它们。")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
