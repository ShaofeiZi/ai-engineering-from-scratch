import time
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def timing_comparison():
    print("=== 计时：列表 vs NumPy ===\n")

    size = 1_000_000

    start = time.perf_counter()
    python_list = [x ** 2 for x in range(size)]
    list_time = time.perf_counter() - start
    print(f"列表推导式: {list_time:.4f}s")

    start = time.perf_counter()
    numpy_array = np.arange(size) ** 2
    numpy_time = time.perf_counter() - start
    print(f"NumPy:      {numpy_time:.4f}s")
    print(f"加速比:     {list_time / numpy_time:.1f}x")


def inline_plotting():
    print("\n=== 内联绘图 ===\n")

    np.random.seed(42)
    x = np.linspace(0, 10, 200)
    y_sin = np.sin(x)
    y_noisy = y_sin + np.random.normal(0, 0.2, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, y_sin, label="sin(x)")
    axes[0].plot(x, y_noisy, alpha=0.5, label="noisy")
    axes[0].set_title("信号 vs 噪声")
    axes[0].legend()

    axes[1].hist(y_noisy - y_sin, bins=30, edgecolor="black")
    axes[1].set_title("噪声分布")

    plt.tight_layout()
    plt.savefig("notebook_plot.png", dpi=100)
    print("图表已保存到 notebook_plot.png")
    print("在 notebook 中，plt.show() 会将图表内联显示。")


def dataframe_display():
    print("\n=== DataFrame 展示 ===\n")

    df = pd.DataFrame({
        "model": ["Linear Regression", "Random Forest", "Neural Network", "XGBoost"],
        "accuracy": [0.72, 0.89, 0.94, 0.91],
        "train_time_sec": [0.1, 2.3, 45.6, 8.2],
        "parameters": [102, 50_000, 1_200_000, 25_000],
    })

    print("在 notebook 中，只需输入 'df' 即可渲染出富文本 HTML 表格：\n")
    print(df.to_string(index=False))

    print(f"\n最佳模型: {df.loc[df['accuracy'].idxmax(), 'model']}")
    print(f"最快模型: {df.loc[df['train_time_sec'].idxmin(), 'model']}")


def memory_check():
    print("\n=== 内存占用 ===\n")

    small = np.random.randn(1000)
    medium = np.random.randn(100_000)
    large = np.random.randn(10_000_000)

    for name, arr in [("1K", small), ("100K", medium), ("10M", large)]:
        size_mb = arr.nbytes / 1e6
        print(f"数组 {name:>4s} 元素: {size_mb:>8.2f} MB")

    print(f"\nPython 进程内存: 大数组约占用 ~{sys.getsizeof(large) / 1e6:.1f} MB")
    print("在 notebook 中，内存会随单元格执行不断累积。重启内核即可释放。")


def magic_command_equivalents():
    print("\n=== 魔法命令等价实现 ===\n")
    print("在 notebook 中，你会使用以下魔法命令：")
    print("  %timeit np.random.randn(10000)    -> 微基准测试")
    print("  %%time long_operation()            -> 实际耗时（墙钟时间）")
    print("  %matplotlib inline                 -> 在单元格中显示图表")
    print("  !pip install package               -> 从 notebook 中安装包")
    print("  %env VAR                           -> 查看环境变量")
    print()

    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        np.random.randn(10000)
    elapsed = time.perf_counter() - start
    per_call = elapsed / iterations * 1e6

    print(f"手动计时（类似 %%timeit）：np.random.randn(10000)")
    print(f"  每次调用 {per_call:.1f} us（共 {iterations} 次迭代）")


if __name__ == "__main__":
    print("Notebook 技巧 - 核心模式\n")
    print("请在 Jupyter notebook 中运行以下代码以查看富文本输出。\n")

    timing_comparison()
    inline_plotting()
    dataframe_display()
    memory_check()
    magic_command_equivalents()
