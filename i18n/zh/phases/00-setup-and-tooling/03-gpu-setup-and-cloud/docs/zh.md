# GPU 配置与云服务

> 用 CPU 训练足以满足学习需要。真正进行训练时，则需要 GPU。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 0 阶段，第 01 课
**Time:** 约 45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 是否可用
- 在 Google Colab 中配置免费的 T4 GPU，开展云端实验
- 对 CPU 与 GPU 的矩阵乘法进行基准测试，并测量加速倍数
- 使用 fp16 经验法则，估算显存能够容纳的最大模型

## 问题

第 1–3 阶段的大多数课程都可以顺畅地在 CPU 上运行。但当你开始训练 CNN、Transformer 或 LLM（第 4 阶段及以后）时，就需要 GPU 加速。在 CPU 上耗时 8 小时的训练任务，使用 GPU 可能只需 10 分钟。

你有三种选择：本地 GPU、云端 GPU，或者免费的 Google Colab。

## 核心概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## 动手构建

### 方案 1：本地 NVIDIA GPU

先检查电脑是否配备 NVIDIA GPU：

```bash
nvidia-smi
```

安装支持 CUDA 的 PyTorch 后，运行以下代码进行验证：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 方案 2：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. 依次选择 Runtime > Change runtime type > T4 GPU
3. 运行 `!nvidia-smi` 进行验证

你可以将本课程的 Notebook 直接上传到 Colab。

### 方案 3：云端 GPU

使用 Lambda Labs、RunPod 或 Vast.ai 时：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没关系

大多数课程都可以在 CPU 上运行。确实需要 GPU 的课程会明确说明，并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手构建：GPU 与 CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，比较 CPU 与 GPU 的耗时
2. 如果没有 GPU，请在 Google Colab 上运行并比较结果
3. 查看你的 GPU 显存容量，并估算它能容纳的最大模型（经验法则：fp16 中每个参数占 2 字节）

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| CUDA | “GPU 编程” | NVIDIA 的并行计算平台，可让代码在 GPU 上运行 |
| VRAM | “GPU 显存” | GPU 上独立于系统内存的视频内存，它会限制模型规模 |
| fp16 | “半精度” | 16 位浮点格式，内存占用是 fp32 的一半，而精度损失通常很小 |
| Tensor Core | “高速矩阵硬件” | 专用于矩阵乘法的 GPU 核心，速度可达普通核心的 4–8 倍 |
