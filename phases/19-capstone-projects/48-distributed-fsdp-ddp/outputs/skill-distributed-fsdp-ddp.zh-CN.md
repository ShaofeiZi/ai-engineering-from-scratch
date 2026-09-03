---
name: distributed-fsdp-ddp
description: 使用从头编写的 DDP 封装和 FSDP 参数分片草案，在 gloo 或 nccl 后端上搭建多 rank 训练。
version: 1.0.0
phase: 19
lesson: 48
tags: [distributed, ddp, fsdp, collectives]
---

## 适用场景

模型可以放入单个设备但需要更高吞吐量（DDP）。模型无法放入单个设备（FSDP）。两种情况都是：同一代码路径的多 rank 训练设置。

## 拉起进程组

```python
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
```

`gloo` 是 CPU 后端；`nccl` 是 GPU 后端。两者实现相同的集合通信接口。

## 封装模型

1. 在 rank 0 上，从你的种子构建模型。
2. 用 DDP 外壳封装它。
3. 外壳的 `__init__` 对每个参数和缓冲区调用 `dist.broadcast(p.data, src=0)`。
4. 每次 `loss.backward()` 之后，训练器调用 `sync_grads()`。
5. `sync_grads()` 调用 `dist.all_reduce(p.grad, op=SUM)` 和 `p.grad.div_(world_size)`。
6. 在每个 rank 上使用相同的平均梯度执行优化器步进。

## 分片参数（FSDP 草案）

1. 展平每个参数，填充至 `world_size` 的整数倍。
2. 在本地保留你的分片；释放其余部分。
3. 前向传播之前，`dist.all_gather(...)` 在每个 rank 上重建完整张量。
4. 前向传播之后，丢弃完整张量。

## 故障模式

- 跳过广播：各 rank 从不同初始化开始，静默发散。
- 求和后遗忘除法：梯度被放大 world_size 倍，优化器步进过大。
- 检查点使用跨设备重命名：非原子化；与第 47 课的陷阱相同。
- 在同一集合通信中混用 CPU 和 CUDA 张量：后端不匹配，运行挂起。
