---
name: checkpoint-save-resume
description: 原子化、分片的检查点，完整捕获 RNG 状态，使被中断的运行可在 epoch 中途恢复并保持相同的损失轨迹。
version: 1.0.0
phase: 19
lesson: 47
tags: [training, durability, resume, sharded-state]
---

## 适用场景

任何训练时长超过集群单次作业时限的运行，任何必须能在节点重启后继续执行的运行，以及任何大到无法放入单个载荷的模型。

## 负载结构

```python
{
  "schema": "ckpt.v1",
  "model": model.state_dict(),
  "optimizer": opt.state_dict(),
  "scheduler": sched.state_dict(),
  "state": {"step": int, "epoch": int, "batch_in_epoch": int, "losses": [float, ...]},
  "rng": {"python": ..., "numpy": ..., "torch_cpu": ..., "torch_cuda": ...},
  "wall_saved_at": time.time(),
}
```

## 原子化保存

1. 将负载写入与目标文件同目录的唯一临时文件。
2. `os.replace(tmp, target)` 原子化替换。
3. 绝不直接写入目标文件名。

## 分片布局

- 每个分片一个 `model.shard-NNN.pt`，按键轮询或按参数组拆分。
- `meta.pt` 携带优化器、调度器、训练状态、RNG 以及分片清单。
- `index.json` 携带每个分片的 `sha256`，也携带 `meta.pt` 的哈希值。
- 加载器在合并前验证每个哈希。

## Epoch 中途恢复

- 保存 `(epoch, batch_in_epoch)`，并与 `step` 放在一起。
- 在恢复 epoch 的第一个批次之前恢复 RNG 状态。
- 将生成器快进跳过已消耗的批次。

## 故障模式

- 跨设备重命名：非原子化，会丢失之前的文件。将临时文件放在同目录下。
- 遗忘 RNG：恢复后的损失偏离基线。运行演示中的断言。
- 遗忘优化器状态：恢复后的下一步会突变，相同条件下的差异会迅速放大。
- 剪枝了错误的检查点：保留最近 K 个以及最优的一个。
