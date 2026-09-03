---
name: prompt-distributed-training-planner
description: 根据模型规模和可用硬件规划分布式训练任务
version: 1.0.0
phase: 10
lesson: 5
tags: [distributed-training, fsdp, deepspeed, tensor-parallelism, pipeline-parallelism, scaling]
---

# 分布式训练规划器

在规划大语言模型的分布式训练时,使用此框架确定并行策略、内存预算、通信开销和预期吞吐量。

## 输入要求

请提供:
- **模型大小**(以十亿计的参数量)
- **目标训练 tokens**(以万亿计)
- **可用 GPU**(类型:A100/H100/H200,数量,互连:NVLink/InfiniBand)
- **GPU 内存**(A100/H100 为 80GB,H200 为 141GB)
- **节点**(每节点 GPU 数,节点数)
- **预算约束**(最大美元成本,最大墙钟时间)

## 步骤 1:内存预算

计算每个组件在单 GPU 上的内存:

| 组件 | 公式 | FP16 | FP32 |
|-----------|---------|------|------|
| 权重 | params x bytes_per_param | params x 2 | params x 4 |
| Adam 优化器 (m + v) | params x 4 x 2 | 始终 8 字节/参数 | 8 字节/参数 |
| 梯度 | params x bytes_per_param | params x 2 | params x 4 |
| 激活(估算) | seq_len x batch x hidden x layers x 2 | 因情况而异 | 因情况而异 |

若总量超过 GPU 内存,则需要分片。按以下顺序尝试:
1. ZeRO-1(仅分片优化器)-- 通信开销最低
2. ZeRO-2(+ 梯度)-- 通信中等
3. FSDP/ZeRO-3(+ 权重)-- 通信开销最高但内存节省最大
4. 若激活仍然过大,加入激活检查点
5. 若单层无法放入单 GPU,加入张量并行

## 步骤 2:并行策略

### 决策树

1. **单层能否放入单 GPU?**
   - 否:需要张量并行。设 TP = 2、4 或 8(节点内)。
   - 是:跳过张量并行。

2. **完整模型(含分片)能否放入单节点内的 GPU?**
   - 否:需要流水线并行。设 PP = 节点数 / 分组数。
   - 是:跳过流水线并行。

3. **剩余多少 GPU 用于数据并行?**
   - DP = total_gpus / (TP x PP)

4. **数据并行组内使用何种分片级别?**
   - 从 FSDP (ZeRO-3) 开始。若通信成为瓶颈则降级到 ZeRO-2 或 ZeRO-1。

### 典型配置

| 模型大小 | GPU 总数 | TP | PP | DP | 分片 |
|-----------|-----------|----|----|-----|----------|
| 7B | 8 | 1 | 1 | 8 | FSDP |
| 13B | 16 | 2 | 1 | 8 | FSDP |
| 70B | 64 | 8 | 1 | 8 | FSDP |
| 70B | 128 | 8 | 2 | 8 | FSDP |
| 405B | 16,384 | 8 | 16 | 128 | FSDP |

## 步骤 3:通信分析

估算每个训练步的通信量:

- **数据并行(all-reduce)**:每步 2 x gradient_size x (N-1)/N
- **FSDP(all-gather + reduce-scatter)**:每步约 3 x weight_size x (N-1)/N(高于 DP)
- **张量并行(每层 all-reduce)**:每步 2 x activation_size x num_layers(需要 NVLink)
- **流水线并行(点对点)**:每个阶段边界 activation_size(开销极小)

若通信时间超过计算时间的 20%,则策略受通信限制。解决方案:
- 梯度累积(降低 all-reduce 频率)
- 通信与计算重叠(FSDP 默认如此)
- 增大 micro-batch 大小(更好的计算通信比)
- 切换到通信开销更低的分片阶段

## 步骤 4:吞吐量与成本估算

**每个训练步的 FLOPS:**
- 前向:~2 x params x tokens_per_batch
- 反向:~4 x params x tokens_per_batch(前向的 2 倍)
- 总计:~6 x params x tokens_per_batch

**训练时间:**
- total_flops = 6 x params x total_tokens
- time_seconds = total_flops / (num_gpus x gpu_tflops x 1e12 x utilization)
- 典型利用率:35-45%(考虑通信、流水线气泡、内存开销)

**成本:**
- total_gpu_hours = num_gpus x time_seconds / 3600
- cost = total_gpu_hours x cost_per_gpu_hour

## 步骤 5:验证清单

启动前:

1. 单 GPU 内存在硬件限制内(预留 10% 余量)
2. 有效批大小与目标匹配(per_gpu_batch x DP x gradient_accumulation_steps)
3. 通信与计算之比低于 20%
4. 流水线气泡比例低于 15%(足够多的 micro-batch)
5. 学习率已按有效批大小缩放
6. 检查点频率考虑了失败概率(大型训练每 1-2 小时保存一次)
7. 已设置梯度裁剪(大模型通常为 1.0)
8. 预热步数与总步数成比例(通常为总步数的 0.1-1%)

## 危险信号

- **TP > 8**:跨节点(通过 InfiniBand)的张量并行几乎总是比流水线并行慢
- **流水线阶段 > 32**:即使有大量 micro-batch,气泡开销也变得显著
- **有效批大小 > 10M tokens**:收益递减,可能损害收敛
- **利用率低于 30%**:受通信限制 —— 重新评估并行策略
- **13B 以上无激活检查点**:反向传播时将耗尽内存
- **单 GPU 批量较小却无梯度累积**:梯度噪声增大;应累积到 256+ 样本的有效批大小
