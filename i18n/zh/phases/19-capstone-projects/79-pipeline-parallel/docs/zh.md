# 管道并行与 Bubble 分析

> 张量并行把矩阵乘法拆到多个 rank 上。管道并行则是把模型按深度拆到多个 rank 上，每个 rank 持有一个 stage。微批次沿着管道流动。开头和结尾那段空闲时间就是 bubble；如何把它压低，就是这门工艺的核心。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 19 阶段 Track C 第 42–49 课
**Time:** 约 90 分钟

## 学习目标

- 把一个顺序模型切成 N 个 stage，并模拟它在 N 个 rank 之间的前向管道。
- 使用 GPipe 调度，也就是先把前向填满，再统一 backward，让 M 个微批次穿过管道，并计算 bubble fraction。
- 把 bubble 与 Megatron-LM 和 PipeDream 使用的交错式 1F1B 调度做比较。
- 说明为什么 stage 划分时更应该追求每个 stage 计算量相等，而不是参数数量相等。

## 问题

一个 70B 参数模型在 fp16 下，仅参数就要占用 140 GB。没有消费级 GPU 能完整装下它。ZeRO-3 会把参数分片到各个 rank 上，但每个 forward step 仍需要各 rank 为当前层执行 allgather，把完整层临时拼出来，因此每层都要支付 log(N) 跳的通信代价。管道并行走的是另一条路：把模型切成 N 个 stage，每个 stage 放到一个 rank 上。layer 1 的前向在 rank 0 上完成后，把激活张量交给 rank 1；rank 1 计算 layer 2，再交给 rank 2；依此类推。backward 则按相反方向流回。因为每个 rank 只持有一个 stage，所以内存近似线性下降；但计算变成了串行流动，于是 bubble 问题随之出现。

bubble 是指管道开始时的空闲时间，也就是等待第一批微批次抵达最后一个 stage；以及结束时的空闲时间，也就是等待最后一批微批次把 backward 彻底回流。对于 M 个微批次、N 个 stage，每个 stage 的 bubble fraction 是 (N-1)/(M+N-1)。当 M=8、N=4 时，它是 27%；当 M=64、N=4 时，它只有 4.5%。只要每一步有足够多的微批次，bubble 就会缩小；这也意味着单个微批次的 batch size 必须变小，而这正是微批次设计要面对的约束。

## 概念

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe 调度

先把全部 M 个微批次依次送入前向传播，把管道填满；在所有前向都结束之后，再统一按相反方向排空 backward。因为每个微批次的激活值都要一直保留到它自己的 backward 开始，内存占用会随 M 线性增长。前向需要 M+N-1 个周期，后向再需要一个 M+N-1 周期。每个 stage 的有效工作量是 2M 个周期，而空闲 bubble 是 2(N-1) 个周期。如果前向和后向都各占一个时间单位，那么 bubble fraction 就是 (N-1)/(M+N-1)。只要让 M 远大于 N，就能把 bubble 隐藏掉。

### 1F1B 调度

交错调度的思路是：一旦某个微批次的前向抵达最后一个 stage，就立刻开始它的 backward，并让它沿管道反向流回。这样每个 stage 会交替执行一次 forward、一次 backward。bubble 仍然是 N-1，但激活内存不再随微批次数增长，而是被管道深度所限制。生产级管道大多采用 1F1B，例如 Megatron 和 PipeDream。本课先实现更简单的 GPipe；1F1B 留作练习扩展。

### 为什么每个 stage 的计算量相等更重要

如果 stage 0 需要 50 ms，而 stage 1 需要 100 ms，那么每个周期都会被 stage 1 卡住。其他 stage 每轮都要空等 50 ms，直到 stage 1 释放结果。按参数量均分是错误的指标：Transformer 的计算主要由注意力和每层 MLP 决定，而 embedding 层参数很多，计算却不重。stage 划分应该尽量让每个 stage 的 FLOPs 接近，而不是让每个 stage 的权重数量接近。

### 微批次与总批次

一条管道会处理 M 个微批次，每个微批次大小为 B。有效总 batch size 是 M*B。一个管道 step 结束时，得到的梯度就是这 M*B 个样本联合的梯度。bubble fraction 取决于 M，而优化器实际感知的是 M*B。调 M 的过程，本质上是在用更低的 bubble 去换取更高的单微批次内存占用；对于 GPipe 来说，M 越大，激活内存也越高。

```figure
cd-pipeline-bubble
```

## 动手构建

`code/main.py` 实现了：

- `PipelineStage`：一个小型 `nn.Module`，持有某个 stage 的参数，并暴露 `forward(activation)`。
- `Pipeline(stages, num_microbatches)`：使用模拟的逐 stage 墙钟时间来编排 GPipe 调度。
- `bubble_fraction(num_stages, num_microbatches)`：闭式公式 (N-1)/(M+N-1)。
- 一个 4-stage 的演示程序：打印逐微批次的执行轨迹，以及测得的 bubble fraction。

运行它：

```bash
python3 code/main.py
```

输出是一张按 stage 和 microbatch 展开的甘特图，以及与闭式公式预测值对比后的 bubble 百分比。

## 生产环境中的常见模式

有三种模式会把管道并行从“概念成立”推进到“工程可用”。

**Activation checkpointing 与 pipeline 天然配套。** 在 GPipe 中，如果有 M 个微批次同时在飞，那么激活内存就是单个微批次激活内存的 M 倍。Activation checkpointing 会在 backward 时重算 forward，用额外计算换内存；这与 pipeline 配合后，才让长序列训练真正可行。

**Stage balance 需要实测，而不是假设。** 生产团队会先跑 profiling pass，在目标硬件上测量每层的真实 FLOPs 和墙钟时间，然后按这个结果做 stage 划分。Megatron-LM 的 `--num-layers-per-stage` 接受一个列表，就是为了在各 stage 单层开销不一致时，允许层数不平均。

**Send-recv 调度必须避免死锁。** 如果管道中每个 stage 都先 send 再 recv，链路就会互相卡死。标准修复方式是交错：偶数 rank 先 send 再 recv，奇数 rank 先 recv 再 send。本课把 rank 调度显式写出来，就是为了让这个模式清楚可见。

## 实际应用

生产模式：

- **Megatron-LM。** 大规模 pipeline parallel 的参考实现。使用 1F1B，并支持把 tensor、pipeline 和 data parallel 组合起来。
- **DeepSpeed Pipeline。** 与 ZeRO 集成得很好；ZeRO-1 + pipeline 是许多超大开源模型的常见组合。
- **PyTorch Pipe。** PyTorch 原生的管道包装器，建立在 `torch.distributed.pipeline.sync.Pipe` 之上。

## 交付成果

第 80 课会把每个 stage 的参数分片写入分片检查点。第 81 课则在概念上把 DDP、ZeRO 和 pipeline 一起组合进端到端演示中，不过为了运行时可控，演示仍然采用的是模拟版 pipeline。

## 练习

1. 实现 1F1B，并验证它的 bubble fraction 与 GPipe 一致，但激活内存是有上界的。
2. 在更深的模型上测量真实的逐 stage 墙钟时间，并按测量结果重新平衡 stage。
3. 在 pipeline 微批次之上叠加梯度累计，并检查得到的梯度是否等于等价总 batch 的前向结果。
4. 把 pipeline 与 activation checkpointing 组合起来，测量内存下降幅度以及增加的计算代价。
5. 把 pipeline 与 DDP 结合，也就是让每个 pipeline rank 再复制到一个数据并行组中，并推演对应的二维调度。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Pipeline | “沿深度做模型并行” | 每个 rank 持有一个 stage，激活值从 stage 流向 stage |
| Bubble | “管道空闲时间” | 开头和结尾各有 N-1 个步骤中，部分 stage 没有工作可做 |
| Microbatch | “总 batch 的切片” | 一个 forward/backward 单元；M 增大时 bubble 会缩小 |
| GPipe | “先填满再排空” | 所有 M 次前向都结束后才开始 backward；激活内存很高 |
| 1F1B | “交错调度” | 每个 stage 交替执行一次 forward 和一次 backward；激活内存有界 |

## 延伸阅读

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- 第 19 阶段第 76 课：调度会直接使用那里的 send/recv 原语
- 第 19 阶段第 78 课：ZeRO 与 pipeline 正交，二者经常组合使用
