# 推测解码与 EAGLE

> 前沿大语言模型每生成一个词元，都需要让数十亿参数完成一次完整前向传播。这次前向传播配置严重过剩：大多数时候，小得多的模型都能正确猜中后续 3～5 个词元，大模型只需*验证*猜测。猜对时，你只用一次计算的代价就得到 5 个词元。推测解码（Leviathan 等，2023）让这一过程在数学上保持精确，EAGLE-3（2025）又把接受率提高到平均每次验证约 4.5 个词元——输出分布相同时可加速 4～5 倍。

**Type:** 构建
**Languages:** Python（使用 numpy）
**Prerequisites:** 阶段 10 第 12 课（推理优化）、阶段 10 第 04 课（预训练 Mini-GPT）
**Time:** 约 75 分钟

## 问题

70B 级模型在 H100 上的典型解码吞吐量是每秒 40～80 个词元。每个词元都要求一次完整前向传播，把全部模型权重从 HBM 读入。不能缩小模型而不改变输出，也不能让批大小超过内存容量。看似已经无路可走——除非可以让模型在一次前向传播中输出不止一个词元。

自回归生成看似天然串行：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在并发机会。如果有一个廉价预测器说“接下来 4 个词元很可能是 [a, b, c, d]”，就可以在大型模型的**一次前向传播**中验证全部 5 个位置，再接受最长的匹配前缀。

Leviathan、Kalai 与 Matias（2023，《Fast Inference from Transformers via Speculative Decoding》）用一条巧妙的接受/拒绝规则，让这一过程保持数学精确，同时保留目标模型的采样分布。输出分布相同，速度提高 2～4 倍。

## 概念

### 双模型设置

- **目标模型** `M_p`：真正希望从中采样的大型、缓慢、高质量模型。分布为 `p(x)`。
- **草稿模型** `M_q`：小型、快速、质量较低的模型。分布为 `q(x)`，规模小 5～30 倍。

每一步如下：

1. 草稿模型以自回归方式提出 `K` 个词元：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型针对全部 `K+1` 个位置只运行*一次*前向传播，为每个候选词元生成 `p(x_k)`。
3. 使用下文修正后的拒绝采样规则，从左到右接受或拒绝各词元，并接受最长匹配前缀。
4. 如果有任意词元被拒绝，就从修正分布采样替代词元并停止；否则，从 `p(· | x_1...x_K)` 采样一个奖励词元。

如果草稿与目标完全一致，每次目标模型前向传播可以得到 K+1 个词元；如果草稿在位置 1 就出错，则只能得到 1 个。

### 精确性规则

推测解码与从 p 直接采样在**分布上可证明等价**。拒绝规则如下：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿与目标一致时（`p ≈ q`），接受概率接近 1；两者不一致时，残差分布经过特殊构造，保证整体样本仍精确服从 `p`。

**贪心情形。** 当 temperature=0 时，只需检查 `argmax(p) == x_t`。若相等就接受，否则输出 `argmax(p)` 并停止。

### 预期加速比

如果草稿模型的逐词元接受率为 `α`，每次目标模型前向传播产生的预期词元数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

当 `α = 0.8, K = 4` 时，`(1 - 0.8^5)/(1 - 0.8) = 3.36`，即每次前向传播产生 3.36 个词元。一次目标前向传播的总成本约为 `cost_q * K + cost_p`（K 个草稿步骤加一次目标验证）。若 `cost_p >> cost_q * K`，吞吐量加速比就是 `3.36× / 1 = 3.36×`。

唯一真正重要的参数是 `α`，它完全取决于草稿模型与目标模型的一致程度。优质草稿决定一切。

### 训练草稿模型：蒸馏

随机的小模型并不是好草稿。标准做法是从目标模型蒸馏：

1. 选择一个小型架构（70B 目标使用约 1B 草稿，7B 目标使用约 500M 草稿）。
2. 在大型文本语料上运行目标模型，存储其下一词元分布。
3. 使用相对于目标分布的 KL 散度训练草稿，而不是使用真实词元训练。

最终，编码任务上的 `α` 通常为 0.6～0.8，自然语言聊天为 0.7～0.85；生产环境可加速 2～3 倍。

### EAGLE：树状草稿 + 特征复用

Li、Wei、Zhang、Zhang（2024，《EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty》）观察到标准推测解码的两个低效之处：

1. 草稿要串行运行 K 步，而且每一步都经过完整网络。但草稿本可以复用最近一次验证时目标模型的特征（隐藏状态）——目标模型已经算出了丰富表示，草稿却在从头重复推导。
2. 草稿输出的是单条线性序列。如果它能输出一棵*候选树*（每个节点有多个猜测），目标模型就能在一次前向传播中借助树注意力掩码并行验证多条候选路径，再选择最长的已接受分支。

EAGLE-1 的变化：
- 草稿输入 = 目标模型在位置 t 的最终隐藏状态，而不是原始词元。
- 草稿架构 = 1 个 Transformer 解码器层，而不是独立小模型。
- 输出 = 每层深度 K = 4～8 个候选组成的树，深度为 4～6。

EAGLE-2（2024）加入动态树拓扑：草稿不确定时让树变宽，自信时保持较窄。在不增加验证成本的情况下，提高 `α_effective`。

EAGLE-3（Li 等，2025，《EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test》）移除了对固定顶层特征的依赖，并使用新的“测试时模拟”损失训练草稿——训练输出与目标模型的测试时分布匹配，而不是与教师强制训练分布匹配。接受率从 0.75（EAGLE-2）提高到 0.82（EAGLE-3），每次验证的平均词元数从 3.0 增至 4.5。

### 树注意力验证

草稿输出一棵树时，目标模型会在一次前向传播中使用**树注意力掩码**验证它——这种因果掩码编码树的拓扑，而不是一条纯线性链。每个词元只能关注其祖先。验证仍然只是一次前向传播与一次矩阵乘法；拓扑掩码只增加少量 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是两个相互竞争的首词元候选，`c, d, e, f` 是第二词元候选，那么全部六个位置都能在一次前向传播中得到验证。输出是所有已接受路径中的最长前缀。

### 何时有效，何时无效

**有效：**
- 具有可预测文本的聊天/补全任务（代码、常见英语、结构化输出）。`α` 较高。
- 解码期间 GPU 计算资源闲置的场景（内存受限阶段）。树状草稿可以利用可用 FLOP。

**无效或没有收益：**
- 高度随机的输出（高温度创意写作）。`α` 会降至接近 `1/|vocab|`。
- 并发极高的批量服务——批处理已经占满 FLOP，几乎没有空间执行树验证。
- 非常小的目标模型，草稿模型与它相比并没有小多少。

生产团队通常报告：聊天任务实际耗时加速 2～3 倍，代码生成加速 3～5 倍，创意写作则接近零收益。

```figure
speculative-decoding
```

## 动手构建

`code/main.py` 包含：

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，执行精确拒绝规则，并验证相对于直接从目标模型采样，其经验 KL < 0.01。
- 一个 EAGLE 风格树状草稿器，以 top-p 分支构建深度为 K 的树。
- 一个树注意力掩码构造器，为验证模型生成正确的因果模式。
- 一个接受率测试框架，在微型语言模型上同时运行两种方案（把一个 GPT-2-small 从 GPT-2-medium 目标模型中蒸馏出来）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 学以致用

- **vLLM** 与 **SGLang** 原生支持推测解码。参数为 `--speculative_model`、`--num_speculative_tokens`；EAGLE-2/3 通过 `--spec_decoding_algorithm eagle` 参数启用。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 与 EAGLE 树。
- **参考草稿模型：** `Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 起草）、`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 起草）。
- **Medusa 头**（Cai 等，2024，《Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads》）：不使用草稿模型，而是在目标模型本身添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付成果

本课会生成 `outputs/skill-speculative-tuning.md`——一项分析目标模型工作负载，并选择草稿模型、K（草稿长度）、树宽、温度和何时回退到普通解码的技能。

## 练习

1. 实现精确拒绝规则并进行实证验证。分别通过 `speculative_decode` 和直接目标模型采样运行 10K 个样本，计算两个输出分布之间的总变差距离，应 < 0.01。

2. 计算加速公式。给定固定 `α` 与 `K`，绘制每次目标前向传播的预期词元数，并找出 α ∈ {0.5, 0.7, 0.9} 时的最优 K。

3. 训练一个微型草稿模型。以 124M GPT-2 为目标，在 100M 词元上使用 KL 损失蒸馏 30M GPT-2 草稿模型，再在留出文本上测量 `α`。预期值为 0.6～0.7。

4. 实现 EAGLE 风格树状起草。不要生成一条链，而是让草稿在每个深度输出 top-3 分支。构建树注意力掩码，验证目标模型会接受最长的正确分支。

5. 测量失效模式。在 temperature=1.5（高随机性）下运行推测解码，证明 α 会坍缩，而且草稿开销会使该算法慢于普通解码。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 目标模型 | “大模型” | 真正希望从中采样的缓慢高质量模型（p 分布） |
| 草稿模型 | “推测器” | 小而快的预测器（q 分布）；规模小 5～30 倍 |
| K / 草稿长度 | “前瞻” | 每次验证所推测的词元数量 |
| α / 接受率 | “命中率” | 草稿提议被接受的逐词元概率 |
| 精确拒绝规则 | “接受测试” | 保持目标分布不变的 r < p/q 比较 |
| 残差分布 | “修正后的 p-q” | (p - q)+ / ||(p - q)+||_1，即拒绝时采样的分布 |
| 树状起草 | “分支式推测” | 草稿输出一棵候选树，再以树结构注意力掩码在一次前向传播中验证 |
| 树注意力掩码 | “拓扑掩码” | 编码树拓扑的因果掩码，使每个节点只关注其祖先 |
| Medusa 头 | “并行输出头” | 目标模型自身的 K 个额外预测头；不需要独立草稿模型 |
| EAGLE 特征复用 | “隐藏状态草稿” | 草稿输入是目标模型最后一层隐藏状态，而不是原始词元，因此草稿更小 |
| 测试时模拟损失 | “EAGLE-3 训练” | 在符合目标模型测试时分布的输出上训练草稿，而不是使用教师强制 |

## 延伸阅读

- [Leviathan、Kalai、Matias，2023——“通过推测解码实现 Transformer 快速推理”](https://arxiv.org/abs/2211.17192)——奠基论文与精确拒绝规则
- [Chen、Borgeaud、Irving 等，2023——“通过推测采样加速大型语言模型解码”](https://arxiv.org/abs/2302.01318)——DeepMind 同期提出的推测采样论文
- [Cai、Li、Geng、Wang、Wang、Zhu、Dao，2024——“Medusa：带多个解码头的简单大语言模型推理加速框架”](https://arxiv.org/abs/2401.10774)——草稿模型的并行头替代方案
- [Li、Wei、Zhang、Zhang，2024——“EAGLE：推测采样需要重新思考特征不确定性”](https://arxiv.org/abs/2401.15077)——特征复用与树状起草
- [Li 等，2024——“EAGLE-2：使用动态草稿树加速语言模型推理”](https://arxiv.org/abs/2406.16858)——动态树拓扑
- [Li 等，2025——“EAGLE-3：通过训练时测试扩展大型语言模型推理加速”](https://arxiv.org/abs/2503.01840)——训练时与测试时匹配
- [Fu、Haotian、Peng 等，2024——“使用前瞻解码打破大语言模型推理的顺序依赖”](https://arxiv.org/abs/2402.02057)——Jacobi/前瞻解码，一种无推测器替代方案
