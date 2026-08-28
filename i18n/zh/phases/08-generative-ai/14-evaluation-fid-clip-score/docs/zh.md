# 评估——FID、CLIP Score 与人类偏好

> 每个生成模型排行榜都会引用 FID、CLIP Score，以及人类偏好竞技场中的胜率。只要研究者有心钻空子，每个数字都有可被利用的失效模式。不了解这些失效模式，就无法分辨真正的改进与刷榜结果。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 8 · 01（分类体系）、阶段 2 · 04（评估指标）
**Time:** 约 45 分钟

## 问题

生成模型要从*样本质量*和*条件遵循程度*两方面接受评判，而二者都没有闭式度量。模型必须渲染 1 万张图像；某种方法必须为它们打分；你还必须相信这些数字能跨模型家族、分辨率和架构进行比较。经历 2014～2026 年大浪淘沙后，留下了三种指标：

- **FID（Fréchet Inception Distance，弗雷歇 Inception 距离）。** 在 Inception 网络的特征空间中，衡量真实分布与生成分布之间的距离。越低越好。
- **CLIP Score。** 生成图像的 CLIP 图像嵌入与提示词的 CLIP 文本嵌入之间的余弦相似度。越高越好，用于衡量提示词遵循程度。
- **人类偏好。** 让两个模型针对同一提示词正面对决，由人类（或 GPT-4 级模型）选择更好的结果，再汇总为 Elo 分数。

你还会遇到 IS（Inception Score，基本已淘汰）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一种都试图修正前一种指标的某项缺陷。

## 概念

![FID、CLIP 与偏好：三个维度，各有不同失效模式](../assets/evaluation.svg)

### FID——样本质量

Heusel 等（2017）提出了以下步骤：

1. 分别从 N 张真实图像和 N 张生成图像中提取 Inception-v3 特征（2048 维）。
2. 对两组特征分别拟合高斯分布：计算均值 `μ_r, μ_g` 与协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

它的含义是特征空间中两个多元高斯分布之间的弗雷歇距离。数值越低，分布越相似。

失效模式：
- **小样本 N 下存在偏差。** FID 对特征分布做均方计算——N 较小时会低估协方差，得到虚低的 FID。务必使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 在 ImageNet 上训练。对于远离 ImageNet 的领域（人脸、艺术作品、含文字图像），FID 没有意义。应改用领域专用的特征提取器。
- **刷指标。** 过拟合 Inception 的先验可以在视觉质量没有提升的情况下得到低 FID。可用下文的 CMMD 克服这一问题。

### CLIP Score——提示词遵循程度

Radford 等（2021）提出。对于一张生成图像 + 提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 3 万张生成图像上取平均值，就得到一个可在模型之间比较的标量。

失效模式：
- **CLIP 自身的盲点。** CLIP 的组合推理能力较弱（“蓝色球体上的红色立方体”经常判断错误）。模型即使没有真正遵循复杂提示词，也可能获得很高的 CLIP Score。
- **短提示词偏差。** 现实数据中，短提示词更容易与 CLIP 图像匹配；较长提示词会在机制上得到更低分数。
- **提示词刷分。** 在提示词中加入“高质量、4k、杰作”，可以抬高 CLIP Score，却不会改善图文绑定。

CMMD（Jayasumana 等，2024）修复了其中一部分问题：它使用 CLIP 特征而非 Inception 特征，并使用最大均值差异而非弗雷歇距离，因此更擅长检测细微质量差异。

### 人类偏好——真实标准

选取一组提示词，用模型 A 和模型 B 分别生成结果。把成对图像展示给人类（或强大的大语言模型裁判），再将胜负汇总为 Elo 或 Bradley-Terry 分数。常见基准包括：

- **PartiPrompts（Google）：** 1600 条多样化提示词，分为 12 类。
- **HPSv2：** 10.7 万条人类标注，广泛用作自动化代理指标。
- **ImageReward：** 13.7 万对提示词-图像偏好数据，采用 MIT 许可证。
- **PickScore：** 在 Pick-a-Pic 的 260 万条偏好数据上训练。
- **Chatbot Arena 风格的图像竞技场：** https://imagearena.ai/ 等。

失效模式：
- **裁判差异。** 非专业人士与专家的偏好不同，两类人都应纳入。
- **提示词分布。** 精心挑选的提示词会偏袒某个模型家族，必须始终记录提示词分布。
- **大语言模型裁判的奖励黑客。** GPT-4 裁判会被漂亮但错误的输出蒙骗。必须结合人类评审交叉验证。

## 组合使用

生产评估报告应包括：

1. 在 1 万～3 万个样本上，对照留出的真实分布计算 FID（样本质量）。
2. 在同一批样本及其提示词上计算 CLIP Score / CMMD（遵循程度）。
3. 在盲测竞技场中对照上一版模型计算胜率（整体偏好）。
4. 失效模式分析：随机抽取 50 个输出，标注已知问题（手部结构、文字渲染、物体数量一致性）。

任何单项指标都是谎言。三项相互印证的指标 + 定性审查，才能构成可信结论。

```figure
gx-fid-distributions
```

## 动手构建

`code/main.py` 在合成“特征向量”上实现 FID、类似 CLIP Score 的指标和 Elo 汇总（以四维向量代替 Inception 特征）。你将看到：

- 在较小 N 和较大 N 上计算 FID——观察偏差。
- 以特征池之间的余弦相似度作为“CLIP Score”。
- 根据合成偏好流执行 Elo 更新规则。

### 第 1 步：用四行代码计算 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第 2 步：CLIP 风格的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第 3 步：Elo 汇总

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 时的 FID。** 经验上，N 小于 1 万时结果不可靠。论文报告低样本量 FID，就是在刷指标。
- **跨分辨率比较 FID。** Inception 的 299×299 缩放会改变特征分布。只能在相同分辨率下比较。
- **只报告一个随机种子。** 至少运行 3 个种子，并报告标准差。
- **通过负向提示词抬高 CLIP Score。** 有些流水线会因过度拟合提示词而提高 CLIP Score。应检查图像是否出现视觉饱和。
- **提示词重叠造成 Elo 偏差。** 如果两个模型在训练中都见过某条基准提示词，Elo 就没有意义。应使用留出的提示词集。
- **付费众包人评偏差。** Prolific、MTurk 的标注者通常更年轻，也更熟悉技术。应混合招募艺术与设计专家。

## 学以致用

2026 年的生产评估协议：

| 支柱 | 最低要求 | 推荐方案 |
|--------|---------|-------------|
| 样本质量 | 在 1 万个样本上对照留出真实数据计算 FID | + 在 5000 个样本上计算 CMMD + 按类别子集计算 FID |
| 提示词遵循程度 | 在 3 万个样本上计算 CLIP Score | + HPSv2 + ImageReward + VQA 式问答 |
| 偏好 | 与基线进行 200 对盲测 | + 2000 对人评 + 大语言模型裁判 + Chatbot Arena |
| 失效分析 | 人工标注 50 个样本 | 人工标注 500 个样本 + 自动安全分类器 |

四个支柱出现在同一份报告中，才能称为结论；只提供一项，就是营销。

## 交付成果

保存 `outputs/skill-eval-report.md`。该技能接收新模型检查点 + 基线，并输出完整评估方案：样本量、指标、失效模式探针和签署标准。

## 练习

1. **简单。** 运行 `code/main.py`，在相同合成分布上比较 N=100 与 N=1000 时的 FID，并报告偏差幅度。
2. **中等。** 用合成的 CLIP 风格特征实现 CMMD（公式参见 Jayasumana 等，2024），比较它与 FID 对质量差异的敏感度。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 的一个子集中选取 1000 对图像与提示词，根据偏好微调一个小型 CLIP 评分器，再测量它与留出集的一致率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| FID | “弗雷歇 Inception 距离” | 真实与生成图像的 Inception 特征各自拟合高斯后，两者之间的弗雷歇距离。 |
| CLIP Score | “图文相似度” | CLIP 图像嵌入与文本嵌入之间的余弦相似度。 |
| CMMD | “FID 的替代品” | 基于 CLIP 特征的 MMD；偏差更小，也不假设高斯分布。 |
| IS | “Inception Score” | exp(E_x[KL(p(y\|x) \|\| p(y))])；与现代模型的相关性很差，已被淘汰。 |
| HPSv2 / ImageReward / PickScore | “学习得到的偏好代理指标” | 在人类偏好数据上训练的小模型，用作自动裁判。 |
| Elo | “国际象棋等级分” | 对成对胜负执行 Bradley-Terry 汇总。 |
| PartiPrompts | “基准提示词集” | Google 精选的 1600 条提示词，覆盖 12 个类别。 |
| FD-DINO | “自监督替代品” | 使用 DINOv2 特征计算的 FD；更适合 ImageNet 以外的领域。 |

## 生产说明：评估也是一种推理负载

在 1 万个样本上计算 FID，意味着生成 1 万张图像。对于单张 L4 上以 1024² 分辨率运行 50 步的 SDXL 基础模型，这相当于约 11 小时的单请求推理。评估预算是真实成本，其形态正是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **尽量增大批次，不必关心延迟。** 离线评估应采用能装入内存的最大静态批次。在 80GB H100 上调用 `pipe(...).images` 并设置 `num_images_per_prompt=8`，其总耗时可比单请求快 4～6 倍。
- **缓存真实特征。** 对真实参考集提取 Inception（FID）或 CLIP（CLIP Score、CMMD）特征只需执行*一次*，并将结果存为 `.npz`。不要在每次评估时重新计算。

对于 CI / 回归门禁：每个 PR 在 500 个样本的子集上运行 FID + CLIP Score（约 30 分钟）；每晚运行完整的 1 万样本 FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel 等（2017），以双时间尺度更新规则训练的 GAN 收敛到局部纳什均衡（FID）](https://arxiv.org/abs/1706.08500)——FID 论文。
- [Jayasumana 等（2024），重新思考 FID：迈向更好的图像生成评估指标（CMMD）](https://arxiv.org/abs/2401.09603)——CMMD。
- [Radford 等（2021），从自然语言监督中学习可迁移视觉模型（CLIP）](https://arxiv.org/abs/2103.00020)——CLIP。
- [Wu 等（2023），HPSv2：全面的人类偏好评分](https://arxiv.org/abs/2306.09341)——HPSv2。
- [Xu 等（2023），ImageReward：学习并评估人类对文生图的偏好](https://arxiv.org/abs/2304.05977)——ImageReward。
- [Yu 等（2023），扩展用于内容丰富文生图的自回归模型（Parti + PartiPrompts）](https://arxiv.org/abs/2206.10789)——PartiPrompts。
- [Stein 等（2023），揭示生成模型评估指标的缺陷](https://arxiv.org/abs/2306.04675)——失效模式综述。
