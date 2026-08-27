# 直接偏好优化家族

> Rafailov 等（2023）证明，RLHF 的最优解可以根据偏好数据写成闭式，因此可以跳过显式奖励模型，直接优化策略。这一洞见催生了一个方法家族——IPO、KTO、SimPO、ORPO、BPO——每种方法都针对 DPO 的一种失败模式。到 2026 年，直接对齐算法用于前沿模型后训练的次数已经超过 PPO。但第 2 课的过度优化曲线仍然适用：DAA 并没有逃离 Goodhart，只是改变了问题出现的位置。

**Type:** 学习
**Languages:** Python（标准库，六种偏好损失比较器）
**Prerequisites:** 阶段 18 · 01（指令遵循作为对齐信号），阶段 18 · 02（奖励黑客），阶段 10 · 08（DPO 基础）
**Time:** 约 75 分钟

## 学习目标

- 从带 KL 项的 RLHF 最优解推导 DPO 闭式形式。
- 说明 IPO、KTO、SimPO、ORPO、BPO 分别修复 DPO 的哪一种失败模式。
- 区分“隐式奖励差距”与“偏好强度”，并解释 IPO 的恒等映射为何重要。
- 解释 Rafailov 等（NeurIPS 2024）为何证明，即使没有显式 RM，DAA 仍会过度优化。

## 问题

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此，奖励由最优策略与参考策略之比隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

把它代入 Bradley-Terry 偏好似然后，配分函数 `Z(x)` 会抵消，因为它只依赖 `x`。剩下的损失只包含策略参数，不再需要奖励模型。这就是 DPO。

问题在于，这项推导假设最优解可以到达、偏好数据位于分布内，而且参考策略是真实的众数锚点。这些假设都不完全成立。这个家族中的每种方法，都在修复其中一个被违反的假设。

## 概念

### DPO（Rafailov 等，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出现的问题：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 没有上界。一个很小的偏好也可能产生任意大的差距。
- 损失会把被选与被拒响应的对数概率推向相反方向。只要被拒响应下降得更快，被选响应的绝对对数概率也可能下降。这就是“被选响应退化”（Degraded Chosen Response）现象。
- 分布外偏好（罕见响应与另一个罕见响应组成的配对）会产生任意的隐式奖励。

### IPO（Azar 等，2024）

恒等偏好优化（Identity Preference Optimization）用偏好概率上的恒等映射取代 log-sigmoid。损失变成针对一个有界目标的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

间隔由 `1/(2 beta)` 约束。偏好强度与隐式奖励差距成正比，因此不会爆炸。

### KTO（Ethayarajh 等，2024）

Kahneman-Tversky 优化完全移除了成对结构。给定单个带标签输出，以及二元“理想”或“不理想”信号，它会映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

收益与损失使用不同权重（损失厌恶）。它的优点是可以使用数量丰富得多的非配对数据。

### SimPO（Meng 等，2024）

简单偏好优化（Simple Preference Optimization）让训练信号与生成过程保持一致。它完全移除参考策略，并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

同时使用间隔 `gamma` 来保持稳定。长度归一化消除了利用 DPO 长度偏差失败模式的动机（从构造上说，更长的 `y_w` 会带来更大的对数概率差距）。

### ORPO（Hong 等，2024）

赔率比偏好优化（Odds-Ratio Preference Optimization）在标准 SFT 负对数似然上增加一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

它不需要参考策略——SFT 项就是正则化器。可以通过单一阶段，从基础模型直接训练到对齐模型，无需单独的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

BPO 识别出被选响应退化问题：DPO 会保留 `y_w > y_l` 的排序，但 `y_w` 的绝对对数概率仍可能下降。BPO 添加一行修正，对被选响应的向下变化施加惩罚。据报告，在 Llama-3.1-8B-Instruct 的数学推理任务上，相对 DPO 提升了 10.1% 的准确率。

### 普遍结论：DAA 仍会过度优化

Rafailov 等在《直接对齐算法中奖励模型过度优化的缩放定律》（NeurIPS 2024）中，使用 DPO、IPO 和 SLiC，在多个数据集与不同 KL 预算下训练策略。真实奖励相对 KL 的曲线呈现出与 Gao 等人相同的先达到峰值、再崩溃的形态。隐式奖励会在训练期间查询分布外样本，而 KL 正则化无法使这一过程稳定。

DAA 并没有逃离 Goodhart，只是把问题出现的表面从“奖励模型被过度优化”换成了“参考策略比值被过度优化”。更好的数据、集成模型和提前停止等通用修复，对二者都适用。

### 如何选择（2026）

- 如果有大量成对偏好数据：使用带保守 beta 的 DPO；长度偏差明显时使用 SimPO。
- 如果有未配对的二元反馈：使用 KTO。
- 如果希望从基础模型开始采用单阶段流水线：使用 ORPO。
- 如果 DPO 日志中出现被选响应对数概率退化：使用 BPO。
- 如果偏好强度差异很大且 DPO 已饱和：使用 IPO。

每个实验室都会在一组测试上运行全部五种方法，并针对每项任务选择获胜者。没有理由认为数学推理与安全任务会共享同一个最优方法。

```figure
dpo-margin
```

## 使用它

`code/main.py` 会在一个真实偏好强度随配对变化的玩具偏好数据集上，比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO）。每种损失都使用一个小型 softmax 策略，在相同的 500 个配对样本上优化，并绘制每种方法的最终胜率、被选响应对数概率漂移和隐式奖励分布范围。

## 交付成果

本课会生成 `outputs/skill-preference-loss-selector.md`。给定数据集统计信息（成对或非配对、偏好强度可变或一致、长度分布）与目标（单阶段或先 SFT 再偏好优化），它会推荐一种偏好损失，并报告该方法防范的失败模式。

## 练习

1. 运行 `code/main.py`，报告 DPO 和 BPO 最终的被选响应对数概率下降值。BPO 应保留更高的被选响应绝对概率，请验证这一点。

2. 修改偏好数据，让所有配对具有相同强度。六种方法中哪一种最稳健？哪一种退化？解释 IPO 在此处的优势。

3. 让被拒响应的平均长度达到被选响应的 2 倍。在不改变其他设置的情况下，以数值展示 DPO 如何利用长度，以及 SimPO 如何修复它。

4. Rafailov 等（NeurIPS 2024）声称 DAA 会过度优化。复现一个单点版本：绘制“被选减被拒”的 KL 散度，并观察 DPO 在 beta 很大时的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 在 DPO 上增加的单行修正，并与 `code/main.py` 中的实现核对。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|------------------------|
| DPO | “不需要奖励模型的 RLHF” | 从 RLHF 闭式最优解推导出的、只包含策略参数的损失 |
| 隐式奖励 | “对数比值” | `beta * log(pi(y\|x) / pi_ref(y\|x))`——由 DPO 隐含定义的奖励 |
| IPO | “有界 DPO” | 用恒等映射取代 log-sigmoid；隐式奖励差距上限为 `1/(2 beta)` |
| KTO | “非配对 DPO” | 在单个标签上使用带损失厌恶的前景理论效用 |
| SimPO | “无参考 DPO” | 按长度归一化的对数似然 + 间隔；不使用参考策略 |
| ORPO | “单阶段 DPO” | NLL + 赔率比偏好项；从基础模型一次训练完成 |
| BPO | “保留被选响应的 DPO” | 在 DPO 上增加惩罚，防止被选响应的绝对对数概率下降 |
| 被选响应退化 | “被选响应概率下降” | 只要被拒响应下降更快，DPO 就会降低被选响应的对数概率 |
| DAA | “直接对齐算法” | 跳过显式 RM 的任何偏好损失方法 |

## 延伸阅读

- [Rafailov 等——直接偏好优化（NeurIPS 2023，arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)
- [Azar 等——理解人类偏好学习的通用理论范式（AISTATS 2024，arXiv:2310.12036）](https://arxiv.org/abs/2310.12036)——IPO
- [Ethayarajh 等——KTO：作为前景理论优化的模型对齐（arXiv:2402.01306）](https://arxiv.org/abs/2402.01306)
- [Meng、Xia、Chen——SimPO（NeurIPS 2024，arXiv:2405.14734）](https://arxiv.org/abs/2405.14734)
- [Hong、Lee、Thorne——ORPO（EMNLP 2024，arXiv:2403.07691）](https://arxiv.org/abs/2403.07691)
- [BPO——行为保持优化（ICLR 2026 OpenReview b97EwMUWu7）](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等——DAA 中 RM 过度优化的缩放定律（NeurIPS 2024，arXiv:2406.02900）](https://arxiv.org/abs/2406.02900)
