---
name: prompt-tensor-debugger
description: 用于逐步调试深度学习代码中张量形状错误的提示词
phase: 1
lesson: 12
---

我在深度学习代码中有子形状错误.

**错误信息:** [点击错误在这里]

**我的子形状:**
- 现在,我们要去看看.
- 现在,我们要去看看.

**我试图做的手术:** 描述它

---

在调试时,请遵循以下过程:

**步骤1:确定操作类型.**
错误是什么操作造成的?
- 矩阵乘法 / 线性层 (内面尺寸必须匹配)
- 广播 (右边的直线,每个必须等于或1)
- 缩 (除了猫尺寸外,所有缩相匹配)
- 曲 (预计具体的排名和道位置)
- 改造 (必须保留全部元素)

**步骤2:写出形状合同.**
对于已确定的操作,明确写出预期的形状:
```
matmul(A, B): A is (..., m, k), B is (..., k, n) -> (..., m, n)
broadcast(A, B): align right, each pair must be (equal) or (one is 1)
cat([A, B], dim=d): all dims match except dim d
Linear(in_f, out_f): input last dim must equal in_f
Conv2d(in_c, out_c, k): input must be (B, in_c, H, W)
```

**步骤3:找到不匹配.**
根据合同,比较实际形状,确定违反规则的确切尺寸.

**选择最小的解决方案.**
选择这个桌子:

| 症状 | 修复 |
|---|---|
| 缺少批量尺寸 | `.unsqueeze(0)` |
| 缺失道尺寸 | `.unsqueeze(1)` |
| 额外尺寸-1维度 | `.squeeze(dim)` |
| 内部色不适合 | `.transpose(-1, -2)` 或检查体重形状 |
| 需要 NCHW 根据 NHWC | `.permute(0, 3, 1, 2)` |
| 需要 NHWC 根据 NCHW | `.permute(0, 2, 3, 1)` |
| 线性空间平面 | `.flatten(1)` 或 `.reshape(B, -1)` |
| 分头: (B,T,D) 到 (B,H,T,D/H) | `.reshape(B, T, H, D//H).transpose(1, 2)` |
| 合并头: (B,H,T,D/H) 到 (B,T,D) | `.transpose(1, 2).reshape(B, T, H*(D//H))` |
| 没有连接的子与 .view() | `.contiguous().view(...)` 或使用 `.reshape(...)` |

**步骤5:检查检查.**
确认所有重塑元素都保留在任何重塑中.确认操作的形状合同现在已满足.

**步骤 6:检查是否有无声虫.**
即使形状相匹配,请检查:
- 广播在预期轴上发生 (不是偶然的)
- 减少是把正确的尺寸总结
- 批量维度 (dim 0) 通过整个前行通过存活
- 转换+重塑是使用的 (不仅重塑) 当维度排序问题

按以下形式格式对您的反应:
```
OPERATION: [what operation failed]
EXPECTED: [shape contract]
ACTUAL: [what shapes were provided]
MISMATCH: [which dimension, why]
FIX: [exact code]
RESULT: [shapes after fix]
```
