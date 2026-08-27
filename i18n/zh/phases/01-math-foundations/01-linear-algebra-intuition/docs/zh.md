# 线性代数直觉

> 每个 AI 模型，本质上都只是戴了一顶华丽帽子的矩阵运算。

**Type:** 学习
**Languages:** Python, Julia
**Prerequisites:** 第 0 阶段
**Time:** 约 1 小时

## 学习目标

- 使用 Python 从零实现向量和矩阵运算（加法、点积与矩阵乘法）
- 从几何角度解释点积、投影和 Gram–Schmidt 过程的作用
- 使用行化简判断一组向量的线性无关性、秩与基
- 将线性代数概念与嵌入、注意力分数和 LoRA 等 AI 应用联系起来

## 问题

随便打开一篇机器学习论文，你很可能在第一页就看到向量、矩阵、点积和变换。如果缺乏线性代数直觉，这些只是符号；具备直觉之后，你就能看清神经网络实际在做什么——在空间中移动点。

你不必成为数学家。你需要先理解这些运算在几何上意味着什么，然后亲手把它们编写出来。

## 核心概念

### 向量是点（也是方向）

向量只是一列数字，但这些数字具有含义——它们是空间中的坐标。

**二维向量 [3, 2]：**

| x | y | 点 |
|---|---|-------|
| 3 | 2 | 该向量在平面上从原点 (0,0) 指向 (3, 2) |

这个向量的长度为 sqrt(3^2 + 2^2) = sqrt(13)，方向指向右上方。

在 AI 中，向量可以表示一切：
- 一个词 → 由 768 个数字组成的向量（它在嵌入空间中的“含义”）
- 一张图片 → 由数百万个像素值组成的向量
- 一个用户 → 由偏好组成的向量

### 矩阵是变换

矩阵把一个向量变换成另一个向量。它可以执行旋转、缩放、拉伸或投影。

```mermaid
graph LR
    subgraph Before
        A["Point A"]
        B["Point B"]
    end
    subgraph Matrix["Matrix Multiplication"]
        M["M (transformation)"]
    end
    subgraph After
        A2["Point A'"]
        B2["Point B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在 AI 中，矩阵就是模型本身：
- 神经网络权重 → 把输入变换为输出的矩阵
- 注意力分数 → 决定关注哪些内容的矩阵
- 嵌入 → 把词映射为向量的矩阵

### 点积衡量相似度

两个向量的点积可以反映它们有多相似。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

Same direction:      a · b > 0  (similar)
Perpendicular:       a · b = 0  (unrelated)
Opposite direction:  a · b < 0  (dissimilar)
```

搜索引擎、推荐系统和 RAG 的工作方式正是如此——寻找点积较高的向量。

### 线性无关

如果一组向量中的任何一个都不能表示成其他向量的线性组合，那么这些向量就是线性无关的。如果 v1、v2、v3 线性无关，它们可以张成三维空间；如果其中一个能由其他向量组合得到，它们就只能张成一个平面。

这对 AI 为什么重要：特征矩阵的列应该线性无关。如果两个特征完全相关（线性相关），模型便无法区分它们各自的影响。这会在回归中造成多重共线性——权重矩阵变得不稳定，输入的微小变化可能引发输出的剧烈波动。

**具体示例：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1 和 v2 线性无关——二者都不是另一个向量的标量倍数或线性组合。但 v3 = 2*v1 + v2，因此 {v1, v2, v3} 是线性相关集合。这三个向量都位于 xy 平面内；无论怎样组合，都无法到达 [0, 0, 1]。你虽然有三个向量，却只有两个自由维度。

放到数据集中理解：如果 feature_3 = 2*feature_1 + feature_2，那么加入 feature_3 不会为模型提供任何新信息。更糟的是，它会使正规方程变成奇异方程，因而不存在唯一的权重解。

### 基与秩

基是一组能够张成整个空间的最小线性无关向量集合。基向量的数量就是空间的维数。

三维空间的标准基是 {[1,0,0], [0,1,0], [0,0,1]}，但三维空间中的任意三个线性无关向量都能构成一组有效的基。选择一组基，也就是选择一个坐标系。

矩阵的秩 = 线性无关列的数量 = 线性无关行的数量。如果 rank < min(rows, cols)，这个矩阵就是秩亏的。这意味着：
- 方程组有无穷多个解（或者无解）
- 变换会丢失信息
- 矩阵不可逆

| 情况 | 秩 | 对机器学习的含义 |
|-----------|------|---------------------|
| 满秩（rank = min(m, n)） | 可达到的最大值 | 存在唯一的最小二乘解，模型条件良好 |
| 秩亏（rank < min(m, n)） | 低于最大值 | 特征存在冗余，权重解有无穷多个，需要正则化 |
| 秩为 1 | 1 | 每一列都是同一个向量的缩放副本，所有数据都位于一条直线上 |
| 接近秩亏（奇异值很小） | 数值意义上的低秩 | 矩阵病态，输入中的微小噪声会引发输出大幅变化；应使用 SVD 截断或岭回归 |

### 投影

把向量 **a** 投影到向量 **b** 上，可以得到 **a** 在 **b** 方向上的分量：

```
proj_b(a) = (a dot b / b dot b) * b
```

残差（a - proj_b(a)）与 b 垂直。这种正交分解是最小二乘拟合的基础。

投影在机器学习中无处不在：
- 线性回归会最小化观测值到列空间的距离——其解本身就是一次投影
- PCA 将数据投影到方差最大的方向
- Transformer 中的注意力会计算查询向量在键向量上的投影

```mermaid
graph LR
    subgraph Projection["Projection of a onto b"]
        direction TB
        O["Origin"] --> |"b (direction)"| B["b"]
        O --> |"a (original)"| A["a"]
        O --> |"proj_b(a)"| P["projection"]
        A -.-> |"residual (perpendicular)"| P
    end
```

**示例：**a = [3, 4]，b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

投影会丢弃 y 分量。这就是最简单形式的降维——舍弃你不关心的方向。

### Gram–Schmidt 过程

Gram–Schmidt 过程可以把任意一组线性无关向量转换为一组标准正交基。“标准正交”意味着每个向量长度为 1，并且任意两个向量彼此垂直。

算法步骤：
1. 取第一个向量并将其归一化
2. 取第二个向量，减去它在第一个向量上的投影，再进行归一化
3. 取第三个向量，减去它在此前所有向量上的投影，再进行归一化
4. 对剩余向量重复上述过程

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

QR 分解内部采用的正是这一过程。Q 是标准正交基，R 保存投影系数。QR 分解可用于：
- 求解线性方程组（比高斯消元更稳定）
- 计算特征值（QR 算法）
- 最小二乘回归（标准数值方法）

```figure
eigen-directions
```

## 动手构建

### 第 1 步：从零实现向量（Python）

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### 第 2 步：从零实现矩阵（Python）

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### 第 3 步：理解它对 AI 的意义

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### 第 4 步：Julia 版本

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### 第 5 步：从零实现线性无关判断与投影（Python）

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 实际使用

下面使用 NumPy 完成同样的工作——这才是实践中真正会采用的方式：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 使用 NumPy 计算秩、投影和 QR 分解

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch——张量是支持自动微分的向量

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积对 x 的梯度就是 y，PyTorch 自动完成了计算。神经网络中的每项操作都建立在矩阵乘法、点积和投影等基础运算之上，自动微分则会跟踪贯穿所有运算的梯度。

你刚刚从零实现了 NumPy 一行代码就能完成的功能，现在也理解了它在底层究竟做了什么。

## 交付成果

本课会产出：
- `outputs/prompt-linear-algebra-tutor.md`——帮助 AI 助手通过几何直觉讲解线性代数的提示词

## 知识关联

本课的每个概念都与现代 AI 的具体组成部分相关：

| 概念 | 出现位置 |
|---------|------------------|
| 点积 | Transformer 中的注意力分数、RAG 中的余弦相似度 |
| 矩阵乘法 | 每一层神经网络、每一种线性变换 |
| 线性无关 | 特征选择、避免多重共线性 |
| 秩 | 判断方程组是否可解、LoRA（低秩适配） |
| 投影 | 线性回归（投影到列空间）、PCA |
| Gram–Schmidt / QR | 数值求解器、特征值计算 |
| 标准正交基 | 稳定的数值计算、白化变换 |

LoRA 值得特别说明。它把权重更新分解成低秩矩阵，从而微调大型语言模型。LoRA 不必更新一个 4096x4096 的权重矩阵（1600 万个参数），而只需更新两个大小分别为 4096x16 和 16x4096 的矩阵（13.1 万个参数）。秩为 16 的约束意味着，LoRA 假设权重更新位于完整 4096 维空间中的一个 16 维子空间内。这正是线性代数解决实际问题的例子。

## 练习

1. 实现 `Vector.angle_between(other)`，返回两个向量之间的夹角（单位为度）
2. 创建一个二维缩放矩阵，使 x 坐标翻倍、y 坐标变为三倍，然后将它应用到向量 [1, 1]
3. 给定 5 个随机的类词向量（维度为 50），使用余弦相似度找出最相似的两个
4. 验证 Gram–Schmidt 的输出确实标准正交：检查任意两个向量的点积都为 0，且每个向量的长度都为 1
5. 创建一个秩为 2 的 3x3 矩阵，使用 `rank()` 方法验证，然后解释这些列向量张成什么几何对象
6. 将向量 [1, 2, 3] 投影到 [1, 1, 1] 上。所得结果在几何上表示什么？

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| Vector | “一根箭头” | 表示 n 维空间中一个点或方向的一列数字 |
| Matrix | “一张数字表” | 将向量从一个空间映射到另一个空间的变换 |
| Dot product | “相乘再求和” | 衡量两个向量方向一致程度的指标，也是相似度搜索的核心 |
| Embedding | “某种 AI 魔法” | 表示某个对象（词、图像、用户）含义的向量 |
| Linear independence | “它们不重叠” | 集合中没有任何一个向量可以表示为其他向量的线性组合 |
| Rank | “有多少维” | 矩阵中线性无关列（或行）的数量 |
| Projection | “影子” | 一个向量沿另一个向量方向的分量 |
| Basis | “坐标轴” | 能够张成整个空间的最小线性无关向量集合 |
| Orthonormal | “互相垂直的单位向量” | 两两垂直且长度均为 1 的向量 |
