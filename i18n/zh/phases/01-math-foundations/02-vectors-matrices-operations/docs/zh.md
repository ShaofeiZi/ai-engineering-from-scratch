# 向量、矩阵与运算

> 每个神经网络，本质上都只是多了几道步骤的矩阵乘法。

**Type:** 构建
**Languages:** Python, Julia
**Prerequisites:** 第 1 阶段，第 01 课（Linear Algebra Intuition）
**Time:** 约 1 小时

## 学习目标

- 构建一个支持逐元素运算、矩阵乘法、转置、行列式和逆矩阵的 Matrix 类
- 区分逐元素乘法与矩阵乘法，并解释二者各自适用的场景
- 仅使用从零实现的 Matrix 类构建一个全连接神经网络层（`relu(W @ x + b)`）
- 解释广播规则，以及神经网络框架如何完成偏置加法

## 问题

你准备构建一个神经网络，读代码时看到了这一行：

```
output = activation(weights @ input + bias)
```

其中的 `@` 表示矩阵乘法，`weights` 是矩阵，`input` 是向量。如果不了解这些运算，这一行就像魔法；如果理解它们，你会发现这三个运算已经构成了一个网络层的完整前向传播。

模型处理的每张图片都是由像素值组成的矩阵，每个词嵌入都是一个向量，每一层神经网络都是一次矩阵变换。不熟练掌握矩阵运算，就无法构建 AI 系统，正如不理解变量就无法编写程序一样。

本课将从零开始培养这种熟练度。

## 核心概念

### 向量：有序数字列表

向量是一列具有方向和长度的数字。在 AI 中，向量用于表示数据点、特征或参数。

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度（模）为 5，也就是经典的 3-4-5 直角三角形。

### 矩阵：数字网格

矩阵是由行和列组成的二维网格。一个 m x n 矩阵有 m 行、n 列。

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

在神经网络中，权重矩阵把输入向量变换成输出向量。一个拥有 784 个输入和 128 个输出的网络层，会使用 128x784 的权重矩阵。

### 为什么形状很重要

矩阵乘法遵循严格规则：`(m x n) @ (n x p) = (m x p)`。两个内部维度必须相等。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

如果 PyTorch 报出形状不匹配错误，原因就在这里。

### 运算一览

| 运算 | 作用 | 在神经网络中的用途 |
|-----------|-------------|-------------------|
| 加法 | 逐元素合并 | 给输出添加偏置 |
| 标量乘法 | 缩放每个元素 | 学习率 × 梯度 |
| 矩阵乘法 | 变换向量 | 网络层的前向传播 |
| 转置 | 交换行与列 | 反向传播 |
| 行列式 | 用单个数字概括矩阵性质 | 检查矩阵是否可逆 |
| 逆矩阵 | 撤销一次变换 | 求解线性方程组 |
| 单位矩阵 | 不改变输入的矩阵 | 初始化、残差连接 |

### 逐元素乘法与矩阵乘法

初学者经常混淆这两种运算。

逐元素乘法：将相同位置的元素相乘。两个矩阵必须具有相同形状。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：计算第一个矩阵各行与第二个矩阵各列的点积，两个内部维度必须相等。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

二者是不同的运算，有不同的结果，也遵循不同的规则。

### 广播

当你把一个偏置向量加到输出矩阵上时，二者的形状并不相同。广播会沿缺失的维度扩展较小的数组，使形状匹配。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

所有现代框架都会自动进行广播。理解这一机制，可以避免代码明明能运行、形状看起来却不匹配时产生困惑。

```figure
vector-projection
```

## 动手构建

### 第 1 步：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：包含核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：查看运行效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第 4 步：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个全连接层：`output = relu(W @ x + b)`。每个神经网络中的每个全连接层，本质上都执行完全相同的操作。

## 实际使用

NumPy 能够用更少的代码、更快几个数量级地完成上面的所有操作。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符会调用 `__matmul__`。NumPy 使用以 C 和 Fortran 编写的优化 BLAS 例程来实现它：数学完全相同，速度却快 100 倍。

NumPy 中的广播：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 会自动把一维偏置广播到两行。这就是所有神经网络框架实现偏置加法的方式。

## 交付成果

本课会产出一份通过几何直觉讲解矩阵运算的提示词，参见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类，是我们在第 3 阶段第 10 课中实现迷你神经网络框架的基础。

## 练习

1. **验证逆矩阵。**计算 `A @ A.inverse_2x2()`，确认结果是单位矩阵。使用三个不同的 2x2 矩阵进行测试。行列式为零时会发生什么？

2. **实现 3x3 逆矩阵。**扩展 Matrix 类，使用伴随矩阵法计算 3x3 矩阵的逆，并与 NumPy 的 `np.linalg.inv` 对照测试。

3. **构建两层网络。**只使用你实现的 Matrix 类（不使用 NumPy）创建一个两层神经网络：输入 (3) -> 隐藏层 (4) -> 输出 (2)。随机初始化权重，运行一次前向传播，并验证所有形状都正确。

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| Vector | “一根箭头” | 一列有序数字；在 AI 中表示高维空间里的一个点 |
| Matrix | “一张数字表” | 一种线性变换，将向量从一个空间映射到另一个空间 |
| Matrix multiply | “把数字乘起来就行” | 计算第一个矩阵每一行与第二个矩阵每一列的点积；顺序很重要 |
| Transpose | “翻转一下” | 交换行与列，把 m x n 矩阵变成 n x m 矩阵；对反向传播至关重要 |
| Determinant | “从矩阵算出的某个数字” | 衡量矩阵对面积（二维）或体积（三维）的缩放程度；为零表示变换压扁了一个维度 |
| Inverse | “撤销矩阵” | 能够逆转该变换的矩阵；只有行列式不为零时才存在 |
| Identity matrix | “最无聊的矩阵” | 相当于乘以 1 的矩阵，用于残差连接（ResNet） |
| Broadcasting | “神奇地修正形状” | 沿缺失维度重复较小数组，使其形状与较大数组匹配 |
| Element-wise | “普通乘法” | 将相同位置的元素相乘；两个数组必须形状相同（或可广播） |

## 延伸阅读

- [3Blue1Brown：线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra)——本课所有运算的可视化直觉
- [NumPy 广播文档](https://numpy.org/doc/stable/user/basics.broadcasting.html)——NumPy 遵循的准确规则
- [Stanford CS229 线性代数复习资料](http://cs229.stanford.edu/section/cs229-linalg.pdf)——面向机器学习的精炼参考资料
