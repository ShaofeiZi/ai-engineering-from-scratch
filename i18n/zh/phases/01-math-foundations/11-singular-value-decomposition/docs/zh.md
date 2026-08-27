# 奇异值分解

> SVD 是线性代数中的瑞士军刀。每个矩阵都有 SVD，每位数据科学家都需要掌握它。

**Type:** 构建
**Languages:** Python, Julia
**Prerequisites:** 第 1 阶段，第 01 课（Linear Algebra Intuition）、第 02 课（Vectors & Matrices Operations）和第 03 课（Matrix Transformations）
**Time:** 约 2 小时

## 学习目标

- 通过幂迭代实现 SVD，并解释 U、Sigma 和 V^T 的几何含义
- 使用截断 SVD 压缩图像，衡量压缩率与重建误差之间的取舍
- 通过 SVD 计算 Moore-Penrose 伪逆，求解超定最小二乘系统
- 将 SVD 与 PCA、推荐系统（潜在因子）以及 NLP 中的潜在语义分析联系起来

## 问题

你有一个 1000x2000 的矩阵。它可能是用户—电影评分矩阵、文档—词频表，也可能是一张图像的像素值。你需要压缩它、去除噪声、发现其中隐藏的结构，或者用它求解最小二乘系统。特征分解只能用于方阵；即便矩阵是方阵，也必须拥有完整的一组线性无关特征向量。

SVD 可以作用于任何矩阵，无论形状和秩如何，都不需要额外条件。它把矩阵分解成三个因子，揭示该矩阵作用于空间时的几何结构。SVD 是整个线性代数中适用范围最广、用途最多的矩阵分解方法。

## 核心概念

### SVD 在几何上做了什么

任何矩阵，无论形状如何，都按顺序执行三项操作：旋转、缩放、再旋转。SVD 会把这组操作明确地分解出来。

```
A = U * Sigma * V^T

      m x n     m x m    m x n    n x n
     (any)    (rotate)  (scale)  (rotate)
```

给定任意矩阵 A，SVD 会将其分解为：
- V^T 在输入空间（n 维）中旋转向量
- Sigma 沿每条轴进行缩放（拉伸或压缩）
- U 把结果旋转到输出空间（m 维）

```mermaid
graph LR
    A["Input space (n-dim)\nData cloud\n(arbitrary orientation)"] -->|"V^T\n(rotate)"| B["Scaled space\nAligned with axes\nthen scaled by Sigma"]
    B -->|"U\n(rotate)"| C["Output space (m-dim)\nRotated to output\norientation"]
```

可以这样理解：你把一个矩阵交给 SVD，它会告诉你：“这个矩阵先通过 V^T 旋转输入球体，再通过 Sigma 将它拉伸成椭球体，最后通过 U 旋转这个椭球体。”奇异值就是椭球各条轴的长度。

### 完整分解

对于形状为 m x n 的矩阵 A：

```
A = U * Sigma * V^T

where:
  U     is m x m, orthogonal (U^T U = I)
  Sigma is m x n, diagonal (singular values on the diagonal)
  V     is n x n, orthogonal (V^T V = I)

The singular values sigma_1 >= sigma_2 >= ... >= sigma_r > 0
where r = rank(A)
```

U 的各列称为左奇异向量，V 的各列称为右奇异向量，Sigma 的对角元素称为奇异值。奇异值始终非负，并且按惯例从大到小排列。

### 左奇异向量、奇异值与右奇异向量

SVD 的每个组成部分都有不同的几何含义。

**右奇异向量（V 的各列）：**它们构成输入空间（R^n）的一组标准正交基。这些方向经过矩阵映射后，会变成输出空间中的正交方向。可以把它们看作定义域最自然的坐标系。

**奇异值（Sigma 的对角元素）：**它们是缩放因子。第 i 个奇异值表示矩阵沿第 i 个右奇异向量方向拉伸多少倍。奇异值为零，表示该方向被矩阵完全压扁。

**左奇异向量（U 的各列）：**它们构成输出空间（R^m）的一组标准正交基。第 i 个左奇异向量，是第 i 个右奇异向量经过缩放后落入输出空间的方向。

三者之间的关系为：

```
A * v_i = sigma_i * u_i

The matrix A takes the i-th right singular vector v_i,
scales it by sigma_i, and maps it to the i-th left singular vector u_i.
```

这个关系逐坐标说明了任意矩阵究竟对空间做了什么。

### 外积形式

SVD 还可以写成若干秩为 1 的矩阵之和：

```
A = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T + ... + sigma_r * u_r * v_r^T

Each term sigma_i * u_i * v_i^T is a rank-1 matrix (an outer product).
The full matrix is the sum of r such matrices, where r is the rank.
```

这种形式是低秩近似的基础。每一项都添加一层结构：第一项捕获最重要的单一模式，第二项捕获次重要模式，依此类推。截断这个和，就能得到任意给定秩下最优的近似。

```
Rank-1 approx:    A_1 = sigma_1 * u_1 * v_1^T
                  (captures the dominant pattern)

Rank-2 approx:    A_2 = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T
                  (captures the two most important patterns)

Rank-k approx:    A_k = sum of top k terms
                  (optimal by the Eckart-Young theorem)
```

### 与特征分解的关系

SVD 与特征分解密切相关。A 的奇异值和奇异向量，可以直接由 A^T A 和 A A^T 的特征值与特征向量得到。

```
A^T A = V * Sigma^T * U^T * U * Sigma * V^T
      = V * Sigma^T * Sigma * V^T
      = V * D * V^T

where D = Sigma^T * Sigma is a diagonal matrix with sigma_i^2 on the diagonal.

So:
- The right singular vectors (V) are eigenvectors of A^T A
- The singular values squared (sigma_i^2) are eigenvalues of A^T A

Similarly:
A A^T = U * Sigma * V^T * V * Sigma^T * U^T
      = U * Sigma * Sigma^T * U^T

So:
- The left singular vectors (U) are eigenvectors of A A^T
- The eigenvalues of A A^T are also sigma_i^2
```

这一联系告诉我们三件事：
1. 奇异值始终是实数且非负，因为它们是半正定矩阵特征值的平方根。
2. 可以通过对 A^T A 做特征分解来计算 SVD，但这会把条件数平方，损失数值精度。专用 SVD 算法可以避免这个问题。
3. 当 A 是方形的对称半正定矩阵时，SVD 与特征分解完全相同。

### 截断 SVD：低秩近似

Eckart-Young-Mirsky 定理指出，在 Frobenius 范数和谱范数下，A 的最佳秩 k 近似，都是保留前 k 个奇异值及对应向量得到的：

```
A_k = U_k * Sigma_k * V_k^T

where:
  U_k     is m x k  (first k columns of U)
  Sigma_k is k x k  (top-left k x k block of Sigma)
  V_k     is n x k  (first k columns of V)

Approximation error = sigma_{k+1}  (in spectral norm)
                    = sqrt(sigma_{k+1}^2 + ... + sigma_r^2)  (in Frobenius norm)
```

这不只是“一种不错的”近似，而是可以证明的最佳秩 k 近似。任何其他秩 k 矩阵与 A 的距离都不会更小。

| 分量 | 相对大小 | 秩为 3 的近似中保留？ |
|-----------|-------------------|------------------------|
| sigma_1 | 最大 | 是 |
| sigma_2 | 大 | 是 |
| sigma_3 | 中等偏大 | 是 |
| sigma_4 | 中等 | 否（误差） |
| sigma_5 | 中等偏小 | 否（误差） |
| sigma_6 | 小 | 否（误差） |
| sigma_7 | 很小 | 否（误差） |
| sigma_8 | 极小 | 否（误差） |

保留前三项时，A_3 会捕获三个最大的奇异值，误差则来自剩余的 sigma_4 到 sigma_8。

如果奇异值衰减很快，较小的 k 就能捕获矩阵的大部分信息；如果衰减很慢，矩阵就没有明显的低秩结构。

### 使用 SVD 压缩图像

灰度图像是像素强度组成的矩阵。一张 800x600 图像包含 480,000 个数值，SVD 可以用少得多的数据近似它。

```
Original image: 800 x 600 = 480,000 values

SVD with rank k:
  U_k:      800 x k values
  Sigma_k:  k values
  V_k:      600 x k values
  Total:    k * (800 + 600 + 1) = k * 1401 values

  k=10:   14,010 values   (2.9% of original)
  k=50:   70,050 values  (14.6% of original)
  k=100: 140,100 values  (29.2% of original)

  The compression ratio improves as k gets smaller,
  but visual quality degrades.
```

关键在于，自然图像的奇异值通常衰减得很快。前几个奇异值捕获宽泛结构，例如形状和渐变；后面的奇异值则捕获细节与噪声。截断到秩 50 时，图像通常看起来几乎与原图相同，却能减少 85% 的存储空间。

### 推荐系统中的 SVD

Netflix Prize 让这种方法广为人知。设想一个用户—电影评分矩阵，其中大多数元素都是缺失的。

```
             Movie1  Movie2  Movie3  Movie4  Movie5
  User1      [  5      ?       3       ?       1  ]
  User2      [  ?      4       ?       2       ?  ]
  User3      [  3      ?       5       ?       ?  ]
  User4      [  ?      ?       ?       4       3  ]

  ? = unknown rating
```

核心思想是：这个评分矩阵具有低秩结构。用户的品味并非完全相互独立，少数潜在因子就能解释大多数偏好，例如动作片与剧情片、老电影与新电影、理性风格与感官刺激。

对填补后的评分矩阵执行 SVD，会得到：
- U：潜在因子空间中的用户画像
- Sigma：每个潜在因子的重要程度
- V^T：潜在因子空间中的电影画像

一个用户对一部电影的预测评分，等于其用户画像与电影画像的点积，再按奇异值加权。低秩近似会填补矩阵中的缺失元素。

实践中通常使用 Simon Funk 的增量 SVD 或 ALS（交替最小二乘）等能够直接处理缺失数据的变体，但核心思想相同：通过 SVD 分解潜在因子。

### NLP 中的 SVD：潜在语义分析

潜在语义分析（LSA，也称潜在语义索引 LSI）会对词项—文档矩阵应用 SVD。

```
             Doc1   Doc2   Doc3   Doc4
  "cat"      [  3      0      1      0  ]
  "dog"      [  2      0      0      1  ]
  "fish"     [  0      4      1      0  ]
  "pet"      [  1      1      1      1  ]
  "ocean"    [  0      3      0      0  ]

After SVD with rank k=2:

  Each document becomes a point in 2D "concept space."
  Each term becomes a point in the same 2D space.
  Documents about similar topics cluster together.
  Terms with similar meanings cluster together.

  "cat" and "dog" end up near each other (land pets).
  "fish" and "ocean" end up near each other (water concepts).
  Doc1 and Doc3 cluster if they share similar topics.
```

LSA 是最早成功从原始文本中捕获语义相似度的方法之一。它之所以有效，是因为同义词往往出现在相似文档中，SVD 因而会把它们归入相同的潜在维度。现代词嵌入（Word2Vec、GloVe）可以看作这一思想的后继方法。

### 使用 SVD 去噪

在含噪数据中，信号集中于最大的几个奇异值，噪声则分散在所有奇异值上。截断 SVD 可以移除噪声底。

**干净信号的奇异值：**

| 分量 | 大小 | 类型 |
|-----------|-----------|------|
| sigma_1 | 非常大 | 信号 |
| sigma_2 | 大 | 信号 |
| sigma_3 | 中等 | 信号 |
| sigma_4 | 接近零 | 可忽略 |
| sigma_5 | 接近零 | 可忽略 |

**含噪信号的奇异值（噪声会影响全部分量）：**

| 分量 | 大小 | 类型 |
|-----------|-----------|------|
| sigma_1 | 非常大 | 信号 |
| sigma_2 | 大 | 信号 |
| sigma_3 | 中等 | 信号 |
| sigma_4 | 小 | 噪声 |
| sigma_5 | 小 | 噪声 |
| sigma_6 | 小 | 噪声 |
| sigma_7 | 小 | 噪声 |

```mermaid
graph TD
    A["All singular values"] --> B{"Clear gap?"}
    B -->|"Above gap"| C["Signal: keep these (top k)"]
    B -->|"Below gap"| D["Noise: discard these"]
    C --> E["Reconstruct with A_k to get denoised version"]
```

这种方法用于信号处理、科学测量和数据清洗。只要矩阵受到加性噪声污染，截断 SVD 都是一种有理论依据的信号—噪声分离方法。

### 通过 SVD 计算伪逆

Moore-Penrose 伪逆 A+ 把矩阵求逆推广到了非方阵和奇异矩阵，而 SVD 使它的计算非常简单。

```
If A = U * Sigma * V^T, then:

A+ = V * Sigma+ * U^T

where Sigma+ is formed by:
  1. Transpose Sigma (swap rows and columns)
  2. Replace each non-zero diagonal entry sigma_i with 1/sigma_i
  3. Leave zeros as zeros

For A (m x n):      A+ is (n x m)
For Sigma (m x n):  Sigma+ is (n x m)
```

伪逆可以求解最小二乘问题。如果 Ax = b 没有精确解（超定系统），那么 x = A+ b 就是最小二乘解，即使 ||Ax - b|| 最小的解。

```
Overdetermined system (more equations than unknowns):

  [1  1]         [3]
  [2  1] x   =   [5]       No exact solution exists.
  [3  1]         [6]

  x_ls = A+ b = V * Sigma+ * U^T * b

  This gives the x that minimizes the sum of squared residuals.
  Same result as the normal equations (A^T A)^(-1) A^T b,
  but numerically more stable.
```

### 数值稳定性优势

对 A^T A 做特征分解会将奇异值平方，因为 A^T A 的特征值是 sigma_i^2。这也会把条件数平方，放大数值误差。

```
Example:
  A has singular values [1000, 1, 0.001]
  Condition number of A: 1000 / 0.001 = 10^6

  A^T A has eigenvalues [10^6, 1, 10^{-6}]
  Condition number of A^T A: 10^6 / 10^{-6} = 10^{12}

  Computing SVD directly: works with condition number 10^6
  Computing via A^T A:     works with condition number 10^{12}
                           (6 extra digits of precision lost)
```

现代 SVD 算法（Golub-Kahan 双对角化）直接作用于 A，从不显式构造 A^T A。因此应始终优先使用 `np.linalg.svd(A)`，而不是 `np.linalg.eig(A.T @ A)`。

### 与 PCA 的联系

PCA 就是对中心化数据执行 SVD。这并非类比，而是完全相同的计算。

```
Given data matrix X (n_samples x n_features), centered (mean subtracted):

Covariance matrix: C = (1/(n-1)) * X^T X

PCA finds eigenvectors of C. But:

  X = U * Sigma * V^T    (SVD of X)

  X^T X = V * Sigma^2 * V^T

  C = (1/(n-1)) * V * Sigma^2 * V^T

So the principal components are exactly the right singular vectors V.
The explained variance for each component is sigma_i^2 / (n-1).

In sklearn, PCA is implemented using SVD, not eigendecomposition.
It is faster and more numerically stable.
```

这意味着，第 10 课关于降维的所有内容，底层实现都是 SVD。PCA 是 SVD 在机器学习中最常见的应用。

```figure
svd-rank-reconstruction
```

## 动手构建

### 第 1 步：使用幂迭代从零实现 SVD

思路是：通过对 A^T A（或 A A^T）执行幂迭代，找到最大的奇异值及其向量；然后对矩阵进行降秩，再重复这一过程，求下一个奇异值。

```python
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V
```

### 第 2 步：测试并与 NumPy 比较

```python
np.random.seed(42)
A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Our singular values:", np.round(S_ours, 4))
print("NumPy singular values:", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")
```

### 第 3 步：图像压缩演示

```python
def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d}  error={error:.4f}  storage={ratio:.1%}")
```

### 第 4 步：去除噪声

```python
np.random.seed(42)
clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))
noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Noisy error:    {np.linalg.norm(noisy - clean):.4f}")
print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
print(f"Improvement:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")
```

### 第 5 步：伪逆

```python
A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD pseudoinverse solution:  {x_svd}")
print(f"np.linalg.lstsq solution:   {x_lstsq}")
print(f"np.linalg.pinv solution:    {x_pinv}")
```

## 实际使用

完整演示位于 `code/svd.py`，运行它可以查看 SVD 如何应用于图像压缩、推荐系统、潜在语义分析和去噪。

```bash
python svd.py
```

`code/svd.jl` 中的 Julia 版本，使用 Julia 原生 `svd()` 函数和 `LinearAlgebra` 包演示相同概念。

```bash
julia svd.jl
```

## 交付成果

本课会产出：
- `outputs/skill-svd.md`——帮助你判断何时以及如何在真实项目中应用 SVD 的技能

## 练习

1. 不使用幂迭代，实现完整 SVD。改为对 A^T A 做特征分解，得到 V 和奇异值，再计算 U = A V Sigma^{-1}。将数值精度与幂迭代版本及 NumPy 进行比较。

2. 加载一张真实灰度图像（或将一张图像转换为灰度），分别以秩 1、5、10、25、50、100 压缩。计算每个秩对应的压缩率和相对误差，并找出图像达到可接受视觉质量时的秩。

3. 构建一个微型推荐系统。创建一个含部分已知元素的 10x8 用户—电影评分矩阵，用每行均值填补缺失项，计算 SVD 并重建秩为 3 的近似矩阵。使用重建矩阵预测缺失评分，并验证预测是否合理。

4. 创建一个包含 3 个合成主题的 100x50 文档—词项矩阵，每个主题关联 5 个词项，再添加噪声。应用 SVD，验证最大的 3 个奇异值明显高于其余值。把文档投影到三维潜在空间，检查同一主题的文档是否聚集在一起。

5. 生成一个干净的低秩矩阵（秩为 3，大小为 50x40），并添加不同强度的 Gaussian 噪声（sigma = 0.1、0.5、1.0、2.0）。对每种噪声强度扫描 k=1 到 40，并根据相对于干净矩阵的重建误差，找出最佳截断秩。绘制最佳 k 如何随噪声强度变化。

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| SVD | “分解任意矩阵” | 把 A 分解为 U Sigma V^T，其中 U 与 V 为正交矩阵，Sigma 是对角元素非负的对角矩阵；适用于任意形状的矩阵 |
| Singular value | “这个分量有多重要” | Sigma 的第 i 个对角元素，衡量矩阵沿第 i 个主方向拉伸多少；始终非负并按降序排列 |
| Left singular vector | “输出方向” | U 的一列；第 i 个右奇异向量按 sigma_i 缩放后映射到输出空间中的方向 |
| Right singular vector | “输入方向” | V 的一列；矩阵把输入空间中的这个方向按 sigma_i 缩放后映射到第 i 个左奇异向量 |
| Truncated SVD | “低秩近似” | 只保留最大的 k 个奇异值及对应向量，得到原矩阵可证明的最佳秩 k 近似（Eckart-Young 定理） |
| Rank | “真实维度” | 非零奇异值的数量，表示矩阵实际使用了多少个独立方向 |
| Pseudoinverse | “广义逆” | V Sigma+ U^T；对非零奇异值取倒数，零值保持不变，用于求解非方阵或奇异矩阵的最小二乘问题 |
| Condition number | “对误差有多敏感” | sigma_max / sigma_min；条件数很大，表示输入的微小变化会造成输出的大幅变化，SVD 可以直接揭示它 |
| Latent factor | “隐藏变量” | SVD 发现的低秩空间中的一个维度；推荐系统中可能代表类型偏好，NLP 中可能代表某个主题 |
| Frobenius norm | “矩阵的总体大小” | 所有元素平方和的平方根，也等于所有奇异值平方和的平方根，用于衡量近似误差 |
| Eckart-Young theorem | “SVD 给出最佳压缩” | 对任意目标秩 k，截断 SVD 在所有秩 k 矩阵中使近似误差最小 |
| Power iteration | “寻找最大特征向量” | 反复让矩阵乘以一个随机向量并归一化，最终收敛到最大特征值对应的特征向量；是许多 SVD 算法的基础 |

## 延伸阅读

- [Gilbert Strang：《线性代数及其应用》第 7 章](https://math.mit.edu/~gs/linearalgebra/)——结合应用深入讲解 SVD
- [3Blue1Brown：SVD 究竟是什么？](https://www.youtube.com/watch?v=vSczTbgc8Rc)——SVD 的几何直觉
- [我们推荐奇异值分解](https://www.ams.org/publicoutreach/feature-column/fcarc-svd)——美国数学学会提供的易懂概览
- [Netflix Prize 与矩阵分解](https://sifter.org/~simon/journal/20061211.html)——Simon Funk 关于推荐系统 SVD 的原始博客文章
- [潜在语义分析](https://en.wikipedia.org/wiki/Latent_semantic_analysis)——SVD 最早的 NLP 应用
- [Trefethen 与 Bau：《数值线性代数》](https://people.maths.ox.ac.uk/trefethen/text.html)——理解 SVD 算法及数值性质的经典教材
