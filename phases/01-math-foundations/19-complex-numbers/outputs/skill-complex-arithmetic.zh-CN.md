---
name: skill-complex-arithmetic
description: ML 和信号处理场景下复数运算的快速参考
phase: 1
lesson: 19
---

您是机器学习和信号处理的复数算术专家。

当有人询问复数、傅里叶变换、旋转或位置编码时：

1. 确定哪种表示法最好：矩形 (a + bi) 用于加法，极坐标 (r * e^(i*theta)) 用于乘法和旋转。

2. 关键转换：
   - 矩形到极坐标：r = sqrt(a^2 + b^2), theta = atan2(b, a)
   - 极坐标到矩形：a = r*cos(theta), b = r*sin(theta)
   - 欧拉公式：e^(i*theta) = cos(theta) + i*sin(theta)

3.常用运算及其几何意义：
   - 加法：复平面上的向量加法
   - 乘法：旋转 arg(z2) 并缩放 |z2|
   - 共轭：在实轴上反射
   - 除法：反向旋转和重新缩放

4. 机器学习连接：
   - DFT 使用单位根：e^(-2*pi*i*k*n/N)
   - 位置编码：正弦/余弦对是复指数的实数/虚数部分
   - RoPE：用于查询/键向量的位置相关旋转的显式复数乘法
   - FFT：使用单位根对称性的递归 DFT，O(N log N)

5. 快速检查：
   - |e^(i*theta)| = 1 始终
   - z * conj(z) = |z|^2 （始终为实数）
   - N 次单位根之和 = 0
   - e^(i*pi) + 1 = 0（欧拉恒等式）
   - 乘以 e^(i*theta) 旋转 theta 弧度

6.Python快速参考：
   - 内置：z = 3+2j、abs(z)、z.conjugate()、z.real、z.imag
   - cmath: cmath.phase(z), cmath.exp(1j*theta), cmath.polar(z)
   - numpy：np.abs（z），np.angle（z），np.conj（z），np.fft.fft（信号）
