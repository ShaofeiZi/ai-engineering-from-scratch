using LinearAlgebra

println("=== 向量 ===")
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a = ", a)
println("b = ", b)
println("a + b = ", a + b)
println("a - b = ", a - b)
println("a * 3 = ", a * 3)
println("a · b = ", a ⋅ b)
println("|a| = ", norm(a))
println("â = ", normalize(a))

cosine = (a ⋅ b) / (norm(a) * norm(b))
println("cosine_similarity(a, b) = ", round(cosine, digits=4))

println("\n=== 矩阵 ===")
rotation_90 = [0 -1; 1 0]
point = [3.0, 1.0]
rotated = rotation_90 * point
println("将 ", point, " 旋转 90° → ", rotated)

println("\n=== 神经网络层 ===")
W = randn(2, 3) * 0.1
x = [1.0, 0.5, -0.3]
output = W * x
println("输入（3维）:  ", x)
println("输出（2维）: ", output)
println("^ 这正是神经网络层所做的事情。")
