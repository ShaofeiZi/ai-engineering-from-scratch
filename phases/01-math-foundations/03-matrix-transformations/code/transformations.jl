using LinearAlgebra


function rotation_2d(theta)
    c, s = cos(theta), sin(theta)
    return [c -s; s c]
end


function rotation_3d_z(theta)
    c, s = cos(theta), sin(theta)
    return [c -s 0; s c 0; 0 0 1]
end


function rotation_3d_x(theta)
    c, s = cos(theta), sin(theta)
    return [1 0 0; 0 c -s; 0 s c]
end


function rotation_3d_y(theta)
    c, s = cos(theta), sin(theta)
    return [c 0 s; 0 1 0; -s 0 c]
end


function scaling_2d(sx, sy)
    return [sx 0; 0 sy]
end


function shearing_2d(kx, ky)
    return [1 kx; ky 1]
end


function demo_basic_transformations()
    println("=" ^ 60)
    println("基本变换")
    println("=" ^ 60)

    point = [1.0, 0.0]
    theta = pi / 4

    rotated = rotation_2d(theta) * point
    println("\n将 (1,0) 旋转 45 度: $(round.(rotated, digits=4))")

    scaled = scaling_2d(2, 3) * [1.0, 1.0]
    println("将 (1,1) 缩放 (2,3): $(round.(scaled, digits=4))")

    sheared = shearing_2d(1, 0) * [1.0, 1.0]
    println("将 (1,1) 剪切 kx=1: $(round.(sheared, digits=4))")

    reflected = [-1 0; 0 1] * [2.0, 1.0]
    println("将 (2,1) 关于 y 轴反射: $(round.(reflected, digits=4))")
end


function demo_unit_square()
    println("\n" * "=" ^ 60)
    println("对单位正方形进行变换")
    println("=" ^ 60)

    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    labels = ["原点", "右", "右上", "上"]

    println("\n原始正方形:")
    for (label, pt) in zip(labels, square)
        println("  $label: $pt")
    end

    transforms = [
        ("旋转 45 度", rotation_2d(pi / 4)),
        ("缩放 (2, 0.5)", scaling_2d(2, 0.5)),
        ("剪切 kx=0.5", shearing_2d(0.5, 0)),
        ("关于 y 轴反射", [-1 0; 0 1]),
    ]

    for (name, M) in transforms
        println("\n$name:")
        for (label, pt) in zip(labels, square)
            result = M * pt
            println("  $label: $pt -> $(round.(result, digits=4))")
        end
        println("  det = $(round(det(M), digits=4))")
    end
end


function demo_composition()
    println("\n" * "=" ^ 60)
    println("变换的组合")
    println("=" ^ 60)

    R = rotation_2d(pi / 2)
    S = scaling_2d(2, 0.5)

    point = [1.0, 0.0]

    result1 = (S * R) * point
    result2 = (R * S) * point

    println("\n点: $point")
    println("先旋转 90 再缩放 (2, 0.5): $(round.(result1, digits=4))")
    println("先缩放 (2, 0.5) 再旋转 90: $(round.(result2, digits=4))")
    println("顺序很重要。")

    println("\ndet(R) = $(round(det(R), digits=4))")
    println("det(S) = $(round(det(S), digits=4))")
    println("det(S * R) = $(round(det(S * R), digits=4))")
    println("det(S) * det(R) = $(round(det(S) * det(R), digits=4))")
end


function demo_3d_rotations()
    println("\n" * "=" ^ 60)
    println("三维旋转")
    println("=" ^ 60)

    point = [1.0, 0.0, 0.0]
    theta = pi / 2

    rz = rotation_3d_z(theta) * point
    rx = rotation_3d_x(theta) * point
    ry = rotation_3d_y(theta) * point

    println("\n点: $point")
    println("绕 z 轴旋转 90: $(round.(rz, digits=4))")
    println("绕 x 轴旋转 90: $(round.(rx, digits=4))")
    println("绕 y 轴旋转 90: $(round.(ry, digits=4))")

    println("\ndet(Rz) = $(round(det(rotation_3d_z(theta)), digits=4))")
    println("det(Rx) = $(round(det(rotation_3d_x(theta)), digits=4))")
    println("det(Ry) = $(round(det(rotation_3d_y(theta)), digits=4))")
    println("所有旋转矩阵的行列式都 = 1。")
end


function demo_eigenvalues()
    println("\n" * "=" ^ 60)
    println("特征值与特征向量")
    println("=" ^ 60)

    matrices = [
        ("对称矩阵", [2 1; 1 2]),
        ("上三角矩阵", [3 1; 0 2]),
        ("缩放矩阵", [3 0; 0 5]),
        ("旋转 90 度", [0 -1; 1 0]),
    ]

    for (name, A) in matrices
        vals = eigvals(A)
        vecs = eigvecs(A)
        println("\n$name: $A")
        println("  特征值: $vals")

        if all(isreal, vals)
            for i in 1:length(vals)
                v = real.(vecs[:, i])
                lam = real(vals[i])
                println("  lambda=$(round(lam, digits=4)), v=$(round.(v, digits=4))")
                println("    A * v = $(round.(A * v, digits=4))")
                println("    l * v = $(round.(lam * v, digits=4))")
            end
        else
            println("  复数特征值: 纯旋转，没有实数特征向量。")
        end
    end
end


function demo_eigendecomposition()
    println("\n" * "=" ^ 60)
    println("特征分解")
    println("=" ^ 60)

    A = Float64[3 1; 0 2]
    F = eigen(A)

    println("\nA = $A")
    println("特征值: $(F.values)")
    println("特征向量（列向量）:")
    display(F.vectors)
    println()

    V = F.vectors
    D = Diagonal(F.values)
    reconstructed = V * D * inv(V)
    println("重构 A = V * D * V^-1:")
    display(round.(reconstructed, digits=4))
    println()
end


function demo_determinant_meaning()
    println("\n" * "=" ^ 60)
    println("行列式作为体积缩放因子")
    println("=" ^ 60)

    cases = [
        ("旋转 45 度", rotation_2d(pi / 4)),
        ("缩放 (2, 3)", scaling_2d(2, 3)),
        ("剪切 kx=1", shearing_2d(1, 0)),
        ("关于 y 轴反射", [-1 0; 0 1]),
        ("奇异矩阵", [1 2; 2 4]),
    ]

    println()
    for (name, M) in cases
        d = det(M)
        if abs(d) < 1e-10
            meaning = "空间坍缩，不可逆"
        elseif d < 0
            meaning = "方向翻转"
        elseif abs(d - 1.0) < 1e-10
            meaning = "面积保持不变"
        else
            meaning = "面积缩放 $(round(abs(d), digits=1)) 倍"
        end
        println("det($name) = $(round(d, digits=4))  ($meaning)")
    end
end


function demo_pca_preview()
    println("\n" * "=" ^ 60)
    println("PCA 预览：协方差矩阵的特征向量")
    println("=" ^ 60)

    cov = [2.0 1.0; 1.0 3.0]
    F = eigen(cov)

    println("\n协方差矩阵: $cov")
    println("特征值（每个主成分方向的方差）: $(F.values)")
    println("特征向量（主成分）:")
    display(F.vectors)
    println()
    println("PCA 选取特征值最大的特征向量作为主成分。")
    println("这里 PC1 捕获了 $(round(F.values[2] / sum(F.values) * 100, digits=1))% 的方差。")
end


demo_basic_transformations()
demo_unit_square()
demo_composition()
demo_3d_rotations()
demo_eigenvalues()
demo_eigendecomposition()
demo_determinant_meaning()
demo_pca_preview()
