# Julia 版支持向量机。通过随机次梯度下降在合页损失上训练线性 SVM，
# 并带 L2 正则化（软间隔），同时包含多项式核与 RBF 核函数。仅使用标准库。参考资料：
#   https://docs.julialang.org/en/v1/manual/control-flow/
#   https://docs.julialang.org/en/v1/stdlib/Random/
#   https://docs.julialang.org/en/v1/manual/arrays/

using Random
using Printf


function dotprod(a::Vector{Float64}, b::Vector{Float64})::Float64
    s = 0.0
    @inbounds for i in 1:length(a)
        s += a[i] * b[i]
    end
    return s
end


function vec_norm(a::Vector{Float64})::Float64
    return sqrt(dotprod(a, a))
end


function linear_kernel(x::Vector{Float64}, z::Vector{Float64})::Float64
    return dotprod(x, z)
end


function polynomial_kernel(x::Vector{Float64}, z::Vector{Float64};
                          degree::Int=3, c::Float64=1.0)::Float64
    return (dotprod(x, z) + c) ^ degree
end


function rbf_kernel(x::Vector{Float64}, z::Vector{Float64};
                   gamma::Float64=0.5)::Float64
    diff = x .- z
    return exp(-gamma * dotprod(diff, diff))
end


function hinge_loss(X::Vector{Vector{Float64}}, ys::Vector{Int},
                   w::Vector{Float64}, b::Float64)::Float64
    n = length(X)
    total = 0.0
    for i in 1:n
        margin = ys[i] * (dotprod(w, X[i]) + b)
        total += max(0.0, 1.0 - margin)
    end
    return total / n
end


function svm_objective(X::Vector{Vector{Float64}}, ys::Vector{Int},
                      w::Vector{Float64}, b::Float64, lambda::Float64)::Float64
    return 0.5 * lambda * dotprod(w, w) + hinge_loss(X, ys, w, b)
end


mutable struct LinearSVM
    w::Vector{Float64}
    b::Float64
    lr::Float64
    lambda::Float64
    n_epochs::Int
    history::Vector{Tuple{Int, Float64}}
end


LinearSVM(; lr::Float64=0.001, lambda::Float64=0.01, n_epochs::Int=1000) =
    LinearSVM(Float64[], 0.0, lr, lambda, n_epochs, Tuple{Int, Float64}[])


function fit_svm!(model::LinearSVM, X::Vector{Vector{Float64}}, ys::Vector{Int};
                 seed::Int=0)
    rng = MersenneTwister(seed)
    n_features = length(X[1])
    n_samples = length(X)
    model.w = zeros(n_features)
    model.b = 0.0
    empty!(model.history)

    for epoch in 0:(model.n_epochs - 1)
        indices = randperm(rng, n_samples)
        for i in indices
            margin = ys[i] * (dotprod(model.w, X[i]) + model.b)
            if margin >= 1
                for j in 1:n_features
                    model.w[j] -= model.lr * model.lambda * model.w[j]
                end
            else
                for j in 1:n_features
                    model.w[j] -= model.lr * (model.lambda * model.w[j] - ys[i] * X[i][j])
                end
                model.b -= model.lr * (-ys[i])
            end
        end
        if epoch % 100 == 0 || epoch == model.n_epochs - 1
            push!(model.history, (epoch, svm_objective(X, ys, model.w, model.b, model.lambda)))
        end
    end
    return model
end


function predict_svm(model::LinearSVM, X::Vector{Vector{Float64}})::Vector{Int}
    return [dotprod(model.w, x) + model.b >= 0 ? 1 : -1 for x in X]
end


function decision_function(model::LinearSVM, X::Vector{Vector{Float64}})::Vector{Float64}
    return [dotprod(model.w, x) + model.b for x in X]
end


function margin_width(model::LinearSVM)::Float64
    n = vec_norm(model.w)
    return n == 0 ? 0.0 : 2.0 / n
end


function find_support_vectors(model::LinearSVM, X::Vector{Vector{Float64}},
                             ys::Vector{Int}; tol::Float64=0.1)::Vector{Int}
    svs = Int[]
    for i in 1:length(X)
        margin = ys[i] * (dotprod(model.w, X[i]) + model.b)
        if abs(margin - 1.0) < tol
            push!(svs, i)
        end
    end
    return svs
end


function svm_accuracy(y_true::Vector{Int}, y_pred::Vector{Int})::Float64
    return sum(y_true .== y_pred) / length(y_true)
end


function generate_linear_data(; n_samples::Int=100, margin::Float64=1.0, seed::Int=42)
    rng = MersenneTwister(seed)
    X = Vector{Vector{Float64}}()
    ys = Int[]
    for _ in 1:n_samples
        x1 = -3.0 + 6.0 * rand(rng)
        x2 = -3.0 + 6.0 * rand(rng)
        val = x1 + x2
        if val > margin / 2
            push!(X, Float64[x1, x2])
            push!(ys, 1)
        elseif val < -margin / 2
            push!(X, Float64[x1, x2])
            push!(ys, -1)
        end
    end
    return X, ys
end


function generate_noisy_data(; n_samples::Int=200, noise::Float64=0.5, seed::Int=42)
    rng = MersenneTwister(seed)
    X = Vector{Vector{Float64}}()
    ys = Int[]
    for _ in 1:n_samples
        x1 = -3.0 + 6.0 * rand(rng)
        x2 = -3.0 + 6.0 * rand(rng)
        val = x1 - 0.5 * x2 + noise * randn(rng)
        push!(X, Float64[x1, x2])
        push!(ys, val > 0 ? 1 : -1)
    end
    return X, ys
end


function generate_circular_data(; n_samples::Int=200, seed::Int=42)
    rng = MersenneTwister(seed)
    X = Vector{Vector{Float64}}()
    ys = Int[]
    for _ in 1:n_samples
        r = 3.0 * rand(rng)
        angle = 2 * pi * rand(rng)
        x1 = r * cos(angle) + 0.1 * randn(rng)
        x2 = r * sin(angle) + 0.1 * randn(rng)
        push!(X, Float64[x1, x2])
        push!(ys, r > 1.5 ? 1 : -1)
    end
    return X, ys
end


function svm_train_test_split(X::Vector{Vector{Float64}}, ys::Vector{Int};
                             test_ratio::Float64=0.2, seed::Int=42)
    rng = MersenneTwister(seed)
    indices = randperm(rng, length(X))
    split = Int(round(length(X) * (1 - test_ratio)))
    train_idx = indices[1:split]
    test_idx = indices[(split + 1):end]
    return (X[train_idx], ys[train_idx], X[test_idx], ys[test_idx])
end


function demo_hinge_loss()
    println("=" ^ 65)
    println("合页损失")
    println("=" ^ 65)
    println()
    margins = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    @printf("  %10s  %12s  %14s\n", "y * f(x)", "合页损失", "对数损失")
    println("  " * "-" ^ 10 * "  " * "-" ^ 12 * "  " * "-" ^ 14)
    for m in margins
        h = max(0.0, 1.0 - m)
        l = log(1 + exp(-m))
        @printf("  %10.1f  %12.3f  %14.3f\n", m, h, l)
    end
    println()
    println("  当 y*f(x) >= 1 时合页损失恰好为零。")
    println("  对数损失永远不会恰好为零。合页损失能得到稀疏模型。")
    println()
end


function demo_linear_svm()
    println("=" ^ 65)
    println("线性 SVM（软间隔）")
    println("=" ^ 65)
    println()
    X, ys = generate_linear_data(n_samples=200, margin=1.0, seed=42)
    X_train, ys_train, X_test, ys_test = svm_train_test_split(X, ys)

    @printf("  数据集：%d 个样本，线性可分\n", length(X))
    @printf("  训练集：%d   测试集：%d\n", length(X_train), length(X_test))

    svm = LinearSVM(lr=0.001, lambda=0.01, n_epochs=500)
    fit_svm!(svm, X_train, ys_train; seed=1)

    train_acc = svm_accuracy(ys_train, predict_svm(svm, X_train))
    test_acc = svm_accuracy(ys_test, predict_svm(svm, X_test))
    @printf("\n  权重：[%.4f, %.4f]\n", svm.w[1], svm.w[2])
    @printf("  偏置：%.4f\n", svm.b)
    @printf("  间隔宽度：%.4f\n", margin_width(svm))
    @printf("  训练集准确率：%.4f\n", train_acc)
    @printf("  测试集准确率：%.4f\n", test_acc)

    svs = find_support_vectors(svm, X_train, ys_train; tol=0.3)
    @printf("  支持向量：%d / %d\n", length(svs), length(X_train))
    println()
end


function demo_c_parameter()
    println("=" ^ 65)
    println("C 参数（正则化权衡）")
    println("=" ^ 65)
    println()
    X, ys = generate_noisy_data(n_samples=300, noise=0.8, seed=42)
    X_train, ys_train, X_test, ys_test = svm_train_test_split(X, ys)

    @printf("  %8s  %8s  %10s  %10s  %8s  %6s\n",
            "C", "lambda", "训练准确率", "测试准确率", "间隔", "支持向量")
    println("  " * "-" ^ 8 * "  " * "-" ^ 8 * "  " * "-" ^ 10 * "  " *
            "-" ^ 10 * "  " * "-" ^ 8 * "  " * "-" ^ 6)
    for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
        lam = 1.0 / (c * length(X_train))
        svm = LinearSVM(lr=0.001, lambda=lam, n_epochs=500)
        fit_svm!(svm, X_train, ys_train; seed=2)
        train_acc = svm_accuracy(ys_train, predict_svm(svm, X_train))
        test_acc = svm_accuracy(ys_test, predict_svm(svm, X_test))
        mw = margin_width(svm)
        n_sv = length(find_support_vectors(svm, X_train, ys_train; tol=0.3))
        @printf("  %8.3f  %8.5f  %10.4f  %10.4f  %8.4f  %6d\n",
                c, lam, train_acc, test_acc, mw, n_sv)
    end
    println()
    println("  小 C（大 lambda）：间隔宽，松弛多，泛化能力更好。")
    println("  大 C（小 lambda）：间隔窄，松弛少，有过拟合风险。")
    println()
end


function demo_kernels()
    println("=" ^ 65)
    println("核函数")
    println("=" ^ 65)
    println()
    x = Float64[1.0, 0.0]
    cases = [
        ("同方向", Float64[2.0, 0.0]),
        ("垂直",   Float64[0.0, 1.0]),
        ("接近",   Float64[1.1, 0.1]),
        ("同向较远", Float64[5.0, 0.0]),
        ("相反",   Float64[-1.0, 0.0]),
    ]
    @printf("  参考点：%s\n", x)
    println()
    @printf("  %-20s  %8s  %10s  %10s  %10s\n",
            "点", "线性", "多项式(d=2)", "多项式(d=3)", "RBF(g=0.5)")
    println("  " * "-" ^ 20 * "  " * "-" ^ 8 * "  " * "-" ^ 10 * "  " *
            "-" ^ 10 * "  " * "-" ^ 10)
    for (name, z) in cases
        k_l = linear_kernel(x, z)
        k_p2 = polynomial_kernel(x, z; degree=2)
        k_p3 = polynomial_kernel(x, z; degree=3)
        k_rbf = rbf_kernel(x, z; gamma=0.5)
        @printf("  %-20s  %8.3f  %10.3f  %10.3f  %10.4f\n",
                name, k_l, k_p2, k_p3, k_rbf)
    end
    println()
    println("  线性核：原始点积。RBF：基于局部性。")
    println()
end


function demo_linear_vs_nonlinear()
    println("=" ^ 65)
    println("线性 SVM 与多项式特征映射的对比")
    println("=" ^ 65)
    println()
    X, ys = generate_circular_data(n_samples=200, seed=42)
    X_train, ys_train, X_test, ys_test = svm_train_test_split(X, ys)

    svm = LinearSVM(lr=0.001, lambda=0.01, n_epochs=500)
    fit_svm!(svm, X_train, ys_train; seed=3)
    train_acc = svm_accuracy(ys_train, predict_svm(svm, X_train))
    test_acc = svm_accuracy(ys_test, predict_svm(svm, X_test))
    @printf("  环形数据上的普通线性 SVM：训练=%.4f  测试=%.4f\n",
            train_acc, test_acc)
    println()

    function augment(X)
        return [Float64[x[1], x[2], x[1] ^ 2, x[2] ^ 2, x[1] * x[2]] for x in X]
    end
    X_train_aug = augment(X_train)
    X_test_aug = augment(X_test)
    svm_aug = LinearSVM(lr=0.0005, lambda=0.01, n_epochs=1000)
    fit_svm!(svm_aug, X_train_aug, ys_train; seed=4)
    train_aug = svm_accuracy(ys_train, predict_svm(svm_aug, X_train_aug))
    test_aug = svm_accuracy(ys_test, predict_svm(svm_aug, X_test_aug))
    println("  多项式特征映射后 (x1, x2, x1^2, x2^2, x1*x2)：")
    @printf("  增广特征上的线性 SVM：训练=%.4f  测试=%.4f\n",
            train_aug, test_aug)
    println()
    println("  核技巧隐式地完成了这一特征映射。")
    println()
end


function demo_support_vectors()
    println("=" ^ 65)
    println("支持向量")
    println("=" ^ 65)
    println()
    X, ys = generate_linear_data(n_samples=200, margin=1.5, seed=42)
    X_train, ys_train, _, _ = svm_train_test_split(X, ys)
    svm = LinearSVM(lr=0.001, lambda=0.01, n_epochs=1000)
    fit_svm!(svm, X_train, ys_train; seed=5)

    margins = [(i, ys_train[i] * (dotprod(svm.w, X_train[i]) + svm.b))
              for i in 1:length(X_train)]
    sort!(margins; by=t -> t[2])

    @printf("  在 %d 个点上完成训练。\n", length(X_train))
    @printf("  权重：[%.4f, %.4f]  偏置：%.4f\n", svm.w[1], svm.w[2], svm.b)
    println()
    println("  按间隔 (y * f(x)) 排序的点：")
    @printf("  %6s  %4s  %8s  %s\n", "索引", "y", "间隔", "角色")
    println("  " * "-" ^ 6 * "  " * "-" ^ 4 * "  " * "-" ^ 8 * "  " * "-" ^ 20)
    for (idx, m) in margins[1:8]
        role = m < 0 ? "误分类" :
               m < 1.0 ? "间隔内" :
               m < 1.2 ? "支持向量" :
                         "安全分类"
        @printf("  %6d  %4d  %8.4f  %s\n", idx, ys_train[idx], m, role)
    end
    println("  ...")
    for (idx, m) in margins[(end - 2):end]
        @printf("  %6d  %4d  %8.4f  安全分类\n", idx, ys_train[idx], m)
    end
    n_sv = sum(1 for (_, m) in margins if 0.7 < m < 1.3)
    n_safe = sum(1 for (_, m) in margins if m >= 1.3)
    n_inside = sum(1 for (_, m) in margins if 0 < m < 0.7)
    println()
    @printf("  支持向量（间隔 ~ 1.0）：%d\n", n_sv)
    @printf("  安全分类（间隔 >> 1）：%d\n", n_safe)
    @printf("  间隔内（0 < 间隔 < 1）：%d\n", n_inside)
    println()
end


function main()
    demo_hinge_loss()
    demo_linear_svm()
    demo_c_parameter()
    demo_kernels()
    demo_linear_vs_nonlinear()
    demo_support_vectors()
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
