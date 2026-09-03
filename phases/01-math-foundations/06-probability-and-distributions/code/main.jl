# Julia 概率与分布。手写 PMF、PDF、采样器（伯努利、分类分布、均匀分布、Box-Muller 正态分布），
# softmax + log-softmax + 交叉熵，边缘分布，中心极限定理演示。
# 仅使用标准库。参考资料：
#   https://docs.julialang.org/en/v1/stdlib/Random/
#   https://docs.julialang.org/en/v1/manual/missing/
#   https://en.wikipedia.org/wiki/Box-Muller_transform

using Random
using Statistics
using Printf


factorial_int(n::Int)::Int = n <= 1 ? 1 : prod(2:n)


function combinations(n::Int, k::Int)::Int
    return factorial_int(n) ÷ (factorial_int(k) * factorial_int(n - k))
end


function conditional_probability(p_a_and_b::Float64, p_b::Float64)
    if p_b == 0.0
        throw(ArgumentError("conditional_probability: P(B) 为零，无法相除"))
    end
    return p_a_and_b / p_b
end


bernoulli_pmf(k::Int, p::Float64) = k == 1 ? p : (1 - p)


categorical_pmf(k::Int, probs::Vector{Float64}) = probs[k + 1]


function poisson_pmf(k::Int, lam::Float64)
    return (lam ^ k) * exp(-lam) / factorial_int(k)
end


function uniform_pdf(x::Float64, a::Float64, b::Float64)
    return a <= x <= b ? 1.0 / (b - a) : 0.0
end


function normal_pdf(x::Float64, mu::Float64, sigma::Float64)
    coeff = 1.0 / (sigma * sqrt(2pi))
    exponent = -0.5 * ((x - mu) / sigma) ^ 2
    return coeff * exp(exponent)
end


function expected_value(values::Vector{Float64}, probs::Vector{Float64})::Float64
    return sum(values .* probs)
end


function variance_of(values::Vector{Float64}, probs::Vector{Float64})::Float64
    mu = expected_value(values, probs)
    return sum(probs .* (values .- mu) .^ 2)
end


function sample_bernoulli(rng::AbstractRNG, p::Float64, n::Int)
    return [rand(rng) < p ? 1 : 0 for _ in 1:n]
end


function sample_categorical(rng::AbstractRNG, probs::Vector{Float64}, n::Int)
    cumulative = cumsum(probs)
    samples = Int[]
    for _ in 1:n
        r = rand(rng)
        idx = findfirst(c -> r <= c, cumulative)
        push!(samples, idx === nothing ? length(probs) - 1 : idx - 1)
    end
    return samples
end


function sample_uniform(rng::AbstractRNG, a::Float64, b::Float64, n::Int)
    return [a + (b - a) * rand(rng) for _ in 1:n]
end


function sample_normal_box_muller(rng::AbstractRNG, mu::Float64, sigma::Float64, n::Int)
    samples = Float64[]
    for _ in 1:n
        # rand(rng) 取值范围为 [0, 1)；防止 u1 == 0 以保证 log(u1) 有限。
        u1 = rand(rng)
        while u1 == 0.0
            u1 = rand(rng)
        end
        u2 = rand(rng)
        z = sqrt(-2 * log(u1)) * cos(2pi * u2)
        push!(samples, mu + sigma * z)
    end
    return samples
end


function softmax(logits::Vector{Float64})
    m = maximum(logits)
    exps = exp.(logits .- m)
    return exps ./ sum(exps)
end


function log_softmax(logits::Vector{Float64})
    m = maximum(logits)
    shifted = logits .- m
    log_sum_exp = m + log(sum(exp.(shifted)))
    return logits .- log_sum_exp
end


function cross_entropy_loss(logits::Vector{Float64}, target_index::Int)
    return -log_softmax(logits)[target_index + 1]
end


function joint_to_marginals(joint::Matrix{Float64})
    marginal_x = vec(sum(joint, dims=2))
    marginal_y = vec(sum(joint, dims=1))
    return marginal_x, marginal_y
end


function check_independence(joint::Matrix{Float64},
                            marginal_x::Vector{Float64},
                            marginal_y::Vector{Float64};
                            tol::Float64=1e-9)::Bool
    for i in eachindex(marginal_x), j in eachindex(marginal_y)
        if abs(joint[i, j] - marginal_x[i] * marginal_y[j]) > tol
            return false
        end
    end
    return true
end


function demonstrate_clt(rng::AbstractRNG, n_per_sample::Int, n_averages::Int)
    averages = Float64[]
    for _ in 1:n_averages
        samples = rand(rng, n_per_sample)
        push!(averages, mean(samples))
    end
    return averages
end


function main()
    rng = MersenneTwister(42)

    println("=" ^ 60)
    println("概率与分布")
    println("=" ^ 60)

    println("\n--- 条件概率 ---")
    p_king_given_face = conditional_probability(4 / 52, 12 / 52)
    @printf("P(King | Face card) = %.4f\n", p_king_given_face)

    println("\n--- PMF：伯努利分布 (p=0.7) ---")
    for k in 0:1
        @printf("  P(X=%d) = %.4f\n", k, bernoulli_pmf(k, 0.7))
    end

    println("\n--- PMF：分类分布 ---")
    cat_probs = Float64[0.1, 0.3, 0.4, 0.2]
    for k in 0:(length(cat_probs) - 1)
        @printf("  P(X=%d) = %.4f\n", k, categorical_pmf(k, cat_probs))
    end

    println("\n--- PMF：泊松分布 (lambda=3) ---")
    for k in 0:9
        @printf("  P(X=%d) = %.4f\n", k, poisson_pmf(k, 3.0))
    end

    println("\n--- PDF：正态分布 (mu=0, sigma=1) ---")
    for x in -3.0:1.0:3.0
        @printf("  f(%+.0f) = %.4f\n", x, normal_pdf(x, 0.0, 1.0))
    end

    println("\n--- 期望与方差 ---")
    die_values = Float64[1, 2, 3, 4, 5, 6]
    die_probs = fill(1 / 6, 6)
    mu = expected_value(die_values, die_probs)
    var = variance_of(die_values, die_probs)
    @printf("  公正骰子：E[X] = %.4f, Var(X) = %.4f, SD = %.4f\n", mu, var, sqrt(var))

    println("\n--- 采样：伯努利分布 (p=0.3, n=20) ---")
    bern = sample_bernoulli(rng, 0.3, 20)
    println("  样本：$bern")
    @printf("  经验均值：%.4f（期望 0.3）\n", mean(bern))

    println("\n--- 采样：分类分布 ---")
    cat_samples = sample_categorical(rng, Float64[0.1, 0.3, 0.4, 0.2], 1000)
    counts = [count(==(i), cat_samples) for i in 0:3]
    println("  1000 次样本的计数：$counts")
    println("  经验频率：$(round.(counts ./ 1000, digits=4))")
    println("  期望：[0.1, 0.3, 0.4, 0.2]")

    println("\n--- 采样：正态分布（Box-Muller）---")
    norm = sample_normal_box_muller(rng, 0.0, 1.0, 10000)
    sample_mean = mean(norm)
    sample_var = var_of_samples(norm)
    println("  来自 N(0, 1) 的 10000 个样本：")
    @printf("  样本均值：%.4f（期望 0）\n", sample_mean)
    @printf("  样本方差：%.4f（期望 1）\n", sample_var)

    println("\n--- Softmax ---")
    logits = Float64[2.0, 1.0, 0.1]
    probs = softmax(logits)
    println("  Logits： $logits")
    println("  Softmax：$(round.(probs, digits=4))")
    @printf("  总和：  %.4f\n", sum(probs))

    println("\n--- 大 logits 的 Softmax（数值稳定性测试）---")
    large_logits = Float64[100, 101, 102]
    probs_large = softmax(large_logits)
    println("  Logits： $large_logits")
    println("  Softmax：$(round.(probs_large, digits=4))")
    println("  （因为在 exp 之前减去了最大值，所以不会溢出）")

    println("\n--- 对数概率 ---")
    lp = log_softmax(logits)
    println("  Logits：     $logits")
    println("  Log-softmax：$(round.(lp, digits=4))")
    println("  验证 exp：   $(round.(exp.(lp), digits=4))")

    println("\n--- 交叉熵损失 ---")
    ce = cross_entropy_loss(Float64[2.0, 1.0, 0.1], 0)
    println("  Logits：[2.0, 1.0, 0.1]，目标：0")
    @printf("  交叉熵损失：%.4f\n", ce)

    println("\n--- 对数概率为何重要 ---")
    word_prob = 0.01
    n_words = 50
    raw_product = word_prob ^ n_words
    log_sum = n_words * log(word_prob)
    @printf("  P(word)^%d = %.2e\n", n_words, raw_product)
    @printf("  对数求和：%.4f（稳定）\n", log_sum)
    @printf("  还原值：  %.2e\n", exp(log_sum))

    println("\n--- 联合分布与边缘分布 ---")
    joint = Float64[0.40 0.10; 0.05 0.45]
    mx, my = joint_to_marginals(joint)
    println("  联合分布（天气 x 伞）：")
    @printf("    晴天，不带伞：%.2f\n", joint[1, 1])
    @printf("    晴天，带伞：  %.2f\n", joint[1, 2])
    @printf("    雨天，不带伞：%.2f\n", joint[2, 1])
    @printf("    雨天，带伞：  %.2f\n", joint[2, 2])
    println("  边缘分布 X（天气）： $mx")
    println("  边缘分布 Y（伞）：   $my")
    println("  是否独立？$(check_independence(joint, mx, my))")

    println("\n--- 中心极限定理 ---")
    println("  对均匀分布 [0, 1) 样本取平均：")
    for n in [1, 2, 5, 30]
        avgs = demonstrate_clt(rng, n, 10000)
        @printf("    n=%2d: 均值=%.4f, 标准差=%.4f\n", n, mean(avgs), std_of_samples(avgs))
    end
    println("  随着 n 增大，标准差减小，分布趋近正态。")

    println("\n" * "=" ^ 60)
    println("所有概率计算完成。")
    println("=" ^ 60)
end


function var_of_samples(xs::Vector{Float64})::Float64
    m = mean(xs)
    return sum((xs .- m) .^ 2) / length(xs)
end


function std_of_samples(xs::Vector{Float64})::Float64
    return sqrt(var_of_samples(xs))
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
