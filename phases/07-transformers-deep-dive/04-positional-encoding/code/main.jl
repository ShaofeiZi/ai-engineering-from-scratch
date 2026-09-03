# 用 Julia 实现位置编码：正弦绝对位置、旋转位置嵌入（RoPE）和 ALiBi
# 偏置矩阵。验证 RoPE 点积仅取决于相对距离。仅使用标准库。资料来源：
#   https://arxiv.org/abs/2104.09864
#   https://arxiv.org/abs/2108.12409
#   https://docs.julialang.org/en/v1/manual/mathematical-operations/

using Random
using Printf


function sinusoidal_pe(n::Int, d::Int; base::Float64=10000.0)::Matrix{Float64}
    n > 0 || throw(ArgumentError("n 必须大于 0"))
    d > 0 || throw(ArgumentError("d 必须大于 0"))
    iseven(d) || throw(ArgumentError("d 必须为偶数，以便组成正弦/余弦对"))
    pe = zeros(n, d)
    for pos in 0:(n - 1)
        for i in 0:(d ÷ 2 - 1)
            theta = pos / (base ^ (2 * i / d))
            pe[pos + 1, 2 * i + 1] = sin(theta)
            pe[pos + 1, 2 * i + 2] = cos(theta)
        end
    end
    return pe
end


function apply_rope(x::Vector{Float64}, pos::Int; base::Float64=10000.0)::Vector{Float64}
    d = length(x)
    iseven(d) || throw(ArgumentError("RoPE 要求嵌入维度为偶数"))
    out = copy(x)
    for i in 0:(d ÷ 2 - 1)
        theta = pos / (base ^ (2 * i / d))
        c = cos(theta)
        s = sin(theta)
        a = x[2 * i + 1]
        b = x[2 * i + 2]
        out[2 * i + 1] = a * c - b * s
        out[2 * i + 2] = a * s + b * c
    end
    return out
end


function dotprod(a::Vector{Float64}, b::Vector{Float64})::Float64
    return sum(a .* b)
end


function alibi_slopes(n_heads::Int)::Vector{Float64}
    n_heads > 0 || throw(ArgumentError("n_heads 必须大于 0"))
    return [2.0 ^ (-8.0 * (h) / n_heads) for h in 1:n_heads]
end


function alibi_bias(n_heads::Int, seq_len::Int; causal::Bool=true)
    slopes = alibi_slopes(n_heads)
    out = Vector{Matrix{Float64}}()
    for m in slopes
        bias = fill(0.0, seq_len, seq_len)
        for i in 1:seq_len
            for j in 1:seq_len
                if causal && j > i
                    bias[i, j] = -Inf
                else
                    bias[i, j] = -m * abs(i - j)
                end
            end
        end
        push!(out, bias)
    end
    return out
end


function demo_sinusoidal()
    println("=== 正弦位置编码 ===")
    pe = sinusoidal_pe(8, 8)
    println("前 4 个位置、前 4 个维度：")
    for pos in 1:4
        row_str = join([@sprintf("%+.3f", pe[pos, j]) for j in 1:4], "  ")
        @printf("  pos=%d: %s\n", pos - 1, row_str)
    end
    println()
end


function demo_rope_relative()
    println("=== RoPE：点积仅取决于相对距离 ===")
    rng = MersenneTwister(0)
    d = 16
    q = randn(rng, d)
    k = randn(rng, d)
    pairs = [(3, 5), (7, 9), (100, 102), (1024, 1026)]
    @printf("%6s  %6s  %4s  %18s\n", "pos_q", "pos_k", "间隔", "<q_rot, k_rot>")
    for (pq, pk) in pairs
        q_rot = apply_rope(q, pq)
        k_rot = apply_rope(k, pk)
        d_prod = dotprod(q_rot, k_rot)
        @printf("%6d  %6d  %4d  %18.6f\n", pq, pk, pk - pq, d_prod)
    end
    println("间隔为 2 的所有行都应产生相同点积。")
    println()
end


function demo_rope_base_scaling()
    println("=== RoPE 基数缩放（面向长上下文的 NTK-aware 方法）===")
    rng = MersenneTwister(1)
    d = 8
    q = randn(rng, d)
    k = randn(rng, d)
    for base in (10000.0, 100000.0, 1_000_000.0)
        q_rot = apply_rope(q, 4096; base=base)
        k_rot = apply_rope(k, 4098; base=base)
        @printf("  基数=%8d  得分=%+.6f\n", Int(base), dotprod(q_rot, k_rot))
    end
    println("基数越大 = 旋转越慢 = 不发生相位环绕的上下文越长。")
    println()
end


function demo_alibi()
    println("=== ALiBi 偏置矩阵 ===")
    n_heads = 4
    slopes = alibi_slopes(n_heads)
    @printf("%d 个头的斜率：%s\n", n_heads,
            join([@sprintf("%.4f", s) for s in slopes], ", "))
    bias = alibi_bias(n_heads, 6; causal=false)
    println("第 1 个头的偏置（token 越近，惩罚越小）：")
    for row in eachrow(bias[1])
        println("  " * join([@sprintf("%+6.2f", v) for v in row], "  "))
    end
    println()
end


function main()
    demo_sinusoidal()
    demo_rope_relative()
    demo_rope_base_scaling()
    demo_alibi()
    println("要点：RoPE 在点积内部编码相对位置；")
    println("ALiBi 完全跳过嵌入。如今正弦位置编码已退居次要位置。")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
