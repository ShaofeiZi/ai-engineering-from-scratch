# 感知机 + 单隐藏层 MLP 的 Julia 实现。先用单层 Rosenblatt
# 感知机实现 AND/OR/NOT，再用手动连接的 XOR 网络展示
# 感知机为何无法解决 XOR，最后训练一个 2-2-1 sigmoid MLP
# 并手动实现反向传播。
# 仅使用标准库。参考资料：
#   https://en.wikipedia.org/wiki/Perceptron
#   https://docs.julialang.org/en/v1/manual/types/#Composite-Types

using Random
using Printf


mutable struct Perceptron
    weights::Vector{Float64}
    bias::Float64
    lr::Float64
end

Perceptron(n_inputs::Int; lr::Float64=0.1) =
    Perceptron(zeros(Float64, n_inputs), 0.0, lr)


function predict(p::Perceptron, inputs::Vector{Float64})::Int
    return sum(p.weights .* inputs) + p.bias >= 0 ? 1 : 0
end


function train!(p::Perceptron, data::Vector{Tuple{Vector{Float64}, Int}}; epochs::Int=100)
    for epoch in 1:epochs
        errors = 0
        for (inputs, target) in data
            pred = predict(p, inputs)
            err = target - pred
            if err != 0
                errors += 1
                p.weights .+= p.lr * err .* inputs
                p.bias += p.lr * err
            end
        end
        if errors == 0
            println("在第 $epoch 轮收敛")
            return
        end
    end
    println("经过 $epochs 轮后未收敛")
end


function test_gate(name::String, n_inputs::Int, data::Vector{Tuple{Vector{Float64}, Int}})
    println("=== $name ===")
    p = Perceptron(n_inputs)
    train!(p, data)
    println("  权重: $(p.weights), 偏置: $(p.bias)")
    for (inputs, expected) in data
        result = predict(p, inputs)
        status = result == expected ? "正确" : "错误"
        println("  $inputs -> $result (期望 $expected) $status")
    end
    println()
end


# 手动连接的 XOR：通过 OR + NAND + AND 实现。展示两层
# 感知机网络可以计算 XOR，而单层感知机则不能。
function xor_network(x1::Float64, x2::Float64)::Int
    or_neuron = Perceptron(2)
    or_neuron.weights = Float64[1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = Float64[-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = Float64[1.0, 1.0]
    and_neuron.bias = -1.5

    h1 = predict(or_neuron, Float64[x1, x2])
    h2 = predict(nand_neuron, Float64[x1, x2])
    return predict(and_neuron, Float64[h1, h2])
end


# 小型训练 MLP：2 个输入 -> 2 个 sigmoid 隐藏神经元 -> 1 个 sigmoid 输出。
mutable struct TwoLayerNetwork
    w_hidden::Matrix{Float64}    # 2x2
    b_hidden::Vector{Float64}    # 2
    w_output::Vector{Float64}    # 2
    b_output::Float64
    lr::Float64
    # 反向传播缓存
    last_input::Vector{Float64}
    hidden_out::Vector{Float64}
    output::Float64
end

function TwoLayerNetwork(; lr::Float64=2.0, seed::Int=0)
    rng = MersenneTwister(seed)
    return TwoLayerNetwork(
        rand(rng, 2, 2) .* 2 .- 1,
        rand(rng, 2) .* 2 .- 1,
        rand(rng, 2) .* 2 .- 1,
        rand(rng) * 2 - 1,
        lr,
        Float64[],
        zeros(Float64, 2),
        0.0,
    )
end


sigmoid(x::Float64)::Float64 = 1.0 / (1.0 + exp(-clamp(x, -500.0, 500.0)))


function forward!(net::TwoLayerNetwork, inputs::Vector{Float64})::Float64
    net.last_input = inputs
    for i in 1:2
        z = net.w_hidden[i, 1] * inputs[1] + net.w_hidden[i, 2] * inputs[2] + net.b_hidden[i]
        net.hidden_out[i] = sigmoid(z)
    end
    z_out = net.w_output[1] * net.hidden_out[1] + net.w_output[2] * net.hidden_out[2] + net.b_output
    net.output = sigmoid(z_out)
    return net.output
end


function backward!(net::TwoLayerNetwork, target::Float64)
    err = target - net.output
    d_output = err * net.output * (1 - net.output)
    saved_w_output = copy(net.w_output)
    hidden_deltas = zeros(Float64, 2)
    for i in 1:2
        h = net.hidden_out[i]
        hidden_deltas[i] = d_output * saved_w_output[i] * h * (1 - h)
    end
    for i in 1:2
        net.w_output[i] += net.lr * d_output * net.hidden_out[i]
    end
    net.b_output += net.lr * d_output
    for i in 1:2, j in 1:2
        net.w_hidden[i, j] += net.lr * hidden_deltas[i] * net.last_input[j]
    end
    for i in 1:2
        net.b_hidden[i] += net.lr * hidden_deltas[i]
    end
end


function train!(net::TwoLayerNetwork, data::Vector{Tuple{Vector{Float64}, Float64}};
                epochs::Int=10000)
    for epoch in 0:(epochs - 1)
        total_err = 0.0
        for (inputs, target) in data
            out = forward!(net, inputs)
            total_err += (target - out) ^ 2
            backward!(net, target)
        end
        if epoch % 2000 == 0
            @printf("  轮次 %d, 误差: %.4f\n", epoch, total_err)
        end
    end
end


function main()
    and_data = Tuple{Vector{Float64}, Int}[
        (Float64[0, 0], 0),
        (Float64[0, 1], 0),
        (Float64[1, 0], 0),
        (Float64[1, 1], 1),
    ]
    or_data = Tuple{Vector{Float64}, Int}[
        (Float64[0, 0], 0),
        (Float64[0, 1], 1),
        (Float64[1, 0], 1),
        (Float64[1, 1], 1),
    ]
    not_data = Tuple{Vector{Float64}, Int}[
        (Float64[0], 1),
        (Float64[1], 0),
    ]
    xor_data = Tuple{Vector{Float64}, Int}[
        (Float64[0, 0], 0),
        (Float64[0, 1], 1),
        (Float64[1, 0], 1),
        (Float64[1, 1], 0),
    ]

    test_gate("AND 门", 2, and_data)
    test_gate("OR 门", 2, or_data)
    test_gate("NOT 门", 1, not_data)

    println("=== XOR 门（单层感知机 - 会失败）===")
    p_xor = Perceptron(2)
    train!(p_xor, xor_data; epochs=1000)
    for (inputs, expected) in xor_data
        result = predict(p_xor, inputs)
        status = result == expected ? "正确" : "错误"
        println("  $inputs -> $result (期望 $expected) $status")
    end
    println()

    println("=== XOR 门（多层网络 - 可行）===")
    for (inputs, expected) in xor_data
        result = xor_network(inputs[1], inputs[2])
        status = result == expected ? "正确" : "错误"
        println("  $inputs -> $result (期望 $expected) $status")
    end
    println()

    println("=== XOR 门（带反向传播的训练两层网络）===")
    xor_train = Tuple{Vector{Float64}, Float64}[
        (Float64[0, 0], 0.0),
        (Float64[0, 1], 1.0),
        (Float64[1, 0], 1.0),
        (Float64[1, 1], 0.0),
    ]
    net = TwoLayerNetwork(lr=2.0)
    train!(net, xor_train; epochs=10000)
    println()
    for (inputs, expected) in xor_train
        result = forward!(net, inputs)
        predicted = result >= 0.5 ? 1 : 0
        @printf("  %s -> %.4f (rounded: %d, expected %d)\n", inputs, result, predicted, Int(expected))
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
