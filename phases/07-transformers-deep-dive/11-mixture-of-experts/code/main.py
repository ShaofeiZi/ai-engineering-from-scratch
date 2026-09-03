"""仅使用标准库实现混合专家模型（MoE）。

实现内容：
- 使用 softmax 门控的 top-k 路由器
- 无辅助损失的偏置更新（DeepSeek-V3）
- 跨多个 token 的专家用量跟踪
"""

import math
import random


def silu(x):
    return x / (1.0 + math.exp(-x))


def make_expert(d_in, d_hidden, rng):
    """微型“专家”：输入 -> SiLU -> 输出。为便于说明而使用线性层。"""
    scale = math.sqrt(2.0 / (d_in + d_hidden))
    W = [[rng.gauss(0, scale) for _ in range(d_hidden)] for _ in range(d_in)]
    return W


def apply_expert(x, W):
    d_hidden = len(W[0])
    out = [0.0] * d_hidden
    for i, xi in enumerate(x):
        if xi == 0.0:
            continue
        for j in range(d_hidden):
            out[j] += xi * W[i][j]
    return [silu(v) for v in out]


def route(hidden, W_router, top_k, bias):
    """返回（top-k 专家索引，这些专家对应的门控权重）。

    偏置会影响专家选择（argmax），但不会影响门控权重——这是
    DeepSeek-V3 的无辅助损失技巧。
    """
    E = len(W_router)
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(E)]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(E), key=lambda i: -biased[i])[:top_k]
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates


def moe_layer_forward(x, experts, W_router, top_k, bias):
    """计算单个 token `x` 的 MoE 输出，返回输出向量。"""
    top_idx, gates = route(x, W_router, top_k, bias)
    d_hidden = len(experts[0][0])
    out = [0.0] * d_hidden
    for e_idx, gate in zip(top_idx, gates):
        h = apply_expert(x, experts[e_idx])
        for j in range(d_hidden):
            out[j] += gate * h[j]
    return out, top_idx


def update_bias(bias, usage_counts, target, gamma):
    """无辅助损失的均衡：根据实际用量与目标用量之差上下调整偏置。"""
    for e in range(len(bias)):
        if usage_counts[e] > target:
            bias[e] -= gamma
        elif usage_counts[e] < target:
            bias[e] += gamma
    return bias


def run_epoch(tokens, experts, W_router, top_k, bias):
    usage = [0] * len(experts)
    for x in tokens:
        _, top_idx = moe_layer_forward(x, experts, W_router, top_k, bias)
        for e in top_idx:
            usage[e] += 1
    return usage


def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p) for p in ps)


def dense_active_params(n_experts, expert_params, top_k, d_model):
    """总参数量和每个 token 的活跃参数量。d_model 用于估算注意力。"""
    total = n_experts * expert_params
    active = top_k * expert_params
    return total, active


def main():
    rng = random.Random(42)
    d_model = 16
    d_hidden = 32
    n_experts = 8
    top_k = 2
    n_tokens = 1000

    experts = [make_expert(d_model, d_hidden, rng) for _ in range(n_experts)]
    W_router = [[rng.gauss(0, 0.3) for _ in range(d_model)] for _ in range(n_experts)]

    # 构造具有一定结构的合成 token，使初始路由并非均匀分布。
    tokens = [[rng.gauss(0, 1) for _ in range(d_model)] for _ in range(n_tokens)]

    bias = [0.0] * n_experts
    target = n_tokens * top_k / n_experts

    print("=== MoE 路由：无辅助损失的均衡 ===")
    print(f"配置：{n_experts} 个专家，top-{top_k}，{n_tokens} 个 token，目标用量 = {target:.0f}（每个专家）")
    print()
    usage = run_epoch(tokens, experts, W_router, top_k, bias)
    print(f"迭代  0  用量：" + " ".join(f"{u:>4}" for u in usage) + f"  熵={entropy(usage):.3f}")

    for it in range(1, 11):
        bias = update_bias(bias, usage, target, gamma=0.15)
        usage = run_epoch(tokens, experts, W_router, top_k, bias)
        print(f"迭代 {it:>2}  用量：" + " ".join(f"{u:>4}" for u in usage) + f"  熵={entropy(usage):.3f}")
    print(f"最大熵（均匀分布）= ln({n_experts}) = {math.log(n_experts):.3f}")
    print()

    print("=== 参数量（每层的 FFN 部分）===")
    ffn_params = d_model * d_hidden * 3  # 类 SwiGLU：W1、W2、W3
    print(f"  简化 MoE：总计={n_experts * ffn_params:>10}  活跃={top_k * ffn_params:>10}")

    # DeepSeek-V3 结构（逐层 FFN；真实模型有 61 层）
    d = 7168
    shared = 1
    routed = 256
    active = 8
    layers = 61
    ffn_full = 3 * d * int(d * 2.67)
    fine_expert = ffn_full // 8
    total_moe_per_layer = (shared + routed) * fine_expert
    active_moe_per_layer = (shared + active) * fine_expert
    print(f"  类 DeepSeek-V3 每层：总计={total_moe_per_layer / 1e9:.1f}B  活跃={active_moe_per_layer / 1e9:.1f}B")
    print(f"  DeepSeek-V3 FFN 总计（×{layers} 层）：总计约 {total_moe_per_layer * layers / 1e9:.0f}B，活跃约 {active_moe_per_layer * layers / 1e9:.0f}B")
    print(f"  Llama-3-70B FFN 总计：约 {32 * ffn_full / 1e9:.0f}B（每个 token 全部激活）")
    print()
    print("要点：活跃 FLOPs 相同，但参数规模大得多。")


if __name__ == "__main__":
    main()
