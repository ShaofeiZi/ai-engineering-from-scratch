"""推测解码服务器——草稿/验证调度器脚手架。

关键架构原语是草稿/验证调度器：草稿模型提出 k 个候选 token；目标模型通过一次
批处理验证它们；提交所有被接受的前缀，并从目标模型重新采样被拒绝的后缀。
此脚手架使用合成 token 概率实现调度器，使接受/拒绝逻辑和吞吐量计算可端到端观察。

运行：python main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 合成模型——小型词表上的概率分布
# ---------------------------------------------------------------------------

VOCAB = list("abcdefghij")


def softmax_from(seed: int) -> list[float]:
    rnd = random.Random(seed)
    weights = [rnd.random() for _ in VOCAB]
    total = sum(weights)
    return [w / total for w in weights]


def sample(dist: list[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(dist):
        acc += p
        if r <= acc:
            return i
    return len(dist) - 1


# ---------------------------------------------------------------------------
# target——希望减少调用次数的昂贵模型
# ---------------------------------------------------------------------------

@dataclass
class TargetModel:
    calls: int = 0
    tokens_verified: int = 0

    def distribution(self, ctx_seed: int) -> list[float]:
        return softmax_from(ctx_seed * 7 + 13)

    def verify(self, draft_tokens: list[int], ctx_seed: int,
               rng: random.Random) -> tuple[list[int], int]:
        """返回（accepted_tokens, resampled_next）。一次目标模型调用即可批量验证
        draft_tokens：目标模型为每个位置生成概率；接受到首次拒绝之前的位置。"""
        self.calls += 1
        self.tokens_verified += len(draft_tokens) + 1
        accepted: list[int] = []
        for pos, tok in enumerate(draft_tokens):
            dist = self.distribution(ctx_seed + pos)
            # 简单接受标准：目标模型对该 token 的概率 >= 0.5 * 最大概率
            if dist[tok] >= 0.5 * max(dist):
                accepted.append(tok)
            else:
                break
        # 在已接受序列之后的位置从目标模型重新采样下一个 token
        ctx = ctx_seed + len(accepted)
        dist = self.distribution(ctx)
        next_tok = sample(dist, rng)
        return accepted, next_tok


# ---------------------------------------------------------------------------
# 草稿模型——成本更低，且大体与目标模型对齐
# ---------------------------------------------------------------------------

@dataclass
class DraftModel:
    calls: int = 0
    alignment: float = 0.80     # 草稿模型选中目标模型选择的 token 的概率

    def propose(self, ctx_seed: int, k: int, rng: random.Random,
                target: TargetModel) -> list[int]:
        self.calls += 1
        draft_tokens: list[int] = []
        for pos in range(k):
            dist = target.distribution(ctx_seed + pos)
            # 以 alignment 概率输出目标模型的最佳项，否则采样邻近项
            if rng.random() < self.alignment:
                draft_tokens.append(max(range(len(dist)), key=lambda i: dist[i]))
            else:
                draft_tokens.append(sample(dist, rng))
        return draft_tokens


# ---------------------------------------------------------------------------
# 解码调度器——推测循环 + 用于对比的贪心基线
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    generated: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    accepted_sum: int = 0

    def acceptance_rate(self, k: int) -> float:
        if self.target_calls == 0:
            return 0.0
        return self.accepted_sum / (self.target_calls * k)

    def tokens_per_target_call(self) -> float:
        return self.generated / max(1, self.target_calls)


def speculative_decode(n_tokens: int, k: int, rng: random.Random,
                       target: TargetModel, draft: DraftModel) -> Metrics:
    m = Metrics()
    ctx_seed = 1
    while m.generated < n_tokens:
        draft_tokens = draft.propose(ctx_seed, k, rng, target)
        m.draft_calls += 1
        accepted, next_tok = target.verify(draft_tokens, ctx_seed, rng)
        m.target_calls += 1
        m.accepted_sum += len(accepted)
        for tok in accepted:
            m.generated += 1
            ctx_seed += 1
            if m.generated >= n_tokens:
                break
        if m.generated < n_tokens:
            m.generated += 1     # 重新采样的 next_tok
            ctx_seed += 1
    return m


def baseline_decode(n_tokens: int, rng: random.Random,
                    target: TargetModel) -> Metrics:
    m = Metrics()
    ctx_seed = 1
    while m.generated < n_tokens:
        target.calls += 1
        m.target_calls += 1
        dist = target.distribution(ctx_seed)
        _ = sample(dist, rng)
        m.generated += 1
        ctx_seed += 1
    return m


# ---------------------------------------------------------------------------
# 扫描——比较不同 k 和草稿模型对齐度下的加速比
# ---------------------------------------------------------------------------

def main() -> None:
    n_tokens = 500
    print(f"=== 解码 {n_tokens} 个 token，对比基线与推测解码 ===")

    target = TargetModel()
    rng = random.Random(7)
    base = baseline_decode(n_tokens, rng, target)
    print(f"基线：目标模型调用 {base.target_calls} 次，"
          f"每次调用 {base.tokens_per_target_call():.2f} 个 token")

    for alignment in (0.60, 0.75, 0.90):
        for k in (2, 4, 6):
            target = TargetModel()
            draft = DraftModel(alignment=alignment)
            rng = random.Random(7)
            m = speculative_decode(n_tokens, k, rng, target, draft)
            speedup = base.target_calls / max(1, m.target_calls)
            print(f"  align={alignment:.2f} k={k}  "
                  f"target_calls={m.target_calls:3d}  "
                  f"acceptance={m.acceptance_rate(k):.2f}  "
                  f"tok/call={m.tokens_per_target_call():.2f}  "
                  f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
