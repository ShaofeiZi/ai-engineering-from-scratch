"""InternVL3-style 原生预训练语料混合器 + ViR 路由模拟器。

三个小工具：
  1. 语料混合规划器 — 给定目标百分比，计算每种模态的步数。
  2. ViR 路由模拟 — 给定查询分布，估算每次请求的平均 token 数。
  3. DvD 吞吐量估算 — 给定编码器 FLOPs 和 LLM FLOPs，选择服务方案。

仅使用标准库。并非真正的训练器；用于演示 InternVL3 运行时的核算逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorpusMix:
    text_pct: float
    interleaved_pct: float
    caption_pct: float
    video_pct: float

    def normalize(self) -> None:
        total = self.text_pct + self.interleaved_pct + self.caption_pct + self.video_pct
        self.text_pct /= total
        self.interleaved_pct /= total
        self.caption_pct /= total
        self.video_pct /= total

    def steps(self, total: int) -> dict:
        return {
            "text":       int(total * self.text_pct),
            "interleaved": int(total * self.interleaved_pct),
            "caption":    int(total * self.caption_pct),
            "video":      int(total * self.video_pct),
        }


@dataclass
class RouterTier:
    name: str
    tokens: int
    fraction: float


def vir_sim(tiers: list[RouterTier]) -> dict:
    avg = sum(t.tokens * t.fraction for t in tiers)
    baseline = max(t.tokens for t in tiers)
    return {"avg_tokens": avg, "baseline": baseline, "ratio": baseline / avg}


def dvd_throughput(encoder_flops: int, llm_flops: int,
                   llm_tokens: int = 128) -> dict:
    colocated = encoder_flops + llm_flops * llm_tokens
    decoupled = max(encoder_flops, llm_flops * llm_tokens)
    return {"colocated": colocated, "decoupled": decoupled,
            "speedup": colocated / decoupled}


def posthoc_vs_native_table() -> None:
    print("\n后接式训练与原生预训练")
    print("-" * 60)
    rows = [
        ("指标",                   "后接式",     "原生"),
        ("-" * 22,                 "-" * 12,     "-" * 12),
        ("总 GPU 小时",            "~30k",       "~300k"),
        ("复用基础 LLM",            "是",         "否"),
        ("对齐负债",                "明显",       "可忽略"),
        ("MMLU 回退",              "-2 至 -8",   "0"),
        ("GSM8K 回退",             "-3 至 -10",  "0"),
        ("语料灵活性",              "仅指令",     "交错数据"),
        ("后续替换基础 LLM",         "可以",       "不可以"),
        ("示例",                    "LLaVA, Qwen-VL v1", "InternVL3, GPT-4o, Chameleon"),
    ]
    for r in rows:
        print(f"  {r[0]:<22}{r[1]:<14}{r[2]}")


def main() -> None:
    print("=" * 60)
    print("INTERNVL3 原生预训练（第 12 阶段，第 10 课）")
    print("=" * 60)

    mix = CorpusMix(text_pct=40, interleaved_pct=35, caption_pct=20, video_pct=5)
    mix.normalize()
    total_steps = 500_000
    steps = mix.steps(total_steps)
    corpus_names = {"text": "文本", "interleaved": "交错数据",
                    "caption": "图注", "video": "视频"}
    print(f"\n语料 MIX（目标 {total_steps:,} 训练步数）")
    print("-" * 60)
    for k, v in steps.items():
        print(f"  {corpus_names[k]:<14}：{v:>8,}  ({v * 100 / total_steps:.1f}%)")
    print("\n40% 的文本下限保留了基础 LLM 能力；交错混合是关键解锁点")
    print("使模型能够在预训练期间学习多图推理。")

    print("\nVIR ROUTING SIMULATION（生产环境查询分布）")
    print("-" * 60)
    tiers = [
        RouterTier("低分辨率照片问答",      256, 0.50),
        RouterTier("中分辨率商品图",         576, 0.30),
        RouterTier("高分辨率文档 + OCR",    2048, 0.20),
    ]
    for t in tiers:
        print(f"  {t.name:<26}  {t.tokens:>5} token × {t.fraction * 100:>4.0f}%")
    r = vir_sim(tiers)
    print(f"\n  平均 token/请求      : {r['avg_tokens']:.0f}")
    print(f"  基线（全部高分辨率） : {r['baseline']}")
    print(f"  相对基线加速比       : {r['ratio']:.2f}x")
    print("  注：50% 的真实查询仅需低分辨率编码")

    print("\nDVD 部署——编码器与 LLM 的并行度")
    print("-" * 60)
    encoder_gflops = 300
    llm_gflops_per_token = 8
    d = dvd_throughput(encoder_gflops, llm_gflops_per_token, 128)
    print(f"  编码器：每张图像 {encoder_gflops} GFLOPs")
    print(f"  LLM    : 每个输出 token {llm_gflops_per_token} GFLOPs，共 128 个 token")
    print(f"  同置总计: {d['colocated']} GFLOPs")
    print(f"  解耦瓶颈: {d['decoupled']} GFLOPs")
    print(f"  加速比: 使用 DvD 时 {d['speedup']:.2f}x")

    posthoc_vs_native_table()


if __name__ == "__main__":
    main()
