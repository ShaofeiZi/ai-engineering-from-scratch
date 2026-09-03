"""受带宽约束的边缘推理 decode 模拟器，使用 Python stdlib。

根据 (weights_bytes / bandwidth_bytes_per_sec) 计算一系列边缘目标上的理论 decode
吞吐量，并与观测基准比较。演示边缘设备上的 decode 受内存而非计算能力限制。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Target:
    name: str
    bandwidth_gb_s: float
    observed_toks_per_s_llama8b_q4: float | None
    notes: str


TARGETS = [
    Target("数据中心 H100 HBM3",       3350, 170,  "参考上限"),
    Target("Jetson AGX Orin",          205,  45,  "边缘与数据中心之间的桥梁"),
    Target("Apple M3 Max",             400,  55,  "统一内存 MPS"),
    Target("Apple M4（MacBook Air）",  120,  25,  "消费级笔记本电脑"),
    Target("Apple A18（iPhone 16）",    60,   8,  "配备 ANE 的手机"),
    Target("Snapdragon 8 Gen 3",        77,   7,  "中高端 Android"),
    Target("Snapdragon X Elite",       135,  22,  "Windows ARM 笔记本电脑"),
    Target("M3 Max 上的 WebGPU",        400,  41,  "浏览器损耗约 25%"),
    Target("Pixel 9 上的 WebGPU",        77,   6,  "移动端浏览器 Chrome 121+"),
]


def ceiling(target: Target, model_gb: float) -> float:
    seconds_per_token = model_gb / target.bandwidth_gb_s
    return 1 / seconds_per_token


def efficiency(observed: float | None, ceiling_val: float) -> str:
    if observed is None:
        return "    -"
    return f"{observed / ceiling_val * 100:4.0f}%"


def main() -> None:
    model_name = "Llama 3.1 8B Q4"
    model_gb = 4.7
    print("=" * 95)
    print(f"边缘解码上限——{model_name}（HBM/DRAM 中占 {model_gb:.1f} GB）")
    print("=" * 95)
    header = f"{'目标':26}  {'带宽（GB/秒）':>9}  {'上限（token/秒）':>16}  {'观测值':>10}  {'效率':>11}  说明"
    print(header)
    print("-" * len(header))
    for t in TARGETS:
        c = ceiling(t, model_gb)
        obs = t.observed_toks_per_s_llama8b_q4
        eff = efficiency(obs, c)
        obs_display = f"{obs:>8.0f}  " if obs is not None else f"{'-':>10}  "
        print(f"{t.name:26}  {t.bandwidth_gb_s:8.0f}   {c:15.1f}   {obs_display}{eff:>11}  {t.notes}")

    print()
    print("解读：带宽决定上限。只有运行时效率低下时，计算能力才会成为关键。")
    print()
    print("=" * 95)
    print("量化影响 — 相同目标，不同格式")
    print("=" * 95)
    iphone_bw = 60.0
    for name, size in [("BF16", 18.8), ("INT8", 9.4), ("Q4 GGUF", 4.7), ("Q3 GGUF", 3.6)]:
        c = 1 / (size / iphone_bw)
        print(f"iPhone 16 + {name:8}  模型={size:5.1f} GB  上限={c:6.1f} token/秒")


if __name__ == "__main__":
    main()
