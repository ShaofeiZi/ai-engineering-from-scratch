"""自托管 LLM 引擎决策树遍历器——使用 Python 标准库。

根据硬件、规模和工作负载选择引擎，并给出说明。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SPECIALIZED_WORKLOAD_MARKERS = (
    "agentic",
    "agent",
    "prefix",
    "智能体",
    "代理式",
    "前缀",
)


def _prefers_sglang(workload: str | Mapping[str, Any]) -> bool:
    """识别中英文描述或稳定的结构化工作负载标签。"""
    if isinstance(workload, Mapping):
        if workload.get("agentic") is True or workload.get("prefix_reuse") is True:
            return True
        stable_values = (
            workload.get("kind"),
            workload.get("type"),
            workload.get("category"),
        )
        return any(
            isinstance(value, str) and _prefers_sglang(value)
            for value in stable_values
        )

    normalized = workload.casefold()
    return any(marker.casefold() in normalized for marker in SPECIALIZED_WORKLOAD_MARKERS)


def pick_engine(hardware: str, scale: str, workload: str | Mapping[str, Any]) -> dict:
    reasons = []
    engine = None

    if hardware == "CPU":
        engine = "llama.cpp"
        reasons.append("硬件为 CPU——只有 llama.cpp 具备竞争力")
        if scale == "single_user":
            reasons.append("单用户开发 → Ollama 封装 llama.cpp，提供一条命令即可使用的体验")
            engine = "Ollama（底层为 llama.cpp）"
    elif hardware == "Apple Silicon":
        engine = "Ollama" if scale == "single_user" else "llama.cpp"
        reasons.append("Apple Silicon → 通过 llama.cpp 使用 Metal（由 Ollama 封装）")
    elif hardware == "AMD":
        engine = "vLLM"
        reasons.append("AMD → vLLM 支持 ROCm；TRT-LLM 仅支持 NVIDIA")
        if _prefers_sglang(workload):
            engine = "SGLang"
            reasons.append("智能体型或前缀密集型负载 → SGLang RadixAttention")
    elif hardware == "NVIDIA Hopper":
        if _prefers_sglang(workload):
            engine = "SGLang"
            reasons.append("Hopper + 智能体型或前缀密集型负载 → SGLang 更为擅长")
        elif scale == "single_user":
            engine = "Ollama"
            reasons.append("Hopper 上的单用户场景属于开发用途 → Ollama 已足够")
        else:
            engine = "vLLM"
            reasons.append("Hopper 生产环境 → vLLM 是通用默认选择")
    elif hardware == "NVIDIA Blackwell":
        engine = "TRT-LLM"
        reasons.append("Blackwell + 吞吐量优先 → TRT-LLM 在 B200/GB200 上领先")
        if scale in ("small_team", "production") and not _prefers_sglang(workload):
            reasons.append("vLLM Blackwell SM120 紧随其后（v0.15.1，2026 年 2 月）")

    if scale == "enterprise":
        reasons.append("1 万名以上用户 → 组合生产栈（阶段 17 · 18）"
                      " + 分离式架构（阶段 17 · 17）+ 缓存感知路由器（阶段 17 · 11）")

    reasons.append("TGI 自 2025 年 12 月 11 日起进入维护模式——新项目默认不选 TGI")

    return {
        "hardware": hardware,
        "scale": scale,
        "workload": workload,
        "engine": engine,
        "reasons": reasons,
    }


SCENARIOS = [
    ("CPU",              "single_user",   "聊天"),
    ("Apple Silicon",    "single_user",   "编码助手"),
    ("NVIDIA Hopper",    "production",    "通用聊天"),
    ("NVIDIA Hopper",    "production",    "智能体多轮任务"),
    ("NVIDIA Blackwell", "enterprise",    "MoE 前沿模型服务"),
    ("AMD",              "production",    "大量复用前缀的 RAG"),
    ("NVIDIA Hopper",    "small_team",    "128K 长上下文"),
]


def main() -> None:
    print("=" * 80)
    print("自托管引擎决策树——硬件 / 规模 / 工作负载")
    print("=" * 80)
    for hw, sc, wl in SCENARIOS:
        d = pick_engine(hw, sc, wl)
        print(f"\n[{hw}] [{sc}] [{wl}]")
        print(f"  → 引擎：{d['engine']}")
        for r in d["reasons"]:
            print(f"    · {r}")


if __name__ == "__main__":
    main()
