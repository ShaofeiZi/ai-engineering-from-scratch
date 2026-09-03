"""多智能体 AI SRE 分诊模拟器——使用 Python 标准库。

三个专业智能体分别提出假设，监督者按共识度排序。
对抗式评估：无法达成共识时升级给人工处理。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentHypothesis:
    agent: str
    root_cause: str
    confidence: float
    evidence: list[str]


def log_agent(incident: str) -> AgentHypothesis:
    # 模拟扫描日志并选出最常见的错误标记
    if "checkout" in incident.lower():
        return AgentHypothesis(
            "LogAgent",
            "/api/llm 上的 KV 缓存激增导致 vLLM OOM",
            0.78,
            ["频率：每分钟 142 个错误", "模式：'kv_cache_allocation_failed'", "节点：pod-gpu-3"],
        )
    return AgentHypothesis("LogAgent", "原因不明", 0.35, ["日志中没有明显模式"])


def metric_agent(incident: str) -> AgentHypothesis:
    # 模拟用 PromQL 查询匹配已知模式
    return AgentHypothesis(
        "MetricAgent",
        "错误激增前 4 分钟，GPU 内存利用率达到 98%",
        0.82,
        ["DCGM_FI_DEV_FB_USED >= 97% 持续 240 秒", "与错误发生的相关系数：0.93"],
    )


def runbook_agent(incident: str) -> AgentHypothesis:
    # 模拟在运维手册仓库中进行向量搜索
    return AgentHypothesis(
        "RunbookAgent",
        "匹配运维手册 RB-017：突发并发下的 KV 缓存 OOM",
        0.88,
        ["运维手册：RB-017", "上次应用：2026-01-14", "安全操作：重启 Pod，并将 --gpu-memory-utilization 降至 0.85"],
    )


def supervisor(hypotheses: list[AgentHypothesis]) -> dict:
    # 对相似根因分组；达成共识会提高置信度
    root_causes = {}
    for h in hypotheses:
        key = h.root_cause.split(" on ")[0].split(" hit ")[0][:30]
        root_causes.setdefault(key, []).append(h)

    ranked = sorted(root_causes.items(), key=lambda kv: -sum(h.confidence for h in kv[1]))
    top_key, top_agents = ranked[0]
    adversarial_agreement = len(top_agents) >= 2
    action = "重启 Pod，并降低 --gpu-memory-utilization"  # 安全操作

    return {
        "top_root_cause": top_key,
        "supporting_agents": [h.agent for h in top_agents],
        "aggregated_confidence": sum(h.confidence for h in top_agents) / len(top_agents),
        "adversarial_agreement": adversarial_agreement,
        "proposed_action": action,
        "safety_gate": "需要人工批准" if not adversarial_agreement else "安全操作已自动批准",
    }


def main() -> None:
    print("=" * 80)
    print("AI SRE 分诊——多智能体调查生产事故")
    print("=" * 80)
    incident = "过去 6 分钟内 /checkout/generate-summary 错误率较高"
    print(f"\n事故：{incident}\n")

    hypotheses = [log_agent(incident), metric_agent(incident), runbook_agent(incident)]
    for h in hypotheses:
        print(f"[{h.agent}] 置信度={h.confidence:.2f}")
        print(f"  根因：{h.root_cause}")
        for e in h.evidence:
            print(f"  - {e}")
        print()

    decision = supervisor(hypotheses)
    print("-" * 80)
    print("监督者")
    print("-" * 80)
    for k, v in decision.items():
        print(f"  {k}: {v}")

    print("\n说明：监督者只会提出范围有限的安全操作。")
    print("涉及拓扑、代码或 IAM 的广泛变更始终会升级给人工负责人。")


if __name__ == "__main__":
    main()
