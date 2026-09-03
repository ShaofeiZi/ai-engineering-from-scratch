"""DevOps 故障排查智能体——K8s 知识图谱 + HITL 审批门禁。

关键架构原语包括：(a) K8s 知识图谱，使根因分析能够从告警对象遍历到其邻居，
并叠加遥测数据；(b) 默认只读的工具接口，每条破坏性命令都必须通过人在回路
审批，而且每条被考虑的命令都会写入审计日志。此脚手架实现了这两部分。

运行：python main.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# K8s 知识图谱——对象 + 遥测叠加边
# ---------------------------------------------------------------------------

@dataclass
class Node:
    kind: str               # "Pod" | "Deployment" | "Node" | "Service" | "Prom" | "Loki"
    name: str
    attrs: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.name}"


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # (src, rel, dst)

    def add(self, n: Node) -> None:
        self.nodes[n.key] = n

    def link(self, src: str, rel: str, dst: str) -> None:
        self.edges.append((src, rel, dst))

    def neighbors(self, key: str) -> list[tuple[str, str]]:
        out = [(rel, dst) for s, rel, dst in self.edges if s == key]
        out += [(rel, src) for src, rel, dst in self.edges if dst == key]
        return out


def build_sample_cluster() -> Graph:
    g = Graph()
    dep = Node("Deployment", "checkout-api",
               {"revision": 42, "image": "checkout-api:v2.41", "deployed_at": "14m ago"})
    rs = Node("ReplicaSet", "checkout-api-abc")
    node = Node("Node", "ip-10-2-3-4", {"kernel": "6.1.109"})
    pods = [Node("Pod", f"checkout-api-abc-{i}", {"phase": "Running"}) for i in range(3)]
    svc = Node("Service", "checkout-api")
    prom = Node("Prom", "error_rate{deployment=checkout-api}",
                {"last_15m": "mean=0.14 up_trend", "threshold": 0.05})
    loki = Node("Loki", "namespace=prod,app=checkout-api",
                {"last_15m": "500 errors on /api/v2/pay, stack = NullHealthz"})

    for n in (dep, rs, node, svc, prom, loki, *pods):
        g.add(n)
    g.link(dep.key, "OWNS", rs.key)
    for p in pods:
        g.link(rs.key, "OWNS", p.key)
        g.link(p.key, "SCHEDULED_ON", node.key)
    g.link(svc.key, "EXPOSES", dep.key)
    g.link(dep.key, "OBSERVED_BY", prom.key)
    g.link(dep.key, "OBSERVED_BY", loki.key)
    return g


# ---------------------------------------------------------------------------
# 假设排序——时效性 * 特异性 * 引用数量
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    title: str
    citations: list[str]
    recency_mins: int
    specificity: float     # 0..1
    path_len: int

    def score(self) -> float:
        recency_w = max(0.0, 1.0 - self.recency_mins / 60.0)
        path_w = 1.0 / (1 + self.path_len)
        return (recency_w * 0.35 +
                self.specificity * 0.35 +
                min(len(self.citations), 5) / 5 * 0.2 +
                path_w * 0.1)


def root_cause(g: Graph, alerted: str) -> list[Hypothesis]:
    """从告警对象向外遍历，收集遥测数据并提出排序后的假设。"""
    hyps: list[Hypothesis] = []
    # 最近的遥测同级节点
    telemetry: list[Node] = []
    for rel, neighbor_key in g.neighbors(alerted):
        n = g.nodes.get(neighbor_key)
        if n and n.kind in ("Prom", "Loki", "Tempo"):
            telemetry.append(n)

    # 假设：若近期部署后观察到错误激增，则可能是发布异常
    dep = g.nodes.get(alerted)
    if dep and dep.kind == "Deployment":
        mins = int(str(dep.attrs.get("deployed_at", "?")).split("m")[0]) if "m" in str(dep.attrs.get("deployed_at", "")) else 999
        hyps.append(Hypothesis(
            title=f"bad rollout: image {dep.attrs.get('image')} fails /healthz",
            citations=[t.name for t in telemetry],
            recency_mins=mins,
            specificity=0.82,
            path_len=0,
        ))

    # 假设：节点级问题（嘈杂邻居 / 内核）
    nodes = [g.nodes[dst] for _, dst in g.neighbors(alerted) if dst.startswith("Node/")]
    if nodes:
        hyps.append(Hypothesis(
            title=f"node-level pressure on {nodes[0].name} (kernel={nodes[0].attrs.get('kernel')})",
            citations=[n.name for n in nodes],
            recency_mins=30,
            specificity=0.45,
            path_len=2,
        ))

    # 假设：服务网格 / DNS
    hyps.append(Hypothesis(
        title="DNS flap in kube-system/coredns",
        citations=[],
        recency_mins=60,
        specificity=0.2,
        path_len=4,
    ))

    return sorted(hyps, key=lambda h: -h.score())


# ---------------------------------------------------------------------------
# 审批门禁 + 审计日志——追踪每条被考虑的命令
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    ts: float
    tool: str
    args: dict
    considered: bool = True
    approved: bool = False
    executed: bool = False
    approver: str | None = None
    result: str | None = None


@dataclass
class Agent:
    graph: Graph
    audit: list[AuditEvent] = field(default_factory=list)
    read_only_tools: tuple = ("kubectl_get", "kubectl_describe", "promql", "logql", "traceql")
    destructive_tools: tuple = ("kubectl_scale", "kubectl_rollback", "kubectl_delete", "argocd_rollback")

    def call(self, tool: str, args: dict, approver: str | None = None) -> AuditEvent:
        ev = AuditEvent(ts=time.time(), tool=tool, args=args)
        if tool in self.read_only_tools:
            ev.executed = True
            ev.result = "ok (read-only)"
        elif tool in self.destructive_tools:
            if approver:
                ev.approved = True
                ev.approver = approver
                ev.executed = True
                ev.result = f"executed by {approver}"
            else:
                ev.result = "blocked: no slack approval"
        else:
            ev.result = "blocked: unknown tool"
        self.audit.append(ev)
        return ev


# ---------------------------------------------------------------------------
# 演示——完整的告警 -> 图遍历 -> 假设排序 -> Slack 门禁流程
# ---------------------------------------------------------------------------

def main() -> None:
    g = build_sample_cluster()
    agent = Agent(graph=g)

    alerted = "Deployment/checkout-api"
    print(f"=== 收到告警：{alerted}（错误率 14%）===")

    # 智能体首先拉取只读遥测数据
    agent.call("promql", {"query": "rate(http_requests_total{status=~'5..'}[5m])"})
    agent.call("logql", {"query": '{app="checkout-api"} |~ "stack"'})

    hyps = root_cause(g, alerted)
    print("\n排序后的假设：")
    for i, h in enumerate(hyps, 1):
        print(f"  #{i} score={h.score():.3f}  {h.title}")
        print(f"     引用：{h.citations}")

    # 智能体提出回滚方案，但必须等待 Slack 审批
    print("\n提出修复方案：")
    ev = agent.call("argocd_rollback", {"app": "checkout-api", "to_revision": 41})
    print(f"  {ev.tool}: {ev.result}")

    # Slack 已批准 -> 智能体执行
    print("\nSlack 审批已由 alice@sre 通过")
    ev = agent.call("argocd_rollback",
                    {"app": "checkout-api", "to_revision": 41},
                    approver="alice@sre")
    print(f"  {ev.tool}: {ev.result}")

    print("\n审计日志：")
    for ev in agent.audit:
        print(" ", json.dumps({
            "tool": ev.tool, "executed": ev.executed,
            "approved": ev.approved, "approver": ev.approver,
            "result": ev.result,
        }))


if __name__ == "__main__":
    main()
