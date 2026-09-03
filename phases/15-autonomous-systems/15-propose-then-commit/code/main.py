"""Propose-then-commit HITL 状态机 — 标准库 Python。

四个阶段：
  1. propose：代理通过幂等键持久化所提议的操作
  2. surface：审查者查看元数据（意图、血缘、影响范围、回滚）
  3. commit：需要正向确认；幂等
  4. verify：在提交后 re-read 目标资源

三个演示：
  - 干净的审批流程
  - 瞬时故障后重试 -> 幂等性捕获
  - rubber-stamp UI 对比 challenge-and-response 检查清单
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field


@dataclass
class Proposal:
    thread_id: str
    action: str
    payload: dict
    intent: str
    lineage: str
    blast_radius: str
    rollback: str

    def key(self) -> str:
        sig = json.dumps({"t": self.thread_id, "a": self.action,
                          "p": self.payload}, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:16]


@dataclass
class Store:
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def all(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def save(self, key: str, record: dict) -> None:
        data = self.all()
        data[key] = record
        with open(self.path, "w") as f:
            json.dump(data, f)


# ---------- 已执行 side-effect 跟踪器（假装是后端） ----------

SIDE_EFFECTS: list[str] = []


def execute(proposal: Proposal) -> bool:
    SIDE_EFFECTS.append(f"{proposal.action}:{json.dumps(proposal.payload)}")
    return True


def verify(proposal: Proposal) -> bool:
    # 在真实系统中，此步骤会重新读取目标资源。
    needle = f"{proposal.action}:{json.dumps(proposal.payload)}"
    return needle in SIDE_EFFECTS


# ---------- 流程 ----------

def propose(store: Store, p: Proposal) -> str:
    k = p.key()
    existing = store.all().get(k)
    if existing:
        print(f"  [提议] 幂等：记录 {k} 已存在"
              f"（status={existing['status']}）")
        return k
    record = {"status": "waiting", **vars(p)}
    store.save(k, record)
    print(f"  [propose] 记录 {k} 已存储，等待审查")
    return k


def surface(store: Store, k: str) -> None:
    r = store.all()[k]
    print(f"  [surface] 提案 {k}")
    # 使用 'name' 而非 'field'，以避免遮蔽 dataclasses.field
    # 以防日后读者在此模块下方添加 dataclass（Ruff F402）。
    for name in ("intent", "lineage", "blast_radius", "rollback"):
        print(f"    {name:<14} {r[name]}")


def rubber_stamp_approve(store: Store, k: str) -> bool:
    r = store.all()
    rec = r[k]
    rec["status"] = "approved"
    rec["ack_mode"] = "rubber_stamp"
    store.save(k, rec)
    print("  [批准：走过场] 点击了通过（无检查清单）")
    return True


def checklist_approve(store: Store, k: str,
                      understood: bool, verified: bool,
                      rollback_ready: bool) -> bool:
    if not (understood and verified and rollback_ready):
        print("  [批准：检查清单] REJECTED（回答不完整）")
        return False
    r = store.all()
    rec = r[k]
    rec["status"] = "approved"
    rec["ack_mode"] = "challenge_response"
    store.save(k, rec)
    print("  [批准：检查清单] APPROVED（三项检查全部通过）")
    return True


def commit(store: Store, k: str) -> bool:
    data = store.all()
    rec = data[k]
    if rec["status"] == "committed":
        print(f"  [提交] 幂等：{k} 已提交，无需重新执行")
        return True
    if rec["status"] != "approved":
        print(f"  [提交] 拒绝：{k} status={rec['status']}")
        return False
    p = Proposal(
        thread_id=rec["thread_id"], action=rec["action"],
        payload=rec["payload"], intent=rec["intent"],
        lineage=rec["lineage"], blast_radius=rec["blast_radius"],
        rollback=rec["rollback"],
    )
    execute(p)
    rec["status"] = "committed"
    store.save(k, rec)
    print(f"  [提交] 已执行；验证结果={verify(p)}")
    return True


# ---------- 演示 ----------

def main() -> None:
    print("=" * 80)
    print("PROPOSE-THEN-COMMIT HITL（第 15 阶段，第 15 课）")
    print("=" * 80)
    tmp = tempfile.mkdtemp()
    store = Store(os.path.join(tmp, "proposals.json"))

    p = Proposal(
        thread_id="t-001",
        action="email.send",
        payload={"to": "team@example.com", "subject": "release"},
        intent="向团队邮件列表宣布 v1.2 发布",
        lineage="发布说明页面 /releases/1.2",
        blast_radius="37 名收件人；误发会造成对外负面影响",
        rollback="无法带内回滚；需后续发送更正邮件",
    )

    print("\n演示 1：干净的审批流程（challenge-and-response）")
    print("-" * 80)
    k = propose(store, p)
    surface(store, k)
    checklist_approve(store, k, understood=True, verified=True, rollback_ready=True)
    commit(store, k)

    print("\n演示 2：审批后重试；幂等性捕获 re-exec")
    print("-" * 80)
    initial = len(SIDE_EFFECTS)
    commit(store, k)  # 重试
    commit(store, k)  # 重试
    print(f"  重试 2 次后的副作用总数：{len(SIDE_EFFECTS)} "
          f"（原为 {initial}）-> 保持幂等")

    print("\n演示 3：rubber-stamp UI vs challenge-and-response")
    print("-" * 80)
    p2 = Proposal(
        thread_id="t-002", action="db.update",
        payload={"row": 42, "col": "status", "val": "closed"},
        intent="关闭一个长期未更新的 issue",
        lineage="定期扫描长期未更新 issue 的看板",
        blast_radius="一行 DB 记录；可在 1 小时备份窗口内恢复",
        rollback="从夜间备份恢复该行",
    )
    k2 = propose(store, p2)
    rubber_stamp_approve(store, k2)
    commit(store, k2)

    p3 = Proposal(
        thread_id="t-003", action="db.drop_table",
        payload={"table": "old_users"},
        intent="依照清理运行手册删除未使用的表",
        lineage="runbook #RB-17",
        blast_radius="破坏性操作；将删除 42 万行；24 小时内无法恢复",
        rollback="从每周备份恢复；最多丢失 6 天数据",
    )
    k3 = propose(store, p3)
    # 审阅者无法勾选 rollback-ready；checklist_approve 拒绝
    ok = checklist_approve(store, k3, understood=True, verified=True,
                           rollback_ready=False)
    # 教学意图：对被拒绝的提案调用 commit()，以便该
    # 日志演示了当 status 仍为
    # "waiting" 而非 "approved" 时，commit() 会拒绝。我们将拒绝行 WANT 到
    # 打印。
    if not ok:
        commit(store, k3)

    print()
    print("=" * 80)
    print("要点：让结构化评审成为阻力最小的路径")
    print("-" * 80)
    print("  幂等键防止在重试时发生 double-execution。")
    print("  持久性使审批即使延迟两天到达依然能生效。")
    print("  Challenge-and-response 检查清单是文档中记录的缓解措施")
    print("  用于 rubber-stamp 审批；EU AI 法案第14条对此有要求。")
    print("  Post-commit 验证消除了“以为已发生”这类问题。")


if __name__ == "__main__":
    main()
