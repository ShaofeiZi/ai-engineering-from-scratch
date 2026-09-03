"""带检查点的工作流：幂等性、前置条件、验证、回滚。

模拟四种场景：
  1. 正常运行
  2. 提交崩溃后重试  -> 幂等性阻止重复执行
  3. 前置条件失败     -> 工作流中止，不触发执行
  4. 验证失败         -> 触发回滚
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass


# ---------- 迷你数据库 ----------

DB = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}


def persist_transfer(txid: str, from_acct: str, to_acct: str, amount: int) -> None:
    DB[f"balance_{from_acct}"] -= amount
    DB[f"balance_{to_acct}"] += amount
    DB["last_transfer_id"] = txid


def rollback_transfer(txid: str, from_acct: str, to_acct: str, amount: int,
                      prior_last_transfer_id: str | None) -> None:
    # 补偿事务：恢复余额和先前的转账 ID。
    DB[f"balance_{from_acct}"] += amount
    DB[f"balance_{to_acct}"] -= amount
    DB["last_transfer_id"] = prior_last_transfer_id


# ---------- 检查点存储 ----------

@dataclass
class Checkpoint:
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def load(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def save(self, k: str, v: dict) -> None:
        # 原子写入：先序列化到同目录临时文件，fsync，再
        # 重命名。如果进程在写入过程中崩溃，原始文件
        # 仍然完好，因此下次重试会找到之前的
        # 幂等记录，而不是截断的 JSON 数据。
        data = self.load()
        data[k] = v
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)


# ---------- 工作流 ----------

def key(txid: str) -> str:
    return hashlib.sha256(txid.encode()).hexdigest()[:12]


def run_transfer(cp: Checkpoint, txid: str, from_acct: str, to_acct: str,
                 amount: int, min_balance: int,
                 inject_crash_after_execute: bool = False,
                 inject_verify_fail: bool = False) -> str:
    k = key(txid)
    record = cp.load().get(k, {"status": "new"})

    # 所有终态的幂等性。同一个 txid 在任意终态
    # 判决之后重试——已提交、已验证、已回滚、
    # 前置条件中止——都必须短路返回原始结果，
    # 而不是重新执行。
    terminal_results = {
        "committed": "idempotent-skip",
        "verified": "ok",
        "rolled-back": "verify-fail-rolled-back",
        "aborted-precondition": "aborted-precondition",
    }
    if record["status"] in terminal_results:
        return terminal_results[record["status"]]

    # 前置条件检查：转账后余额必须保持 >= min_balance
    if DB[f"balance_{from_acct}"] - amount < min_balance:
        cp.save(k, {"status": "aborted-precondition", "txid": txid})
        return "aborted-precondition"

    # 捕获先前状态，以便回滚时精确恢复（而非简单反转）。
    prior_last_transfer_id = DB["last_transfer_id"]

    # 在副作用发生之前记录意图，这样在 save 和
    # persist_transfer 之间崩溃时，会留下 "committed" 标记，
    # 重试时可以检测并短路。只有在后续读取（下方）
    # 确认副作用已落地后，才提升为 "verified"。
    #
    # 持久性细微缺口（课程权衡）：如果进程在
    # cp.save 之后、persist_transfer 之前崩溃，重试会看到
    # status == "committed" 并返回 "idempotent-skip"，
    # 即使转账实际上从未执行。生产系统通过以下方式
    # 消除此缺口：(a) 将幂等键传递到副作用本身，
    # 让目标 DB 强制 exactly-once，或
    # (b) 对 "committed" 增加目标的事后读取校验，
    # 这正是下方验证步骤在非崩溃路径中所做的。
    cp.save(k, {"status": "committed", "txid": txid,
                "from_acct": from_acct, "to_acct": to_acct,
                "amount": amount,
                "prior_last_transfer_id": prior_last_transfer_id})
    persist_transfer(txid, from_acct, to_acct, amount)
    if inject_crash_after_execute:
        raise RuntimeError("simulated crash after execute")

    # 事后验证
    if inject_verify_fail or DB["last_transfer_id"] != txid:
        rollback_transfer(txid, from_acct, to_acct, amount, prior_last_transfer_id)
        cp.save(k, {"status": "rolled-back", "txid": txid})
        return "verify-fail-rolled-back"

    cp.save(k, {"status": "verified", "txid": txid})
    return "ok"


# ---------- 驱动程序 ----------

def main() -> None:
    print("=" * 80)
    print("检查点与回滚（第 15 阶段，第 16 课）")
    print("=" * 80)

    tmp = tempfile.mkdtemp()
    print()
    print("场景 1：正常运行")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp1.json"))
    out = run_transfer(cp, "tx-001", "A", "B", 100, min_balance=200)
    print(f"  结果={out}  DB={DB}")

    print("\n场景 2：提交过程中崩溃，重试（幂等性捕获）")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp2.json"))
    try:
        run_transfer(cp, "tx-002", "A", "B", 100, min_balance=200,
                     inject_crash_after_execute=True)
    except RuntimeError as e:
        print(f"  崩溃: {e}")
    # 崩溃后重试
    out = run_transfer(cp, "tx-002", "A", "B", 100, min_balance=200)
    print(f"  重试结果={out}  DB={DB}")

    print("\n场景 3：前置条件失败（余额将低于最小值）")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp3.json"))
    out = run_transfer(cp, "tx-003", "A", "B", 10_000, min_balance=200)
    print(f"  结果={out}  DB={DB}")

    print("\n场景 4：验证失败 -> 回滚")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp4.json"))
    balances_before = dict(DB)
    out = run_transfer(cp, "tx-004", "A", "B", 100, min_balance=200,
                       inject_verify_fail=True)
    balances_after = dict(DB)
    print(f"  结果={out}  前后余额是否相等="
          f"{balances_before == balances_after}")

    print()
    print("=" * 80)
    print("要点：幂等性 + 前置条件 + 验证 + 回滚")
    print("-" * 80)
    print("  四个部分，缺一不可。各自覆盖不同的故障类别：")
    print("  幂等性   -> 崩溃时重试安全")
    print("  前置条件 -> 审批与提交之间的状态漂移")
    print("  验证     -> 副作用在我们以为发生时并未发生")
    print("  回滚     -> 已知不良状态已恢复或告警")
    print("  第 14 条运维解读：检查点可查询，回滚已演练，")
    print("  审计链在部署中存活。")


if __name__ == "__main__":
    main()
