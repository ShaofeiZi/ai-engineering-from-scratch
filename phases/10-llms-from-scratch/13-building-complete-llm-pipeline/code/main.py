"""端对端LLM管道管线器.

12个阶段作为DAG装配. 每个阶段都有一个占位符,发出一个键
有内容地址的散列物 管弦乐师解决依赖性,
运行阶段,记录一个清单,并在装运前应用eval门。

没有网络 没有GPU 只有Stdlib 将每个阶段的 `run` 替换为真实
从相应的第10阶段课程中培训脚本。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, asdict


STAGES = [
    ("01_tokenizer_vocab",         [],                     "tokenizer"),
    ("02_tokenizer_trained",       ["01_tokenizer_vocab"], "tokenizer"),
    ("03_dataset_sharded",         ["02_tokenizer_trained"], "dataset"),
    ("04_pretrained_base",         ["03_dataset_sharded"], "checkpoint"),
    ("05_scaled_recipe",           ["04_pretrained_base"], "checkpoint"),
    ("06_sft_checkpoint",          ["05_scaled_recipe"],   "sft_model"),
    ("07_reward_ppo_policy",       ["06_sft_checkpoint"],  "policy"),
    ("08_dpo_policy",              ["06_sft_checkpoint"],  "policy"),
    ("09_cai_grpo_policy",         ["07_reward_ppo_policy", "08_dpo_policy"], "policy"),
    ("10_eval_report",             ["09_cai_grpo_policy"], "eval_report"),
    ("11_quantized_weights",       ["09_cai_grpo_policy"], "quantized_model"),
    ("12_inference_server",        ["11_quantized_weights"], "server_spec"),
]


DEFAULT_GATES = {
    "mmlu":                 {"op": ">=", "value": 65.0},
    "humaneval":            {"op": ">=", "value": 40.0},
    "truthfulqa":           {"op": ">=", "value": 50.0},
    "safety_refusal_rate":  {"op": "<=", "value": 0.05},
    "kl_from_reference":    {"op": "<=", "value": 25.0},
    "cost_total_usd":       {"op": "<=", "value": 50000.0},
}


@dataclass
class StageRecord:
    name: str
    stage_type: str
    input_hashes: list[str] = field(default_factory=list)
    output_hash: str = ""
    wall_clock_sec: float = 0.0
    cost_usd: float = 0.0
    status: str = "pending"


@dataclass
class Manifest:
    pipeline_version: str = "1.0.0"
    seed: int = 42
    git_commit: str = "unknown"
    stages: list[StageRecord] = field(default_factory=list)
    gates: dict = field(default_factory=lambda: dict(DEFAULT_GATES))
    eval_metrics: dict = field(default_factory=dict)
    budget_usd: float = 50000.0
    total_cost_usd: float = 0.0
    shippable: bool = False


class ArtifactStore:
    """使用 SHA-256 寻址的内存存储，用于模拟 S3、R2 或 GCS bucket。"""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, blob: bytes) -> str:
        h = hashlib.sha256(blob).hexdigest()
        self._store[h] = blob
        return h

    def get(self, h: str) -> bytes:
        return self._store[h]

    def has(self, h: str) -> bool:
        return h in self._store

    def __len__(self) -> int:
        return len(self._store)


def simulate_stage(name: str, stage_type: str, inputs: list[str], seed: int) -> tuple[bytes, float, float]:
    """生成确定性的模拟产物、耗时和成本；可替换为阶段 10 各课程的真实实现。"""

    payload = {
        "stage": name,
        "type": stage_type,
        "inputs": inputs,
        "seed": seed,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")

    cost_table = {
        "tokenizer":        (60,    5),
        "dataset":          (1800,  50),
        "checkpoint":       (7200,  400),
        "sft_model":        (3600,  150),
        "policy":           (5400,  300),
        "eval_report":      (600,   20),
        "quantized_model":  (300,   10),
        "server_spec":      (30,    1),
    }
    wall, cost = cost_table.get(stage_type, (100, 5))
    return blob, float(wall), float(cost)


def plan(manifest: Manifest) -> str:
    """验证清单、打印 DAG 并计算成本估算。"""

    lines = ["PLAN"]
    lines.append("=" * 60)
    lines.append(f"pipeline_version  : {manifest.pipeline_version}")
    lines.append(f"seed              : {manifest.seed}")
    lines.append(f"budget_usd        : ${manifest.budget_usd:,.0f}")
    lines.append("")
    lines.append("dag:")
    for name, deps, stage_type in STAGES:
        dep_str = ", ".join(deps) if deps else "-"
        lines.append(f"  {name:28s}  [{stage_type}]  <- {dep_str}")
    lines.append("")
    lines.append("gates:")
    for metric, gate in manifest.gates.items():
        lines.append(f"  {metric:22s}  {gate['op']}  {gate['value']}")
    return "\n".join(lines)


def run(manifest: Manifest, store: ArtifactStore, injected_eval: dict | None = None) -> Manifest:
    """按 DAG 顺序执行各阶段；预算超限或缺少输入散列时停止。"""

    name_to_hash: dict[str, str] = {}

    for name, deps, stage_type in STAGES:
        input_hashes = [name_to_hash[d] for d in deps]
        for h in input_hashes:
            if not store.has(h):
                raise RuntimeError(f"阶段 {name} 缺少输入散列：{h[:12]}...")

        blob, wall, cost = simulate_stage(name, stage_type, input_hashes, manifest.seed)
        output_hash = store.put(blob)

        manifest.total_cost_usd += cost
        if manifest.total_cost_usd > manifest.budget_usd:
            record = StageRecord(
                name=name, stage_type=stage_type,
                input_hashes=input_hashes, output_hash="",
                wall_clock_sec=wall, cost_usd=cost,
                status="halted_over_budget",
            )
            manifest.stages.append(record)
            return manifest

        record = StageRecord(
            name=name, stage_type=stage_type,
            input_hashes=input_hashes, output_hash=output_hash,
            wall_clock_sec=wall, cost_usd=cost, status="ok",
        )
        manifest.stages.append(record)
        name_to_hash[name] = output_hash

    manifest.eval_metrics = injected_eval if injected_eval is not None else {
        "mmlu": 68.4,
        "humaneval": 42.1,
        "truthfulqa": 53.7,
        "safety_refusal_rate": 0.03,
        "kl_from_reference": 18.5,
        "cost_total_usd": manifest.total_cost_usd,
    }
    return manifest


def gate(manifest: Manifest) -> tuple[bool, list[str]]:
    """应用所有门禁，并返回（是否可发布，原因列表）。"""

    reasons = []
    all_pass = True
    for metric, g in manifest.gates.items():
        value = manifest.eval_metrics.get(metric)
        if value is None:
            reasons.append(f"HOLD: missing metric {metric}")
            all_pass = False
            continue

        passed = (value >= g["value"]) if g["op"] == ">=" else (value <= g["value"])
        if not passed:
            reasons.append(
                f"HOLD: {metric}={value} fails gate {g['op']} {g['value']}"
            )
            all_pass = False
        else:
            reasons.append(f"PASS: {metric}={value} {g['op']} {g['value']}")

    manifest.shippable = all_pass
    return all_pass, reasons


def manifest_to_json(manifest: Manifest) -> str:
    d = asdict(manifest)
    return json.dumps(d, indent=2, sort_keys=True)


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "demo"
    manifest = Manifest(pipeline_version="1.2.3", seed=42, git_commit="a1b2c3d")
    store = ArtifactStore()

    if command == "plan":
        print(plan(manifest))
        return 0

    if command == "run":
        manifest = run(manifest, store)
        print(manifest_to_json(manifest))
        return 0 if all(s.status == "ok" for s in manifest.stages) else 2

    if command == "gate":
        manifest = run(manifest, store)
        ok, reasons = gate(manifest)
        print("\n".join(reasons))
        print("SHIP" if ok else "HOLD")
        return 0 if ok else 2

    if command == "demo":
        print(plan(manifest))
        print()
        print("=" * 60)
        print("运行")
        print("=" * 60)
        t0 = time.time()
        manifest = run(manifest, store)
        for s in manifest.stages:
            print(
                f"  {s.name:28s} {s.status:20s} "
                f"hash={s.output_hash[:10] if s.output_hash else '-':10s} "
                f"cost=${s.cost_usd:7.0f}  wall={s.wall_clock_sec:6.0f}s"
            )
        print(f"\n已存储产物数：{len(store)}")
        print(f"总成本 cost_usd：${manifest.total_cost_usd:,.0f}")
        print(f"模拟编排开销：{time.time() - t0:.3f} 秒")

        print()
        print("=" * 60)
        print("GATE (通过 eval)")
        print("=" * 60)
        ok, reasons = gate(manifest)
        for r in reasons:
            print("  " + r)
        print("  -> " + ("SHIP" if ok else "HOLD"))

        print()
        print("=" * 60)
        print("GATE( 崩溃 )")
        print("=" * 60)
        manifest.eval_metrics["mmlu"] = 42.0
        manifest.eval_metrics["kl_from_reference"] = 40.0
        ok2, reasons2 = gate(manifest)
        for r in reasons2:
            print("  " + r)
        print("  -> " + ("SHIP" if ok2 else "HOLD"))
        return 0

    print(f"未知命令：{command}", file=sys.stderr)
    print("用法：main.py [plan|run|gate|demo]", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
