"""自主研究智能体——计划/执行/验证树搜索脚手架。

关键架构原语是对实验节点执行最佳优先树搜索，其中包含预算受限的扩展、逐节点
沙箱执行，以及新颖性 x 质量 x 预算评分函数。LLM 规划器和实际 PyTorch 实验均
由 stub 替代，因此无需真实计算即可端到端观察树搜索骨架。

运行：python main.py
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# 实验节点——（假设、配置、结果）元组
# ---------------------------------------------------------------------------

@dataclass
class Node:
    node_id: int
    parent: int | None
    hypothesis: str
    config: dict[str, object]
    result: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    novelty: float = 0.5
    quality: float = 0.0
    failure: str | None = None

    def score(self, remaining_budget: float) -> float:
        budget_weight = min(1.0, remaining_budget / 10.0)
        return self.novelty * 0.4 + self.quality * 0.5 + budget_weight * 0.1


# ---------------------------------------------------------------------------
# stub 规划器——通过小幅修改扩展来提出子节点
# ---------------------------------------------------------------------------

def expand(node: Node, next_id: int) -> list[Node]:
    """每次改变一个配置维度来提出子节点。"""
    children: list[Node] = []
    base_cfg = node.config
    # 改变稀疏度
    for sp in (4, 8, 16):
        cfg = dict(base_cfg, sparsity_top=sp)
        children.append(Node(node_id=next_id, parent=node.node_id,
                             hypothesis=f"sparsity top-{sp}",
                             config=cfg))
        next_id += 1
    # 改变学习率
    for lr in (3e-4, 1e-3):
        cfg = dict(base_cfg, lr=lr)
        children.append(Node(node_id=next_id, parent=node.node_id,
                             hypothesis=f"lr={lr}",
                             config=cfg))
        next_id += 1
    return children


# ---------------------------------------------------------------------------
# 沙箱执行——stub 实现；返回伪造但可复现的指标
# ---------------------------------------------------------------------------

def run_experiment(node: Node, rng: random.Random) -> None:
    """模拟在沙箱容器中运行实验。
    真实构建会调用 shell 执行：
      docker run --network=none --memory=8g --cpus=2 --read-only ...
    并从挂载的输出卷捕获 stdout 和指标文件。"""
    sp = node.config.get("sparsity_top", 8)
    lr = node.config.get("lr", 3e-4)
    # 根据超参数构造 loss（稀疏度越小通常越好，但存在限度）
    ideal_sp = 8
    loss = 3.0 - 0.3 * (1 - abs(sp - ideal_sp) / 16) + rng.gauss(0, 0.05)
    loss += 0.0001 * abs(lr - 3e-4) * 1000
    node.result = {"loss": round(loss, 3), "sparsity_top": sp, "lr": lr}
    node.cost_usd = 1.2 + rng.uniform(0, 0.4)
    node.quality = max(0.0, 1.0 - (loss - 2.5) / 1.5)
    node.novelty = 0.5 + rng.uniform(-0.1, 0.2)
    # 模拟偶发失败
    if rng.random() < 0.1:
        node.failure = "oom_killed_by_cgroup"
        node.quality = 0.0


# ---------------------------------------------------------------------------
# 验证步骤——评分前对结果进行健全性检查
# ---------------------------------------------------------------------------

def verify(node: Node) -> bool:
    if node.failure:
        return False
    if node.result.get("loss", 99) > 4.0:
        node.failure = "loss_diverged"
        return False
    return True


# ---------------------------------------------------------------------------
# 树搜索——受预算和最大深度约束的最佳优先搜索
# ---------------------------------------------------------------------------

@dataclass
class Tree:
    root: Node
    nodes: dict[int, Node] = field(default_factory=dict)
    frontier: list = field(default_factory=list)  # (neg_score, counter, node_id)
    counter: int = 0
    budget: float = 30.0
    spent: float = 0.0
    max_nodes: int = 24

    def push(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.counter += 1
        remaining = self.budget - self.spent
        heapq.heappush(self.frontier, (-node.score(remaining), self.counter, node.node_id))

    def pop(self) -> Node | None:
        while self.frontier:
            _, _, nid = heapq.heappop(self.frontier)
            return self.nodes[nid]
        return None


def tree_search(seed: str, rng: random.Random) -> Tree:
    root = Node(node_id=0, parent=None, hypothesis=seed, config={"sparsity_top": 8, "lr": 3e-4})
    root.novelty = 1.0
    root.quality = 0.5
    tree = Tree(root=root)
    tree.push(root)

    next_id = 1
    while tree.frontier and len(tree.nodes) < tree.max_nodes:
        cur = tree.pop()
        if cur is None:
            break
        if tree.spent >= tree.budget:
            print(f"    预算已耗尽，当前花费 ${tree.spent:.2f}")
            break
        if cur.node_id != 0:
            run_experiment(cur, rng)
            tree.spent += cur.cost_usd
            ok = verify(cur)
            flag = "ok " if ok else "FAIL"
            print(f"    [{flag}] 节点 #{cur.node_id:02d}  假设='{cur.hypothesis}'  "
                  f"loss={cur.result.get('loss','?'):>5}  "
                  f"$={cur.cost_usd:.2f}  cum=${tree.spent:.2f}")
            if not ok:
                continue
        # 扩展最有希望的节点
        children = expand(cur, next_id)
        next_id += len(children)
        for ch in children:
            tree.push(ch)

    return tree


# ---------------------------------------------------------------------------
# 最佳分支选择与报告撰写 stub
# ---------------------------------------------------------------------------

def best_branch(tree: Tree) -> list[Node]:
    done = [n for n in tree.nodes.values() if n.result and not n.failure]
    if not done:
        return []
    best = max(done, key=lambda n: n.quality)
    # 回溯到根节点
    chain = [best]
    while chain[-1].parent is not None:
        chain.append(tree.nodes[chain[-1].parent])
    return list(reversed(chain))


def main() -> None:
    print("=== 自主研究智能体：树搜索（预算 $30）===")
    rng = random.Random(7)
    seed = "investigate sparsity patterns in attention maps of sub-1B transformers"
    tree = tree_search(seed, rng)
    print()
    print(f"已探索节点：{len(tree.nodes)}")
    print(f"预算花费  ：${tree.spent:.2f} / ${tree.budget:.2f}")
    print(f"失败节点  ：{sum(1 for n in tree.nodes.values() if n.failure)}")

    branch = best_branch(tree)
    print(f"\n最佳分支（长度 {len(branch)}）：")
    for n in branch:
        print(f"  #{n.node_id:02d} {n.hypothesis}   q={n.quality:.2f}  loss={n.result.get('loss','?')}")

    print("\n（此处将运行 writer + reviewer + red-team 步骤；"
          "脚手架中使用 stub 实现）")


if __name__ == "__main__":
    main()
