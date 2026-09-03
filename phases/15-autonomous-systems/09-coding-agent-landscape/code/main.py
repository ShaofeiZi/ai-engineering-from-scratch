"""CodeAct 与 JSON tool-call 脚手架对比 — 纯标准库 Python。

两个脚手架使用相同的桩 "模型"（确定性规则），因此对比
将脚手架与模型质量隔离开来。指标：
  - 已解决的测试任务数
  - 消耗的轮数
  - per-action 爆炸半径（一次操作可触及的文件数）

这里的要点是教学性的：脚手架即 load-bearing.，OpenHands
（arXiv:2407.16741）明确押注了 CodeAct；JSON 工具调用
在由提供商控制执行器的托管服务中占据主导地位。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# ---------- 微型世界：一个小型内存“仓库” ----------

INITIAL_REPO = {
    "app.py": "def add(a, b):\n    return a - b\n",
    "util.py": "def lower(s):\n    return s.upper()\n",
    "cli.py": "VERSION = 'v0.0'\n",
}

TESTS = [
    ("app.py", "add(2, 3) == 5"),
    ("util.py", "lower('AB') == 'ab'"),
    ("cli.py", "VERSION == 'v1.0'"),
]

# 按路径替换：当测试失败时，桩“模型”所应用的替换。
# 将该表集中定义，避免在以下两者之间重复 if/elif 链
# 两个脚手架之间，并避免在 TESTS 后续扩展时出现 UnboundLocalError。
FIXES: dict[str, tuple[str, str]] = {
    "app.py": ("a - b", "a + b"),
    "util.py": ("s.upper()", "s.lower()"),
    "cli.py": ("v0.0", "v1.0"),
}


def run_tests(repo: dict[str, str]) -> list[bool]:
    """确定性桩：针对仓库字符串模拟测试套件。"""
    results = []
    for path, _expr in TESTS:
        src = repo.get(path, "")
        passed = False
        if path == "app.py":
            passed = "return a + b" in src
        elif path == "util.py":
            passed = "return s.lower()" in src
        elif path == "cli.py":
            passed = "VERSION = 'v1.0'" in src
        results.append(passed)
    return results


def _apply_fix(repo: dict[str, str], path: str) -> bool:
    """就地应用 per-path 修复。当且仅当应用了修复时返回 True。"""
    rule = FIXES.get(path)
    if rule is None:
        return False
    old, new = rule
    repo[path] = repo[path].replace(old, new)
    return True


# ---------- JSON tool-call 脚手架：每轮一次操作 ----------

@dataclass
class JsonScaffold:
    repo: dict[str, str] = field(default_factory=lambda: dict(INITIAL_REPO))
    turns: int = 0

    def step(self) -> str:
        """根据当前失败的测试，每次返回一个 JSON 操作。"""
        self.turns += 1
        results = run_tests(self.repo)
        for (path, _), ok in zip(TESTS, results, strict=True):
            if ok:
                continue
            if _apply_fix(self.repo, path):
                return json.dumps({"tool": "edit", "path": path})
        return json.dumps({"tool": "done"})

    def blast_radius(self) -> int:
        return 1  # 每次操作只触及一个文件

    def run(self, max_turns: int = 10) -> tuple[int, int]:
        for _ in range(max_turns):
            action = self.step()
            if json.loads(action).get("tool") == "done":
                break
        passed = sum(run_tests(self.repo))
        return passed, self.turns


# ---------- CodeAct 脚手架：一段代码片段可触及多个文件 ----------

@dataclass
class CodeActScaffold:
    repo: dict[str, str] = field(default_factory=lambda: dict(INITIAL_REPO))
    turns: int = 0
    # 跟踪观察到的单次操作触及文件数的最大值。
    # 这比 len(repo) 的静态上界更符合实际，因为
    # 如果有人添加了未经测试的辅助函数，它不会悄然膨胀。
    worst_touched: int = 0

    def step(self) -> str:
        """返回一段 Python 代码片段，可一次性编辑多个文件。"""
        self.turns += 1
        # 单个 "代码片段" 操作会一次重写所有失败的文件。
        snippet_lines = []
        results = run_tests(self.repo)
        for (path, _), ok in zip(TESTS, results, strict=True):
            if ok:
                continue
            if _apply_fix(self.repo, path):
                snippet_lines.append(f"fs.write('{path}', ...)")
        self.worst_touched = max(self.worst_touched, len(snippet_lines))
        if not snippet_lines:
            return "done()"
        return "; ".join(snippet_lines)

    def blast_radius(self) -> int:
        # 观察到的最坏情况：单次操作触及的文件数。
        return self.worst_touched

    def run(self, max_turns: int = 10) -> tuple[int, int]:
        for _ in range(max_turns):
            action = self.step()
            if action == "done()":
                break
        passed = sum(run_tests(self.repo))
        return passed, self.turns


# ---------- 驱动程序 ----------

def report(name: str, passed: int, turns: int, blast: int) -> None:
    total = len(TESTS)
    print(f"  {name:<18}  通过 {passed}/{total}  轮次 {turns:>2}  "
          f"影响范围 {blast}")


def main() -> None:
    print("=" * 70)
    print("CODEACT 与 JSON 工具调用脚手架（第 15 阶段，第 9 课）")
    print("=" * 70)
    print()
    print("相同的桩模型、含三个缺陷的玩具仓库，仅对比脚手架。")
    print("-" * 70)

    js = JsonScaffold()
    passed, turns = js.run()
    report("JSON tool-call", passed, turns, js.blast_radius())

    ca = CodeActScaffold()
    passed, turns = ca.run()
    report("CodeAct (stub)", passed, turns, ca.blast_radius())

    print()
    print("=" * 70)
    print("要点：脚手架不是摆设。它就是产品本身。")
    print("-" * 70)
    print("  相同的模型，两种脚手架，不同的轮数。")
    print("  CodeAct 将多次编辑压缩为一次操作。")
    print("  代价是爆炸半径：CodeAct 需要加固的沙箱")
    print("  隔离（OpenHands 使用 Docker）。JSON tool-calls 则通过")
    print("  构造本身获得安全性，因为每次操作都经过独立验证。")
    print("  两者都不是绝对更优；取舍取决于需要审计的内容。")


if __name__ == "__main__":
    main()
