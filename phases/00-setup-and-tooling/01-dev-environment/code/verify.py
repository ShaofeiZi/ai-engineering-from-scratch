"""面向学习路线的环境预检工具（AI Engineering from Scratch）。

课程：phases/00-setup-and-tooling/01-dev-environment/docs/en.md
在开始某条学习路线之前，请从仓库根目录运行本文件。
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable


class ChineseHelpFormatter(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, prefix or "用法：")


@dataclass(frozen=True)
class Result:
    ok: bool
    detail: str


@dataclass(frozen=True)
class Probe:
    label: str
    run: Callable[[], Result]
    fix: str


@dataclass(frozen=True)
class Route:
    label: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    next_command: str
    manual: tuple[str, ...] = ()


def command_result(command: str, minimum_major: int | None = None) -> Result:
    path = shutil.which(command)
    if path is None:
        return Result(False, f"在 PATH 中未找到 {command!r}")

    try:
        process = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Result(False, f"无法运行 {path}: {exc}")

    output = (process.stdout or process.stderr).strip().splitlines()
    detail = output[0] if output else f"退出码 {process.returncode}，且没有版本输出"
    if process.returncode != 0:
        return Result(False, detail)

    if minimum_major is not None:
        digits = "".join(character if character.isdigit() else " " for character in detail)
        parts = digits.split()
        if not parts:
            return Result(False, f"无法从 {detail!r} 中解析版本号")
        major = int(parts[0])
        if major < minimum_major:
            return Result(False, f"检测到 {detail}；需要版本 {minimum_major}+")

    return Result(True, f"{detail}，路径：{path}")


def python_result() -> Result:
    version = platform.python_version()
    executable = sys.executable
    if sys.version_info < (3, 11):
        return Result(False, f"在 {executable} 检测到 Python {version}；需要 Python 3.11+")
    return Result(True, f"Python {version}，路径：{executable}")


def module_result(module: str) -> Result:
    if importlib.util.find_spec(module) is None:
        return Result(False, f"{sys.executable} 无法导入 {module!r}")
    return Result(True, f"可通过 {sys.executable} 导入")


def gpu_result() -> Result:
    if importlib.util.find_spec("torch") is None:
        return Result(False, "未安装 PyTorch，因此未检查加速器后端")

    try:
        import torch
    except Exception as exc:
        return Result(False, f"无法导入 PyTorch：{type(exc).__name__}: {exc}")

    if torch.cuda.is_available():
        return Result(True, f"CUDA: {torch.cuda.get_device_name(0)}")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return Result(True, "Apple MPS 可用")
    return Result(True, "仅使用 CPU；入门课程不要求 GPU")


def git_fix() -> str:
    system = platform.system()
    if system == "Darwin":
        return "运行 `xcode-select --install`，然后运行 `git --version`。"
    if system == "Windows":
        return "运行 `winget install --id Git.Git -e`，然后运行 `git --version`。"
    return "运行 `sudo apt-get update && sudo apt-get install -y git`，然后运行 `git --version`。"


PROBES = {
    "python": Probe(
        "Python 3.11+",
        python_result,
        "使用 `uv python install 3.12` 安装，激活该环境，再用 `python3` 重新运行。",
    ),
    "git": Probe("Git", lambda: command_result("git"), git_fix()),
    "node": Probe(
        "Node.js 20+",
        lambda: command_result("node", minimum_major=20),
        "运行 `fnm install 22 && fnm use 22`，然后运行 `node --version`。",
    ),
    "npx": Probe(
        "npx",
        lambda: command_result("npx"),
        "安装 Node.js 22，然后运行 `npm install -g npm` 和 `npx --version`。",
    ),
    "cargo": Probe(
        "Rust cargo",
        lambda: command_result("cargo"),
        "使用 rustup 安装 Rust，重启 shell，然后运行 `cargo --version`。",
    ),
    "julia": Probe(
        "Julia",
        lambda: command_result("julia"),
        "使用 juliaup 安装 Julia，重启 shell，然后运行 `julia --version`。",
    ),
    "numpy": Probe(
        "NumPy",
        lambda: module_result("numpy"),
        "激活课程环境，然后运行 `python3 -m pip install numpy`。",
    ),
    "matplotlib": Probe(
        "Matplotlib",
        lambda: module_result("matplotlib"),
        "激活课程环境，然后运行 `python3 -m pip install matplotlib`。",
    ),
    "jupyter": Probe(
        "Jupyter",
        lambda: module_result("jupyter"),
        "激活课程环境，然后运行 `python3 -m pip install jupyter`。",
    ),
    "torch": Probe(
        "PyTorch",
        lambda: module_result("torch"),
        "激活课程环境，然后运行 `python3 -m pip install torch`。",
    ),
    "gpu": Probe(
        "加速器后端",
        gpu_result,
        "GPU 为可选项。如需检测 CUDA 或 Apple MPS，请先安装 PyTorch。",
    ),
}


BASE_OPTIONAL = ("node", "npx", "numpy", "matplotlib", "jupyter", "torch", "gpu", "cargo", "julia")

ROUTES = {
    "beginner": Route(
        "入门课程",
        ("python", "git"),
        BASE_OPTIONAL,
        "python3 phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.py",
    ),
    "ml-foundations": Route(
        "数学与 ML 基础",
        ("python", "git", "numpy"),
        ("matplotlib", "jupyter", "torch", "gpu", "julia"),
        "python3 phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.py",
    ),
    "llm-engineering": Route(
        "LLM 工程",
        ("python", "git"),
        ("numpy", "torch", "gpu", "node", "npx", "cargo"),
        "python3 phases/11-llm-engineering/01-prompt-engineering/code/prompt_engineering.py",
    ),
    "agents": Route(
        "智能体工程",
        ("python", "git"),
        ("node", "npx", "numpy", "torch"),
        "python3 phases/14-agent-engineering/01-the-agent-loop/code/main.py",
    ),
    "mcp": Route(
        "Model Context Protocol (MCP)",
        ("python", "git"),
        ("node", "npx"),
        "python3 phases/13-tools-and-protocols/06-mcp-fundamentals/code/main.py",
    ),
    "agent-skills": Route(
        "Agent Skills 工程",
        ("python", "git", "node", "npx"),
        (),
        "python3 phases/13-tools-and-protocols/22-skills-and-agent-sdks/code/main.py",
        (
            "选择一个支持 skill 的宿主，并确认已完成安装。",
            "选择用户级或项目级 skill 作用域，并确认该作用域可写。",
        ),
    ),
    "certification": Route(
        "Claude 认证备考",
        ("python", "git"),
        ("node", "npx"),
        "打开 certifications/claude/GETTING_STARTED.md 并选择一条认证路线。",
        ("如使用 AI 导师，请确认所选宿主可以读取仓库 skills。",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅检查开始所选课程路线所需的工具。",
        add_help=False,
        formatter_class=ChineseHelpFormatter,
    )
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示此帮助信息并退出")
    parser.add_argument(
        "--route",
        choices=tuple(ROUTES),
        default="beginner",
        help="要准备的学习路线（默认：beginner）",
    )
    parser.add_argument(
        "--show-later",
        action="store_true",
        help="同时检查当前可选或后续课程才需要的工具",
    )
    return parser.parse_args()


def print_probe(key: str, required: bool) -> bool:
    probe = PROBES[key]
    result = probe.run()
    if result.ok:
        status = "PASS"
    elif required:
        status = "FAIL"
    else:
        status = "LATER"
    timing = "当前必需" if required else "当前可选或后续需要"
    print(f"  [{status}] {probe.label} ({timing})")
    print(f"         {result.detail}")
    if not result.ok:
        print(f"         修复方法：{probe.fix}")
    return result.ok


def main() -> int:
    args = parse_args()
    route = ROUTES[args.route]

    print("\n=== AI Engineering from Scratch：环境检查 ===\n")
    print(f"学习路线：{route.label}（`--route {args.route}`）\n")

    passed = 0
    for key in route.required:
        passed += int(print_probe(key, required=True))

    if route.optional and args.show_later:
        print("\n当前可选或后续需要的工具：")
        for key in route.optional:
            print_probe(key, required=False)
    elif route.optional:
        print(
            f"\n已跳过后续检查：开始学习暂不需要这 {len(route.optional)} 个工具。"
            "需要检查时，请添加 `--show-later`。"
        )

    if route.manual:
        print("\n手动检查：")
        for item in route.manual:
            print(f"  [MANUAL] {item}")

    total = len(route.required)
    print(f"\n结果：{passed}/{total} 项必需检查通过")
    if passed == total:
        print(f"已可开始学习{route.label}。")
        print(f"下一步：{route.next_command}\n")
        return 0

    print("环境尚未就绪。请逐一执行上方的修复命令，然后重新运行本预检。\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
