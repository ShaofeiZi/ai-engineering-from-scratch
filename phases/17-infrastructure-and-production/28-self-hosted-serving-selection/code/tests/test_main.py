"""自托管引擎工作负载分类的回归测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "main.py"
SPEC = importlib.util.spec_from_file_location("serving_selection_main", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ServingSelectionTests(unittest.TestCase):
    def test_english_agentic_workload_chooses_sglang(self) -> None:
        result = MODULE.pick_engine("NVIDIA Hopper", "production", "Agentic multi-turn")
        self.assertEqual(result["engine"], "SGLang")

    def test_chinese_agentic_workload_chooses_sglang(self) -> None:
        result = MODULE.pick_engine("NVIDIA Hopper", "production", "智能体多轮任务")
        self.assertEqual(result["engine"], "SGLang")

    def test_chinese_prefix_workload_chooses_sglang_on_amd(self) -> None:
        result = MODULE.pick_engine("AMD", "production", "大量复用前缀的 RAG")
        self.assertEqual(result["engine"], "SGLang")

    def test_stable_category_chooses_sglang(self) -> None:
        result = MODULE.pick_engine("NVIDIA Hopper", "production", {"category": "prefix_heavy"})
        self.assertEqual(result["engine"], "SGLang")

    def test_stable_boolean_chooses_sglang(self) -> None:
        result = MODULE.pick_engine("AMD", "production", {"prefix_reuse": True})
        self.assertEqual(result["engine"], "SGLang")

    def test_general_workload_chooses_vllm(self) -> None:
        result = MODULE.pick_engine("NVIDIA Hopper", "production", {"kind": "general_chat"})
        self.assertEqual(result["engine"], "vLLM")


if __name__ == "__main__":
    unittest.main()
