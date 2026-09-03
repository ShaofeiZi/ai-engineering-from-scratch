"""群聊策略的中英文控制流回归测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "main.py"
SPEC = importlib.util.spec_from_file_location("group_chat_main", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GroupChatPolicyTests(unittest.TestCase):
    def test_coder_revises_after_english_review(self) -> None:
        result = MODULE.coder_policy([MODULE.Msg("reviewer", "Review: please fix this")])
        self.assertIn("a + b", result)

    def test_coder_revises_after_chinese_review(self) -> None:
        result = MODULE.coder_policy([MODULE.Msg("reviewer", "审查：请修复这个问题")])
        self.assertIn("a + b", result)

    def test_selector_routes_english_approval_to_manager(self) -> None:
        pool = [MODULE.Msg("reviewer", "review: APPROVED")]
        self.assertEqual(MODULE.llm_style_selector(pool, MODULE.AGENTS), "manager")

    def test_selector_routes_chinese_approval_to_manager(self) -> None:
        pool = [MODULE.Msg("reviewer", "审查：已批准")]
        self.assertEqual(MODULE.llm_style_selector(pool, MODULE.AGENTS), "manager")

    def test_manager_terminates_for_chinese_approval(self) -> None:
        pool = [MODULE.Msg("reviewer", "审查通过")]
        self.assertEqual(MODULE.manager_policy(pool), "TERMINATE")

    def test_unapproved_review_returns_to_coder(self) -> None:
        pool = [MODULE.Msg("reviewer", "review: not approved")]
        self.assertEqual(MODULE.llm_style_selector(pool, MODULE.AGENTS), "coder")

    def test_chinese_unapproved_review_returns_to_coder(self) -> None:
        pool = [MODULE.Msg("reviewer", "审查：未通过，请继续修复")]
        self.assertEqual(MODULE.llm_style_selector(pool, MODULE.AGENTS), "coder")


if __name__ == "__main__":
    unittest.main()
