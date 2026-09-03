"""动作执行器的双语显示与稳定语义回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import main


class ApplyActionTests(unittest.TestCase):
    def click(self, element_desc: str, *, element_id: str | None = None) -> str:
        action = {"action": "click", "element_desc": element_desc}
        if element_id is not None:
            action["element_id"] = element_id
        return main.apply_action(main.BrowserState(), action).page

    def test_original_english_search_and_booking_text_still_works(self) -> None:
        self.assertEqual(self.click("Search flights"), "search")
        self.assertEqual(self.click("Book now"), "confirmation")
        self.assertEqual(self.click("Submit"), "confirmation")

    def test_original_english_login_flow_still_works(self) -> None:
        self.assertEqual(self.click("Login"), "login")
        self.assertEqual(self.click("Forgot password"), "reset_sent")

    def test_chinese_display_text_is_supported(self) -> None:
        self.assertEqual(self.click("搜索航班"), "search")
        self.assertEqual(self.click("立即预订"), "confirmation")
        self.assertEqual(self.click("忘记密码"), "reset_sent")

    def test_stable_element_id_is_independent_of_display_language(self) -> None:
        self.assertEqual(self.click("Rechercher", element_id="search"), "search")
        self.assertEqual(self.click("予約する", element_id="book"), "confirmation")

    def test_unknown_element_keeps_current_page(self) -> None:
        state = main.BrowserState(page="results")
        result = main.apply_action(
            state, {"action": "click", "element_desc": "Help"}
        )
        self.assertEqual(result.page, "results")


if __name__ == "__main__":
    unittest.main()
