import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "statistics.py"
SPEC = importlib.util.spec_from_file_location("lesson_statistics", MODULE_PATH)
statistics_lesson = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(statistics_lesson)


class EffectInterpretationTests(unittest.TestCase):
    def test_interpret_cohens_d_returns_stable_machine_labels(self):
        self.assertEqual(statistics_lesson.interpret_cohens_d(0.0), "negligible")
        self.assertEqual(statistics_lesson.interpret_cohens_d(0.2), "small")
        self.assertEqual(statistics_lesson.interpret_cohens_d(0.5), "medium")
        self.assertEqual(statistics_lesson.interpret_cohens_d(0.8), "large")
        self.assertEqual(statistics_lesson.interpret_cohens_d(-1.1), "large")

    def test_format_effect_interpretation_keeps_chinese_in_display_layer(self):
        self.assertEqual(
            statistics_lesson.format_effect_interpretation("negligible"),
            "可忽略",
        )
        self.assertEqual(statistics_lesson.format_effect_interpretation("small"), "小")
        self.assertEqual(statistics_lesson.format_effect_interpretation("medium"), "中")
        self.assertEqual(statistics_lesson.format_effect_interpretation("large"), "大")
        self.assertEqual(
            statistics_lesson.format_effect_interpretation("custom"),
            "custom",
        )

    def test_ab_test_simulator_exposes_machine_readable_effect_label(self):
        result = statistics_lesson.ab_test_simulator(n_per_group=8, true_effect=0.0)
        self.assertIn(
            result["effect_interpretation"],
            {"negligible", "small", "medium", "large"},
        )


if __name__ == "__main__":
    unittest.main()
