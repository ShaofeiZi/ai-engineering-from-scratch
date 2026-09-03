import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "fourier.py"
SPEC = importlib.util.spec_from_file_location("lesson_fourier", MODULE_PATH)
fourier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fourier)


class FourierPromptOutputTests(unittest.TestCase):
    def test_write_prompt_output_populates_missing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"

            fourier.write_prompt_output(output_dir=output_dir)

            canonical = output_dir / "prompt-spectral-analyzer.md"
            zh_companion = output_dir / "prompt-spectral-analyzer.zh-CN.md"
            self.assertTrue(canonical.exists())
            self.assertTrue(zh_companion.exists())
            self.assertEqual(
                canonical.read_text(encoding="utf-8"),
                fourier.PROMPT_SPECTRAL_ANALYZER_EN,
            )
            self.assertEqual(
                zh_companion.read_text(encoding="utf-8"),
                fourier.PROMPT_SPECTRAL_ANALYZER_ZH_CN,
            )

    def test_write_prompt_output_preserves_existing_canonical_english(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            output_dir.mkdir()
            canonical = output_dir / "prompt-spectral-analyzer.md"
            canonical_text = "canonical english prompt\n"
            canonical.write_text(canonical_text, encoding="utf-8")

            fourier.write_prompt_output(output_dir=output_dir)

            self.assertEqual(canonical.read_text(encoding="utf-8"), canonical_text)
            zh_companion = output_dir / "prompt-spectral-analyzer.zh-CN.md"
            self.assertTrue(zh_companion.exists())
            self.assertEqual(
                zh_companion.read_text(encoding="utf-8"),
                fourier.PROMPT_SPECTRAL_ANALYZER_ZH_CN,
            )

    def test_prompt_outputs_keep_english_and_chinese_in_separate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"

            fourier.write_prompt_output(output_dir=output_dir)

            canonical = (output_dir / "prompt-spectral-analyzer.md").read_text(
                encoding="utf-8"
            )
            zh_companion = (
                output_dir / "prompt-spectral-analyzer.zh-CN.md"
            ).read_text(encoding="utf-8")

            self.assertIn("You are a spectral analysis expert.", canonical)
            self.assertNotIn("你是一位频谱分析专家", canonical)
            self.assertIn("你是一位频谱分析专家", zh_companion)
            self.assertNotIn("You are a spectral analysis expert.", zh_companion)


if __name__ == "__main__":
    unittest.main()
