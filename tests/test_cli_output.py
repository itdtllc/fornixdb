"""Token-budget-aware output: recall costs the consuming AI context, so the
consumer can cap it (--max-chars / MCP max_chars)."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from fornixdb.cli import fit_chars, main


class TestFitChars(unittest.TestCase):
    def test_no_budget_passthrough(self):
        lines = ["a" * 100, "b" * 100]
        self.assertEqual(fit_chars(lines, None), (lines, 0))
        self.assertEqual(fit_chars(lines, 0), (lines, 0))

    def test_keeps_whole_blocks_best_first(self):
        lines = ["first hit", "second hit", "third hit"]
        kept, omitted = fit_chars(lines, len("first hit\nsecond hit"))
        self.assertEqual(kept, ["first hit", "second hit"])
        self.assertEqual(omitted, 1)

    def test_under_budget_keeps_all(self):
        kept, omitted = fit_chars(["a", "b"], 1000)
        self.assertEqual((kept, omitted), (["a", "b"], 0))

    def test_first_block_longer_than_budget_truncates(self):
        # something beats nothing: the best hit is trimmed, not dropped
        kept, omitted = fit_chars(["x" * 50, "y"], 10)
        self.assertEqual(len(kept), 1)
        self.assertLessEqual(len(kept[0]), 10)
        self.assertTrue(kept[0].endswith("…"))
        self.assertEqual(omitted, 1)


class TestCliMaxChars(unittest.TestCase):
    def _run(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(list(argv))
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_recall_respects_budget(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            for i in range(5):
                self._run("--db", db, "--no-shared", "store",
                          "--gist", f"budget test fact number {i} with padding")
            full = self._run("--db", db, "--no-shared", "recall", "budget test fact")
            capped = self._run("--db", db, "--no-shared", "recall",
                               "budget test fact", "--max-chars", "100")
        self.assertEqual(full.count("budget test fact"), 5)
        self.assertLess(capped.count("budget test fact"), 5)
        self.assertIn("more — raise --max-chars", capped)


if __name__ == "__main__":
    unittest.main()
