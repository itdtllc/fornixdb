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


class TestFeelCli(unittest.TestCase):
    """`fornixdb feel <reading>` — the literal-reading path is platform-neutral
    (no pmset), so it runs everywhere; the battery path is Mac-only by design."""

    def _run(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(list(argv))
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_literal_reading_becomes_a_feel_memory(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            out = self._run("--db", db, "--no-shared", "feel",
                            "lid closed", "--sensor", "lid")
            self.assertIn("feel[lid]: lid closed", out)
            found = self._run("--db", db, "--no-shared", "recall", "lid closed")
        self.assertIn("feel[lid]: lid closed", found)


DOC = """# Top

Top body.

## Child

Child body.
"""


class TestImportMarkdownCli(unittest.TestCase):
    def _run(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(list(argv))
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_doc_mode_chunks_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            doc = Path(d) / "doc.md"
            doc.write_text(DOC)
            out = self._run("--db", db, "--no-shared", "import-markdown", str(doc))
            again = self._run("--db", db, "--no-shared", "import-markdown", str(doc))
            recall = self._run("--db", db, "--no-shared", "recall", "child")
        self.assertIn("imported 2", out)         # Top + Child
        self.assertIn("imported 0", again)       # idempotent re-run
        self.assertIn("skipped 2", again)
        self.assertIn("Child", recall)

    def test_frontmatter_mode(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            Path(d, "fact.md").write_text(
                "---\nname: a-fact\ndescription: a one liner\n---\nbody\n")
            out = self._run("--db", db, "--no-shared",
                            "import-markdown", d, "--frontmatter")
        self.assertIn("imported 1", out)

    def test_export_then_reimport_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            out_dir = str(Path(d) / "export")
            self._run("--db", db, "--no-shared", "store",
                      "--gist", "a fact to export", "--name", "exported-fact")
            out = self._run("--db", db, "--no-shared", "export-markdown", out_dir)
            self.assertTrue((Path(out_dir) / "exported-fact.md").exists())
            self.assertTrue((Path(out_dir) / "FornixDB.md").exists())
            db2 = str(Path(d) / "m2.db")
            self._run("--db", db2, "--no-shared",
                      "import-markdown", out_dir, "--frontmatter")
            recall = self._run("--db", db2, "--no-shared", "recall", "fact to export")
        self.assertIn("exported 1", out)
        self.assertIn("a fact to export", recall)


class TestEvalEmptyStoreGuard(unittest.TestCase):
    """eval/answer-eval on a 0-memory store must fail loudly, not report a
    false 0% that looks like a total regression (the silent-empty-store
    footgun: --db is a PRE-subcommand global, easy to omit)."""

    def test_eval_empty_store_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "empty.db")            # fresh store: 0 memories
            golden = str(Path(d) / "golden.jsonl")    # never read — guard short-circuits
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = main(["--db", db, "eval", golden])
            self.assertEqual(rc, 2)                    # not 0 (false pass), not a crash
            msg = err.getvalue()
            self.assertIn("0 memories", msg)
            self.assertIn("--db", msg)                 # points at the fix

    def test_answer_eval_empty_store_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "empty.db")
            golden = str(Path(d) / "golden.jsonl")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = main(["--db", db, "answer-eval", golden])
            self.assertEqual(rc, 2)
            self.assertIn("0 memories", err.getvalue())


if __name__ == "__main__":
    unittest.main()
