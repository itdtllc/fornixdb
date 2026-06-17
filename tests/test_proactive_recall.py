import os
import tempfile
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic: keyword recall, no model

from fornixdb.adapters.claude_code_recall import (HEADER, _format_block,
                                                  main, proactive_recall,
                                                  relevant_memories)
from fornixdb.adapters.native_memory import set_ingest_mode
from fornixdb.core import MemoryStore
from fornixdb.multistore import set_config


def file_store(tmp):
    return MemoryStore(db_path=Path(tmp) / "t.db")


class TestProactiveRecall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    PROMPT = "how does the deploy script read its configuration"

    def _seed(self, gist="the deploy script reads configuration from the env",
              **kw):
        return self.s.store(gist, gist, kind=kw.pop("kind", "semantic"), **kw)

    # --------------------------------------------------------- relevance gate

    def test_silence_when_nothing_relevant(self):
        # zero token overlap with PROMPT: keyword recall (vectors off in tests)
        # finds no anchor, so a relevant-floor miss is the same as silence
        self._seed("monte carlo simulation draws thousands of random samples")
        self.assertIsNone(proactive_recall(self.s, self.PROMPT, session_id="s1"))

    def test_block_surfaced_for_a_relevant_prompt(self):
        mid = self._seed()
        block = proactive_recall(self.s, self.PROMPT, session_id="s1")
        self.assertIsNotNone(block)
        self.assertIn(HEADER, block)
        self.assertIn(f"#{mid}", block)

    def test_floor_filters_low_cosine_rows(self):
        # stub recall to exercise the vector floor deterministically
        self.s.recall = lambda *a, **k: [
            {"id": 1, "kind": "semantic", "gist": "strong", "vec_cos": 0.55},
            {"id": 2, "kind": "semantic", "gist": "weak", "vec_cos": 0.10}]
        out = relevant_memories(self.s, "x", floor=0.30)
        self.assertEqual([r["id"] for r in out], [1])

    def test_keyword_anchor_rows_pass_without_vectors(self):
        # no vec_cos (keyword-only recall) is a literal token anchor — trusted
        self.s.recall = lambda *a, **k: [
            {"id": 9, "kind": "semantic", "gist": "anchor", "vec_cos": None}]
        self.assertEqual([r["id"] for r in relevant_memories(self.s, "x")], [9])

    # ------------------------------------------------------------ budget / form

    def test_limit_caps_injected_rows(self):
        for i in range(5):
            self._seed(f"the deploy script reads configuration variant {i}")
        block = proactive_recall(self.s, self.PROMPT, session_id="s1", limit=3)
        # header + at most 3 memory lines
        self.assertLessEqual(len(block.splitlines()), 4)

    def test_provenance_flag_on_auto_captured(self):
        self._seed("the deploy script configuration session notes",
                   kind="episodic", source="claude-code-transcript")
        block = proactive_recall(self.s, self.PROMPT, session_id="s1")
        self.assertIn("[auto-captured]", block)

    def test_format_block_drops_trailing_lines_over_budget(self):
        rows = [{"id": i, "kind": "semantic", "gist": "g" * 100} for i in range(5)]
        block = _format_block(rows, max_chars=len(HEADER) + 130)
        self.assertLessEqual(len(block), len(HEADER) + 130)
        self.assertIn(HEADER, block)

    # --------------------------------------------------------------- switches

    def test_trivial_prompt_skipped(self):
        self._seed()
        self.assertIsNone(proactive_recall(self.s, "ok", session_id="s1"))

    def test_explicit_mode_disables_injection(self):
        self._seed()
        set_ingest_mode(self.s, "explicit")
        self.assertIsNone(proactive_recall(self.s, self.PROMPT, session_id="s1"))

    def test_config_switch_disables_injection(self):
        self._seed()
        set_config(self.s, "proactive_recall", "off")
        self.assertIsNone(proactive_recall(self.s, self.PROMPT, session_id="s1"))

    # ----------------------------------------------------- cross-turn dedup

    def test_same_memory_not_reinjected_within_session(self):
        self._seed()
        first = proactive_recall(self.s, self.PROMPT, session_id="s1")
        self.assertIsNotNone(first)
        # same prompt, same session: the only hit was already injected -> silence
        second = proactive_recall(self.s, self.PROMPT, session_id="s1")
        self.assertIsNone(second)
        # a DIFFERENT session starts fresh
        self.assertIsNotNone(proactive_recall(self.s, self.PROMPT, session_id="s2"))

    # ------------------------------------------------------------------ main

    def test_main_reads_stdin_and_prints_block(self):
        import io
        import json
        from contextlib import redirect_stdout
        mid = self._seed()
        payload = json.dumps({"prompt": self.PROMPT, "session_id": "s1"})
        buf = io.StringIO()
        # main opens its own store from --db; point it at ours
        argv = ["--db", str(Path(self.tmp.name) / "t.db")]
        import sys
        old_stdin = sys.stdin
        sys.stdin = io.StringIO(payload)
        try:
            with redirect_stdout(buf):
                rc = main(argv)
        finally:
            sys.stdin = old_stdin
        self.assertEqual(rc, 0)
        self.assertIn(f"#{mid}", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
