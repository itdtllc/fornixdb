"""Lower-friction capture (§15.2 #1): jot a candidate cheaply mid-work, review
and promote at a checkpoint. Candidates are NOT memories until promoted."""

import tempfile
import unittest
from pathlib import Path

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.adapters.mcp_server import CORE_TOOLS, FornixMCP


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestCandidateCore(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_jot_stages_not_stores(self):
        self.s.jot("a fleeting thought")
        self.assertEqual(len(self.s.candidates()), 1)
        # not a memory: nothing recallable, no memory row
        self.assertEqual(self.s.conn.execute(
            "SELECT count(*) c FROM memory").fetchone()["c"], 0)

    def test_candidates_oldest_first(self):
        self.s.jot("first")
        self.s.jot("second")
        notes = [c["note"] for c in self.s.candidates()]
        self.assertEqual(notes, ["first", "second"])

    def test_discard_by_id(self):
        a = self.s.jot("keep")
        b = self.s.jot("drop")
        self.assertEqual(self.s.discard_candidates(ids=[b]), 1)
        self.assertEqual([c["id"] for c in self.s.candidates()], [a])

    def test_discard_all(self):
        self.s.jot("x")
        self.s.jot("y")
        self.assertEqual(self.s.discard_candidates(), 2)
        self.assertEqual(self.s.candidates(), [])

    def test_session_scope(self):
        self.s.jot("s1 note", session_id="s1")
        self.s.jot("s2 note", session_id="s2")
        self.assertEqual(len(self.s.candidates(session_id="s1")), 1)
        self.assertEqual(len(self.s.candidates()), 2)


class TestCandidateMCP(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.srv = FornixMCP(db_path=Path(self.tmp.name) / "m.db", shared=False)

    def tearDown(self):
        self.srv.store.close()
        self.tmp.cleanup()

    def test_jot_then_review_then_promote_then_clear(self):
        self.srv.jot(note="owner prefers dark mode")
        self.srv.jot(note="bug in export path")
        review = self.srv.review_candidates()
        self.assertIn("2 pending", review)
        # promote keepers via the existing batch tool
        self.srv.remember_many(items=[
            {"title": "dark-mode-pref", "content": "owner prefers dark mode"}])
        self.assertEqual(self.srv.store.conn.execute(
            "SELECT count(*) c FROM memory WHERE superseded_time IS NULL"
        ).fetchone()["c"], 1)
        # drop the staging area
        self.assertIn("discarded all", self.srv.review_candidates(clear=True))
        self.assertEqual(self.srv.store.candidates(), [])

    def test_empty_jot_is_handled(self):
        self.assertIn("nothing to jot", self.srv.jot(note="   "))

    def test_review_discard_subset(self):
        self.srv.jot(note="a")
        self.srv.jot(note="b")
        ids = [c["id"] for c in self.srv.store.candidates()]
        out = self.srv.review_candidates(discard=[ids[0]])
        self.assertIn("discarded 1", out)
        self.assertEqual(len(self.srv.store.candidates()), 1)

    def test_jot_and_review_are_optional_on_by_default(self):
        # consistent with every non-core tool: shipped enabled, disable-able
        self.assertNotIn("jot", CORE_TOOLS)
        self.assertNotIn("review_candidates", CORE_TOOLS)


if __name__ == "__main__":
    unittest.main()
