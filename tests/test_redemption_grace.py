"""A redemption that the next scan reverses is not a redemption.

The suppression scan re-derives from the host transcripts every time it runs.
Those transcripts do not change, so a memory redeemed today still shows the same
pushes and the same zero references tomorrow and re-qualifies at once — which
silently undid every deliberate "this one matters" signal the system has: an
explicit undo, a `show`, a `mark_helpful`, a rewrite. Suppression must be
re-earned from pushes that happened AFTER the redemption.
"""
import unittest

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.suppress import _spare_redeemed


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestRedemptionRecordsItsBaseline(unittest.TestCase):

    def test_redeeming_records_the_pushes_it_overruled(self):
        s = mem_store()
        mid = s.store("a memory", embedder=False)
        s.suppress_proactive({mid: (9, 0)})
        s.clear_proactive_suppression([mid], "cli_undo")
        row = s.conn.execute(
            "SELECT proactive_suppressed_at, redeemed_pushes FROM memory "
            "WHERE id = ?", (mid,)).fetchone()
        self.assertIsNone(row["proactive_suppressed_at"])
        self.assertEqual(row["redeemed_pushes"], 9)

    def test_a_never_redeemed_memory_has_no_baseline(self):
        s = mem_store()
        mid = s.store("a memory", embedder=False)
        row = s.conn.execute("SELECT redeemed_pushes FROM memory WHERE id = ?",
                             (mid,)).fetchone()
        self.assertIsNone(row["redeemed_pushes"])

    def test_clearing_an_unsuppressed_row_records_nothing(self):
        # a no-op redeem must not stamp a baseline that would shield the row
        s = mem_store()
        mid = s.store("a memory", embedder=False)
        self.assertEqual(s.clear_proactive_suppression([mid], "cli_undo"), 0)
        row = s.conn.execute("SELECT redeemed_pushes FROM memory WHERE id = ?",
                             (mid,)).fetchone()
        self.assertIsNone(row["redeemed_pushes"])


class TestSpareRedeemed(unittest.TestCase):

    def _redeemed(self, store, pushes_at_redemption):
        mid = store.store("a memory", embedder=False)
        store.suppress_proactive({mid: (pushes_at_redemption, 0)})
        store.clear_proactive_suppression([mid], "cli_undo")
        return mid

    def test_the_same_stale_evidence_no_longer_re_suppresses(self):
        s = mem_store()
        mid = self._redeemed(s, 9)
        kept, spared = _spare_redeemed(s, {mid: (9, 0)}, 8)
        self.assertEqual(kept, {})
        self.assertEqual(spared, {mid: 0})

    def test_new_pushes_short_of_the_bar_still_spare_it(self):
        s = mem_store()
        mid = self._redeemed(s, 9)
        kept, spared = _spare_redeemed(s, {mid: (14, 0)}, 8)   # 5 since
        self.assertEqual(kept, {})
        self.assertEqual(spared, {mid: 5})

    def test_suppression_is_re_earned_once_the_bar_is_cleared_again(self):
        s = mem_store()
        mid = self._redeemed(s, 9)
        kept, spared = _spare_redeemed(s, {mid: (17, 0)}, 8)   # 8 since
        self.assertEqual(kept, {mid: (17, 0)})
        self.assertEqual(spared, {})

    def test_a_never_redeemed_candidate_is_untouched(self):
        s = mem_store()
        mid = s.store("a memory", embedder=False)
        kept, spared = _spare_redeemed(s, {mid: (9, 0)}, 8)
        self.assertEqual(kept, {mid: (9, 0)})
        self.assertEqual(spared, {})

    def test_empty_candidate_set_is_handled(self):
        self.assertEqual(_spare_redeemed(mem_store(), {}, 8), ({}, {}))

    def test_a_show_redemption_also_sticks(self):
        # the redemption paths that are not explicit undos matter just as much
        s = mem_store()
        mid = s.store("a memory", embedder=False)
        s.suppress_proactive({mid: (10, 0)})
        s.clear_proactive_suppression([mid], "show")
        kept, spared = _spare_redeemed(s, {mid: (10, 0)}, 8)
        self.assertEqual(kept, {})
        self.assertIn(mid, spared)


class TestScanAndApplyHonorsIt(unittest.TestCase):

    def test_a_redeemed_row_survives_a_full_apply_pass(self):
        import json
        import tempfile
        from pathlib import Path
        from fornixdb.suppress import scan_and_apply
        s = mem_store()
        mid = s.store("a memory", embedder=False)
        s.suppress_proactive({mid: (9, 0)})
        s.clear_proactive_suppression([mid], "cli_undo")
        with tempfile.TemporaryDirectory() as tmp:
            block = ("[FornixDB · possibly-relevant past]\n"
                     f"#{mid} 2026-08-01 sem  a gist")
            lines = [json.dumps({"type": "attachment",
                                 "attachment": {"content": block,
                                                "hookEvent": "UserPromptSubmit"}})
                     for _ in range(9)]
            Path(tmp, "s.jsonl").write_text("\n".join(lines), encoding="utf-8")
            report = scan_and_apply(s, tmp, apply=True)
        self.assertIn(mid, report["spared_since_redemption"])
        row = s.conn.execute("SELECT proactive_suppressed_at FROM memory "
                             "WHERE id = ?", (mid,)).fetchone()
        self.assertIsNone(row["proactive_suppressed_at"])


if __name__ == "__main__":
    unittest.main()
