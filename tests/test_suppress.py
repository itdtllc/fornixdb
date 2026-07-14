"""Per-memory proactive-push suppression (schema v13).

Covers the rule, the store mechanics + redemption, the shared push filter across
L3/L4/L5, the recall/show/timeline invariant, the self-correcting scan, and
migration idempotence.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"   # deterministic keyword recall, no model

from fornixdb import suppress
from fornixdb.core import MemoryStore
from fornixdb.multistore import set_config
from fornixdb.proactive import push_suppressed, relevant_memories


def _store(tmp):
    return MemoryStore(db_path=Path(tmp) / "t.db")


class TestRule(unittest.TestCase):
    def _scan(self, per_memory):
        return {"per_memory": {i: {"impressions": p, "referenced": r}
                               for i, (p, r) in per_memory.items()}}

    def test_chronic_never_referenced_suppresses(self):
        to_sup, earned = suppress.classify(
            self._scan({1: (8, 0), 2: (20, 0)}), 8, 0)
        self.assertEqual(to_sup, {1: (8, 0), 2: (20, 0)})
        self.assertEqual(earned, set())

    def test_below_push_threshold_is_spared(self):
        to_sup, _ = suppress.classify(self._scan({1: (7, 0), 2: (3, 0)}), 8, 0)
        self.assertEqual(to_sup, {})

    def test_referenced_row_is_never_suppressed_and_is_redemption_candidate(self):
        to_sup, earned = suppress.classify(
            self._scan({1: (20, 1), 2: (34, 17)}), 8, 0)
        self.assertEqual(to_sup, {})
        self.assertEqual(earned, {1, 2})

    def test_thresholds_are_config_overridable(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            self.assertEqual(suppress.thresholds(s), (8, 0))
            set_config(s, "suppress_min_pushed", "12")
            set_config(s, "suppress_max_referenced", "2")
            self.assertEqual(suppress.thresholds(s), (12, 2))
            # with max_referenced=2, a row referenced twice still suppresses
            to_sup, earned = suppress.classify(
                self._scan({1: (12, 2), 2: (12, 3)}), 12, 2)
            self.assertEqual(to_sup, {1: (12, 2)})
            self.assertEqual(earned, {2})


class TestStoreMechanics(unittest.TestCase):
    def test_suppress_stamps_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            mid = s.store("chronic never-referenced note", kind="semantic")
            self.assertEqual(s.suppress_proactive({mid: (9, 0)}), 1)
            row = s.show(mid, reinforce=False)
            # NOTE: show(reinforce=False) does not redeem
            self.assertIsNotNone(row["proactive_suppressed_at"])
            self.assertEqual(row["suppressed_pushed"], 9)
            # re-running does not re-stamp (already suppressed)
            self.assertEqual(s.suppress_proactive({mid: (9, 0)}), 0)

    def test_suppress_skips_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            self.assertEqual(s.suppress_proactive({99999: (9, 0)}), 0)

    def test_list_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            a = s.store("alpha noisy", kind="semantic")
            b = s.store("beta noisy", kind="semantic")
            s.suppress_proactive({a: (10, 0), b: (8, 0)})
            listed = {r["id"] for r in s.proactive_suppressed()}
            self.assertEqual(listed, {a, b})
            self.assertEqual(s.clear_proactive_suppression([a], "test"), 1)
            self.assertEqual({r["id"] for r in s.proactive_suppressed()}, {b})
            # clearing a non-suppressed id is a no-op (0), not an error
            self.assertEqual(s.clear_proactive_suppression([a], "test"), 0)

    def test_predicate_respects_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            mid = s.store("noisy", kind="semantic")
            s.suppress_proactive({mid: (9, 0)})
            row = dict(s.conn.execute(
                "SELECT * FROM memory WHERE id = ?", (mid,)).fetchone())
            self.assertTrue(push_suppressed(s, row))
            set_config(s, "proactive_suppression", "off")
            self.assertFalse(push_suppressed(s, row))   # feature off -> no-op

    def test_unstamped_row_is_never_suppressed(self):
        self.assertFalse(push_suppressed(None, {"gist": "x"}))


class TestRedemption(unittest.TestCase):
    def _suppressed(self, s):
        mid = s.store("chronic pushed note about ledgers", kind="semantic")
        s.suppress_proactive({mid: (10, 0)})
        return mid

    def test_show_redeems(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            mid = self._suppressed(s)
            s.show(mid)   # reinforce=True by default
            self.assertEqual(s.proactive_suppressed(), [])

    def test_mark_helpful_redeems(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            mid = self._suppressed(s)
            s.mark_helpful(mid)
            self.assertEqual(s.proactive_suppressed(), [])

    def test_set_gist_redeems(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            mid = self._suppressed(s)
            s.set_gist(mid, "a materially rewritten gist")
            self.assertEqual(s.proactive_suppressed(), [])

    def test_supersede_clears_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            old = self._suppressed(s)
            new = s.store("the corrected fact", kind="semantic")
            s.supersede(old, new)
            self.assertEqual(s.proactive_suppressed(), [])

    def test_plain_recall_does_NOT_redeem(self):
        # a suppressed row appearing as a low-ranked recall hit must not clear
        # suppression (else it never sticks) — only reinforce=True does
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            mid = self._suppressed(s)
            hits = s.recall("ledgers")            # count_recall=True, reinforce=False
            self.assertIn(mid, {r["id"] for r in hits})   # still RETURNED
            self.assertEqual(len(s.proactive_suppressed()), 1)  # still suppressed


class TestPushFilterAndInvariant(unittest.TestCase):
    def test_suppressed_excluded_from_L3_but_recall_show_still_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            keep = s.store("quarterly widget revenue forecast", kind="semantic")
            mute = s.store("quarterly widget revenue appendix", kind="semantic")
            s.suppress_proactive({mute: (12, 0)})
            rows = relevant_memories(s, "quarterly widget revenue", channel="L3")
            ids = {r["id"] for r in rows}
            self.assertIn(keep, ids)
            self.assertNotIn(mute, ids)              # muted from the push stream
            # INVARIANT: explicit recall + show still return the suppressed row
            self.assertIn(mute, {r["id"] for r in s.recall("appendix")})
            self.assertIsNotNone(s.show(mute, reinforce=False))

    def test_feature_off_pushes_suppressed_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            mute = s.store("orphaned turbine maintenance log", kind="semantic")
            s.suppress_proactive({mute: (12, 0)})
            set_config(s, "proactive_suppression", "off")
            rows = relevant_memories(s, "turbine maintenance log", channel="L3")
            self.assertIn(mute, {r["id"] for r in rows})

    def test_suppressed_excluded_from_L5_field(self):
        from fornixdb.field import run_field
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            mute = s.store("solar inverter derating curve note", kind="semantic")
            s.suppress_proactive({mute: (12, 0)})
            fr = run_field(s, "solar inverter derating curve")
            self.assertNotIn(mute, set(fr.rows))


class TestScanApplyAndSelfCorrect(unittest.TestCase):
    def _transcript(self, path, blocks):
        """blocks: list of ('push', id, channel) | ('cite', id) -> one JSONL file."""
        lines = []
        for b in blocks:
            if b[0] == "push":
                lines.append(json.dumps({"type": "attachment", "attachment": {
                    "content": f"[FornixDB possibly-relevant past]\n#{b[1]} note",
                    "hookEvent": b[2]}}))
            else:
                lines.append(json.dumps({"type": "assistant",
                                         "message": {"content": f"per #{b[1]} we ..."}}))
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def test_scan_applies_and_then_redeems_when_referenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            noisy = s.store("never-referenced chronic push", kind="semantic")
            useful = s.store("proven useful chronic push", kind="semantic")
            tdir = Path(tmp) / "transcripts"
            tdir.mkdir()
            # noisy: pushed 9x, never cited. useful: pushed 9x, never cited (yet).
            blocks = []
            for _ in range(9):
                blocks += [("push", noisy, "UserPromptSubmit"),
                           ("push", useful, "UserPromptSubmit")]
            self._transcript(tdir / "s1.jsonl", blocks)
            rep = suppress.scan_and_apply(s, str(tdir), apply=True)
            self.assertEqual(rep["applied"]["newly_suppressed"], 2)
            self.assertEqual({r["id"] for r in s.proactive_suppressed()},
                             {noisy, useful})
            # now `useful` earns a citation in a later session -> next scan redeems it
            blocks2 = []
            for _ in range(9):
                blocks2 += [("push", noisy, "UserPromptSubmit"),
                            ("push", useful, "UserPromptSubmit")]
            blocks2.append(("cite", useful))
            self._transcript(tdir / "s2.jsonl", blocks2)
            rep2 = suppress.scan_and_apply(s, str(tdir), apply=True)
            self.assertEqual(rep2["applied"]["redeemed"], 1)
            self.assertEqual(rep2["applied"]["redeemed_ids"], [useful])
            self.assertEqual({r["id"] for r in s.proactive_suppressed()}, {noisy})

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            noisy = s.store("dry-run chronic push", kind="semantic")
            tdir = Path(tmp) / "t"
            tdir.mkdir()
            self._transcript(tdir / "s.jsonl",
                             [("push", noisy, "UserPromptSubmit")] * 9)
            rep = suppress.scan_and_apply(s, str(tdir), apply=False)
            self.assertEqual(rep["candidate_count"], 1)
            self.assertNotIn("applied", rep)
            self.assertEqual(s.proactive_suppressed(), [])


class TestMigrationIdempotence(unittest.TestCase):
    def test_reopen_keeps_columns_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.db"
            s = MemoryStore(db_path=p)
            mid = s.store("x", kind="semantic")
            s.suppress_proactive({mid: (9, 0)})
            s.conn.close()
            s2 = MemoryStore(db_path=p)       # re-run _migrate + schema script
            cols = [r[1] for r in s2.conn.execute("PRAGMA table_info(memory)")]
            for c in ("proactive_suppressed_at", "suppressed_pushed",
                      "suppressed_referenced"):
                self.assertIn(c, cols)
            self.assertEqual(len(s2.proactive_suppressed()), 1)   # data survived


if __name__ == "__main__":
    unittest.main()
