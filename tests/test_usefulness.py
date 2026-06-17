"""Per-memory usefulness feedback (§15.2 #6): the positive counterpart to the
query-conditional negative feedback. An explicit "this helped" endorsement is
durable and query-INDEPENDENT — it raises a memory's ranking everywhere (scaled
by relevance, so it never lifts an irrelevant row), reinforces it against
staleness, and surfaces in the session-start usefulness rollup."""

import unittest

from fornixdb.core import (MemoryStore, FrozenStoreError, USEFULNESS_WEIGHT,
                           HELPFUL_BUMP)
from fornixdb.db import connect
from fornixdb.multistore import set_config


def mem_store():
    # keyword-only in-memory store: deterministic, no vector auto-embedding
    return MemoryStore(conn=connect(":memory:"))


class TestMarkHelpful(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.a = self.s.store("the alpha fact about retirement budgets", name="alpha")
        self.b = self.s.store("the beta fact about retirement budgets", name="beta")

    def tearDown(self):
        self.s.close()

    def _row(self, mid):
        return dict(self.s.conn.execute(
            "SELECT * FROM memory WHERE id = ?", (mid,)).fetchone())

    def test_columns_exist_and_default_zero(self):
        r = self._row(self.a)
        self.assertEqual(r["helpful_count"], 0)
        self.assertIsNone(r["last_helpful"])

    def test_increments_and_stamps(self):
        out = self.s.mark_helpful(self.b)
        self.assertEqual(out["helpful_count"], 1)
        r = self._row(self.b)
        self.assertEqual(r["helpful_count"], 1)
        self.assertIsNotNone(r["last_helpful"])
        # each call accumulates
        self.s.mark_helpful(self.b)
        self.assertEqual(self._row(self.b)["helpful_count"], 2)

    def test_bumps_salience_and_reinforces(self):
        before = self._row(self.b)
        self.s.mark_helpful(self.b)
        after = self._row(self.b)
        self.assertAlmostEqual(after["salience"],
                               min(before["salience"] + HELPFUL_BUMP, 1.0))
        # reinforced → staleness anchor moved (a helped memory was just confirmed)
        self.assertIsNotNone(after["last_reinforced"])

    def test_by_name_and_missing(self):
        self.assertEqual(self.s.mark_helpful("alpha")["id"], self.a)
        with self.assertRaises(ValueError):
            self.s.mark_helpful("does-not-exist")

    def test_frozen_store_refuses(self):
        set_config(self.s, "frozen", "1")               # `config frozen on`
        self.s.__dict__.pop("_frozen_cache", None)      # invalidate the cache
        with self.assertRaises(FrozenStoreError):
            self.s.mark_helpful(self.a)


class TestUsefulnessRanking(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.a = self.s.store("the alpha fact about retirement budgets", name="alpha")
        self.b = self.s.store("the beta fact about retirement budgets", name="beta")

    def tearDown(self):
        self.s.close()

    def test_endorsement_lifts_a_relevant_memory(self):
        base = {m["id"]: m["score"] for m in self.s.recall("retirement budgets")}
        self.s.mark_helpful(self.b)
        self.s.mark_helpful(self.b)
        after = {m["id"]: m["score"] for m in self.s.recall("retirement budgets")}
        self.assertGreater(after[self.b], base[self.b])   # b rose
        self.assertEqual(after[self.a], base[self.a])      # a unchanged
        self.assertGreater(after[self.b], after[self.a])   # and now leads

    def test_endorsement_never_surfaces_an_irrelevant_memory(self):
        # endorse b heavily, then run a query b has nothing to do with
        for _ in range(20):
            self.s.mark_helpful(self.b)
        hits = [m["id"] for m in self.s.recall("completely unrelated zebra topic")]
        self.assertNotIn(self.b, hits)

    def test_usefulness_bonus_is_bounded(self):
        for _ in range(100):
            self.s.mark_helpful(self.b)
        row = self._fetch(self.b)
        self.assertLessEqual(self.s._usefulness(row), USEFULNESS_WEIGHT)
        self.assertGreater(self.s._usefulness(row), 0.0)

    def _fetch(self, mid):
        return dict(self.s.conn.execute(
            "SELECT * FROM memory WHERE id = ?", (mid,)).fetchone())


class TestUsefulnessRollup(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_cold_store_rollup_empty(self):
        self.s.store("untouched note", name="cold")
        self.assertEqual(self.s.top_useful(), [])

    def test_ordering_endorsed_first_then_recalled(self):
        a = self.s.store("alpha budget note", name="a")
        b = self.s.store("beta budget note", name="b")
        # b gets recalls only; a gets an endorsement
        self.s.recall("budget note")          # bumps recall_count on both
        self.s.mark_helpful(a)
        top = self.s.top_useful()
        self.assertEqual(top[0]["id"], a)     # endorsed outranks recalled-only
        self.assertTrue(any(r["id"] == b for r in top))

    def test_brief_exposes_useful_key(self):
        m = self.s.store("budget note", name="m")
        self.s.mark_helpful(m)
        b = self.s.brief()
        self.assertIn("useful", b)
        self.assertEqual(b["useful"][0]["id"], m)


if __name__ == "__main__":
    unittest.main()
