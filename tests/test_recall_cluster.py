"""Recall cluster: combined subject+time, staleness flags, spreading activation."""

import unittest
from datetime import datetime, timedelta

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from test_vectors import FakeEmbedder


def _backdate(store, mem_id, days):
    old = (datetime.now() - timedelta(days=days)).isoformat()
    store.conn.execute(
        "UPDATE memory SET event_time=?, recorded_time=?, last_recalled=NULL WHERE id=?",
        (old, old, mem_id))
    store.conn.commit()


class TestCombinedSubjectTime(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))
        self.old = self.s.store("fixed the cadence bug in april", "x", kind="episodic")
        self.new = self.s.store("fixed the cadence bug again in june", "x", kind="episodic")
        _backdate(self.s, self.old, 70)   # ~april
        _backdate(self.s, self.new, 5)    # ~june

    def tearDown(self):
        self.s.close()

    def _window(self, days_back_start, days_back_end=0):
        now = datetime.now()
        return ((now - timedelta(days=days_back_start)).isoformat(),
                (now - timedelta(days=days_back_end)).isoformat())

    def test_window_filters_keyword_recall(self):
        since, until = self._window(90, 30)   # the april window
        rows = self.s.recall("cadence bug", since=since, until=until, embedder=False)
        self.assertEqual([r["id"] for r in rows], [self.old])
        since, until = self._window(30)       # the june window
        rows = self.s.recall("cadence bug", since=since, until=until, embedder=False)
        self.assertEqual([r["id"] for r in rows], [self.new])

    def test_window_filters_vector_neighbors_too(self):
        from fornixdb.vectors import backfill
        backfill(self.s, FakeEmbedder())
        since, until = self._window(90, 30)
        rows = self.s.recall("cadence bug", since=since, until=until,
                             embedder=FakeEmbedder())
        self.assertEqual([r["id"] for r in rows], [self.old])

    def test_spanned_event_overlaps_window(self):
        span = self.s.store("long project span", "x", kind="episodic",
                            event_time=(datetime.now() - timedelta(days=100)).isoformat(),
                            event_time_end=(datetime.now() - timedelta(days=40)).isoformat())
        since, until = self._window(60, 30)   # span ends inside this window
        rows = self.s.recall("project span", since=since, until=until, embedder=False)
        self.assertIn(span, [r["id"] for r in rows])


class TestStaleness(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))

    def tearDown(self):
        self.s.close()

    def test_old_unreinforced_fact_flags(self):
        mid = self.s.store("the API endpoint is v2", "x", kind="semantic")
        _backdate(self.s, mid, 200)           # past the 120d semantic half-life
        rows = self.s.recall("API endpoint", embedder=False)
        self.assertGreaterEqual(rows[0]["stale_days"], 199)
        mem = self.s.show(mid, reinforce=False)
        self.assertGreaterEqual(mem["stale_days"], 199)

    def test_fresh_and_reinforced_do_not_flag(self):
        fresh = self.s.store("new fact", "x", kind="semantic")
        rows = self.s.recall("new fact", embedder=False)
        self.assertIsNone(rows[0]["stale_days"])
        old = self.s.store("old but used fact", "x", kind="semantic")
        _backdate(self.s, old, 300)
        self.s.show(old)                      # reinforcement resets the anchor
        rows = self.s.recall("old but used", embedder=False)
        self.assertIsNone(rows[0]["stale_days"])

    def test_episodic_history_never_flags(self):
        mid = self.s.store("ancient session", "x", kind="episodic")
        _backdate(self.s, mid, 400)
        rows = self.s.recall("ancient session", embedder=False)
        self.assertIsNone(rows[0]["stale_days"])

    def test_kind_thresholds_differ(self):
        ref = self.s.store("a how-to", "x", kind="reference")
        _backdate(self.s, ref, 150)           # > semantic 120, < reference 180
        rows = self.s.recall("how to", embedder=False)
        self.assertIsNone(rows[0]["stale_days"])


class TestSpreadingActivation(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))
        self.hub = self.s.store("the render pipeline design", "x")
        self.spoke = self.s.store("seam color correction fix", "x")
        self.dead = self.s.store("obsolete fix", "x")
        self.s.link(self.hub, self.spoke, "refines")
        self.s.link(self.dead, self.hub, "relates")
        self.s.tombstone(self.dead)

    def tearDown(self):
        self.s.close()

    def test_neighbors_attached_both_directions(self):
        rows = self.s.recall("render pipeline", related=True, embedder=False)
        hub = next(r for r in rows if r["id"] == self.hub)
        rel = {(n["relation"], n["id"]) for n in hub["related"]}
        self.assertIn(("refines", self.spoke), rel)
        # incoming direction is visible from the spoke's side
        rows = self.s.recall("seam color correction", related=True, embedder=False)
        spoke = next(r for r in rows if r["id"] == self.spoke)
        self.assertIn(("refines-by", self.hub),
                      {(n["relation"], n["id"]) for n in spoke["related"]})

    def test_tombstoned_neighbors_excluded_and_capped(self):
        rows = self.s.recall("render pipeline", related=True, embedder=False)
        hub = next(r for r in rows if r["id"] == self.hub)
        self.assertNotIn(self.dead, [n["id"] for n in hub["related"]])
        for i in range(6):
            extra = self.s.store(f"extra link {i}", "x")
            self.s.link(self.hub, extra)
        rows = self.s.recall("render pipeline", related=True, embedder=False)
        hub = next(r for r in rows if r["id"] == self.hub)
        self.assertLessEqual(len(hub["related"]), 3)

    def test_off_by_default(self):
        rows = self.s.recall("render pipeline", embedder=False)
        self.assertNotIn("related", rows[0])


if __name__ == "__main__":
    unittest.main()
