import unittest
from datetime import datetime, timedelta

from fornixdb.consolidate import mark_done, status
from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.multistore import set_config


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


def _age(store, mem_id, days):
    """Backdate a memory's recall anchor to simulate the passage of time."""
    old = (datetime.now() - timedelta(days=days)).isoformat()
    store.conn.execute(
        "UPDATE memory SET recorded_time = ?, last_recalled = NULL, event_time = ? "
        "WHERE id = ?", (old, old, mem_id))
    store.conn.commit()


class TestEffectiveSalience(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def test_fresh_memory_undecayed(self):
        mid = self.s.store("fresh", salience=0.8)
        row = dict(self.s.conn.execute("SELECT * FROM memory WHERE id=?", (mid,)).fetchone())
        self.assertAlmostEqual(self.s.effective_salience(row), 0.8, places=2)

    def test_old_episodic_decays_to_floor(self):
        mid = self.s.store("ancient session", kind="episodic", salience=0.8)
        _age(self.s, mid, 700)
        row = dict(self.s.conn.execute("SELECT * FROM memory WHERE id=?", (mid,)).fetchone())
        self.assertAlmostEqual(self.s.effective_salience(row), 0.05, places=3)

    def test_feedback_floor_protects_rules(self):
        mid = self.s.store("owner rule never recalled", kind="feedback", salience=0.7)
        _age(self.s, mid, 2000)
        row = dict(self.s.conn.execute("SELECT * FROM memory WHERE id=?", (mid,)).fetchone())
        self.assertAlmostEqual(self.s.effective_salience(row), 0.35, places=3)

    def test_reinforcement_resets_decay(self):
        mid = self.s.store("recalled often", kind="episodic", salience=0.6)
        _age(self.s, mid, 300)
        self.s.show(mid)  # recall → last_recalled = now
        row = dict(self.s.conn.execute("SELECT * FROM memory WHERE id=?", (mid,)).fetchone())
        self.assertGreater(self.s.effective_salience(row), 0.5)

    def test_meta_override(self):
        set_config(self.s, "decay_halflife_episodic", "9000")
        mid = self.s.store("slow-decay store", kind="episodic", salience=0.8)
        _age(self.s, mid, 700)
        row = dict(self.s.conn.execute("SELECT * FROM memory WHERE id=?", (mid,)).fetchone())
        self.assertGreater(self.s.effective_salience(row), 0.7)

    def test_decay_affects_brief_ordering(self):
        stale = self.s.store("stale loud fact", salience=0.9)
        _age(self.s, stale, 600)
        fresh = self.s.store("fresh quiet fact", salience=0.6)
        b = self.s.brief()
        self.assertEqual(b["salient"][0]["id"], fresh)
        self.assertEqual(b["salient"][1]["id"], stale)  # sunk, not gone


class TestConsolidateStatus(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def test_never_consolidated_is_due(self):
        st = status(self.s)
        self.assertTrue(st["due"])
        self.assertEqual(st["reason"], "never consolidated")

    def test_done_then_not_due(self):
        mark_done(self.s)
        st = status(self.s)
        self.assertFalse(st["due"])
        self.assertEqual(st["new_sessions"], 0)

    def test_session_threshold_triggers(self):
        mark_done(self.s)
        for i in range(10):
            self.s.record_session(f"s{i}", started=datetime.now().isoformat())
        st = status(self.s)
        self.assertTrue(st["due"])
        self.assertIn("new sessions", st["reason"])

    def test_day_threshold_via_override(self):
        mark_done(self.s)
        set_config(self.s, "last_consolidated",
                   (datetime.now() - timedelta(days=8)).isoformat())
        st = status(self.s)
        self.assertTrue(st["due"])
        self.assertIn("since last", st["reason"])


if __name__ == "__main__":
    unittest.main()
