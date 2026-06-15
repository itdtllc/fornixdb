import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from fornixdb.core import MemoryStore
from fornixdb.multistore import set_config
from fornixdb.tiers import load_detail, tier_down, tier_status


def file_store(tmp):
    return MemoryStore(db_path=Path(tmp) / "t.db")


def _age(store, mem_id, days):
    old = (datetime.now() - timedelta(days=days)).isoformat()
    store.conn.execute(
        "UPDATE memory SET recorded_time=?, last_recalled=NULL, event_time=? WHERE id=?",
        (old, old, mem_id))
    store.conn.commit()


class TestTiers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)

    def tearDown(self):
        self.s.close()  # Windows can't delete an open db file
        self.tmp.cleanup()

    def test_consolidate_and_restore(self):
        mid = self.s.store("old session", "the full detail text", kind="episodic",
                           salience=0.3)
        _age(self.s, mid, 90)
        moved = tier_down(self.s)
        self.assertEqual(moved["consolidated"], 1)
        raw = self.s.conn.execute("SELECT detail, retention_tier FROM memory WHERE id=?",
                                  (mid,)).fetchone()
        self.assertIsNone(raw["detail"])
        self.assertEqual(raw["retention_tier"], "consolidated")
        mem = self.s.show(mid, reinforce=False)  # transparent restore
        self.assertEqual(mem["detail"], "the full detail text")

    def test_cold_and_restore(self):
        mid = self.s.store("ancient session", "cold detail body", kind="episodic",
                           salience=0.2)
        _age(self.s, mid, 400)
        moved = tier_down(self.s)
        self.assertEqual(moved["cold"], 1)
        mem = self.s.show(mid, reinforce=False)
        self.assertEqual(mem["detail"], "cold detail body")
        self.assertEqual(mem["retention_tier"], "cold")
        arcs = list((Path(self.tmp.name) / "t.archive").glob("*.jsonl.gz"))
        self.assertEqual(len(arcs), 1)

    def test_cold_archives_isolated_per_store_in_shared_dir(self):
        # Two AIs' stores in one dir (e.g. ~/.fornixdb/memory.db + artist.db)
        # must not share an archive file: memory_id is per-store, so a shared
        # file would let one store restore another's detail.
        a = MemoryStore(db_path=Path(self.tmp.name) / "memory.db")
        b = MemoryStore(db_path=Path(self.tmp.name) / "artist.db")
        try:
            ida = a.store("alpha", "ALPHA detail", kind="episodic", salience=0.2)
            idb = b.store("beta", "BETA detail", kind="episodic", salience=0.2)
            self.assertEqual(ida, idb)  # both fresh → same autoincrement id
            for s, mid in ((a, ida), (b, idb)):
                _age(s, mid, 400)
                self.assertEqual(tier_down(s)["cold"], 1)
            # each restores its OWN detail despite the shared directory
            self.assertEqual(a.show(ida, reinforce=False)["detail"], "ALPHA detail")
            self.assertEqual(b.show(idb, reinforce=False)["detail"], "BETA detail")
            self.assertEqual(
                sorted(p.name for p in Path(self.tmp.name).glob("*.archive")),
                ["artist.archive", "memory.archive"])
        finally:
            a.close()
            b.close()

    def test_fresh_and_feedback_untouched(self):
        fresh = self.s.store("fresh session", "detail", kind="episodic")
        rule = self.s.store("owner rule", "detail", kind="feedback", salience=0.4)
        _age(self.s, rule, 900)
        moved = tier_down(self.s)
        self.assertEqual(moved, {"consolidated": 0, "cold": 0, "pressure": False})
        self.assertEqual(tier_status(self.s).get("hot"), 2)
        self.assertIsNotNone(self.s.show(fresh, reinforce=False)["detail"])

    def test_dry_run_moves_nothing(self):
        mid = self.s.store("old", "d", kind="episodic", salience=0.2)
        _age(self.s, mid, 400)
        moved = tier_down(self.s, dry_run=True)
        self.assertEqual(moved["cold"], 1)
        self.assertEqual(tier_status(self.s), {"hot": 1})

    def test_pressure_includes_semantic(self):
        mid = self.s.store("stale fact", "detail " * 50, salience=0.3)  # semantic
        _age(self.s, mid, 200)
        moved = tier_down(self.s)
        self.assertEqual(moved["consolidated"], 0)  # no pressure → untouched
        set_config(self.s, "max_store_mb", "0.000001")  # force pressure
        moved = tier_down(self.s)
        self.assertTrue(moved["pressure"])
        self.assertGreaterEqual(moved["consolidated"] + moved["cold"], 1)


if __name__ == "__main__":
    unittest.main()
