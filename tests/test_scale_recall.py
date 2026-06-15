"""Bulk-load + recall-at-scale smoke test. Loads a few hundred filler memories
with distinctive needles buried in them and asserts recall still finds the
needles (keyword-only, so it needs no embedder in CI). The heavy, vector-backed
version is examples/scale_recall_test.py — this just guards the path."""

import os
import tempfile
import unittest
from pathlib import Path

from fornixdb.core import MemoryStore

NEEDLES = [
    ("The Zephyr-9 turbopump primes to 4.7 bar before ignition.",
     "Zephyr-9 turbopump prime pressure"),
    ("The Quokka build agent runs on rack B17 slot 4.",
     "where does the Quokka build agent run"),
    ("The Verdigris secret lives in vault path kv/teams/atlas/verdigris.",
     "Verdigris secret vault path"),
    ("Falcon-tier accounts cap at 1450 webhooks per minute.",
     "Falcon-tier webhook cap"),
    ("Sensor array T-12 reports in kelvin.",
     "what unit does sensor array T-12 report in"),
]


class TestScaleRecall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "scale.db")
        self.s = MemoryStore(db_path=self.db)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_needles_found_in_a_few_hundred_filler(self):
        needle_ids = []
        n_filler = 250
        step = n_filler // len(NEEDLES)
        ni = 0
        for i in range(n_filler):
            if ni < len(NEEDLES) and i == ni * step:
                needle_ids.append(self.s.store(NEEDLES[ni][0], kind="semantic"))
                ni += 1
            self.s.store(f"Routine note number {i}: a service was updated and "
                         f"metrics returned to baseline.", kind="semantic")

        self.assertGreaterEqual(self.s.stats()["memories"], n_filler)

        hitk = 0
        for nid, (_fact, query) in zip(needle_ids, NEEDLES):
            rows = self.s.recall(query, limit=5, embedder=False)  # keyword-only
            if nid in [r["id"] for r in rows]:
                hitk += 1
        # distinctive needles must be findable among the filler
        self.assertEqual(hitk, len(NEEDLES))

    def test_footprint_is_measurable(self):
        for i in range(120):
            self.s.store(f"note {i} about a routine deploy", kind="semantic")
        self.s.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size = os.path.getsize(self.db)
        per = size / self.s.stats()["memories"]
        self.assertGreater(per, 0)          # bytes/memory is a real, positive number
        self.assertLess(per, 50_000)        # sanity: not absurdly large


if __name__ == "__main__":
    unittest.main()
