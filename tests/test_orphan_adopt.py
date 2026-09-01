"""Orphan-tombstone adoption (2026-07-25 audit finding, live case #534→#536):
the forget-then-rewrite flow tombstones a row BEFORE its successor exists, an
order in which supersede() can never record lineage. store() repairs it at the
rewrite — a near-identical row stored within the adoption window writes
superseded_by on the recent successor-less tombstone."""

import unittest
from datetime import datetime, timedelta

from fornixdb import prospective, vectors
from fornixdb.core import MemoryStore
from fornixdb.db import connect

from test_vectors import FakeEmbedder

TEXT = "Reminder: evaluate the parallel dissent trial emitted beats and referenced rate"
REWRITE = TEXT + " updated"


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestOrphanAdoption(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.emb = FakeEmbedder()

    def put(self, text, kind="episodic", **kw):
        return self.s.store(text, kind=kind, embedder=self.emb, **kw)

    def row(self, mid):
        return dict(self.s.conn.execute(
            "SELECT superseded_by, superseded_time FROM memory WHERE id = ?",
            (mid,)).fetchone())

    def test_forget_then_rewrite_adopts(self):
        old = self.put(TEXT)
        self.s.tombstone(old)
        new = self.put(REWRITE)
        r = self.row(old)
        self.assertEqual(r["superseded_by"], new)
        self.assertIsNotNone(self.s.conn.execute(
            "SELECT 1 FROM memory_link WHERE memory_id = ? AND related_id = ? "
            "AND relation = 'supersedes'", (new, old)).fetchone())

    def test_distinct_content_not_adopted(self):
        old = self.put("the greenhouse thermostat was replaced on the north bench")
        self.s.tombstone(old)
        self.put(TEXT)
        self.assertIsNone(self.row(old)["superseded_by"])

    def test_stale_tombstone_outside_window_not_adopted(self):
        old = self.put(TEXT)
        self.s.tombstone(old)
        stale = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
        self.s.conn.execute(
            "UPDATE memory SET superseded_time = ? WHERE id = ?", (stale, old))
        self.s.conn.commit()
        self.put(REWRITE)
        self.assertIsNone(self.row(old)["superseded_by"])

    def test_existing_successor_untouched(self):
        old = self.put(TEXT)
        mid = self.put(TEXT + " (second edition)")
        self.s.supersede(old, mid)
        self.put(REWRITE)
        self.assertEqual(self.row(old)["superseded_by"], mid)

    def test_kind_mismatch_not_adopted(self):
        old = self.put(TEXT, kind="semantic")
        self.s.tombstone(old)
        self.put(REWRITE, kind="episodic")
        self.assertIsNone(self.row(old)["superseded_by"])

    def test_no_vectors_noop(self):
        old = self.s.store(TEXT, kind="episodic", embedder=False)
        self.s.tombstone(old)
        self.s.store(REWRITE, kind="episodic", embedder=False)
        self.assertIsNone(self.row(old)["superseded_by"])

    def test_reminder_forget_then_recreate_flow(self):
        # the literal #534→#536 sequence, through the remind() path — which
        # embeds via AUTO resolution, so the suite-wide vectors-off env switch
        # (tests/__init__) is cleared locally per its own instructions
        import os
        os.environ["FORNIXDB_VECTORS"] = "on"
        vectors.set_default_embedder(self.emb)
        try:
            r1 = prospective.remind(self.s, "evaluate the dissent trial",
                                    "in 2 hours")
            self.s.tombstone(r1["id"])
            r2 = prospective.remind(self.s, "evaluate the dissent trial",
                                    "in 2 hours",
                                    detail="FIRST read the how-to doc")
            self.assertEqual(self.row(r1["id"])["superseded_by"], r2["id"])
        finally:
            os.environ["FORNIXDB_VECTORS"] = "off"
            vectors.set_default_embedder(None)


if __name__ == "__main__":
    unittest.main()
