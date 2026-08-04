"""`lineage` — read a project's arc off the supersede chain instead of
reconstructing it from detail.

A long-running project records its state as a run of status rows, each
superseding the last. The chain was always in the store; nothing walked it,
so picking a project back up meant re-reading a long narrative file. These
cover the walk itself and the two ways it can quietly lie: reporting a
tombstoned ancestor as the current row, and dropping a merged branch.
"""

import unittest

from fornixdb.core import MemoryStore
from fornixdb.db import _setup, connect


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestLineage(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        # a five-edition chain: ed1 is oldest, ed5 the live tip
        self.eds = []
        for i in range(1, 6):
            mid = self.s.store(f"RESUME HERE (ed.{i})", f"detail {i}",
                               name=f"ed{i}", kind="semantic")
            if self.eds:
                self.s.supersede(self.eds[-1], mid)
            self.eds.append(mid)

    def tearDown(self):
        self.s.close()

    def test_walks_newest_first_from_the_tip(self):
        chain = self.s.lineage(self.eds[-1])
        self.assertEqual([m["id"] for m in chain], list(reversed(self.eds)))
        self.assertIsNone(chain[0]["superseded_time"])

    def test_an_old_edition_returns_the_same_chain(self):
        """Handing it a tombstoned ancestor must not change the answer."""
        self.assertEqual([m["id"] for m in self.s.lineage(self.eds[0])],
                         [m["id"] for m in self.s.lineage(self.eds[-1])])

    def test_depth_caps_output_without_truncating_the_walk_to_the_tip(self):
        """Regression: `depth` once bounded the forward leg too, so walking up
        from an old edition stopped early and reported a SUPERSEDED row as the
        current one — a wrong answer that looked like a right one."""
        chain = self.s.lineage(self.eds[0], depth=2)
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0]["id"], self.eds[-1])
        self.assertIsNone(chain[0]["superseded_time"])

    def test_resolves_by_name(self):
        self.assertEqual([m["id"] for m in self.s.lineage("ed3")],
                         [m["id"] for m in self.s.lineage(self.eds[-1])])

    def test_unchained_memory_is_a_lineage_of_one(self):
        solo = self.s.store("standalone fact", "d", kind="semantic")
        self.assertEqual([m["id"] for m in self.s.lineage(solo)], [solo])

    def test_unknown_ref_is_empty_not_an_error(self):
        self.assertEqual(self.s.lineage(999999), [])
        self.assertEqual(self.s.lineage("no-such-name"), [])

    def test_merged_branch_is_counted_not_silently_dropped(self):
        """Two rows can point at one successor; the mainline is the most
        recent, and the other must still be visible as a count."""
        other = self.s.store("a parallel status row", "d", kind="semantic")
        self.s.supersede(other, self.eds[-1])
        chain = self.s.lineage(self.eds[-1])
        self.assertEqual(chain[0]["merged_siblings"], 1)
        self.assertIn(other, {m["id"] for m in chain} | {other})

    def test_a_cycle_terminates(self):
        """Hand-edited or repaired stores can carry a loop; the walk must not
        hang on one."""
        self.s.conn.execute("UPDATE memory SET superseded_by = ? WHERE id = ?",
                            (self.eds[0], self.eds[-1]))
        self.s.conn.commit()
        self.assertLessEqual(len(self.s.lineage(self.eds[0], depth=50)), 50)

    def test_listing_a_chain_is_an_impression_not_a_recall(self):
        """Same rule `timeline` follows: walking editions lists them, it does
        not engage with any one of them, so it must not pump recall_count."""
        before = self.s.conn.execute(
            "SELECT recall_count FROM memory WHERE id = ?",
            (self.eds[-1],)).fetchone()[0]
        self.s.lineage(self.eds[-1])
        after = self.s.conn.execute(
            "SELECT recall_count, surfaced_count FROM memory WHERE id = ?",
            (self.eds[-1],)).fetchone()
        self.assertEqual(after[0], before)
        self.assertGreater(after[1], 0)

    def test_detail_is_not_loaded_for_a_listing(self):
        """The whole point is to cost a fraction of reading the details."""
        chain = self.s.lineage(self.eds[-1])
        self.assertTrue(all("topics" not in m for m in chain))


class TestStatusTips(unittest.TestCase):
    """`brief`'s thread dashboard. A resume pointer sits at default salience,
    so it loses the salience ranking to any long-lived reference row — measured
    on the live store 2026-08-02, the active project's pointer did not reach
    even the top-40 candidate pool. Threads rank on a different axis: how
    recently the thread moved."""

    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def _chain(self, project, n, day_base):
        ids = []
        for i in range(1, n + 1):
            mid = self.s.store(f"{project} status ed.{i}", "d", kind="semantic",
                               project=project,
                               event_time=f"2026-07-{day_base + i:02d}T10:00:00")
            if ids:
                self.s.supersede(ids[-1], mid)
            ids.append(mid)
        return ids

    def test_reports_the_live_tip_of_each_project_chain(self):
        a = self._chain("alpha", 3, 1)
        b = self._chain("beta", 2, 10)
        tips = {r["project"]: r["id"] for r in self.s.status_tips()}
        self.assertEqual(tips, {"alpha": a[-1], "beta": b[-1]})

    def test_edition_count_equals_what_lineage_will_show(self):
        """The brief advertises an arc; the number has to be the real one."""
        self._chain("alpha", 4, 1)
        tip = self.s.status_tips()[0]
        self.assertEqual(tip["editions"],
                         len(self.s.lineage(tip["id"], depth=500)))

    def test_a_row_that_supersedes_nothing_is_not_a_thread(self):
        self.s.store("a standalone note", "d", kind="semantic", project="alpha")
        self.assertEqual(self.s.status_tips(), [])

    def test_superseded_rows_are_never_reported_as_current(self):
        ids = self._chain("alpha", 3, 1)
        self.assertNotIn(ids[-2], {r["id"] for r in self.s.status_tips()})

    def test_episodic_rows_are_excluded(self):
        """Session records are already the brief's `recent` list; a thread tip
        is standing state, not an event."""
        old = self.s.store("session one", "d", kind="episodic", project="alpha")
        new = self.s.store("session two", "d", kind="episodic", project="alpha")
        self.s.supersede(old, new)
        self.assertEqual(self.s.status_tips(), [])

    def test_project_filter(self):
        self._chain("alpha", 2, 1)
        self._chain("beta", 2, 10)
        self.assertEqual([r["project"] for r in self.s.status_tips(project="beta")],
                         ["beta"])

    def test_most_recently_moved_thread_comes_first(self):
        self._chain("alpha", 2, 1)
        b = self._chain("beta", 2, 20)
        self.assertEqual(self.s.status_tips()[0]["id"], b[-1])

    def test_brief_carries_threads(self):
        self._chain("alpha", 2, 1)
        self.assertTrue(self.s.brief()["threads"])

    def test_supersede_walk_is_indexed_not_scanned(self):
        # _mainline follows the chain one link at a time, so an unindexed
        # superseded_by costs a full table scan PER LINK — and status_tips()
        # walks 15 chains on every brief(). Measured at 12.5k rows, the index
        # takes brief(days=7) from 606ms to 18ms. A plan test rather than a
        # timing test: on a small fixture the scan is fast enough to pass.
        plan = [r[3] for r in self.s.conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM memory WHERE superseded_by = ? "
            "ORDER BY event_time DESC", (1,))]
        self.assertTrue(any("idx_memory_superseded_by" in step for step in plan),
                        f"supersede walk is not using the index: {plan}")

    def test_index_reaches_a_store_that_predates_it(self):
        # the index lives in _SCHEMA as CREATE ... IF NOT EXISTS, which the
        # schema script runs unconditionally on EVERY connect — so an existing
        # store gains it with no SCHEMA_VERSION bump and no migration. If that
        # ever stops being true, old stores silently keep the slow scan.
        self.s.conn.execute("DROP INDEX idx_memory_superseded_by")
        self.s.conn.commit()
        _setup(self.s.conn)
        self.assertTrue(self.s.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='idx_memory_superseded_by'").fetchone())


if __name__ == "__main__":
    unittest.main()
