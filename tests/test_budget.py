"""Disk budget + boundary policy (Design §13.2) and the standalone freeze."""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from fornixdb.budget import (_vacuum, enforce, footprint_bytes,
                             prune_candidates, shrink, status as budget_status)
from fornixdb.core import (DiskBudgetExceededError, FrozenStoreError,
                           MemoryStore)
from fornixdb.multistore import set_config
from fornixdb.tiers import tier_down


def file_store(tmp):
    return MemoryStore(db_path=Path(tmp) / "b.db")


def _age(store, mem_id, days):
    old = (datetime.now() - timedelta(days=days)).isoformat()
    store.conn.execute(
        "UPDATE memory SET recorded_time=?, last_recalled=NULL, event_time=? WHERE id=?",
        (old, old, mem_id))
    store.conn.commit()


def _fill(store, n=120, kind="episodic", salience=0.3, prefix="filler"):
    ids = []
    for i in range(n):
        ids.append(store.store(f"{prefix} session {i}", "x" * 4000,
                               kind=kind, salience=salience))
    return ids


class TestFrozen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)
        self.mid = self.s.store("kept fact", "detail body", kind="semantic")
        self.other = self.s.store("second fact", "more detail")
        set_config(self.s, "frozen", "on")

    def tearDown(self):
        self.s.close()  # Windows can't delete an open db file
        self.tmp.cleanup()

    def test_all_content_mutation_blocked(self):
        for call in (
            lambda: self.s.store("new", "x"),
            lambda: self.s.tag(self.mid, "topic"),
            lambda: self.s.link(self.mid, self.other),
            lambda: self.s.supersede(self.mid, self.other),
            lambda: self.s.tombstone(self.mid),
            lambda: self.s.set_name(self.mid, "slug"),
            lambda: self.s.set_gist(self.mid, "rewritten"),
            lambda: self.s.record_session("sid"),
        ):
            with self.assertRaises(FrozenStoreError):
                call()

    def test_recall_works_without_reinforcement(self):
        rows = self.s.recall("kept fact", embedder=False)
        self.assertTrue(any(r["id"] == self.mid for r in rows))
        mem = self.s.show(self.mid)  # reinforce=True must be a silent no-op
        self.assertEqual(mem["detail"], "detail body")
        raw = self.s.conn.execute(
            "SELECT recall_count, last_recalled FROM memory WHERE id=?",
            (self.mid,)).fetchone()
        self.assertEqual(raw["recall_count"], 0)
        self.assertIsNone(raw["last_recalled"])

    def test_unfreeze_restores_writes(self):
        set_config(self.s, "frozen", "off")
        self.assertFalse(self.s.frozen())
        self.s.store("works again", "x")
        self.assertEqual(self.s.conn.execute(
            "SELECT recall_count FROM memory WHERE id=?", (self.mid,)
        ).fetchone()["recall_count"], 0)
        self.s.show(self.mid)  # reinforcement is back too
        self.assertEqual(self.s.conn.execute(
            "SELECT recall_count FROM memory WHERE id=?", (self.mid,)
        ).fetchone()["recall_count"], 1)


class TestBudget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)

    def tearDown(self):
        self.s.close()  # Windows can't delete an open db file
        self.tmp.cleanup()

    def test_no_budget_is_never_delete(self):
        _fill(self.s, n=20)
        before = self.s.stats()["memories"]
        result = enforce(self.s)
        self.assertFalse(result["over_after"])
        self.assertIsNone(result["tiered"])
        self.assertEqual(self.s.stats()["memories"], before)

    def test_footprint_counts_db_wal_archive(self):
        mid = self.s.store("old", "cold detail", kind="episodic", salience=0.2)
        _age(self.s, mid, 400)
        tier_down(self.s)  # creates an archive file
        fp = footprint_bytes(self.s)
        self.assertGreater(fp["db"], 0)
        self.assertGreater(fp["archive"], 0)
        self.assertEqual(fp["total"], fp["db"] + fp["wal"] + fp["archive"])

    def test_freeze_policy_refuses_at_cap(self):
        _fill(self.s, n=120)
        set_config(self.s, "disk_budget_mb", "0.05")  # far below the floor
        with self.assertRaises(DiskBudgetExceededError):
            self.s.store("one more", "x")
        # everything already stored stays recallable
        self.assertTrue(self.s.recall("filler session", embedder=False))
        # raising the budget unblocks
        set_config(self.s, "disk_budget_mb", "100")
        self.s.store("fits now", "x")

    def test_prune_policy_makes_room(self):
        _fill(self.s, n=150)
        before = self.s.stats()["memories"]
        _vacuum(self.s)  # budget below the REAL data size, not WAL bloat
        fp = footprint_bytes(self.s)["total"]
        budget_mb = (fp / 2) / 1e6
        set_config(self.s, "disk_budget_mb", f"{budget_mb:.3f}")
        set_config(self.s, "budget_policy", "prune")
        new_id = self.s.store("the newcomer", "y")  # must be accepted
        self.assertIsNotNone(self.s.show(new_id, reinforce=False))
        self.assertLess(self.s.stats()["memories"], before)
        self.assertLessEqual(footprint_bytes(self.s)["total"], budget_mb * 1e6)
        # pruned rows are really gone
        gone = self.s.conn.execute(
            "SELECT count(*) c FROM memory WHERE gist LIKE 'filler%'"
        ).fetchone()["c"]
        self.assertLess(gone, 150)

    def test_prune_holds_cap_under_sustained_logging(self):
        # the owner's actual worry: a lot of data logged over time. A prune cap
        # must hold the footprint across MANY writes (not just accept one
        # newcomer), and load-bearing memories must survive while filler is shed.
        set_config(self.s, "disk_budget_mb", "0.4")
        set_config(self.s, "budget_policy", "prune")
        needle = self.s.store("owner rule: keep this", "load-bearing detail",
                              kind="feedback")          # feedback is pruned last
        for i in range(250):
            self.s.store(f"log line {i}", "x" * 3000, kind="episodic")
        self.assertFalse(budget_status(self.s)["over_budget"])   # cap held throughout
        self.assertLessEqual(footprint_bytes(self.s)["total"], 0.4 * 1e6)
        # the high-value memory survived the forgetting; filler did not all fit
        self.assertIsNotNone(self.s.show(needle, reinforce=False))
        self.assertTrue(self.s.recall("owner rule keep this", embedder=False))
        self.assertLess(self.s.stats()["memories"], 251)

    def test_prune_order_tombstoned_first_feedback_last(self):
        dead = self.s.store("tombstoned junk", "x", kind="episodic", salience=0.9)
        self.s.tombstone(dead)
        epi = self.s.store("old session", "x", kind="episodic", salience=0.2)
        sem = self.s.store("a fact", "x", kind="semantic", salience=0.2)
        fee = self.s.store("owner rule", "x", kind="feedback", salience=0.1)
        order = [c["id"] for c in prune_candidates(self.s)]
        self.assertEqual(order[0], dead)          # tombstoned before any live row
        self.assertEqual(order[-1], fee)          # feedback last of all
        self.assertLess(order.index(epi), order.index(sem))

    def test_prune_survivors_feedback(self):
        rule = self.s.store("owner rule", "load-bearing", kind="feedback")
        _fill(self.s, n=150)
        fp = footprint_bytes(self.s)["total"]
        set_config(self.s, "disk_budget_mb", f"{(fp / 2) / 1e6:.3f}")
        set_config(self.s, "budget_policy", "prune")
        enforce(self.s)
        self.assertIsNotNone(self.s.show(rule, reinforce=False))

    def test_prune_compacts_cold_archive(self):
        ids = _fill(self.s, n=120, salience=0.05)
        for mid in ids[:40]:
            _age(self.s, mid, 400)
        tier_down(self.s)  # 40 rows go cold → archive file exists
        arcs = list((Path(self.tmp.name) / "b.archive").glob("*.jsonl.gz"))
        self.assertTrue(arcs)
        fp = footprint_bytes(self.s)["total"]
        set_config(self.s, "disk_budget_mb", f"{(fp / 3) / 1e6:.3f}")
        set_config(self.s, "budget_policy", "prune")
        enforce(self.s)
        # archived entries of pruned rows were dropped (file shrunk or gone)
        for arc in arcs:
            if arc.exists():
                import gzip
                import json
                with gzip.open(arc, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        mid = json.loads(line)["memory_id"]
                        self.assertIsNotNone(self.s.conn.execute(
                            "SELECT 1 FROM memory WHERE id=?", (mid,)).fetchone())

    def test_enforce_dry_run_changes_nothing(self):
        _fill(self.s, n=120)
        before = self.s.stats()["memories"]
        fp = footprint_bytes(self.s)["total"]
        set_config(self.s, "disk_budget_mb", f"{(fp / 2) / 1e6:.3f}")
        set_config(self.s, "budget_policy", "prune")
        result = enforce(self.s, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertGreater(result["pruned"]["candidates"], 0)
        self.assertEqual(self.s.stats()["memories"], before)

    def test_status_reports(self):
        st = budget_status(self.s)
        self.assertIsNone(st["budget_mb"])
        self.assertEqual(st["policy"], "freeze")  # safe default
        self.assertFalse(st["frozen"])
        self.assertFalse(st["over_budget"])
        set_config(self.s, "disk_budget_mb", "0.001")
        st = budget_status(self.s)
        self.assertTrue(st["over_budget"])
        set_config(self.s, "disk_budget_mb", "off")
        self.assertIsNone(budget_status(self.s)["budget_mb"])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            set_config(self.s, "budget_policy", "delete-everything")
        with self.assertRaises(ValueError):
            set_config(self.s, "disk_budget_mb", "-5")
        with self.assertRaises(ValueError):
            set_config(self.s, "disk_budget_mb", "lots")


class TestShrink(unittest.TestCase):
    """One-shot shrink-to-target (FornixDB #164): 'reduce this space to X' —
    true deletion to a named size, standing cap and policy untouched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)
        self.keep = self.s.store("the load-bearing fact", "important detail",
                                 kind="feedback", salience=0.9)
        _fill(self.s, n=120)
        _vacuum(self.s)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_shrinks_to_target_without_touching_cap(self):
        before = footprint_bytes(self.s)["total"]
        target_mb = (before / 2) / 1e6
        result = shrink(self.s, target_mb)
        self.assertTrue(result["reached"])
        self.assertLessEqual(footprint_bytes(self.s)["total"], target_mb * 1e6)
        self.assertGreater(result["pruned"]["deleted"], 0)
        # standing cap and policy were never set and stay unset
        st = budget_status(self.s)
        self.assertIsNone(st["budget_mb"])
        self.assertEqual(st["policy"], "freeze")
        # the policy default (freeze) did NOT block the shrink: the command
        # itself is the consent. Feedback is forgotten last — it survived.
        self.assertIsNotNone(self.s.show(self.keep, reinforce=False))

    def test_already_under_target_is_a_noop(self):
        before = self.s.stats()["memories"]
        result = shrink(self.s, 10_000)
        self.assertTrue(result["reached"])
        self.assertIsNone(result["pruned"])
        self.assertEqual(self.s.stats()["memories"], before)

    def test_dry_run_deletes_nothing(self):
        before = self.s.stats()["memories"]
        fp = footprint_bytes(self.s)["total"]
        result = shrink(self.s, (fp / 2) / 1e6, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertGreater(result["pruned"]["candidates"], 0)
        self.assertEqual(self.s.stats()["memories"], before)

    def test_unreachable_target_reports_honestly(self):
        # even an empty db file has a size floor — 0.001 MB is unreachable
        result = shrink(self.s, 0.001)
        self.assertFalse(result["reached"])
        self.assertGreater(result["after_mb"], 0.001)

    def test_frozen_store_refuses(self):
        set_config(self.s, "frozen", "on")
        with self.assertRaises(FrozenStoreError):
            shrink(self.s, 0.1)

    def test_invalid_target(self):
        for bad in (0, -5):
            with self.assertRaises(ValueError):
                shrink(self.s, bad)


class TestMachineUsage(unittest.TestCase):
    """Machine-wide usage: every store on the box, per-AI + total, via the
    registry each store joins on open."""

    def setUp(self):
        import os
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = Path(self.tmp.name) / "reg.json"
        self._env = {k: os.environ.get(k)
                     for k in ("FORNIXDB_REGISTRY", "FORNIXDB_SHARED_DB")}
        os.environ["FORNIXDB_REGISTRY"] = str(self.reg)
        os.environ["FORNIXDB_SHARED_DB"] = str(Path(self.tmp.name) / "no-shared.db")
        self.a = MemoryStore(db_path=Path(self.tmp.name) / "alpha.db")
        self.a.store("alpha fact", "x" * 500)
        set_config(self.a, "store_label", "Alpha")
        self.b = MemoryStore(db_path=Path(self.tmp.name) / "beta.db")
        self.b.store("beta fact one")
        self.b.store("beta fact two")
        import json
        self.reg.write_text(json.dumps([
            str((Path(self.tmp.name) / "alpha.db").resolve()),
            str((Path(self.tmp.name) / "beta.db").resolve()),
            str(Path(self.tmp.name) / "gone.db"),  # dead entry → pruned
        ]))

    def tearDown(self):
        import os
        self.a.close(); self.b.close()
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_rollup_labels_counts_total_and_prune(self):
        import json
        from fornixdb.budget import machine_usage
        u = machine_usage()
        by_label = {s["label"]: s for s in u["stores"]}
        self.assertEqual(set(by_label), {"Alpha", "beta"})  # label, else stem
        self.assertEqual(by_label["Alpha"]["memories"], 1)
        self.assertEqual(by_label["beta"]["memories"], 2)
        self.assertAlmostEqual(u["total_mb"],
                               round(sum(s["mb"] for s in u["stores"]), 3))
        self.assertGreater(u["total_mb"], 0)
        # the dead entry was pruned from the registry file
        self.assertNotIn("gone.db", self.reg.read_text())

    def test_temp_stores_do_not_register(self):
        # setUp's stores live under the temp dir, so connect() skipped them;
        # only the hand-written registry content exists
        import json
        from fornixdb.db import _register_store
        before = self.reg.read_text()
        _register_store(Path(self.tmp.name) / "alpha.db")
        self.assertEqual(self.reg.read_text(), before)
        # a non-temp path DOES register (no file is created at the path).
        # Compare resolved: the registry stores resolved paths, and on Windows
        # a drive-less str(Path) differs from its resolved C:\ form (W2).
        fake = Path("/Users/nobody/fornix-fake-store.db")
        _register_store(fake)
        self.assertIn(str(fake.resolve()), json.loads(self.reg.read_text()))


class TestMachineCap(unittest.TestCase):
    """Machine-wide cap across ALL stores (config machine_budget_mb --shared).
    A writing store fixes what it can on its own side only — it never deletes
    another AI's memories."""

    def setUp(self):
        import json, os
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self._env = {k: os.environ.get(k)
                     for k in ("FORNIXDB_REGISTRY", "FORNIXDB_SHARED_DB")}
        os.environ["FORNIXDB_REGISTRY"] = str(base / "reg.json")
        os.environ["FORNIXDB_SHARED_DB"] = str(base / "shared.db")
        self.shared = MemoryStore(db_path=base / "shared.db")
        self.mine = MemoryStore(db_path=base / "mine.db")
        self.other = MemoryStore(db_path=base / "other.db")
        _fill(self.mine, n=80)
        _fill(self.other, n=80, prefix="other")
        for s in (self.mine, self.other):
            _vacuum(s)
        (base / "reg.json").write_text(json.dumps(
            [str((base / "mine.db").resolve()),
             str((base / "other.db").resolve())]))

    def tearDown(self):
        import os
        for s in (self.shared, self.mine, self.other):
            s.close()
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        self.tmp.cleanup()

    def _total(self):
        from fornixdb.budget import machine_usage
        return machine_usage()["total_mb"]

    def test_freeze_refuses_writes_machine_wide(self):
        set_config(self.shared, "machine_budget_mb", f"{self._total() / 2:.3f}")
        with self.assertRaises(DiskBudgetExceededError) as ctx:
            self.mine.store("one more")
        self.assertIn("machine-wide cap", str(ctx.exception))
        # the OTHER store's memories were never touched
        self.assertEqual(self.other.stats()["memories"], 80)

    def test_prune_sheds_own_share_only(self):
        before_other = self.other.stats()["memories"]
        # modest cap: mine must shed its share but stays above its file floor
        # (WAL noise from the open shared/other connections counts too)
        cap = self._total() * 0.85
        set_config(self.shared, "machine_budget_mb", f"{cap:.3f}")
        set_config(self.shared, "machine_budget_policy", "prune")
        self.mine.store("newcomer")          # triggers machine enforcement
        self.assertLessEqual(self._total(), cap + 0.02)
        self.assertLess(self.mine.stats()["memories"], 81)   # own pruned
        self.assertEqual(self.other.stats()["memories"], before_other)
        self.assertTrue(self.mine.recall("newcomer", embedder=False))

    def test_prune_refuses_when_own_store_cannot_fix_it(self):
        # cap below what the OTHER store alone occupies: mine can't fix it
        from fornixdb.budget import _path_footprint
        other_mb = _path_footprint(Path(self.tmp.name) / "other.db")
        set_config(self.shared, "machine_budget_mb", f"{other_mb / 2:.3f}")
        set_config(self.shared, "machine_budget_policy", "prune")
        with self.assertRaises(DiskBudgetExceededError) as ctx:
            self.mine.store("cannot fit")
        self.assertIn("cannot fix that alone", str(ctx.exception))
        self.assertEqual(self.other.stats()["memories"], 80)  # untouched

    def test_usage_reports_cap(self):
        from fornixdb.budget import machine_usage
        # a FRESH shared tier (created in setUp) carries the install default:
        # min(20% of free disk, 500 MB), flagged for review
        u = machine_usage()
        self.assertIsNotNone(u["machine_budget_mb"])
        self.assertLessEqual(u["machine_budget_mb"], 500)
        self.assertGreater(u["machine_budget_mb"], 0)
        self.assertTrue(u["machine_budget_defaulted"])
        # the owner touching the cap IS the review — the flag clears
        set_config(self.shared, "machine_budget_mb", "123")
        u = machine_usage()
        self.assertEqual(u["machine_budget_mb"], 123)
        self.assertFalse(u["machine_budget_defaulted"])
        self.assertFalse(u["over_budget"])
        set_config(self.shared, "machine_budget_mb", "off")
        u = machine_usage()
        self.assertIsNone(u["machine_budget_mb"])
        self.assertFalse(u["machine_budget_defaulted"])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            set_config(self.shared, "machine_budget_mb", "-1")
        with self.assertRaises(ValueError):
            set_config(self.shared, "machine_budget_policy", "explode")
        set_config(self.shared, "machine_budget_mb", "off")  # clears


if __name__ == "__main__":
    unittest.main()
