"""Multi-agent / multi-process / multi-thread safety on one machine.

The concurrent actors on a real box: the MCP server process, PostToolUse hook
processes (one per tool call), a voice host's threads, CLI runs, and admin
passes (dream/shrink) — all against the same store files. These tests pin the
guarantees that make that safe:

  - connect() serializes schema migration (no 'duplicate column name' loser)
  - MemoryStore is shareable across threads (one connection per thread)
  - write_txn makes read-then-act sequences atomic (supersede name-handoff)
  - prospective.due() delivers each reminder to exactly ONE concurrent host
  - side-log appends never tear a line; the registry never goes torn/dark

Processes use multiprocessing spawn (the macOS/Windows default) so nothing
here depends on fork-inherited state — exactly like the real hook processes.
"""

import json
import multiprocessing as mp
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"

from fornixdb import prospective
from fornixdb.core import MemoryStore
from fornixdb.db import SCHEMA_VERSION, append_log_line, connect

THREADS = 8
WRITES_PER_THREAD = 20


# ---------------------------------------------------------------- process
# workers (top-level: spawn pickles them by name)

def _worker_store_rows(db_path: str, start_evt, n: int, tag: str, q) -> None:
    """Open the store and write n rows; report ('ok', count) or ('err', msg)."""
    try:
        start_evt.wait(30)
        store = MemoryStore(db_path=db_path)
        for i in range(n):
            store.store(f"proc {tag} row {i}", kind="semantic")
        store.close()
        q.put(("ok", n))
    except Exception as e:  # noqa: BLE001 — the test wants the message
        q.put(("err", f"{type(e).__name__}: {e}"))


def _worker_connect_only(db_path: str, start_evt, q) -> None:
    """Just open the store (running any pending migration) and close."""
    try:
        start_evt.wait(30)
        conn = connect(db_path)
        conn.close()
        q.put(("ok", 1))
    except Exception as e:  # noqa: BLE001
        q.put(("err", f"{type(e).__name__}: {e}"))


def _worker_due(db_path: str, start_evt, q) -> None:
    """Poll due() once, like a host heartbeat; report how many rows it won."""
    try:
        start_evt.wait(30)
        store = MemoryStore(db_path=db_path)
        rows = prospective.due(store)
        store.close()
        q.put(("ok", len(rows)))
    except Exception as e:  # noqa: BLE001
        q.put(("err", f"{type(e).__name__}: {e}"))


def _run_procs(target, args_list):
    """Spawn one process per args tuple, release them together, collect
    (results, errors)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    start_evt = ctx.Event()
    procs = [ctx.Process(target=target, args=(*args, start_evt, *rest, q))
             for args, rest in args_list]
    for p in procs:
        p.start()
    start_evt.set()
    results, errors = [], []
    for _ in procs:
        kind, val = q.get(timeout=60)
        (results if kind == "ok" else errors).append(val)
    for p in procs:
        p.join(timeout=30)
    return results, errors


class ConcurrencyCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "store.db")

    def tearDown(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------- threads

class TestThreadSafety(ConcurrencyCase):
    def test_one_store_shared_across_threads(self):
        """The exact shape that used to raise sqlite3.ProgrammingError
        ('objects created in a thread can only be used in that same thread')
        and forced hosts into per-thread stores."""
        store = MemoryStore(db_path=self.db_path)
        errors: list[str] = []
        barrier = threading.Barrier(THREADS)

        def work(tag: int) -> None:
            try:
                barrier.wait(10)
                for i in range(WRITES_PER_THREAD):
                    mid = store.store(f"thread {tag} row {i}", kind="semantic")
                    got = store.show(mid)
                    assert got and f"thread {tag} row {i}" in got["gist"]
                    store.recall(f"thread {tag}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=work, args=(t,)) for t in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(errors, [])
        n = store.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        self.assertEqual(n, THREADS * WRITES_PER_THREAD)
        store.close()

    def test_each_thread_gets_its_own_connection(self):
        store = MemoryStore(db_path=self.db_path)
        conns = {}

        def grab(tag: int) -> None:
            conns[tag] = id(store.conn)

        threads = [threading.Thread(target=grab, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(len(set(conns.values())), 3)
        # ... and the same thread keeps getting the same one
        self.assertIs(store.conn, store.conn)
        store.close()

    def test_close_reopens_on_next_use(self):
        store = MemoryStore(db_path=self.db_path)
        store.store("before close", kind="semantic")
        store.close()
        # a closed store handle is not poisoned — next use reopens
        self.assertEqual(
            store.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0], 1)
        store.close()

    def test_injected_conn_stays_fixed(self):
        """The test-suite idiom MemoryStore(conn=connect(':memory:')) must
        keep its single fixed connection (per-thread would open a DIFFERENT
        empty :memory: db per thread)."""
        store = MemoryStore(conn=connect(":memory:"))
        self.assertIs(store.conn, store.conn)
        store.store("in memory", kind="semantic")
        self.assertEqual(
            store.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0], 1)
        store.close()


# ---------------------------------------------------------------- write_txn

class TestWriteTxn(ConcurrencyCase):
    def test_rolls_back_on_exception(self):
        store = MemoryStore(db_path=self.db_path)
        mid = store.store("keep me", kind="semantic")
        with self.assertRaises(RuntimeError):
            with store.write_txn() as conn:
                conn.execute("UPDATE memory SET gist = 'clobbered' WHERE id = ?",
                             (mid,))
                raise RuntimeError("boom")
        self.assertEqual(store.show(mid)["gist"], "keep me")
        store.close()

    def test_reentrant_joins_outer(self):
        store = MemoryStore(db_path=self.db_path)
        with store.write_txn() as conn:
            conn.execute("INSERT INTO topic(name) VALUES ('outer')")
            with store.write_txn() as inner:
                self.assertIs(inner, conn)
                inner.execute("INSERT INTO topic(name) VALUES ('inner')")
            # inner exit must NOT have committed/closed the outer transaction
            self.assertTrue(conn.in_transaction)
        n = store.conn.execute("SELECT COUNT(*) FROM topic").fetchone()[0]
        self.assertEqual(n, 2)
        store.close()

    def test_supersede_name_handoff_is_atomic(self):
        """Two processes superseding the same named row: exactly one live
        holder of the name afterwards, never two."""
        store = MemoryStore(db_path=self.db_path)
        old = store.store("the fact", kind="semantic", name="the-name")
        a = store.store("successor a", kind="semantic")
        b = store.store("successor b", kind="semantic")
        t1 = threading.Thread(target=store.supersede, args=(old, a))
        t2 = threading.Thread(target=store.supersede, args=(old, b))
        t1.start(); t2.start(); t1.join(10); t2.join(10)
        holders = store.conn.execute(
            "SELECT id FROM memory WHERE name = 'the-name'").fetchall()
        self.assertEqual(len(holders), 1)
        store.close()


class TestCompositeWrites(ConcurrencyCase):
    """store()'s insert transaction covers the row, its topics, and any
    _in_txn companion row — all land or none do."""

    def test_remind_never_leaves_a_dud(self):
        """If the prospective row can't be written, the memory row must not
        survive either (a 'reminder' with no delivery state reads as a memory
        of the intention but never fires)."""
        store = MemoryStore(db_path=self.db_path)
        before = store.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        store.conn.execute("ALTER TABLE prospective RENAME TO prospective_gone")
        store.conn.commit()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                prospective.remind(store, "doomed", "in 5 minutes")
            after = store.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            self.assertEqual(after, before)
        finally:
            store.conn.execute(
                "ALTER TABLE prospective_gone RENAME TO prospective")
            store.conn.commit()
        # and the healthy path still writes both rows together
        out = prospective.remind(store, "take the bread out", "in 5 minutes")
        pros = store.conn.execute(
            "SELECT 1 FROM prospective WHERE memory_id = ?",
            (out["id"],)).fetchone()
        self.assertIsNotNone(pros)
        store.close()

    def test_store_rolls_back_row_and_topics_with_companion(self):
        store = MemoryStore(db_path=self.db_path)

        def bad_companion(conn, mem_id):
            raise RuntimeError("companion failed")

        with self.assertRaises(RuntimeError):
            store.store("half-born", kind="semantic",
                        topics=["alpha", "beta"], _in_txn=bad_companion)
        n_mem = store.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        n_top = store.conn.execute("SELECT COUNT(*) FROM topic").fetchone()[0]
        self.assertEqual((n_mem, n_top), (0, 0))
        store.close()


# ---------------------------------------------------------------- processes

class TestMultiProcess(ConcurrencyCase):
    def test_concurrent_writers(self):
        """Hook processes + MCP server + CLI all writing at once."""
        n_procs, n_rows = 4, 10
        results, errors = _run_procs(
            _worker_store_rows,
            [((self.db_path,), (n_rows, f"p{i}")) for i in range(n_procs)])
        self.assertEqual(errors, [])
        store = MemoryStore(db_path=self.db_path)
        n = store.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        self.assertEqual(n, n_procs * n_rows)
        store.close()

    def test_migration_race_single_winner(self):
        """Open a store whose schema lags the code from several processes at
        once. Pre-serialization both probed the stale shape and both ran the
        ALTERs — the loser died with 'duplicate column name'. Now exactly one
        migrates; the rest wait, re-probe, and no-op."""
        conn = connect(self.db_path)
        conn.close()
        raw = sqlite3.connect(self.db_path)
        # Rewind to the pre-v13 shape: drop the suppression columns and mark
        # the version stale, exactly what an 0.8.7-era store looks like.
        if sqlite3.sqlite_version_info < (3, 35):
            raw.close()
            self.skipTest("needs DROP COLUMN (sqlite >= 3.35)")
        for col in ("proactive_suppressed_at", "suppressed_pushed",
                    "suppressed_referenced"):
            raw.execute(f"ALTER TABLE memory DROP COLUMN {col}")
        raw.execute("UPDATE meta SET value = '12' WHERE key = 'schema_version'")
        raw.commit()
        raw.close()

        results, errors = _run_procs(
            _worker_connect_only, [((self.db_path,), ()) for _ in range(4)])
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 4)

        check = sqlite3.connect(self.db_path)
        cols = [r[1] for r in check.execute("PRAGMA table_info(memory)")]
        self.assertEqual(cols.count("proactive_suppressed_at"), 1)
        version = check.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        self.assertEqual(int(version), SCHEMA_VERSION)
        check.close()

    def test_due_reminder_delivered_exactly_once(self):
        """Several hosts polling the same store: one reminder, one delivery."""
        store = MemoryStore(db_path=self.db_path)
        prospective.remind(store, "take the bread out", "in 1 minute")
        store.conn.execute("UPDATE prospective SET due = '2000-01-01T00:00:00'")
        store.conn.commit()
        store.close()

        results, errors = _run_procs(
            _worker_due, [((self.db_path,), ()) for _ in range(4)])
        self.assertEqual(errors, [])
        self.assertEqual(sum(results), 1,
                         f"reminder must fire exactly once, got {results}")


# ---------------------------------------------------------------- side files

class TestSideFiles(ConcurrencyCase):
    def test_append_log_line_concurrent_lines_stay_whole(self):
        path = Path(self._tmp.name) / "log.jsonl"
        payload = {"filler": "x" * 2000}  # big enough to tempt a split write
        barrier = threading.Barrier(THREADS)

        def work(tag: int) -> None:
            barrier.wait(10)
            for i in range(WRITES_PER_THREAD):
                append_log_line(path, json.dumps({**payload, "tag": tag, "i": i}))

        threads = [threading.Thread(target=work, args=(t,)) for t in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), THREADS * WRITES_PER_THREAD)
        for line in lines:
            json.loads(line)  # every line parses — none torn

    def test_registry_recovers_from_torn_file(self):
        """A half-written registry must not put the machine rollup
        permanently dark — the next connect rewrites it whole."""
        reg = Path(self._tmp.name) / "registry.json"
        reg.write_text('["/some/store.db", "/oth')  # torn mid-write
        # a path that can never live under tempfile.gettempdir() (temp-dir
        # stores deliberately skip registration)
        fake = Path("/nonexistent-fornix-test/fake-store.db")
        old = os.environ.get("FORNIXDB_REGISTRY")
        os.environ["FORNIXDB_REGISTRY"] = str(reg)
        try:
            from fornixdb.db import _register_store
            _register_store(fake)
        finally:
            if old is None:
                os.environ.pop("FORNIXDB_REGISTRY", None)
            else:
                os.environ["FORNIXDB_REGISTRY"] = old
        data = json.loads(reg.read_text())
        self.assertIn(str(fake.resolve()), data)


# ---------------------------------------------------------------- config

class TestBusyTimeoutConfig(ConcurrencyCase):
    def test_configured_busy_timeout_applies_on_connect(self):
        conn = connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO meta(key, value) "
                     "VALUES ('busy_timeout_ms', '12000')")
        conn.commit()
        conn.close()
        conn2 = connect(self.db_path)
        ms = conn2.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertEqual(ms, 12000)
        conn2.close()

    def test_garbage_value_falls_back_to_default(self):
        conn = connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO meta(key, value) "
                     "VALUES ('busy_timeout_ms', 'fast please')")
        conn.commit()
        conn.close()
        conn2 = connect(self.db_path)
        ms = conn2.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertEqual(ms, 5000)
        conn2.close()


if __name__ == "__main__":
    unittest.main()
