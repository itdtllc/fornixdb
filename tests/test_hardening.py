"""Security hardening (assessment 2026-06-12, FornixDB #176): file
permissions (B1), shared-tier writer provenance (B3), auto-capture
provenance flags at recall (B4)."""

import contextlib
import io
import os
import sqlite3
import stat
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from fornixdb.cli import main
from fornixdb.core import AUTO_CAPTURE_SOURCES, MemoryStore
from fornixdb.db import SCHEMA_VERSION, _restrict_to_owner_path, connect
from fornixdb.tiers import tier_down


def _run_cli(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(list(argv))
    return rc, buf.getvalue()


@unittest.skipUnless(os.name == "posix", "POSIX permission bits")
class TestFilePermissions(unittest.TestCase):
    """B1: memories are personal data — stores this package creates must be
    owner-only. Existing files are never re-chmodded (a deliberate loosening
    by the owner is respected)."""

    def _mode(self, path: Path) -> int:
        return stat.S_IMODE(path.stat().st_mode)

    def test_new_db_is_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            conn = connect(db)
            try:
                for f in (db, db.with_name("m.db-wal"), db.with_name("m.db-shm")):
                    if f.exists():
                        self.assertEqual(self._mode(f) & 0o077, 0,
                                         f"{f.name} readable by group/other")
            finally:
                conn.close()

    def test_new_parent_dir_is_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "newdir" / "m.db"
            connect(db).close()
            self.assertEqual(self._mode(db.parent), 0o700)

    def test_existing_db_permissions_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            connect(db).close()
            os.chmod(db, 0o644)  # owner loosens it on purpose
            connect(db).close()  # re-open: not creating
            self.assertEqual(self._mode(db), 0o644)

    def test_restrict_helper_file_and_dir(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x"
            f.write_text("hi")
            os.chmod(f, 0o644)
            _restrict_to_owner_path(f)
            self.assertEqual(self._mode(f), 0o600)
            sub = Path(d) / "sub"
            sub.mkdir(mode=0o755)
            _restrict_to_owner_path(sub, is_dir=True)
            self.assertEqual(self._mode(sub), 0o700)

    def test_cold_archive_is_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            s = MemoryStore(db_path=Path(d) / "t.db")
            try:
                mid = s.store("ancient session", "cold detail body",
                              kind="episodic", salience=0.2)
                old = (datetime.now() - timedelta(days=400)).isoformat()
                s.conn.execute("UPDATE memory SET recorded_time=?, "
                               "last_recalled=NULL, event_time=? WHERE id=?",
                               (old, old, mid))
                s.conn.commit()
                self.assertEqual(tier_down(s)["cold"], 1)
            finally:
                s.close()
            arcs = list((Path(d) / "t.archive").glob("*.jsonl.gz"))
            self.assertEqual(len(arcs), 1)
            self.assertEqual(self._mode(arcs[0]) & 0o077, 0,
                             "cold archive readable by group/other")


@unittest.skipUnless(os.name == "nt", "Windows ACL hardening")
class TestFilePermissionsWindows(unittest.TestCase):
    """B1 on Windows: os.chmod can't restrict access (it only flips the
    read-only bit), so a new store's ACL must be reset to the current user via
    icacls. Verified during the fresh-install test on a Windows box. Principal
    names below are the US-English defaults."""

    def _icacls(self, path: Path) -> str:
        return subprocess.run(["icacls", str(path)], capture_output=True,
                              text=True).stdout

    def test_new_db_acl_is_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            connect(db).close()
            out = self._icacls(db)
            user = os.environ.get("USERNAME") or ""
            self.assertIn(user, out)            # current user granted
            self.assertNotIn("(I)", out)        # inheritance was reset
            for broad in ("Everyone", "Authenticated Users", "BUILTIN\\Users"):
                self.assertNotIn(broad, out, f"{broad} can read the store")


class TestWriterMigration(unittest.TestCase):
    """B3 schema v5: existing stores gain the nullable writer column."""

    def test_v4_store_gains_writer_column(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "old.db"
            conn = sqlite3.connect(db)  # a pre-v5 memory table: no writer
            conn.execute("""CREATE TABLE memory (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE, kind TEXT NOT NULL,
                event_time TEXT NOT NULL, event_time_end TEXT,
                recorded_time TEXT NOT NULL, session_id TEXT, project TEXT,
                gist TEXT NOT NULL, detail TEXT,
                salience REAL NOT NULL DEFAULT 0.5,
                retention_tier TEXT NOT NULL DEFAULT 'hot',
                source TEXT, source_ref TEXT, last_recalled TEXT,
                last_reinforced TEXT, recall_count INTEGER NOT NULL DEFAULT 0,
                superseded_by INTEGER, superseded_time TEXT)""")
            conn.commit()
            conn.close()
            conn = connect(db)
            try:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(memory)")]
                self.assertIn("writer", cols)
                version = conn.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()[0]
                self.assertEqual(version, str(SCHEMA_VERSION))
            finally:
                conn.close()


class TestWriterProvenance(unittest.TestCase):
    """B3: shared-tier rows say which agent wrote them; every agent reads the
    shared tier with full trust, so the weakest model on the machine must not
    be able to launder anonymous rows into it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old_shared = os.environ.get("FORNIXDB_SHARED_DB")
        os.environ["FORNIXDB_SHARED_DB"] = str(Path(self.tmp.name) / "shared.db")
        self.db = str(Path(self.tmp.name) / "alpha.db")

    def tearDown(self):
        if self._old_shared is None:
            os.environ.pop("FORNIXDB_SHARED_DB", None)
        else:
            os.environ["FORNIXDB_SHARED_DB"] = self._old_shared
        self.tmp.cleanup()

    def test_shared_write_stamped_and_flagged(self):
        rc, _ = _run_cli("--db", self.db, "store", "--shared",
                         "--gist", "owner prefers metric units everywhere")
        self.assertEqual(rc, 0)
        rc, out = _run_cli("--db", self.db, "recall", "metric units")
        self.assertEqual(rc, 0)
        self.assertIn("[by alpha]", out)  # store_label unset → filename stem

    def test_shared_write_uses_store_label(self):
        _run_cli("--db", self.db, "config", "store_label", "Test-Agent")
        _run_cli("--db", self.db, "store", "--shared",
                 "--gist", "owner prefers metric units everywhere")
        _, out = _run_cli("--db", self.db, "recall", "metric units")
        self.assertIn("[by Test-Agent]", out)

    def test_own_store_write_not_stamped(self):
        _run_cli("--db", self.db, "store",
                 "--gist", "private working note about metric units")
        store = MemoryStore(db_path=self.db)
        try:
            row = store.conn.execute("SELECT writer FROM memory").fetchone()
            self.assertIsNone(row["writer"])
        finally:
            store.close()


class TestAutoCaptureFlag(unittest.TestCase):
    """B4: machine-ingested rows (transcript back-fill, SessionEnd capture —
    no owner review) are flagged at recall so consumers weigh provenance;
    owner-mediated sources stay unflagged."""

    def test_source_classes(self):
        self.assertIn("claude-code-transcript", AUTO_CAPTURE_SOURCES)
        for owner_mediated in ("cli", "mcp", "markdown-import"):
            self.assertNotIn(owner_mediated, AUTO_CAPTURE_SOURCES)

    def test_auto_captured_flagged_in_cli_and_mcp(self):
        from fornixdb.adapters.mcp_server import _line
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            _run_cli("--db", db, "--no-shared", "store",
                     "--source", "claude-code-transcript",
                     "--gist", "session digest about the widget refactor")
            _run_cli("--db", db, "--no-shared", "store",
                     "--gist", "owner-stated fact about the widget refactor")
            _, out = _run_cli("--db", db, "--no-shared", "recall",
                              "widget refactor")
            flagged = [l for l in out.splitlines() if "[auto-captured]" in l]
            self.assertEqual(len(flagged), 1)
            self.assertIn("session digest", flagged[0])
            store = MemoryStore(db_path=db)
            try:
                rows = [dict(r) for r in store.conn.execute(
                    "SELECT * FROM memory ORDER BY id")]
            finally:
                store.close()
            self.assertIn("[auto-captured]", _line(rows[0]))
            self.assertNotIn("[auto-captured]", _line(rows[1]))


if __name__ == "__main__":
    unittest.main()
