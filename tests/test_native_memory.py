"""Native-memory ingest (§15.2 #2): FornixDB FOLLOWS a host AI's native memory
directory downstream — additive, idempotent, deduped — and never owns it. Mode
control (explicit/passive/both) lets the user turn background activity off."""

import os
import tempfile
import unittest
from pathlib import Path

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.adapters.native_memory import (auto_background_enabled, ingest,
                                             ingest_mode, native_dir,
                                             set_ingest_mode, set_native_dir)
from fornixdb.adapters import claude_code_session_end as hook


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


def write_memory_dir(d: Path):
    (d / "MEMORY.md").write_text("# index\n- pointer line\n")  # skipped on import
    (d / "dark-mode.md").write_text(
        "---\nname: dark-mode\ndescription: owner prefers dark mode\n"
        "metadata:\n  type: user\n---\n\nThe owner prefers a dark UI theme.\n")
    (d / "build-cmd.md").write_text(
        "---\nname: build-cmd\ndescription: build with make release\n---\n\n"
        "Run `make release` to build.\n")


class TestModeControl(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_default_mode_is_passive(self):
        self.assertEqual(ingest_mode(self.s), "passive")
        self.assertTrue(auto_background_enabled(self.s))

    def test_explicit_disables_background(self):
        set_ingest_mode(self.s, "explicit")
        self.assertEqual(ingest_mode(self.s), "explicit")
        self.assertFalse(auto_background_enabled(self.s))

    def test_both_enables_background(self):
        set_ingest_mode(self.s, "both")
        self.assertTrue(auto_background_enabled(self.s))

    def test_bad_mode_rejected(self):
        self.assertIn("must be one of", set_ingest_mode(self.s, "nonsense"))
        self.assertEqual(ingest_mode(self.s), "passive")  # unchanged


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_memory_dir(self.dir)
        self.s = mem_store()

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_ingest_follows_native_dir(self):
        r = ingest(self.s, str(self.dir))
        self.assertTrue(r["ok"])
        self.assertEqual(r["imported"], 2)            # MEMORY.md skipped
        names = {row["name"] for row in self.s.conn.execute(
            "SELECT name FROM memory")}
        self.assertEqual(names, {"dark-mode", "build-cmd"})

    def test_ingest_is_idempotent(self):
        ingest(self.s, str(self.dir))
        r2 = ingest(self.s, str(self.dir))           # re-run: nothing new
        self.assertEqual(r2["imported"], 0)
        self.assertEqual(self.s.conn.execute(
            "SELECT count(*) c FROM memory").fetchone()["c"], 2)

    def test_content_dedup_same_fact_new_name(self):
        ingest(self.s, str(self.dir))
        # same fact (gist) re-slugged under a new filename: must NOT double-store
        (self.dir / "dark-mode-v2.md").write_text(
            "---\nname: dark-mode-v2\ndescription: owner prefers dark mode\n---\n\n"
            "dup under a new name\n")
        r = ingest(self.s, str(self.dir))
        self.assertEqual(r["imported"], 0)
        self.assertEqual(self.s.conn.execute(
            "SELECT count(*) c FROM memory WHERE gist = 'owner prefers dark mode'"
        ).fetchone()["c"], 1)

    def test_no_dir_configured(self):
        self.assertFalse(ingest(self.s)["ok"])

    def test_source_tag_marks_native_origin(self):
        ingest(self.s, str(self.dir))
        sources = {r["source"] for r in self.s.conn.execute(
            "SELECT DISTINCT source FROM memory")}
        self.assertEqual(sources, {"claude-code-native"})


class TestHookRespectsMode(unittest.TestCase):
    """The session-end hook must run native ingest only in passive/both, and
    explicit mode must suppress even passive session capture."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_memory_dir(self.dir)
        self.s = mem_store()
        set_native_dir(self.s, str(self.dir))

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_explicit_mode_suppresses_capture(self):
        set_ingest_mode(self.s, "explicit")
        msg = hook.capture_session(self.s, "/nonexistent/transcript.jsonl")
        self.assertIn("explicit", msg)  # short-circuited before the missing file

    def test_passive_mode_allows_capture_path(self):
        set_ingest_mode(self.s, "passive")
        # passive lets capture proceed (then hits the missing-file branch, not
        # the explicit short-circuit) — proving the mode gate opened
        msg = hook.capture_session(self.s, "/nonexistent/transcript.jsonl")
        self.assertNotIn("explicit", msg)


if __name__ == "__main__":
    unittest.main()
