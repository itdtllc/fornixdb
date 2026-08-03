"""_transcripts_source: which transcript pool a push-usefulness CLI command
reads. The old hardcoded ~/.claude/projects default silently scanned another
host's sessions against a second-consumer store (a second consumer's, live 2026-07-26) —
the cross-store id-collision mode of the 2026-07-03 phantom-credit bug. The
resolver's precedence is flag > env FORNIXDB_TRANSCRIPTS > the STORE's
transcripts_path config > the Claude host default, matching dream's refreshes."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from fornixdb.cli import _transcripts_source, main
from fornixdb.core import MemoryStore
from fornixdb.multistore import set_config


class TestTranscriptsSource(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(db_path=Path(self._tmp.name) / "t.db")
        self._env = os.environ.pop("FORNIXDB_TRANSCRIPTS", None)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()
        if self._env is not None:
            os.environ["FORNIXDB_TRANSCRIPTS"] = self._env

    def test_explicit_flag_wins(self):
        set_config(self.store, "transcripts_path", "/cfg/dir")
        os.environ["FORNIXDB_TRANSCRIPTS"] = "/env/dir"
        try:
            self.assertEqual(_transcripts_source(self.store, "/flag/dir"),
                             "/flag/dir")
        finally:
            del os.environ["FORNIXDB_TRANSCRIPTS"]

    def test_empty_flag_means_skip(self):
        # value's documented "empty string to skip" stays a skip, never a
        # fall-through to config (that would scan a pool the caller opted out of)
        set_config(self.store, "transcripts_path", "/cfg/dir")
        self.assertIsNone(_transcripts_source(self.store, ""))

    def test_env_beats_config(self):
        set_config(self.store, "transcripts_path", "/cfg/dir")
        os.environ["FORNIXDB_TRANSCRIPTS"] = "/env/dir"
        try:
            self.assertEqual(_transcripts_source(self.store, None), "/env/dir")
        finally:
            del os.environ["FORNIXDB_TRANSCRIPTS"]

    def test_store_config_beats_host_default(self):
        # the second-consumer mode: a second-consumer store with its own pool must NEVER
        # default to another host's sessions
        set_config(self.store, "transcripts_path", "/her/own/transcripts")
        self.assertEqual(_transcripts_source(self.store, None),
                         "/her/own/transcripts")

    def test_off_sentinel_disables(self):
        # same sentinels dream honors (consolidate.py): disabled is disabled
        os.environ["FORNIXDB_TRANSCRIPTS"] = "off"
        try:
            self.assertIsNone(_transcripts_source(self.store, None))
        finally:
            del os.environ["FORNIXDB_TRANSCRIPTS"]
        set_config(self.store, "transcripts_path", "OFF")
        self.assertIsNone(_transcripts_source(self.store, None))

    def test_unset_falls_back_to_claude_default(self):
        self.assertEqual(_transcripts_source(self.store, None),
                         "~/.claude/projects")


class TestScanCliUsesStoreConfig(unittest.TestCase):
    def setUp(self):
        # the suite pins FORNIXDB_TRANSCRIPTS=off so tests never scan this
        # machine's transcripts; this test provides its own pool
        self._env = os.environ.pop("FORNIXDB_TRANSCRIPTS", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["FORNIXDB_TRANSCRIPTS"] = self._env

    def test_usefulness_scan_reads_configured_pool(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "t.db")
            pool = Path(d) / "pool"
            pool.mkdir()
            (pool / "s.jsonl").write_text(
                json.dumps({"type": "attachment",
                            "attachment": {"hookEvent": "UserPromptSubmit",
                                           "content": "[FornixDB · possibly-"
                                           "relevant past — …]\n#36 gist"}}),
                encoding="utf-8")
            s = MemoryStore(db_path=db)
            set_config(s, "transcripts_path", str(pool))
            s.close()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = main(["--db", db, "usefulness-scan"])
            self.assertFalse(rc)
            self.assertIn(str(pool), out.getvalue())
            self.assertIn("push impressions: 1", out.getvalue())


if __name__ == "__main__":
    unittest.main()
