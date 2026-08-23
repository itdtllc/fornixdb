"""Consolidated config view + health/doctor pass: a single place to see how a
store is set up, with suggested defaults (notably a disk cap, which is the one
recommended setting NOT applied out of the box)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from fornixdb import doctor
from fornixdb.core import MemoryStore, FrozenStoreError
from fornixdb.multistore import get_config, set_config


class _FileStoreCase(unittest.TestCase):
    """doctor reports on a real, file-backed store (budget/footprint math needs
    a db file on disk — an in-memory store has no path)."""

    def setUp(self):
        self._db = tempfile.mktemp(suffix=".db")
        self.s = MemoryStore(db_path=self._db)

    def tearDown(self):
        self.s.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(self._db + suffix)
            except OSError:
                pass


class TestConfigOverview(_FileStoreCase):
    def test_overview_covers_the_key_settings(self):
        keys = {k for k, _ in doctor.config_overview(self.s)}
        for expected in ("capture_mode", "ingest_mode", "vectors",
                         "disk_budget", "frozen", "MCP tools",
                         "proactive_recall", "session_capture"):
            self.assertIn(expected, keys)

    def test_reflects_a_changed_setting(self):
        set_config(self.s, "capture_mode", "explicit")
        rows = dict(doctor.config_overview(self.s))
        self.assertEqual(rows["capture_mode"], "explicit")

    def test_every_overview_item_has_a_default(self):
        # a read-only `config` run must be able to show a default for every
        # option, so none is left blank/unset for the user
        keys = {k for k, _ in doctor.config_overview(self.s)}
        self.assertTrue(keys <= set(doctor.CONFIG_DEFAULTS),
                        f"missing defaults for: {keys - set(doctor.CONFIG_DEFAULTS)}")

    def test_format_config_shows_defaults_when_provided(self):
        out = doctor.format_config(doctor.config_overview(self.s),
                                   doctor.CONFIG_DEFAULTS)
        self.assertIn("[default:", out)


class TestSuggestedDefaults(_FileStoreCase):
    def test_disk_budget_suggested_and_unsatisfied_by_default(self):
        rows = {r["key"]: r for r in doctor.suggested_settings(self.s)}
        self.assertIn("disk_budget_mb", rows)
        self.assertFalse(rows["disk_budget_mb"]["satisfied"])  # never-delete default
        self.assertGreaterEqual(int(rows["disk_budget_mb"]["suggested"]), 1)

    def test_code_defaults_report_satisfied(self):
        rows = {r["key"]: r for r in doctor.suggested_settings(self.s)}
        for k in ("budget_policy", "capture_mode", "vectors",
                  "proactive_recall", "session_capture"):
            self.assertTrue(rows[k]["satisfied"], f"{k} should be satisfied")

    def test_suggested_budget_is_bounded_by_ceiling(self):
        from fornixdb.db import DEFAULT_MACHINE_CAP_MAX_MB
        self.assertLessEqual(doctor.suggested_disk_budget_mb(self.s),
                             DEFAULT_MACHINE_CAP_MAX_MB)

    def test_apply_suggested_sets_only_unsatisfied(self):
        applied = doctor.apply_suggested(self.s)
        self.assertTrue(any(a.startswith("disk_budget_mb") for a in applied))
        # capture_mode was already at the default → not re-applied
        self.assertFalse(any(a.startswith("capture_mode") for a in applied))
        # and now the cap is actually set
        self.assertIsNotNone(get_config(self.s, "disk_budget_mb"))
        # second run is a no-op (idempotent — nothing left unsatisfied here)
        self.assertEqual(doctor.apply_suggested(self.s), [])

    def test_apply_refused_on_frozen_store(self):
        set_config(self.s, "frozen", "1")
        self.s.__dict__.pop("_frozen_cache", None)
        with self.assertRaises(FrozenStoreError):
            doctor.apply_suggested(self.s)


class TestDiagnose(_FileStoreCase):
    def test_schema_row_ok(self):
        rows = doctor.diagnose(self.s, host_paths=())
        self.assertTrue(any(r["level"] == "ok" and "schema" in r["msg"]
                            for r in rows))

    def test_missing_hooks_warn_when_no_settings_file(self):
        rows = doctor.diagnose(self.s, host_paths=())
        warns = [r["msg"] for r in rows if r["level"] == "warn"]
        self.assertTrue(any("SessionEnd" in m for m in warns))
        self.assertTrue(any("UserPromptSubmit" in m for m in warns))

    def test_hooks_detected_when_module_in_settings(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "settings.json"
            f.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [
                {"command": "py -m fornixdb.adapters.claude_code_recall --db x"}]}]}}))
            hs = doctor.host_hook_status([str(f)])
        wired = {r["hook"]: r["wired"] for r in hs["hooks"]}
        self.assertTrue(wired["UserPromptSubmit recall"])
        self.assertFalse(wired["SessionEnd capture"])  # not in this file

    def test_no_budget_emits_info(self):
        rows = doctor.diagnose(self.s, host_paths=())
        self.assertTrue(any(r["level"] == "info" and "budget" in r["msg"]
                            for r in rows))


if __name__ == "__main__":
    unittest.main()


class TestConfigIntegrityScannerGaps(_FileStoreCase):
    """Keys that ARE read, but not by a literal get_config("name") call — the
    scanner has to be told about each one or it reports a healthy store as sick.
    Every warning on the live store was one of these: 37 of them, all false.
    """

    def _flagged(self):
        import re
        return {m.group(1) for m in
                (re.search(r"config '([^']+)'", r["msg"])
                 for r in doctor.config_integrity(self.s)) if m}

    def test_per_session_writeback_key_is_not_flagged(self):
        # built by proactive._writeback_key(session_id), so the literal scan
        # cannot see it — and one lands per session, forever
        set_config(self.s, "writeback_hint_shown_abc-123", "1")
        self.assertNotIn("writeback_hint_shown_abc-123", self._flagged())

    def test_keys_read_through_a_named_constant_are_not_flagged(self):
        # mcp_server._TOOLS_ENABLED_KEY and reproject.UNDO_KEY
        set_config(self.s, "mcp_tools_enabled", "see_image")
        set_config(self.s, "reproject_undo", "[]")
        flagged = self._flagged()
        self.assertNotIn("mcp_tools_enabled", flagged)
        self.assertNotIn("reproject_undo", flagged)

    def test_a_genuinely_unread_key_is_still_flagged(self):
        # the check must still do its job — this is the bug class it exists for
        set_config(self.s, "prooactive_recall", "on")     # typo, nothing reads it
        self.assertIn("prooactive_recall", self._flagged())

    def test_every_session_scoped_prefix_is_recognized(self):
        from fornixdb.db import SESSION_SCOPED_META_PREFIXES
        for prefix in SESSION_SCOPED_META_PREFIXES:
            set_config(self.s, f"{prefix}some-session-id", "x")
        flagged = self._flagged()
        for prefix in SESSION_SCOPED_META_PREFIXES:
            self.assertNotIn(f"{prefix}some-session-id", flagged, prefix)
