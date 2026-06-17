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
