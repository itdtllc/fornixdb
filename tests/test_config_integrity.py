"""doctor config-integrity: a config key with no runtime reader does nothing —
flag keys SET in a store that no code reads (typo / stale / declarative-only,
the "L1 was set but never read" class), and guard at dev time that every
documented dial actually HAS a reader."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"

from fornixdb import doctor
from fornixdb.core import MemoryStore
from fornixdb.db import connect


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


def _put(store, key, value="x"):
    # write meta directly: we test KEY-NAME recognition, not value validation
    store.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                       (key, value))
    store.conn.commit()


def _warned_keys(store):
    return {row["msg"].split("'")[1] for row in doctor.config_integrity(store)}


class TestConfigIntegrity(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_unknown_key_is_flagged(self):
        _put(self.s, "proactive_reall")   # typo of proactive_recall
        self.assertIn("proactive_reall", _warned_keys(self.s))

    def test_legit_static_keys_not_flagged(self):
        for k in ("capture_mode", "proactive_recall", "usefulness_floor_adapt",
                  "project_scoped_pulse", "cross_pulse_dedup", "active_project",
                  "project_aliases", "store_label"):
            _put(self.s, k)
        self.assertEqual(_warned_keys(self.s), set())

    def test_indirectly_read_keys_not_flagged(self):
        for k in ("frozen", "mcp_tools_disabled", "machine_budget_mb"):
            _put(self.s, k)
        self.assertEqual(_warned_keys(self.s) & {"frozen", "mcp_tools_disabled",
                                                 "machine_budget_mb"}, set())

    def test_dynamic_prefix_keys_not_flagged(self):
        for k in ("decay_halflife_semantic", "decay_floor_episodic",
                  "active_project_session_abc", "cadence_turn_abc",
                  "cadence_episode_abc", "proactive_injected_abc"):
            _put(self.s, k, "1")
        self.assertEqual(_warned_keys(self.s), set())

    def test_clean_store_has_no_integrity_warnings(self):
        # schema_version is written on connect; it is a read key, not a smell
        self.assertEqual(doctor.config_integrity(self.s), [])


class TestIntegritySurfacedInDiagnose(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = MemoryStore(db_path=Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_integrity_rows_surface_in_diagnose(self):
        _put(self.s, "totally_made_up_key", "1")
        msgs = " ".join(r["msg"] for r in doctor.diagnose(self.s))
        self.assertIn("totally_made_up_key", msgs)


class TestReaderCoverageGuard(unittest.TestCase):
    """The dev-time guard the spec calls out: a documented dial with no reader
    (L1 declarative-only) must fail here, not silently in a user's store."""

    def test_every_levels_dial_has_a_reader(self):
        from fornixdb import levels
        readers = doctor.config_readers()
        for lv in levels.LEVELS:
            if lv.dial:
                self.assertIn(lv.dial, readers,
                              f"level dial '{lv.dial}' has no runtime reader")

    def test_behavioral_toggles_have_readers(self):
        readers = doctor.config_readers()
        for k in ("capture_mode", "ingest_mode", "session_capture",
                  "proactive_recall", "usefulness_floor_adapt",
                  "project_scoped_pulse", "cross_pulse_dedup", "rhythmic_recall",
                  "associative_recall", "frozen", "active_project",
                  "project_aliases"):
            self.assertIn(k, readers, f"config '{k}' has no runtime reader")


if __name__ == "__main__":
    unittest.main()
