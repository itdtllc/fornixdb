"""L4 rhythmic in-thought recall — the portable cadence controller."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic: keyword recall, no model

from fornixdb.adapters.native_memory import set_ingest_mode
from fornixdb.cadence import Episode, _overlap, pulse
from fornixdb.core import MemoryStore
from fornixdb.multistore import set_config


def file_store(tmp):
    return MemoryStore(db_path=Path(tmp) / "t.db")


class TestOverlap(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertEqual(_overlap("render the octopus shot", "render the octopus shot"), 1.0)

    def test_disjoint_is_zero(self):
        self.assertEqual(_overlap("octopus background", "tax bracket income"), 0.0)

    def test_empty_is_zero(self):
        self.assertEqual(_overlap("", "anything"), 0.0)


class TestPulse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)
        set_config(self.s, "rhythmic_recall", "on")  # L4 is opt-in; enable to test it
        self.s.store("Octopus logo lives at assets/fornixdb_icon.png",
                     kind="reference")
        self.s.store("Flux Redux preserves the reference background",
                     kind="reference")

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_relevant_thought_pulses(self):
        ep = Episode()
        block = pulse(self.s, "now generating the octopus logo training set", ep)
        self.assertIsNotNone(block)
        self.assertIn("octopus", block.lower())
        self.assertEqual(ep.pulse_count, 1)
        self.assertTrue(ep.pulsed_ids)

    def test_trivial_thought_silent(self):
        self.assertIsNone(pulse(self.s, "ok", Episode()))

    def test_episode_dedup_no_resurface(self):
        ep = Episode()
        self.assertIsNotNone(pulse(self.s, "looking up the octopus logo asset path", ep))
        # a DIFFERENT thought that would hit the same memory must not re-surface it
        again = pulse(self.s, "where is that octopus icon file located again", ep)
        self.assertIsNone(again)

    def test_debounce_thought_must_move(self):
        ep = Episode()
        self.assertIsNotNone(pulse(self.s, "flux redux preserves the background", ep))
        # near-identical thought (token overlap above the debounce) → no new pulse
        self.assertIsNone(pulse(self.s, "flux redux preserves the background now", ep))

    def test_max_pulses_budget(self):
        ep = Episode()
        ep.pulse_count = 4  # at the default budget
        self.assertIsNone(pulse(self.s, "octopus logo training set generation", ep))

    def test_config_off_silent(self):
        set_config(self.s, "rhythmic_recall", "off")
        self.assertIsNone(pulse(self.s, "octopus logo training set generation", Episode()))

    def test_ingest_explicit_off(self):
        set_ingest_mode(self.s, "explicit")
        self.assertIsNone(pulse(self.s, "octopus logo training set generation", Episode()))


if __name__ == "__main__":
    unittest.main()
