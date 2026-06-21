"""Cross-pulse dedup: a memory pushed once this session by EITHER rung (L3's
per-turn hook or an L4 tick) is not pushed again that session. L3 and L4 share
one per-session injected set; reversible via `config cross_pulse_dedup off`."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic keyword recall

from fornixdb.cadence import Episode, pulse
from fornixdb.core import MemoryStore
from fornixdb.multistore import set_config
from fornixdb.proactive import injected_this_session, mark_injected


def file_store(tmp):
    return MemoryStore(db_path=Path(tmp) / "t.db")


class TestCrossPulseDedup(unittest.TestCase):
    THOUGHT = "looking up the octopus logo asset path again"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)
        self.mid = self.s.store("Octopus logo lives at assets/fornixdb_icon.png",
                                kind="reference")

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_l4_skips_what_l3_already_injected(self):
        # L3 pushed this memory earlier in the session (recorded in the shared set)
        mark_injected(self.s, "s1", [self.mid])
        # L4 now hits the same memory on a fresh episode → must stay silent
        self.assertIsNone(pulse(self.s, self.THOUGHT, Episode(), session_id="s1"))

    def test_l4_pulse_records_into_shared_session_set(self):
        block = pulse(self.s, self.THOUGHT, Episode(), session_id="s1")
        self.assertIsNotNone(block)
        self.assertIn(self.mid, injected_this_session(self.s, "s1"))

    def test_other_session_not_affected(self):
        mark_injected(self.s, "s1", [self.mid])
        # a different session has its own set → the memory still surfaces
        self.assertIsNotNone(pulse(self.s, self.THOUGHT, Episode(), session_id="s2"))

    def test_dedup_off_falls_back_to_episode_only(self):
        set_config(self.s, "cross_pulse_dedup", "off")
        mark_injected(self.s, "s1", [self.mid])
        # with dedup off the session set is ignored → L4 still surfaces it
        self.assertIsNotNone(pulse(self.s, self.THOUGHT, Episode(), session_id="s1"))

    def test_dedup_off_does_not_write_session_set(self):
        set_config(self.s, "cross_pulse_dedup", "off")
        pulse(self.s, self.THOUGHT, Episode(), session_id="s1")
        self.assertNotIn(self.mid, injected_this_session(self.s, "s1"))


if __name__ == "__main__":
    unittest.main()
