"""The operating-levels ladder (L0–L6) as a configurable surface: cumulative
set/toggle, locked floor, planned rungs, and incoherence detection."""

import unittest

from fornixdb import levels
from fornixdb.core import MemoryStore
from fornixdb.multistore import capture_mode, get_config, set_config


class LevelsCase(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(db_path=":memory:")

    def tearDown(self):
        self.s.close()

    # ---- defaults: a fresh store is at the top built rung (L5) -------------
    def test_fresh_store_defaults_to_top_built_rung(self):
        # capture suggest + proactive + rhythmic + parallel on are all defaults
        rung, incoherent = levels.current_rung(self.s)
        self.assertEqual(rung, "L5")
        self.assertFalse(incoherent)

    def test_only_l0_is_locked(self):
        # L0 (the keyed store floor) is always on and cannot be turned off
        self.assertTrue(levels.is_on(self.s, "L0"))
        with self.assertRaises(ValueError):
            levels.toggle(self.s, "L0", False)
        # L1 is on by default but CAN be turned off (e.g. a microcontroller
        # that supports only keyed get/put, not ranked recall)
        self.assertTrue(levels.is_on(self.s, "L1"))
        levels.toggle(self.s, "L1", False)
        self.assertFalse(levels.is_on(self.s, "L1"))

    def test_rung_l0_is_keyed_only_store(self):
        levels.set_rung(self.s, "L0")  # the microcontroller config
        for lid in ("L1", "L2", "L3", "L4"):
            self.assertFalse(levels.is_on(self.s, lid))
        self.assertEqual(levels.current_rung(self.s)[0], "L0")

    def test_rung_l1_is_recall_only(self):
        levels.set_rung(self.s, "L1")
        self.assertTrue(levels.is_on(self.s, "L1"))
        self.assertFalse(levels.is_on(self.s, "L2"))
        self.assertEqual(levels.current_rung(self.s)[0], "L1")

    def test_toggle_l1_off_cascades_everything_above(self):
        levels.toggle(self.s, "L1", False)
        for lid in ("L2", "L3", "L4"):
            self.assertFalse(levels.is_on(self.s, lid))
        self.assertEqual(levels.current_rung(self.s)[0], "L0")

    # ---- set_rung is cumulative -------------------------------------------
    def test_set_rung_enables_below_disables_above(self):
        levels.set_rung(self.s, "L3")
        self.assertTrue(levels.is_on(self.s, "L2"))
        self.assertTrue(levels.is_on(self.s, "L3"))
        self.assertFalse(levels.is_on(self.s, "L4"))
        rung, incoherent = levels.current_rung(self.s)
        self.assertEqual(rung, "L3")
        self.assertFalse(incoherent)

    def test_set_rung_l2_disables_recall_autonomy_above(self):
        levels.set_rung(self.s, "L2")
        self.assertEqual(capture_mode(self.s), "suggest")  # untouched default
        self.assertFalse(levels.is_on(self.s, "L3"))
        self.assertFalse(levels.is_on(self.s, "L4"))
        self.assertEqual(levels.current_rung(self.s)[0], "L2")

    def test_set_rung_preserves_richer_capture_choice(self):
        set_config(self.s, "capture_mode", "auto")
        levels.set_rung(self.s, "L4")  # re-enables L2 — must NOT downgrade auto
        self.assertEqual(capture_mode(self.s), "auto")

    # ---- single-level toggle cascades -------------------------------------
    def test_toggle_off_disables_above(self):
        levels.toggle(self.s, "L2", False)
        self.assertFalse(levels.is_on(self.s, "L2"))
        self.assertFalse(levels.is_on(self.s, "L3"))  # cascaded
        self.assertFalse(levels.is_on(self.s, "L4"))  # cascaded
        self.assertEqual(capture_mode(self.s), "explicit")
        self.assertEqual(levels.current_rung(self.s)[0], "L1")

    def test_toggle_on_enables_below(self):
        levels.set_rung(self.s, "L2")          # L3/L4 off
        levels.toggle(self.s, "L4", True)      # must pull L3 up too
        self.assertTrue(levels.is_on(self.s, "L3"))
        self.assertTrue(levels.is_on(self.s, "L4"))
        self.assertEqual(levels.current_rung(self.s)[0], "L4")

    # ---- planned rungs are inert ------------------------------------------
    def test_planned_levels_off_and_unsettable(self):
        for lid in ("L6",):
            self.assertFalse(levels.is_on(self.s, lid))
            with self.assertRaises(ValueError):
                levels.toggle(self.s, lid, True)
            with self.assertRaises(ValueError):
                levels.set_rung(self.s, lid)

    # ---- L5 is built and default-on (0.5.0); stepping down still works -----
    def test_l5_default_on_and_revertible(self):
        from fornixdb.multistore import get_config
        self.assertTrue(levels.is_on(self.s, "L5"))    # dial_default=on
        self.assertEqual(levels.current_rung(self.s)[0], "L5")
        levels.set_rung(self.s, "L4")                   # stepping back down
        self.assertIn(get_config(self.s, "parallel_recall"), ("off", "0", "false"))
        self.assertEqual(levels.current_rung(self.s)[0], "L4")
        self.assertFalse(levels.current_rung(self.s)[1])
        levels.set_rung(self.s, "L5")                   # and back up
        self.assertTrue(levels.is_on(self.s, "L5"))
        self.assertEqual(get_config(self.s, "parallel_recall"), "on")
        self.assertFalse(levels.current_rung(self.s)[1])  # coherent both ways

    # ---- incoherence: a gap left by direct `config` edits -----------------
    def test_incoherent_when_high_on_low_off(self):
        set_config(self.s, "capture_mode", "explicit")  # L2 off, L3/L4 still on
        rung, incoherent = levels.current_rung(self.s)
        self.assertEqual(rung, "L1")     # contiguous prefix stops at the gap
        self.assertTrue(incoherent)
        # set_rung re-normalizes to a clean prefix
        levels.set_rung(self.s, "L4")
        self.assertFalse(levels.current_rung(self.s)[1])

    def test_unknown_level_raises(self):
        with self.assertRaises(ValueError):
            levels.level("L9")

    # ---- the dials actually take effect when a rung is selected -----------
    def test_l1_off_makes_recall_exact_only(self):
        # L1's dial must really gate retrieval, not just be declarative
        self.s.store("The pool guy comes on Tuesdays.",
                     name="Pool cleaning schedule", kind="semantic")
        # L1 on (default): a fuzzy subject query finds it via ranking
        self.assertTrue(self.s.recall("pool guy day"))
        # drop to L0 (L1 off): no ranked recall, but exact name still works
        levels.toggle(self.s, "L1", False)
        self.assertEqual(self.s.recall("pool guy day"), [])
        self.assertTrue(self.s.recall("Pool cleaning schedule"))  # keyed get
        # back on restores ranked recall
        levels.toggle(self.s, "L1", True)
        self.assertTrue(self.s.recall("pool guy day"))

    def test_l2_l3_l4_dials_are_wired(self):
        # each rung writes the config key its runtime gate reads
        from fornixdb.multistore import capture_mode, get_config
        levels.set_rung(self.s, "L2")  # capture on, proactive+rhythmic off
        self.assertIn(capture_mode(self.s), ("suggest", "auto"))   # L2 gate
        self.assertIn(get_config(self.s, "proactive_recall"), ("off", "0", "false"))  # L3
        self.assertIn(get_config(self.s, "rhythmic_recall"), ("off", "0", "false"))   # L4

    # ---- formatter is total and marks the rung ----------------------------
    def test_format_ladder_lists_all_and_marks_rung(self):
        out = levels.format_ladder(self.s)
        for lid in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
            self.assertIn(lid, out)
        self.assertIn("current rung", out)
        self.assertIn("on (locked)", out)


if __name__ == "__main__":
    unittest.main()
