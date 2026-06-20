"""Interactive configuration wizard: scripted-input drive of `fornixdb
configure` — review-and-confirm timing, cumulative ladder, frozen handling."""

import unittest

from fornixdb import levels, wizard
from fornixdb.core import MemoryStore
from fornixdb.multistore import capture_mode, get_config, set_config


class _Script:
    """Feeds canned answers to `ask`; records everything printed."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.log: list[str] = []

    def ask(self, prompt=""):
        self.log.append(prompt)
        return self.answers.pop(0)

    def out(self, *parts):
        self.log.append(" ".join(str(p) for p in parts))

    def text(self) -> str:
        return "\n".join(self.log)


class WizardCase(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(db_path=":memory:")

    def tearDown(self):
        self.s.close()

    def _run(self, *answers):
        sc = _Script(*answers)
        res = wizard.run_configure(self.s, ask=sc.ask, out=sc.out, db_label="x")
        return res, sc

    # default fresh store sits at L4 with capture=suggest; the build prompts are
    # rung, capture-style, session, vectors, ingest, budget (no policy when off),
    # then the MCP-tools mode (keep/minimal/custom)

    def test_keep_everything_writes_nothing(self):
        res, _ = self._run("", "", "", "", "", "", "")  # all kept, no confirm
        self.assertEqual(res["applied"], [])
        self.assertFalse(res["aborted"])
        self.assertEqual(levels.current_rung(self.s)[0], "L4")

    def test_lower_rung_and_confirm(self):
        res, _ = self._run("L3", "", "", "", "", "", "", "y")
        self.assertIn("operating_level", res["applied"])
        self.assertEqual(levels.current_rung(self.s)[0], "L3")
        self.assertFalse(levels.is_on(self.s, "L4"))
        self.assertTrue(levels.is_on(self.s, "L3"))

    def test_decline_at_confirm_writes_nothing(self):
        res, _ = self._run("", "auto", "", "", "", "", "", "n")
        self.assertTrue(res["aborted"])
        self.assertEqual(res["applied"], [])
        self.assertEqual(capture_mode(self.s), "suggest")  # unchanged

    def test_change_capture_flavor(self):
        res, _ = self._run("", "auto", "", "", "", "", "", "y")
        self.assertEqual(capture_mode(self.s), "auto")
        self.assertEqual(levels.current_rung(self.s)[0], "L4")  # rung untouched

    def test_drop_to_l1_skips_capture_flavor(self):
        # rung, session, vectors, ingest, budget, tools, confirm — NO capture ask
        res, sc = self._run("L1", "", "", "", "", "", "y")
        self.assertEqual(levels.current_rung(self.s)[0], "L1")
        self.assertEqual(capture_mode(self.s), "explicit")  # L2 off
        self.assertNotIn("capture style", sc.text())

    def test_set_budget_then_policy(self):
        res, _ = self._run("", "", "", "", "", "1000", "freeze", "", "y")
        self.assertEqual(get_config(self.s, "disk_budget_mb"), "1000")
        self.assertEqual(get_config(self.s, "budget_policy"), "freeze")

    def test_reprompt_on_invalid_then_accept(self):
        # invalid rung 'L9' re-prompts, then 'L2' accepted; L2 still asks
        # capture-style: rung(x2), capture, session, vectors, ingest, budget,
        # tools, confirm
        res, sc = self._run("L9", "L2", "", "", "", "", "", "", "y")
        self.assertEqual(levels.current_rung(self.s)[0], "L2")
        self.assertIn("not one of", sc.text())

    def test_tools_minimal_disables_optionals(self):
        from fornixdb.adapters.mcp_server import tools_disabled
        res, _ = self._run("", "", "", "", "", "", "minimal", "y")
        self.assertTrue(res["applied"])              # tool:* entries applied
        disabled = set(tools_disabled(self.s))
        self.assertTrue(disabled)                    # some optional tools off
        self.assertNotIn("recall_memory", disabled)  # core never disabled

    def test_frozen_decline_leaves_store_frozen(self):
        set_config(self.s, "frozen", "on")
        res, _ = self._run("n")  # decline unfreeze
        self.assertTrue(res["aborted"])
        self.assertTrue(self.s.frozen())

    def test_frozen_accept_unfreezes_then_runs(self):
        set_config(self.s, "frozen", "on")
        res, _ = self._run("y", "", "", "", "", "", "", "")  # unfreeze, keep all
        self.assertFalse(self.s.frozen())
        self.assertFalse(res["aborted"])


if __name__ == "__main__":
    unittest.main()
