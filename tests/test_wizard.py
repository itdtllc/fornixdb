"""Interactive configuration wizard: scripted-input drive of `fornixdb
configure` — review-and-confirm timing, cumulative ladder, frozen handling."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fornixdb import cli, levels, wizard
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

    # default fresh store sits at L5 with capture=suggest; the build prompts are
    # rung, dissent (asked at L5), capture-style, session, vectors, ingest,
    # budget (no policy when off), floor-log, transcripts-path, then the
    # MCP-tools mode (keep/minimal/custom)

    def test_keep_everything_writes_nothing(self):
        res, _ = self._run("", "", "", "", "", "", "", "", "", "")  # all kept, no confirm
        self.assertEqual(res["applied"], [])
        self.assertFalse(res["aborted"])
        self.assertEqual(levels.current_rung(self.s)[0], "L5")

    def test_lower_rung_and_confirm(self):
        res, _ = self._run("L3", "", "", "", "", "", "", "", "", "y")
        self.assertIn("operating_level", res["applied"])
        self.assertEqual(levels.current_rung(self.s)[0], "L3")
        self.assertFalse(levels.is_on(self.s, "L4"))

    def test_l5_offered_and_asks_dissent(self):
        # at the L5 default rung the dissent question appears; turning it on:
        # rung, dissent, capture, session, vectors, ingest, budget, floor-log,
        # tools, confirm
        res, sc = self._run("", "on", "", "", "", "", "", "", "", "", "y")
        self.assertIn("parallel_dissent", res["applied"])
        self.assertEqual(levels.current_rung(self.s)[0], "L5")
        self.assertTrue(levels.is_on(self.s, "L5"))  # default-on since 0.5.0
        self.assertEqual(get_config(self.s, "parallel_dissent"), "on")
        self.assertIn("dissent", sc.text())

    def test_below_l5_never_asks_dissent(self):
        # stepping down to L4 drops the dissent ask: rung, capture, session,
        # vectors, ingest, budget, floor-log, tools, confirm
        res, sc = self._run("L4", "", "", "", "", "", "", "", "", "y")
        self.assertIn("operating_level", res["applied"])
        self.assertNotIn("dissent", sc.text())
        self.assertEqual(levels.current_rung(self.s)[0], "L4")
        self.assertTrue(levels.is_on(self.s, "L3"))

    def test_decline_at_confirm_writes_nothing(self):
        res, _ = self._run("", "", "auto", "", "", "", "", "", "", "", "n")
        self.assertTrue(res["aborted"])
        self.assertEqual(res["applied"], [])
        self.assertEqual(capture_mode(self.s), "suggest")  # unchanged

    def test_change_capture_flavor(self):
        res, _ = self._run("", "", "auto", "", "", "", "", "", "", "", "y")
        self.assertEqual(capture_mode(self.s), "auto")
        self.assertEqual(levels.current_rung(self.s)[0], "L5")  # rung untouched

    def test_drop_to_l1_skips_capture_flavor(self):
        # rung, session, vectors, ingest, budget, floor-log, tools, confirm — NO capture ask
        res, sc = self._run("L1", "", "", "", "", "", "", "", "y")
        self.assertEqual(levels.current_rung(self.s)[0], "L1")
        self.assertEqual(capture_mode(self.s), "explicit")  # L2 off
        self.assertNotIn("capture style", sc.text())

    def test_set_budget_then_policy(self):
        res, _ = self._run("", "", "", "", "", "", "1000", "freeze", "", "", "", "y")
        self.assertEqual(get_config(self.s, "disk_budget_mb"), "1000")
        self.assertEqual(get_config(self.s, "budget_policy"), "freeze")

    def test_reprompt_on_invalid_then_accept(self):
        # invalid rung 'L9' re-prompts, then 'L2' accepted; L2 still asks
        # capture-style: rung(x2), capture, session, vectors, ingest, budget,
        # floor-log, tools, confirm
        res, sc = self._run("L9", "L2", "", "", "", "", "", "", "", "", "y")
        self.assertEqual(levels.current_rung(self.s)[0], "L2")
        self.assertIn("not one of", sc.text())

    def test_tools_minimal_disables_optionals(self):
        from fornixdb.adapters.mcp_server import tools_disabled
        res, _ = self._run("", "", "", "", "", "", "", "", "", "minimal", "y")
        self.assertTrue(res["applied"])              # tool:* entries applied
        disabled = set(tools_disabled(self.s))
        self.assertTrue(disabled)                    # some optional tools off
        self.assertNotIn("recall_memory", disabled)  # core never disabled

    def test_enable_floor_log(self):
        # keep rung/dissent/capture/session/vectors/ingest/budget, turn floor-log
        # ON, keep tools, confirm
        res, _ = self._run("", "", "", "", "", "", "", "on", "", "", "y")
        self.assertIn("floor_log", res["applied"])
        self.assertEqual(get_config(self.s, "floor_log"), "on")

    def test_set_transcripts_path_reprompts_on_bad_dir(self):
        # keep rung/dissent/capture/session/vectors/ingest/budget/floor-log,
        # then a nonexistent dir re-prompts, a real one is accepted; keep tools
        with tempfile.TemporaryDirectory() as td:
            res, sc = self._run("", "", "", "", "", "", "", "",
                                "/no/such/dir/zzz", td, "", "y")
            self.assertIn("transcripts_path", res["applied"])
            self.assertEqual(get_config(self.s, "transcripts_path"), td)
            self.assertIn("not an existing directory", sc.text())
            self.assertIn("ids collide", sc.text())  # the cross-store caution

    def test_transcripts_path_off_clears(self):
        with tempfile.TemporaryDirectory() as td:
            set_config(self.s, "transcripts_path", td)
            res, _ = self._run("", "", "", "", "", "", "", "", "off", "", "y")
            self.assertIn("transcripts_path", res["applied"])
            self.assertEqual(get_config(self.s, "transcripts_path") or "", "")

    def test_frozen_decline_leaves_store_frozen(self):
        set_config(self.s, "frozen", "on")
        res, _ = self._run("n")  # decline unfreeze
        self.assertTrue(res["aborted"])
        self.assertTrue(self.s.frozen())

    def test_frozen_accept_unfreezes_then_runs(self):
        set_config(self.s, "frozen", "on")
        res, _ = self._run("y", "", "", "", "", "", "", "", "", "", "")  # unfreeze, keep all
        self.assertFalse(self.s.frozen())
        self.assertFalse(res["aborted"])


class ResolveConfigureStoreCase(unittest.TestCase):
    """`fornix-config` runs with no args; resolve_configure_store picks which
    registered store to configure (asking only when there is more than one)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.artist = self.dir / "fornix-artist.db"
        self.memory = self.dir / "fornix-memory.db"
        for p in (self.artist, self.memory):
            MemoryStore(db_path=str(p)).close()  # materialize real store files
        self.reg = self.dir / "fornix-stores.json"
        self._env = {k: os.environ.get(k) for k in
                     ("FORNIXDB_REGISTRY", "FORNIXDB_SHARED_DB", "FORNIXDB_DB")}
        os.environ["FORNIXDB_REGISTRY"] = str(self.reg)
        # a shared-tier path the registry will list but the picker must exclude
        os.environ["FORNIXDB_SHARED_DB"] = str(self.dir / "fornix-shared.db")
        os.environ.pop("FORNIXDB_DB", None)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def _write_registry(self, *paths):
        self.reg.write_text(json.dumps([str(p) for p in paths]))

    def test_explicit_db_is_respected(self):
        self._write_registry(self.artist, self.memory)
        args = SimpleNamespace(db="chosen.db", cmd="configure")
        cli.resolve_configure_store(args, ask=self._fail_ask, out=lambda *_: None)
        self.assertEqual(args.db, "chosen.db")

    def test_single_store_auto_selected_without_prompt(self):
        self._write_registry(self.memory)
        args = SimpleNamespace(db=None, cmd="configure")
        cli.resolve_configure_store(args, ask=self._fail_ask, out=lambda *_: None)
        self.assertEqual(Path(args.db), self.memory)

    def test_shared_tier_excluded_from_candidates(self):
        shared = Path(os.environ["FORNIXDB_SHARED_DB"])
        MemoryStore(db_path=str(shared)).close()
        self._write_registry(self.memory, shared)  # only memory is a consumer store
        args = SimpleNamespace(db=None, cmd="configure")
        cli.resolve_configure_store(args, ask=self._fail_ask, out=lambda *_: None)
        self.assertEqual(Path(args.db), self.memory)

    def test_multiple_stores_prompts_and_picks(self):
        self._write_registry(self.artist, self.memory)  # 1) artist 2) memory
        args = SimpleNamespace(db=None, cmd="configure")
        cli.resolve_configure_store(args, ask=lambda *_: "2", out=lambda *_: None)
        self.assertEqual(Path(args.db), self.memory)

    def test_no_registry_leaves_db_none(self):
        os.environ.pop("FORNIXDB_REGISTRY", None)
        os.environ["FORNIXDB_REGISTRY"] = str(self.dir / "absent.json")
        args = SimpleNamespace(db=None, cmd="configure")
        cli.resolve_configure_store(args, ask=self._fail_ask, out=lambda *_: None)
        self.assertIsNone(args.db)

    @staticmethod
    def _fail_ask(*_a, **_k):
        raise AssertionError("should not prompt")


if __name__ == "__main__":
    unittest.main()
