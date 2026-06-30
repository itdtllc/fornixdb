"""Claude Code L4 cadence adapter — the tool-call-seam metronome hook.

No model: FORNIXDB_VECTORS=off makes recall deterministic keyword matching, so a
literal token anchor in the seeded memory is what a pulse fires on.
"""

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"

from fornixdb.adapters import claude_code_cadence as cc
from fornixdb.adapters.native_memory import set_ingest_mode
from fornixdb.core import MemoryStore
from fornixdb.multistore import set_config


def file_store(tmp):
    return MemoryStore(db_path=Path(tmp) / "t.db")


class TestBuildThought(unittest.TestCase):
    def test_post_includes_response(self):
        t = cc.build_thought("PostToolUse", "Bash",
                             {"command": "grep foo"}, "matched in bar.py")
        self.assertIn("Bash", t)
        self.assertIn("grep foo", t)
        self.assertIn("matched in bar.py", t)

    def test_pre_omits_response(self):
        t = cc.build_thought("PreToolUse", "Bash",
                             {"command": "grep foo"}, "should be ignored")
        self.assertIn("grep foo", t)
        self.assertNotIn("should be ignored", t)

    def test_bounded(self):
        big = {"x": "a" * 5000}
        t = cc.build_thought("PostToolUse", "Write", big, "b" * 5000)
        # tool name + two bounded slices; nowhere near the raw 10k
        self.assertLess(len(t), cc.THOUGHT_CHARS * 2 + 50)

    def test_fornixdb_own_tool_yields_no_thought(self):
        # A memory write/read self-matches the memory it touches — never pulse.
        for name in ("mcp__fornixdb__remember", "mcp__fornixdb__show_memory",
                     "mcp__fornixdb__recall_memory", "mcp__fornixdb__supersede"):
            self.assertEqual(
                cc.build_thought("PostToolUse", name,
                                 {"ref": "407"}, "#407 some memory text"), "",
                f"{name} should produce no pulse thought")

    def test_renamed_server_still_caught_by_basename(self):
        # Robust to a host that named the MCP server something other than fornixdb.
        self.assertEqual(
            cc.build_thought("PostToolUse", "mcp__memory__remember",
                             {"title": "x", "content": "y"}, "stored #99"), "")

    def test_non_fornixdb_tool_still_pulses(self):
        # A like-named tool on a non-MCP path, or any ordinary tool, is unaffected.
        self.assertNotEqual(
            cc.build_thought("PostToolUse", "Bash",
                             {"command": "remember the milk"}, "ok"), "")
        self.assertNotEqual(
            cc.build_thought("PostToolUse", "Read",
                             {"file_path": "/x/link.py"}, "contents"), "")


class TestTurnAndEpisode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_bump_turn_increments(self):
        self.assertEqual(cc._current_turn(self.s, "S"), "0")
        cc.bump_turn(self.s, "S")
        self.assertEqual(cc._current_turn(self.s, "S"), "1")
        cc.bump_turn(self.s, "S")
        self.assertEqual(cc._current_turn(self.s, "S"), "2")

    def test_round_trip_same_turn(self):
        now = time.time()
        ep = cc.load_episode(self.s, "S", now)
        ep.pulsed_ids.update({3, 7})
        ep.last_query = "grep foo"
        ep.pulse_count = 2
        cc.save_episode(self.s, "S", ep, now)
        ep2 = cc.load_episode(self.s, "S", now + 1)
        self.assertEqual(ep2.pulsed_ids, {3, 7})
        self.assertEqual(ep2.last_query, "grep foo")
        self.assertEqual(ep2.pulse_count, 2)

    def test_new_turn_resets_episode(self):
        now = time.time()
        ep = cc.load_episode(self.s, "S", now)
        ep.pulse_count = 3
        ep.pulsed_ids.add(9)
        cc.save_episode(self.s, "S", ep, now)
        cc.bump_turn(self.s, "S")           # the L3 hook fired a new user turn
        ep2 = cc.load_episode(self.s, "S", now + 1)
        self.assertEqual(ep2.pulse_count, 0)
        self.assertEqual(ep2.pulsed_ids, set())

    def test_idle_gap_resets_episode(self):
        now = time.time()
        ep = cc.load_episode(self.s, "S", now)
        ep.pulse_count = 2
        cc.save_episode(self.s, "S", ep, now)
        ep2 = cc.load_episode(self.s, "S", now + cc.IDLE_RESET_SECONDS + 1)
        self.assertEqual(ep2.pulse_count, 0)


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "t.db")
        s = file_store(self.tmp.name)
        set_config(s, "rhythmic_recall", "on")  # L4 is opt-in; enable to test it
        s.store("DEPLOY RULE: always run the migration script before deploy.",
                kind="semantic", name="deploy-migration-rule")
        s.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, payload):
        buf = io.StringIO()
        stdin = io.StringIO(json.dumps(payload))
        old = cc.sys.stdin
        cc.sys.stdin = stdin
        try:
            with redirect_stdout(buf):
                rc = cc.main(["--db", self.db])
        finally:
            cc.sys.stdin = old
        return rc, buf.getvalue()

    def test_relevant_tool_call_injects(self):
        rc, out = self._run({
            "hook_event_name": "PostToolUse", "session_id": "S",
            "tool_name": "Bash",
            "tool_input": {"command": "kubectl deploy"},
            "tool_response": "preparing deploy of service",
        })
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        ctx = payload["hookSpecificOutput"]["additionalContext"]
        self.assertIn("migration", ctx.lower())
        self.assertEqual(payload["hookSpecificOutput"]["hookEventName"],
                         "PostToolUse")

    def test_irrelevant_tool_call_silent(self):
        rc, out = self._run({
            "hook_event_name": "PostToolUse", "session_id": "S",
            "tool_name": "Read",
            "tool_input": {"file_path": "/etc/hosts"},
            "tool_response": "127.0.0.1 localhost",
        })
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "")

    def test_empty_thought_silent(self):
        rc, out = self._run({"hook_event_name": "PostToolUse",
                             "session_id": "S"})
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "")

    def test_explicit_ingest_mode_off(self):
        s = MemoryStore(db_path=self.db)
        set_ingest_mode(s, "explicit")
        s.close()
        rc, out = self._run({
            "hook_event_name": "PostToolUse", "session_id": "S",
            "tool_name": "Bash",
            "tool_input": {"command": "kubectl deploy"},
            "tool_response": "preparing deploy",
        })
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "")

    def test_no_reinject_same_memory_within_turn(self):
        call = {
            "hook_event_name": "PostToolUse", "session_id": "S",
            "tool_name": "Bash",
            "tool_input": {"command": "kubectl deploy prod"},
            "tool_response": "deploy of prod service starting",
        }
        rc1, out1 = self._run(call)
        self.assertTrue(out1.strip())          # first call surfaces the rule
        # a moved-on second tool call in the SAME turn must not re-surface it
        call2 = dict(call, tool_input={"command": "kubectl deploy staging now"},
                     tool_response="deploy of staging service starting")
        rc2, out2 = self._run(call2)
        self.assertEqual(out2.strip(), "")


if __name__ == "__main__":
    unittest.main()
