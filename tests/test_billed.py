import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"

from fornixdb import billed


def _usage_line(ctx_tokens: int, content=None) -> str:
    return json.dumps({"type": "assistant", "message": {
        "usage": {"input_tokens": 100, "cache_read_input_tokens": ctx_tokens - 100,
                  "cache_creation_input_tokens": 0},
        "content": content or []}})


def _push_line(text: str, hook_event: str = "UserPromptSubmit") -> str:
    return json.dumps({"type": "attachment", "attachment": {
        "type": "hook_success", "hookEvent": hook_event, "content": text}})


class TestAnalyzeTranscript(unittest.TestCase):
    def _write(self, lines) -> Path:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        f.write("\n".join(lines))
        f.close()
        self.addCleanup(os.unlink, f.name)
        return Path(f.name)

    def test_no_usage_records_returns_none(self):
        p = self._write([_push_line("[FornixDB · possibly-relevant past]\n#1 x")])
        self.assertIsNone(billed.analyze_transcript(p))

    def test_billed_total_and_request_count(self):
        p = self._write([_usage_line(10_000), _usage_line(20_000)])
        r = billed.analyze_transcript(p)
        self.assertEqual(r["requests"], 2)
        self.assertEqual(r["billed"], 30_000)
        self.assertEqual(r["events"], [])

    def test_push_before_first_request_enters_at_index_zero(self):
        text = "[FornixDB · possibly-relevant past]\n#1 " + "x" * 396
        p = self._write([_push_line(text), _usage_line(10_000), _usage_line(10_000)])
        r = billed.analyze_transcript(p)
        (idx, tok, kind) = r["events"][0]
        self.assertEqual(idx, 0)
        self.assertEqual(tok, round(len(text) / 4))
        self.assertEqual(kind, "push:L3")
        # re-read by both requests
        self.assertEqual(billed.token_turns(r["events"], r["requests"]), tok * 2)

    def test_push_channels(self):
        base = "[FornixDB · possibly-relevant past]\n"
        p = self._write([
            _push_line(base + "#1 a", "UserPromptSubmit"),
            _push_line(base + "#2 b", "PostToolUse"),
            _push_line(base + "settled: x\n#3 c", "PostToolUse"),
            _usage_line(10_000),
        ])
        kinds = [k for _, _, k in billed.analyze_transcript(p)["events"]]
        self.assertEqual(kinds, ["push:L3", "push:L4", "push:L5"])

    def test_non_fornixdb_attachments_and_tools_ignored(self):
        other_call = [{"type": "tool_use", "id": "t1", "name": "Read",
                       "input": {"file_path": "/x"}}]
        other_result = json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "this text mentions fornixdb but is a file read"}]}})
        p = self._write([
            _usage_line(10_000, other_call), other_result,
            json.dumps({"type": "attachment", "attachment": {
                "type": "hook_success", "hookEvent": "UserPromptSubmit",
                "content": "some other hook mentioning FornixDB by name"}}),
            _usage_line(10_000),
        ])
        self.assertEqual(billed.analyze_transcript(p)["events"], [])

    def test_mcp_call_and_result_attributed_by_id(self):
        call = [{"type": "tool_use", "id": "m1", "name": "mcp__fornixdb__recall_memory",
                 "input": {"query": "q" * 100}}]
        result = json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "m1", "content": "r" * 400}]}})
        p = self._write([_usage_line(10_000, call), result, _usage_line(10_000)])
        r = billed.analyze_transcript(p)
        kinds = {k: tok for _, tok, k in r["events"]}
        self.assertIn("call", kinds)
        self.assertIn("result", kinds)
        self.assertEqual(kinds["result"], 100)  # 400 chars / 4
        # call entered after request 1 (idx 1), result too: each re-read once
        self.assertEqual(billed.token_turns(r["events"], 2),
                         kinds["call"] + kinds["result"])

    def test_event_after_last_request_costs_nothing(self):
        text = "[FornixDB · possibly-relevant past]\n#9 tail"
        p = self._write([_usage_line(10_000), _push_line(text)])
        r = billed.analyze_transcript(p)
        self.assertEqual(billed.token_turns(r["events"], r["requests"]), 0)


if __name__ == "__main__":
    unittest.main()
