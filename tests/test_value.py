"""The one-shot value summary: cost always present; reach + used are optional
(so it answers on any store, air-gapped or without the host's flat memory)."""
import json
import os
import pathlib
import tempfile
import unittest

os.environ["FORNIXDB_VECTORS"] = "off"

from fornixdb import value
from fornixdb.core import MemoryStore
from fornixdb.db import connect


class TestValue(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))
        self.s.store("a fact about deploy config", name="m")

    def tearDown(self):
        self.s.close()

    def test_cost_always_present_optional_skipped(self):
        r = value.report(self.s, transcripts=None)
        self.assertEqual(r["memories"], 1)
        self.assertIn("total_tokens", r["cost"]["fixed_per_session"])
        self.assertNotIn("used", r)   # no transcripts -> skipped
        self.assertNotIn("reach", r)  # no flat-memory paths -> skipped

    def test_format_runs_without_optional(self):
        out = value.format_report(value.report(self.s, transcripts=None))
        self.assertIn("How useful has FornixDB been?", out)
        self.assertIn("COST", out)
        self.assertIn("not measured", out)  # reach fallback line
        # net verdict is still the FIRST line — honest "unknown" without data
        self.assertTrue(out.splitlines()[0].startswith(
            "Estimated net tokens: unknown"))

    def test_used_signal_from_transcripts(self):
        with tempfile.TemporaryDirectory() as d:
            lines = [
                {"type": "attachment", "attachment": {
                    "hookEvent": "UserPromptSubmit",
                    "content": "possibly-relevant past\n#1 gist"}},
                {"type": "assistant", "message": {
                    "content": [{"type": "text", "text": "per #1 do the thing"}]}},
            ]
            pathlib.Path(d, "s.jsonl").write_text(
                "\n".join(json.dumps(x) for x in lines), encoding="utf-8")
            r = value.report(self.s, transcripts=d)
            self.assertIn("used", r)
            self.assertEqual(r["used"]["referenced"], 1)
            self.assertIn("USED", value.format_report(r))

    def _transcript_dir(self, d, cite=True):
        lines = [
            {"type": "attachment", "attachment": {
                "hookEvent": "UserPromptSubmit",
                "content": "possibly-relevant past\n#1 gist"}},
        ]
        if cite:
            lines.append({"type": "assistant", "message": {
                "content": [{"type": "text", "text": "per #1 do the thing"}]}})
        pathlib.Path(d, "s.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    def test_net_verdict_is_first_line_with_breakdown(self):
        with tempfile.TemporaryDirectory() as d:
            self._transcript_dir(d, cite=True)
            r = value.report(self.s, transcripts=d)
            self.assertIn("net", r)
            n = r["net"]
            cost = n["measured_cost_per_session"]["total"]
            # 1 referenced push/session at the assumed band, minus measured cost
            for k, v in value.REDERIVE_TOKENS.items():
                self.assertEqual(n["net_tokens_per_session"][k], v - cost)
            out = value.format_report(r)
            first = out.splitlines()[0]
            self.assertTrue(first.startswith("Estimated tokens SAVED")
                            or first.startswith("Estimated EXTRA tokens"))
            self.assertIn("Supporting data", out)
            self.assertIn("ASSUMED", out)          # the band is printed, not hidden
            self.assertIn("not counted", out)

    def test_net_extra_when_nothing_referenced(self):
        with tempfile.TemporaryDirectory() as d:
            self._transcript_dir(d, cite=False)
            r = value.report(self.s, transcripts=d)
            # no referenced pushes -> pure cost -> EXTRA verdict, negative net
            self.assertLess(r["net"]["net_tokens_per_session"]["high"], 0)
            self.assertTrue(value.format_report(r).splitlines()[0]
                            .startswith("Estimated EXTRA tokens"))

    def test_logging_reminder_tracks_floor_log_config(self):
        out = value.format_report(value.report(self.s, transcripts=None))
        self.assertIn("Logging is OFF", out)       # fresh store defaults off
        from fornixdb.multistore import set_config
        set_config(self.s, "floor_log", "on")
        out = value.format_report(value.report(self.s, transcripts=None))
        self.assertIn("Logging is ON", out)


if __name__ == "__main__":
    unittest.main()
