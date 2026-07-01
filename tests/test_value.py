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


if __name__ == "__main__":
    unittest.main()
