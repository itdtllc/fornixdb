import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"

from fornixdb import usefulness_scan as us


class TestAttribute(unittest.TestCase):
    def test_push_then_cite_is_referenced(self):
        events = [("push", {36, 99}, "UserPromptSubmit"), ("cite", {36}, None)]
        t, _ = us.attribute(events)
        self.assertEqual(t[36], {"impressions": 1, "referenced": 1})
        self.assertEqual(t[99], {"impressions": 1, "referenced": 0})  # never cited

    def test_cite_before_push_does_not_count(self):
        # a citation with no preceding push (e.g. an explicit show/pull) is no credit
        t, _ = us.attribute([("cite", {7}, None), ("push", {7}, "UserPromptSubmit")])
        self.assertEqual(t[7], {"impressions": 1, "referenced": 0})

    def test_repush_without_citation_is_ignored(self):
        # pushed, pushed again (first injection went unused), then cited once:
        # two impressions, one referenced (only the latest injection is credited)
        t, _ = us.attribute([("push", {5}, "UserPromptSubmit"),
                             ("push", {5}, "PostToolUse"), ("cite", {5}, None)])
        self.assertEqual(t[5], {"impressions": 2, "referenced": 1})

    def test_one_citation_credits_one_injection(self):
        t, _ = us.attribute([("push", {5}, "UserPromptSubmit"),
                             ("cite", {5}, None), ("cite", {5}, None)])
        # second citation has nothing pending -> still just one referenced
        self.assertEqual(t[5], {"impressions": 1, "referenced": 1})

    def test_citation_credited_to_the_injecting_channel(self):
        # L3 pushed and cited; L4 pushed and never cited -> per-channel split
        _, pc = us.attribute([("push", {1}, "UserPromptSubmit"),
                              ("cite", {1}, None),
                              ("push", {2}, "PostToolUse")])
        self.assertEqual(pc["L3"], {"impressions": 1, "referenced": 1})
        self.assertEqual(pc["L4"], {"impressions": 1, "referenced": 0})

    def test_l5_prelabel_survives_channel_normalization(self):
        _, pc = us.attribute([("push", {3}, "L5"), ("cite", {3}, None)])
        self.assertEqual(pc["L5"], {"impressions": 1, "referenced": 1})


class TestParseAndScan(unittest.TestCase):
    def _session_file(self, d, name="s.jsonl"):
        def block(content):
            return {"type": "attachment",
                    "attachment": {"hookEvent": "UserPromptSubmit",
                                   "content": content}}
        def assistant(text):
            return {"type": "assistant",
                    "message": {"content": [{"type": "text", "text": text}]}}
        lines = [
            block("[FornixDB · possibly-relevant past — …]\n"
                  "#36 some gist\n#99 other gist"),
            assistant("Per #36 we should branch first."),   # uses 36, not 99
            {"type": "user", "message": {"content": "ok"}},
        ]
        p = Path(d) / name
        p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
        return p

    def test_iter_events_extracts_block_and_citation(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._session_file(d)
            evs = list(us.iter_events(p))
            # pushes carry a 4th field: the block's measured size in chars
            self.assertEqual(evs[0][:3], ("push", {36, 99}, "UserPromptSubmit"))
            self.assertGreater(evs[0][3], 0)
            self.assertIn(("cite", {36}, None), evs)

    def test_settled_block_is_labeled_l5(self):
        # a SETTLED field block carries its direction line; a degraded field
        # block is L4 behavior and keeps the hook-event label
        with tempfile.TemporaryDirectory() as d:
            lines = [
                {"type": "attachment",
                 "attachment": {"hookEvent": "PostToolUse",
                                "stdout": "[FornixDB · possibly-relevant past — …]\n"
                                          "settled: pool · 2026-06-29 · knowledge+recent\n"
                                          "#12 mortar gist"}},
                {"type": "attachment",
                 "attachment": {"hookEvent": "PostToolUse",
                                "stdout": "[FornixDB · possibly-relevant past — …]\n"
                                          "#13 loner gist"}},
            ]
            p = Path(d) / "s.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
            evs = list(us.iter_events(p))
            self.assertEqual(evs[0][:3], ("push", {12}, "L5"))
            self.assertEqual(evs[1][:3], ("push", {13}, "PostToolUse"))

    def test_scan_aggregates_and_rates(self):
        with tempfile.TemporaryDirectory() as d:
            self._session_file(d)
            s = us.scan(d)
            self.assertEqual(s["sessions"], 1)
            self.assertEqual(s["impressions"], 2)      # 36 + 99
            self.assertEqual(s["referenced"], 1)       # only 36
            self.assertEqual(s["reference_rate"], 0.5)

    def test_scan_measures_injected_block_sizes(self):
        block = ("[FornixDB · possibly-relevant past — …]\n"
                 "#36 some gist\n#99 other gist")
        with tempfile.TemporaryDirectory() as d:
            self._session_file(d)                       # writes the block above
            s = us.scan(d)
            self.assertEqual(s["injected_chars"], len(block))
            self.assertEqual(s["injected_tokens"], round(len(block) / 4))
            # the cost is attributed to the injecting channel (L3 here)
            self.assertEqual(s["by_channel"]["L3"]["injected_tokens"],
                             s["injected_tokens"])

    def test_outcomes_from_scan(self):
        s = {"per_memory": {36: {"impressions": 1, "referenced": 1},
                            99: {"impressions": 2, "referenced": 0},
                            7: {"impressions": 0, "referenced": 0}}}
        out = us.outcomes_from_scan(s)
        self.assertEqual(out[36], "useful")
        self.assertEqual(out[99], "noise")
        self.assertNotIn(7, out)                        # never pushed

    def test_scan_missing_path_is_empty(self):
        s = us.scan("/no/such/dir")
        self.assertEqual(s["impressions"], 0)
        self.assertIn("no injected blocks", us.format_report(s))


class TestReferencedCountsFromScan(unittest.TestCase):
    def test_maps_every_pushed_id_to_its_reference_count(self):
        s = {"per_memory": {36: {"impressions": 3, "referenced": 2},
                            99: {"impressions": 4, "referenced": 0}}}
        # every pushed id is present (0 included) so --apply also resets memories
        # that have gone quiet — an idempotent absolute set.
        self.assertEqual(us.referenced_counts_from_scan(s), {36: 2, 99: 0})

    def test_empty_scan(self):
        self.assertEqual(us.referenced_counts_from_scan({}), {})


if __name__ == "__main__":
    unittest.main()
