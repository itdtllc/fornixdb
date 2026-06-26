import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic

from fornixdb.core import MemoryStore
from fornixdb.floor_stats import (load_records, outcomes_from_store,
                                  recommend_floor, summarize)


def _rec(**kw):
    base = {"channel": "L3", "id": 1, "kind": "semantic", "vec_cos": 0.5,
            "eff_floor": 0.45, "base_floor": 0.45, "margin": 0.05,
            "decision": "surfaced", "gist": "g", "query": "q"}
    base.update(kw)
    return base


class TestLoadAndSummarize(unittest.TestCase):
    def test_load_skips_blank_and_corrupt_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "floor_log.jsonl"
            p.write_text(json.dumps(_rec()) + "\n\n"
                         + "{not json}\n"
                         + json.dumps(_rec(id=2)) + "\n", encoding="utf-8")
            recs = load_records(p)
            self.assertEqual([r["id"] for r in recs], [1, 2])

    def test_load_missing_file_is_empty(self):
        self.assertEqual(load_records("/no/such/floor_log.jsonl"), [])
        self.assertEqual(load_records(None), [])

    def test_summarize_counts_and_distributions(self):
        recs = [
            _rec(id=1, decision="surfaced", vec_cos=0.55, eff_floor=0.40),  # lowered
            _rec(id=2, decision="surfaced", vec_cos=0.46, eff_floor=0.45),  # unchanged
            _rec(id=3, decision="below_floor", vec_cos=0.30, eff_floor=0.55,  # raised
                 margin=-0.25, channel="L4"),
        ]
        s = summarize(recs)
        self.assertEqual(s["records"], 3)
        self.assertEqual(s["by_decision"], {"surfaced": 2, "below_floor": 1})
        self.assertEqual(s["by_channel"], {"L3": 2, "L4": 1})
        self.assertEqual(s["dial_activity"],
                         {"raised": 1, "lowered": 1, "unchanged": 1})
        self.assertEqual(s["surfaced_cosine"]["n"], 2)
        self.assertEqual(s["surfaced_cosine"]["max"], 0.55)
        self.assertEqual(s["below_floor_cosine"]["n"], 1)
        self.assertEqual(s["top_surfaced_ids"][0], {"id": 1, "times": 1})

    def test_summarize_without_outcomes_has_no_recommendation(self):
        self.assertNotIn("recommendation", summarize([_rec()]))


class TestRecommendFloor(unittest.TestCase):
    def test_clean_separation_suggests_floor_in_the_gap(self):
        rec = recommend_floor(useful_cos=[0.50, 0.60], noise_cos=[0.30, 0.40])
        self.assertEqual(rec["verdict"], "clean_separation")
        self.assertTrue(0.40 < rec["suggested_floor"] < 0.50)
        self.assertEqual(rec["drops_noise"], 2)

    def test_overlap_reports_useful_cost(self):
        rec = recommend_floor(useful_cos=[0.35, 0.50], noise_cos=[0.30, 0.45])
        self.assertEqual(rec["verdict"], "overlap_no_lossless_floor")
        # 0.35 <= noise_max(0.45) -> raising to noise_max costs that 1 useful row
        self.assertEqual(rec["floor_at_noise_max_drops_useful"], 1)

    def test_no_noise_is_insufficient_evidence(self):
        self.assertEqual(recommend_floor([0.5], [])["verdict"],
                         "insufficient_evidence")

    def test_only_noise_is_raise_safe(self):
        rec = recommend_floor([], [0.30, 0.42])
        self.assertEqual(rec["verdict"], "raise_safe")
        self.assertGreater(rec["suggested_floor"], 0.42)


class TestOutcomesFromStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = MemoryStore(db_path=Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_use_outcomes_classify_useful_noise_unknown(self):
        useful = self.s.store("a helpful fact", "a helpful fact", kind="semantic")
        noise = self.s.store("a pushed but ignored fact", "x", kind="semantic")
        unknown = self.s.store("a brand new fact", "x", kind="semantic")
        self.s.mark_helpful(useful)               # helpful_count > 0 -> useful
        self.s.record_surfaced([noise])           # surfaced, never used -> noise
        out = outcomes_from_store(self.s, [useful, noise, unknown])
        self.assertEqual(out[useful], "useful")
        self.assertEqual(out[noise], "noise")
        self.assertEqual(out[unknown], "unknown")

    def test_empty_ids(self):
        self.assertEqual(outcomes_from_store(self.s, []), {})

    def test_summarize_with_outcomes_adds_recommendation(self):
        recs = [_rec(id=10, vec_cos=0.55), _rec(id=20, vec_cos=0.35)]
        s = summarize(recs, outcomes={10: "useful", 20: "noise"})
        self.assertIn("outcome", s)
        self.assertEqual(s["recommendation"]["verdict"], "clean_separation")


if __name__ == "__main__":
    unittest.main()
