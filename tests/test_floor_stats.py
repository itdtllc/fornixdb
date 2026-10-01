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


class TestPullChannel(unittest.TestCase):
    """Pull decisions (L1) go to the same log, and readers keep push and pull
    apart: they sit on different floors."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.log = Path(self.tmp.name) / "floor_log.jsonl"
        self.s = MemoryStore(db_path=self.db)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def _logged(self):
        return load_records(self.log)

    def test_channels_for(self):
        from fornixdb.floor_stats import channels_for
        self.assertEqual(channels_for(None), ("L3", "L4", "L5", "?"))
        self.assertEqual(channels_for("push"), ("L3", "L4", "L5", "?"))
        self.assertEqual(channels_for("pull"), ("L1",))
        self.assertIsNone(channels_for("all"))
        self.assertEqual(channels_for("L1, L3"), ("L1", "L3"))

    def test_load_filters_by_channel(self):
        from fornixdb.floor_stats import PUSH_CHANNELS
        untagged = _rec(id=3)
        del untagged["channel"]          # pre-channel records were all pushes
        self.log.write_text("\n".join(json.dumps(r) for r in (
            _rec(id=1, channel="L1"), _rec(id=2, channel="L4"), untagged)) + "\n",
            encoding="utf-8")
        self.assertEqual([r["id"] for r in load_records(self.log)], [1, 2, 3])
        self.assertEqual([r["id"] for r in load_records(self.log,
                          channels=PUSH_CHANNELS)], [2, 3])
        self.assertEqual([r["id"] for r in load_records(self.log,
                          channels=("L1",))], [1])

    def test_pull_decisions_classified_against_the_include_floor(self):
        from fornixdb.core import VECTOR_MIN_COS
        from fornixdb.multistore import set_config
        from fornixdb.proactive import log_pull_decisions
        set_config(self.s, "floor_log", "on")
        rows = [
            {"id": 1, "kind": "semantic", "gist": "a", "vec_cos": 0.62, "raw_cos": 0.62},
            {"id": 2, "kind": "semantic", "gist": "b", "vec_cos": 0.0, "raw_cos": 0.21},
            {"id": 9, "kind": "semantic", "gist": "c", "vec_cos": 0.7, "_store": "shared"},
            {"id": 3, "kind": "episodic", "gist": "d"},      # keyword-only store row
        ]
        log_pull_decisions(self.s, "the query", rows)
        recs = self._logged()
        self.assertEqual([r["id"] for r in recs], [1, 2, 3])   # peer row skipped
        self.assertTrue(all(r["channel"] == "L1" for r in recs))
        self.assertEqual([r["decision"] for r in recs],
                         ["surfaced", "keyword_anchor", "keyword_anchor"])
        self.assertEqual([r["rank"] for r in recs], [1, 2, 4])
        self.assertEqual(recs[0]["base_floor"], VECTOR_MIN_COS)
        # the UNFLOORED cosine, so a keyword anchor shows how far under it sat
        self.assertEqual(recs[1]["vec_cos"], 0.21)
        self.assertLess(recs[1]["margin"], 0)
        self.assertIsNone(recs[2]["vec_cos"])

    def test_abstained_rows_are_marked(self):
        from fornixdb.multistore import set_config
        from fornixdb.proactive import log_pull_decisions
        set_config(self.s, "floor_log", "on")
        log_pull_decisions(self.s, "q", [{"id": 1, "vec_cos": 0.4, "raw_cos": 0.4}],
                           abstained=True)
        self.assertEqual(self._logged()[0]["decision"], "abstained")

    def test_pull_logging_is_noop_when_off(self):
        from fornixdb.proactive import log_pull_decisions
        log_pull_decisions(self.s, "q", [{"id": 1, "vec_cos": 0.6}])
        self.assertFalse(self.log.exists())

    def test_cli_recall_logs_and_floor_stats_reads_channels_apart(self):
        import contextlib
        import io
        from fornixdb.cli import main
        from fornixdb.multistore import set_config
        mid = self.s.store("the deploy script reads its config from env",
                           "detail", kind="semantic")
        set_config(self.s, "floor_log", "on")
        # one old push record beside the new pull
        self.log.write_text(json.dumps(_rec(id=77, channel="L4")) + "\n",
                            encoding="utf-8")
        base = ["--db", str(self.db), "--no-shared"]
        with contextlib.redirect_stdout(io.StringIO()):
            main(base + ["recall", "deploy script config"])
        pulls = [r for r in self._logged() if r["channel"] == "L1"]
        self.assertEqual([r["id"] for r in pulls], [mid])
        self.assertEqual(pulls[0]["query"], "deploy script config")

        def stats(*extra):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main(base + ["--json", "floor-stats", *extra])
            return json.loads(buf.getvalue())
        self.assertEqual(stats()["by_channel"], {"L4": 1})          # push default
        pull = stats("--channel", "pull")
        self.assertEqual(pull["by_channel"], {"L1": 1})
        # no transcripts: recall_count would label every pull useful, so no join
        self.assertNotIn("outcome", pull)
        self.assertIn("--transcripts", pull["outcome_note"])
        self.assertEqual(stats("--channel", "all")["records"], 2)


class TestPullOutcomesFromScan(unittest.TestCase):
    def test_pull_flag_reads_pull_counts(self):
        from fornixdb.usefulness_scan import outcomes_from_scan
        scan = {"per_memory": {
            1: {"impressions": 2, "referenced": 0,
                "pull_impressions": 1, "pull_referenced": 1},
            2: {"impressions": 0, "referenced": 0,
                "pull_impressions": 3, "pull_referenced": 0},
            3: {"impressions": 1, "referenced": 1,
                "pull_impressions": 0, "pull_referenced": 0},
        }}
        self.assertEqual(outcomes_from_scan(scan), {1: "noise", 3: "useful"})
        self.assertEqual(outcomes_from_scan(scan, pull=True),
                         {1: "useful", 2: "noise"})
