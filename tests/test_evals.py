"""The eval harness itself must be trustworthy before its numbers are."""

import json
import tempfile
import unittest
from pathlib import Path

from fornixdb.core import MemoryStore
from fornixdb.evals import (format_report, load_golden, load_history,
                            record_run, run, run_case)


class TestEvals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = MemoryStore(db_path=Path(self.tmp.name) / "e.db")
        self.alpha = self.s.store("the quarterly tax filing deadline rule",
                                  "detail", name="tax-deadline")
        self.beta = self.s.store("chart colors are reserved per item", "detail")
        self.gamma = self.s.store("unrelated grocery list", "detail")

    def tearDown(self):
        self.s.close()  # Windows can't delete an open db file
        self.tmp.cleanup()

    def _golden(self, cases) -> Path:
        p = Path(self.tmp.name) / "golden.jsonl"
        p.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
        return p

    def test_hit_by_id_and_by_name(self):
        path = self._golden([
            {"query": "tax filing deadline", "expect": ["tax-deadline"]},
            {"query": "chart colors reserved", "expect": [self.beta]},
        ])
        report = run(self.s, path, embedder=False)
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["hit@1"], 1.0)
        self.assertEqual(report["hit@k"], 1.0)
        self.assertEqual(report["mrr"], 1.0)
        self.assertEqual(report["misses"], [])

    def test_eval_run_does_not_perturb_use_signals(self):
        # the fence must not move what it measures: recall_count feeds the
        # _usefulness ranking bonus and the push floor, and an eval sweep is
        # not genuine use — repeated fence runs were silently inflating it.
        path = self._golden([
            {"query": "tax filing deadline", "expect": ["tax-deadline"]},
        ])
        before = self.s.conn.execute(
            "SELECT recall_count, last_recalled FROM memory WHERE id = ?",
            (self.alpha,)).fetchone()
        run(self.s, path, embedder=False)
        after = self.s.conn.execute(
            "SELECT recall_count, last_recalled FROM memory WHERE id = ?",
            (self.alpha,)).fetchone()
        self.assertEqual(tuple(before), tuple(after))

    def test_miss_is_reported_with_what_ranked(self):
        path = self._golden([
            {"query": "grocery list", "expect": [self.alpha], "note": "wrong on purpose"},
        ])
        report = run(self.s, path, embedder=False)
        self.assertEqual(report["hit@k"], 0.0)
        self.assertEqual(report["mrr"], 0.0)
        miss = report["misses"][0]
        self.assertIn(self.gamma, miss["got"])      # shows what DID rank
        self.assertIn("MISS", format_report(report))

    def test_rank_and_reciprocal_rank(self):
        # both rows match "reserved OR rule"-ish tokens; expect the worse one
        case = {"query": "colors per item", "expect": [self.beta], "k": 5}
        r = run_case(self.s, case, embedder=False)
        self.assertEqual(r["rank"], 1)
        self.assertEqual(r["rr"], 1.0)

    def test_drift_when_rank1_asserted_but_not_first(self):
        # beta wins on token overlap; assert alpha SHOULD be #1 -> silent drift
        case = {"query": "chart colors reserved per item rule",
                "expect": [self.alpha], "rank1": True}
        r = run_case(self.s, case, embedder=False)
        self.assertTrue(r["hitk"])          # still found within k
        self.assertGreater(r["rank"], 1)    # but no longer first
        self.assertTrue(r["drifted"])

    def test_no_drift_when_rank1_asserted_and_first(self):
        case = {"query": "tax filing deadline", "expect": ["tax-deadline"],
                "rank1": True}
        r = run_case(self.s, case, embedder=False)
        self.assertEqual(r["rank"], 1)
        self.assertFalse(r["drifted"])

    def test_no_drift_without_rank1_assertion(self):
        # same losing-rank situation, but the case never claimed rank 1
        case = {"query": "chart colors reserved per item rule",
                "expect": [self.alpha]}
        r = run_case(self.s, case, embedder=False)
        self.assertGreater(r["rank"], 1)
        self.assertFalse(r["drifted"])

    def test_rank1_miss_is_a_miss_not_drift(self):
        # a rank1 case that falls out of top-k entirely is caught by hit@k,
        # not double-counted as drift
        case = {"query": "grocery list", "expect": [self.alpha], "rank1": True}
        r = run_case(self.s, case, embedder=False)
        self.assertFalse(r["hitk"])
        self.assertFalse(r["drifted"])

    def test_drift_aggregated_and_reported(self):
        path = self._golden([
            {"query": "tax filing deadline", "expect": ["tax-deadline"], "rank1": True},
            {"query": "chart colors reserved per item rule",
             "expect": [self.alpha], "rank1": True},
        ])
        report = run(self.s, path, embedder=False)
        self.assertEqual(len(report["drift"]), 1)
        out = format_report(report)            # drift shows even without --verbose
        self.assertIn("DRIFT: 1", out)
        self.assertIn("asserted rank 1, now rank", out)

    def test_record_run_accumulates_history(self):
        # live-store tracking: each recorded run appends, so a precision
        # decline as the store grows is visible across sessions
        path = Path(self.tmp.name) / "eval_history.jsonl"
        rec = record_run({"cases": 2, "hit@1": 1.0, "hit@k": 1.0, "mrr": 1.0,
                          "drift": []}, path, store=self.s)
        self.assertEqual(rec["hit@1"], 1.0)
        self.assertEqual(rec["store_memories"], self.s.stats()["memories"])
        record_run({"cases": 2, "hit@1": 0.5, "hit@k": 1.0, "mrr": 0.75,
                    "drift": [{"q": "x"}]}, path, store=self.s)
        hist = load_history(path)
        self.assertEqual([h["hit@1"] for h in hist], [1.0, 0.5])   # trend visible
        self.assertEqual(hist[1]["drift"], 1)                       # count, not list

    def test_load_history_missing_file_is_empty(self):
        self.assertEqual(load_history(Path(self.tmp.name) / "never.jsonl"), [])

    def test_abstain_case_correct_and_leak(self):
        # negative golden cases (#191): recall SHOULD report nothing relevant.
        path = self._golden([
            {"query": "tax filing deadline", "expect": ["tax-deadline"]},   # positive
            {"query": "zzz nonexistent topic qqq", "expect_abstain": True},  # correct abstain
            {"query": "tax deadline", "expect_abstain": True},               # LEAK (real hit)
        ])
        report = run(self.s, path, embedder=False)
        self.assertEqual(report["cases"], 1)            # only the positive counts here
        self.assertEqual(report["hit@1"], 1.0)
        self.assertEqual(report["abstain_cases"], 2)
        self.assertEqual(report["abstain_correct"], 1)
        self.assertEqual(len(report["abstain_leaks"]), 1)
        out = format_report(report)
        self.assertIn("abstain: 1/2", out)
        self.assertIn("LEAK", out)

    def test_load_golden_accepts_abstain_without_expect(self):
        path = self._golden([{"query": "x", "expect_abstain": True}])
        self.assertEqual(len(load_golden(path)), 1)

    def test_name_follows_supersession(self):
        v2 = self.s.store("tax deadline rule, revised", "detail")
        self.s.supersede(self.alpha, v2)            # name handle moves to v2
        case = {"query": "tax deadline revised", "expect": ["tax-deadline"]}
        r = run_case(self.s, case, embedder=False)
        self.assertEqual(r["expected"], [v2])       # golden tracks the live version
        self.assertTrue(r["hitk"])

    def test_unresolved_expectation_warns_not_crashes(self):
        case = {"query": "tax filing", "expect": ["no-such-name", self.alpha]}
        r = run_case(self.s, case, embedder=False)
        self.assertEqual(r["unresolved"], ["no-such-name"])
        self.assertTrue(r["hitk"])                  # the resolvable one still scores

    def test_wrong_store_warns(self):
        path = self._golden([
            {"query": "anything", "expect": ["a-name-from-another-store"]},
        ])
        report = run(self.s, path, embedder=False)
        self.assertIn("wrong store?", format_report(report))

    def test_golden_validation(self):
        path = self._golden([{"query": "x"}])       # no expect
        with self.assertRaises(ValueError):
            load_golden(path)
        # comments and blank lines are fine
        p = Path(self.tmp.name) / "g2.jsonl"
        p.write_text("# comment\n\n" + json.dumps(
            {"query": "tax", "expect": [self.alpha]}), encoding="utf-8")
        self.assertEqual(len(load_golden(p)), 1)


if __name__ == "__main__":
    unittest.main()
