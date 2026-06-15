"""End-to-end A/B answer harness (FornixDB #4): does recall improve the AI's
ANSWER, not just its ranking? The machinery is tested with a FAKE answerer so
scoring is verified without the Claude API — the API is the default answerer in
production, injected here."""

import json
import tempfile
import unittest
from pathlib import Path

from fornixdb.answer_eval import (fact_present, load_answer_golden, run,
                                  run_case)
from fornixdb.core import MemoryStore
from fornixdb.db import connect


def context_answerer(question, context):
    """A model that perfectly uses recall: it echoes the context (so a fact in
    the recalled notes lands in the answer) and otherwise knows nothing."""
    return context or "I don't know."


class TestAnswerEval(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))

    def tearDown(self):
        self.s.close()

    def test_fact_present_all_and_any(self):
        self.assertTrue(fact_present("the port is 8188", ["8188"]))
        self.assertTrue(fact_present("PORT 8188", ["8188"]))   # case-insensitive
        self.assertFalse(fact_present("the port is 9000", ["8188"]))
        self.assertTrue(fact_present("a and c", ["a", "z"], match="any"))
        self.assertFalse(fact_present("a and c", ["a", "z"], match="all"))

    def test_lift_when_recall_supplies_the_fact(self):
        self.s.store("The relay listens on port 8188", name="relay-port")
        r = run_case(self.s, {"query": "relay port", "answer_contains": ["8188"]},
                     context_answerer, embedder=False)
        self.assertTrue(r["recalled"])
        self.assertFalse(r["a_has"])   # parametric path (no context) can't know
        self.assertTrue(r["b_has"])    # recall supplied it
        self.assertTrue(r["lift"])

    def test_both_miss_when_recall_surfaces_wrong_memory(self):
        # the only memory is unrelated, so recall context never carries the fact
        self.s.store("Pizza dough rests for 24 hours", name="dough")
        r = run_case(self.s, {"query": "relay port", "answer_contains": ["8188"]},
                     context_answerer, embedder=False)
        self.assertFalse(r["lift"])
        self.assertFalse(r["b_has"])

    def test_parametric_already_knows_is_not_lift(self):
        # answerer knows the fact even with no context → no marginal value
        def knows_everything(question, context):
            return "the answer is 8188"
        self.s.store("The relay listens on port 8188", name="relay-port")
        r = run_case(self.s, {"query": "relay port", "answer_contains": ["8188"]},
                     knows_everything, embedder=False)
        self.assertTrue(r["a_has"])
        self.assertTrue(r["b_has"])
        self.assertFalse(r["lift"])    # B was right, but so was A

    def test_regression_is_flagged(self):
        # A knows; B (given context) loses the fact → recall hurt
        def forgets_with_context(question, context):
            return "I don't know." if context else "the answer is 8188"
        self.s.store("The relay listens on port 8188", name="relay-port")
        r = run_case(self.s, {"query": "relay port", "answer_contains": ["8188"]},
                     forgets_with_context, embedder=False)
        self.assertTrue(r["regressed"])
        self.assertFalse(r["lift"])

    def test_run_aggregates_and_records(self):
        self.s.store("The relay listens on port 8188", name="relay-port")
        self.s.store("Atlas runs the Llama-70B model on the server", name="atlas")
        with tempfile.TemporaryDirectory() as d:
            golden = Path(d) / "answers.jsonl"
            golden.write_text("\n".join(json.dumps(c) for c in [
                {"query": "relay port", "answer_contains": ["8188"]},
                {"query": "what model is Atlas", "answer_contains": ["Llama-70B"]},
            ]), encoding="utf-8")
            report = run(self.s, golden, context_answerer, embedder=False)
            self.assertEqual(report["cases"], 2)
            self.assertEqual(report["lift_count"], 2)   # recall earned both
            self.assertEqual(report["lift"], 1.0)

            from fornixdb.answer_eval import record_run
            hist = Path(d) / "hist.jsonl"
            rec = record_run(report, hist, store=self.s)
            self.assertEqual(rec["lift"], 1.0)
            self.assertEqual(rec["store_memories"], 2)
            self.assertEqual(len(hist.read_text().strip().splitlines()), 1)

    def test_load_golden_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.jsonl"
            bad.write_text(json.dumps({"query": "no facts here"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_answer_golden(bad)


if __name__ == "__main__":
    unittest.main()
