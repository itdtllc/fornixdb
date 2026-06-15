"""Marginal-recall benefit harness (FornixDB #190): does FornixDB hold what the
flat MEMORY.md has already lost? The harness must classify honestly — overlap
with the seed files is NOT benefit."""

import json
import tempfile
import unittest
from pathlib import Path

from fornixdb.benefit import (classify, coverage, golden_marginal,
                              scan_flat_baseline)
from fornixdb.core import MemoryStore
from fornixdb.db import connect


class TestBenefit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.s = MemoryStore(conn=connect(":memory:"))
        # a flat system: two topic files; MEMORY.md indexes both, but a tiny
        # cap truncates so only the FIRST entry "loads" at session start
        (self.dir / "alpha_fact.md").write_text("alpha body", encoding="utf-8")
        (self.dir / "beta_fact.md").write_text("beta body", encoding="utf-8")
        (self.dir / "MEMORY.md").write_text(
            "- [Alpha](alpha_fact.md) hook\n"
            "- [Beta](beta_fact.md) hook\n", encoding="utf-8")
        # cap that keeps only the Alpha line in the loaded slice
        self.baseline = scan_flat_baseline(
            self.dir / "MEMORY.md", self.dir, cap_chars=30)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_classify_three_buckets(self):
        in_flat = {"name": "alpha_fact", "kind": "semantic"}
        on_disk = {"name": "beta_fact", "kind": "semantic"}     # past the cap
        native = {"name": "git-squash-rule", "kind": "semantic"}  # no file
        episodic = {"name": None, "kind": "episodic"}
        self.assertEqual(classify(in_flat, self.baseline), "in_flat")
        self.assertEqual(classify(on_disk, self.baseline), "on_disk_only")
        self.assertEqual(classify(native, self.baseline), "fornix_only")
        self.assertEqual(classify(episodic, self.baseline), "fornix_only")

    def test_truncation_detected(self):
        self.assertTrue(self.baseline["truncated"])             # full > cap
        wide = scan_flat_baseline(self.dir / "MEMORY.md", self.dir,
                                  cap_chars=10_000)
        self.assertFalse(wide["truncated"])

    def test_coverage_counts_and_deltas(self):
        self.s.store("alpha body", name="alpha_fact")           # in_flat
        self.s.store("beta body", name="beta_fact")             # on_disk_only
        self.s.store("the git squash rule", name="git-squash-rule")  # fornix_only
        self.s.store("session log", kind="episodic")            # fornix_only
        cov = coverage(self.s, self.baseline)
        self.assertEqual(cov["total"], 4)
        self.assertEqual(cov["buckets"],
                         {"in_flat": 1, "on_disk_only": 1, "fornix_only": 2})
        self.assertEqual(cov["marginal_at_startup"], 3)   # all but in_flat
        self.assertEqual(cov["marginal_content"], 2)      # the fornix_only pair
        self.assertEqual(cov["by_kind"]["episodic"]["fornix_only"], 1)

    def test_golden_marginal_counts_prevented_surprises(self):
        beta = self.s.store("beta body about quarterly filing", name="beta_fact")
        native = self.s.store("the deploy runbook lives in fornix only",
                              name="deploy-runbook")
        golden = self.dir / "g.jsonl"
        golden.write_text("\n".join(json.dumps(c) for c in [
            {"query": "quarterly filing", "expect": ["beta_fact"]},
            {"query": "deploy runbook", "expect": ["deploy-runbook"]},
        ]), encoding="utf-8")
        g = golden_marginal(self.s, self.baseline, golden, embedder=False)
        self.assertEqual(g["answered"], 2)
        self.assertEqual(g["startup_marginal"], 2)   # neither is free at startup
        self.assertEqual(g["content_marginal"], 1)   # only the runbook is fornix_only


if __name__ == "__main__":
    unittest.main()
