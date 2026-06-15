"""Negative feedback (owner decisions 2026-06-12): explicit-only signal,
query-conditional penalty. A memory marked irrelevant for a query is
downweighted for similar queries, fully ranked for everything else, and
never hidden or deleted."""

import unittest
import zlib

from fornixdb.core import MemoryStore, FrozenStoreError
from fornixdb.db import connect
from fornixdb.multistore import set_config
from fornixdb.vectors import backfill

DIM = 64
SYNONYMS = {"glitch": "artifact", "bug": "artifact"}


class FakeEmbedder:
    name = "fake:onehot"

    def embed(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * DIM
            for tok in t.lower().split():
                tok = "".join(c for c in tok if c.isalnum())
                if tok:
                    vec[zlib.crc32(SYNONYMS.get(tok, tok).encode()) % DIM] += 1.0
            out.append(vec)
        return out


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestMarkAndPenalty(unittest.TestCase):
    """Keyword-only path: works with no model installed."""

    def setUp(self):
        self.s = mem_store()
        # the wrong-but-dominant hit (high salience) vs the right answer
        self.bad = self.s.store("Estimator video pipeline notes", salience=1.0)
        self.good = self.s.store("Estimator video pipeline final design", salience=0.2)

    def recall_ids(self, query):
        return [r["id"] for r in self.s.recall(query, embedder=False)]

    def test_same_query_downweights_and_reorders(self):
        query = "estimator video pipeline"
        self.assertEqual(self.recall_ids(query)[0], self.bad)
        self.s.mark_irrelevant(self.bad, query, embedder=False)
        rows = self.s.recall(query, embedder=False)
        self.assertEqual(rows[0]["id"], self.good)
        flagged = {r["id"]: r.get("neg_feedback") for r in rows}
        self.assertTrue(flagged[self.bad])
        self.assertFalse(flagged.get(self.good))

    def test_similar_query_triggers_token_overlap(self):
        self.s.mark_irrelevant(self.bad, "estimator video pipeline", embedder=False)
        rows = self.s.recall("video pipeline estimator tools", embedder=False)
        self.assertEqual(rows[0]["id"], self.good)

    def test_unrelated_query_unaffected(self):
        # query-conditional: the memory stays fully ranked elsewhere
        self.s.mark_irrelevant(self.bad, "monte carlo settings", embedder=False)
        rows = self.s.recall("estimator video pipeline", embedder=False)
        self.assertEqual(rows[0]["id"], self.bad)
        self.assertFalse(any(r.get("neg_feedback") for r in rows))

    def test_never_hidden(self):
        query = "estimator video pipeline"
        self.s.mark_irrelevant(self.bad, query, embedder=False)
        self.assertIn(self.bad, self.recall_ids(query))

    def test_retract_restores_and_remark_reactivates(self):
        query = "estimator video pipeline"
        fid = self.s.mark_irrelevant(self.bad, query, embedder=False)
        self.assertEqual(self.recall_ids(query)[0], self.good)
        self.s.retract_feedback(fid)
        self.assertEqual(self.recall_ids(query)[0], self.bad)
        # the row is kept (never delete) and re-marking reactivates it
        fid2 = self.s.mark_irrelevant(self.bad, query, embedder=False)
        self.assertEqual(fid2, fid)
        self.assertEqual(self.recall_ids(query)[0], self.good)

    def test_list_feedback(self):
        fid = self.s.mark_irrelevant(self.bad, "estimator video pipeline",
                                     embedder=False)
        rows = self.s.list_feedback(self.bad)
        self.assertEqual([r["id"] for r in rows], [fid])
        self.assertIsNone(rows[0]["retracted"])
        self.s.retract_feedback(fid)
        self.assertIsNotNone(self.s.list_feedback(self.bad)[0]["retracted"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.s.mark_irrelevant(999, "whatever", embedder=False)
        with self.assertRaises(ValueError):
            self.s.mark_irrelevant(self.bad, "   ", embedder=False)

    def test_frozen_store_refuses(self):
        set_config(self.s, "frozen", "on")
        with self.assertRaises(FrozenStoreError):
            self.s.mark_irrelevant(self.bad, "estimator video pipeline",
                                   embedder=False)


class TestVectorSimilarity(unittest.TestCase):
    """Associative path: a paraphrase of the marked query triggers via the
    stored query vector even with too little token overlap for the fallback."""

    def setUp(self):
        self.s = mem_store()
        self.emb = FakeEmbedder()
        self.bad = self.s.store("Chart bug triage notes", salience=1.0)
        self.good = self.s.store("Chart bug root cause and fix", salience=0.2)
        backfill(self.s, self.emb)

    def test_paraphrase_triggers_vector_match(self):
        # "the bug" vs "the glitch": Jaccard 1/3 < 0.5, but the synonym table
        # maps both to the same direction → cosine 1.0 ≥ threshold
        self.s.mark_irrelevant(self.bad, "the chart bug", embedder=self.emb)
        rows = self.s.recall("the chart glitch", embedder=self.emb)
        self.assertEqual(rows[0]["id"], self.good)
        self.assertTrue(any(r["id"] == self.bad and r.get("neg_feedback")
                            for r in rows))

    def test_keyword_marked_row_still_works_under_vectors(self):
        # feedback stored without a model (vector NULL) falls back to token
        # overlap even when recall itself runs with an embedder
        self.s.mark_irrelevant(self.bad, "chart bug triage", embedder=False)
        rows = self.s.recall("chart bug triage", embedder=self.emb)
        self.assertEqual(rows[0]["id"], self.good)


if __name__ == "__main__":
    unittest.main()
