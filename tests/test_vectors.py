import unittest
import zlib

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.vectors import backfill, cosine, from_blob, similar, to_blob

# Deterministic fake embedder: tokens hash to one-hot dimensions, with a
# synonym table so semantically-related words share a direction — letting us
# test associative recall with zero keyword overlap and no model download.
SYNONYMS = {
    "automobile": "car", "vehicle": "car",
    "sparkled": "twinkle", "sparkle": "twinkle", "glint": "twinkle",
    "glitch": "artifact", "bug": "artifact",
}
DIM = 64


class FakeEmbedder:
    name = "fake:onehot"

    def embed(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * DIM
            for tok in t.lower().split():
                tok = "".join(c for c in tok if c.isalnum())
                if not tok:
                    continue
                tok = SYNONYMS.get(tok, tok)
                # stable hash — builtin hash() is randomized per process
                vec[zlib.crc32(tok.encode()) % DIM] += 1.0
            out.append(vec)
        return out


class FailingEmbedder:
    name = "fake:broken"

    def embed(self, texts):
        raise RuntimeError("boom")


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestVectorPrimitives(unittest.TestCase):
    def test_blob_roundtrip(self):
        v = [0.1, -2.5, 3.0]
        self.assertEqual(len(to_blob(v)), 12)
        back = from_blob(to_blob(v))
        for a, b in zip(v, back):
            self.assertAlmostEqual(a, b, places=5)

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)  # zero vector safe


class TestBackfillAndSimilar(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.emb = FakeEmbedder()
        self.car = self.s.store("Bought a new car for the commute")
        self.twinkle = self.s.store("Fixed the eye twinkle artifact in VRT lipsync")
        self.misc = self.s.store("Quarterly taxes filed")

    def test_backfill_idempotent(self):
        self.assertEqual(backfill(self.s, self.emb), 3)
        self.assertEqual(backfill(self.s, self.emb), 0)

    def test_similar_finds_synonyms(self):
        backfill(self.s, self.emb)
        ranked = similar(self.s, self.emb, "automobile")
        self.assertEqual(ranked[0][0], self.car)
        self.assertGreater(ranked[0][1], 0.0)

    def test_similar_excludes_superseded(self):
        backfill(self.s, self.emb)
        newer = self.s.store("Sold the car again")
        backfill(self.s, self.emb)
        self.s.supersede(self.car, newer)
        ids = [mid for mid, _ in similar(self.s, self.emb, "automobile")]
        self.assertNotIn(self.car, ids)
        self.assertIn(newer, ids)


class TestHybridRecall(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.emb = FakeEmbedder()
        self.twinkle = self.s.store("Fixed the eye twinkle artifact in VRT lipsync")
        self.s.store("Wrote the quarterly report")
        backfill(self.s, self.emb)

    def test_zero_keyword_overlap_recall(self):
        # "sparkle glitch" shares no token with the memory; synonyms map both
        rows = self.s.recall("sparkle glitch", embedder=self.emb)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["id"], self.twinkle)

    def test_keyword_only_unchanged(self):
        rows = self.s.recall("sparkle glitch", embedder=False)
        self.assertEqual(rows, [])

    def test_broken_embedder_never_breaks_recall(self):
        rows = self.s.recall("twinkle", embedder=FailingEmbedder())
        self.assertTrue(rows)  # falls back to the FTS result

    def test_keyword_and_vector_agreement_ranks_first(self):
        also = self.s.store("twinkle twinkle little star nursery rhyme")
        backfill(self.s, self.emb)
        # both match "twinkle" by keyword; vector similarity to "artifact"
        # context should keep the VRT memory on top
        rows = self.s.recall("twinkle glitch lipsync", embedder=self.emb)
        self.assertEqual(rows[0]["id"], self.twinkle)
        self.assertIn(also, [r["id"] for r in rows])

    def test_auto_embedder_skipped_when_no_vectors(self):
        s2 = mem_store()
        s2.store("plain store, no vectors anywhere")
        rows = s2.recall("plain store")  # embedder=None auto path
        self.assertTrue(rows)
        self.assertIsNone(s2._auto_embedder)

    def test_recall_fts_keep_widens_result_for_blend(self):
        # Regression (eval #17): a row with weak keyword but the strongest
        # vector match fell outside the FTS top-`limit`, then was re-added in
        # the blend with relevance 0 — its bm25 erased, sinking it rank 1->4.
        # The fix keeps the OVERFETCHED keyword rows (each with real bm25) so
        # the blend sees them; _recall_fts(keep=N) returns up to N such rows.
        s = mem_store()
        for i in range(8):
            s.store(f"shared marker row {i}")
        narrow = s._recall_fts("shared marker", "OR", 2, None, None, False)
        wide = s._recall_fts("shared marker", "OR", 2, None, None, False, keep=7)
        self.assertEqual(len(narrow), 2)            # default: keep == limit
        self.assertEqual(len(wide), 7)              # widened for the blend
        self.assertTrue(all(r["relevance"] > 0 for r in wide))  # real bm25, not 0

    def test_abstain_on_out_of_store_query(self):
        # the #191 gate: a query with no real match must NOT pose noise as an
        # answer. "twinkle artifact" recalls the VRT row; an unrelated query
        # finds nothing relevant.
        from fornixdb.core import recall_has_answer
        hit = self.s.recall("twinkle artifact", embedder=self.emb)
        self.assertTrue(recall_has_answer(hit))
        miss = self.s.recall("zucchini parliament saxophone", embedder=self.emb)
        self.assertFalse(recall_has_answer(miss))

    def test_recall_widens_keep_when_embedder_blends(self):
        # recall() must ask for the wider keyword set whenever vectors will
        # blend, so the eval #17 erasure can't recur end to end.
        s = mem_store()
        s.store("shared marker seed")
        backfill(s, self.emb)
        seen = {}
        orig = MemoryStore._recall_fts

        def spy(self, q, mode, limit, *a, **k):
            seen["keep"] = k.get("keep")
            return orig(self, q, mode, limit, *a, **k)

        MemoryStore._recall_fts = spy
        try:
            s.recall("shared marker", embedder=self.emb, limit=2)
        finally:
            MemoryStore._recall_fts = orig
        self.assertIsNotNone(seen["keep"])
        self.assertGreater(seen["keep"], 2)         # wider than `limit`


if __name__ == "__main__":
    unittest.main()
