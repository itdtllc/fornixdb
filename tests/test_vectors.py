import os
import unittest
import zlib

# Vectors are a default dependency now, so a real model would auto-embed and
# perturb keyword-only expectations. Force the auto path OFF for deterministic
# tests; cases that exercise vectors pass an explicit embedder or flip this
# back locally (the env switch only gates the auto path, not explicit embedders).
os.environ["FORNIXDB_VECTORS"] = "off"

from fornixdb.core import MemoryStore, recall_has_answer
from fornixdb.db import connect
from fornixdb.vectors import (backfill, cosine, cosines_for, from_blob,
                              similar, to_blob)

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


class TestSenseModalityPrefix(unittest.TestCase):
    """A sense capture's 1-3-word caption structurally under-overlaps long
    queries; the embedded head must say what the row MEANS ("heard: X")."""

    def setUp(self):
        self.s = mem_store()
        self.emb = FakeEmbedder()

    def _row(self, mid):
        return self.s.conn.execute(
            "SELECT id, name, gist, detail, source FROM memory WHERE id = ?",
            (mid,)).fetchone()

    def test_sense_rows_embed_with_modality_prefix(self):
        from fornixdb.vectors import _chunk_texts
        cases = [("senses:sound", "heard: acoustic guitar"),
                 ("senses:sight", "saw: acoustic guitar"),
                 ("senses:feel", "felt: acoustic guitar"),
                 ("senses:sonar", "sensed: acoustic guitar")]  # unknown sense
        for source, want in cases:
            mid = self.s.store("acoustic guitar", source=source)
            self.assertEqual(_chunk_texts(self._row(mid))[0], want)

    def test_non_sense_rows_unchanged(self):
        from fornixdb.vectors import _chunk_texts
        mid = self.s.store("acoustic guitar", source="chat")
        self.assertEqual(_chunk_texts(self._row(mid))[0], "acoustic guitar")
        plain = self.s.store("acoustic guitar")
        self.assertEqual(_chunk_texts(self._row(plain))[0], "acoustic guitar")

    def test_refresh_senses_reembeds_only_sense_rows(self):
        from fornixdb.vectors import refresh_senses
        cap = self.s.store("whistling in the demo room", source="senses:sound")
        chat = self.s.store("Chat about the whistling demo")
        backfill(self.s, self.emb)          # both embedded, caption pre-prefix?
        # simulate a pre-prefix store: overwrite the caption's head chunk
        # with the bare-gist embedding
        self.s.conn.execute(
            "UPDATE embedding SET vector = ? WHERE memory_id = ? AND chunk = 0",
            (to_blob(self.emb.embed(["whistling in the demo room"])[0]), cap))
        self.s.conn.commit()
        n = refresh_senses(self.s, self.emb)
        self.assertEqual(n, 1)              # only the sense row, not the chat
        head = from_blob(self.s.conn.execute(
            "SELECT vector FROM embedding WHERE memory_id = ? AND chunk = 0",
            (cap,)).fetchone()[0])
        want = self.emb.embed(["heard: whistling in the demo room"])[0]
        self.assertGreater(cosine(head, want), 0.999)
        self.assertEqual(refresh_senses(self.s, self.emb), 1)  # idempotent


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

    def test_auto_path_off_falls_back_to_keyword(self):
        # with the auto path off (env, set at module top), recall still works —
        # keyword + time — and no model is resolved.
        s2 = mem_store()
        s2.store("plain store, keyword only")
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

    def test_blend_keep_is_limit_independent(self):
        # Regression (2026-07-16): the blend keep-set scaled with `limit`, so a
        # mid-ranked bm25 row kept its keyword relevance in a wide fetch but
        # lost it in a narrow one — the same query returned a DIFFERENT ORDER
        # at different limits (live: eval #17 rank 4 at k=5 vs rank 2 at 15).
        # A row's score must not depend on how many rows the caller asked for,
        # so the keep depth is a constant.
        s = mem_store()
        s.store("shared marker seed")
        backfill(s, self.emb)
        keeps = []
        orig = MemoryStore._recall_fts

        def spy(self, q, mode, limit, *a, **k):
            keeps.append(k.get("keep"))
            return orig(self, q, mode, limit, *a, **k)

        MemoryStore._recall_fts = spy
        try:
            s.recall("shared marker", embedder=self.emb, limit=3)
            s.recall("shared marker", embedder=self.emb, limit=12)
        finally:
            MemoryStore._recall_fts = orig
        self.assertEqual(len(set(keeps)), 1)        # same keep at every limit


class TestEmbedOnWrite(unittest.TestCase):
    """store() embeds inline when the store uses vectors, so a memory written
    via remember/store is recallable by meaning immediately — not only after a
    manual `embed` backfill (the drift this closes)."""

    def setUp(self):
        self.s = mem_store()
        self.emb = FakeEmbedder()

    def _emb_count(self, mid):
        return self.s.conn.execute(
            "SELECT count(*) FROM embedding WHERE memory_id = ?", (mid,)).fetchone()[0]

    def test_explicit_embedder_embeds_on_store(self):
        mid = self.s.store("a car", "the automobile parked", embedder=self.emb)
        self.assertGreater(self._emb_count(mid), 0)
        # and it is recallable by meaning with zero keyword overlap
        self.assertIn(mid, [m for m, _ in similar(self.s, self.emb, "vehicle")])

    def test_auto_path_embeds_when_store_uses_vectors(self):
        # a store "uses vectors" once a model is resolved; a subsequent plain
        # store() (embedder=None) then auto-embeds via _resolve_embedder
        self.s._auto_embedder = self.emb            # simulate a resolved model
        mid = self.s.store("a vehicle", "shiny")    # no embedder argument
        self.assertGreater(self._emb_count(mid), 0)

    def test_keyword_only_store_does_not_embed(self):
        # fresh store, no vectors yet, no model resolvable: stays keyword-only
        mid = self.s.store("plain", "no vectors here")
        self.assertEqual(self._emb_count(mid), 0)
        self.assertIsNone(self.s._auto_embedder)

    def test_embedder_false_skips(self):
        self.s._auto_embedder = self.emb
        mid = self.s.store("x", "y", embedder=False)
        self.assertEqual(self._emb_count(mid), 0)

    def test_embed_failure_never_blocks_write(self):
        mid = self.s.store("still", "saved", embedder=FailingEmbedder())
        self.assertTrue(self.s.conn.execute(
            "SELECT 1 FROM memory WHERE id = ?", (mid,)).fetchone())  # write held
        self.assertEqual(self._emb_count(mid), 0)


class TestVectorsDefaultOn(unittest.TestCase):
    """Vectors are ON by default (model2vec ships as a dependency): a fresh
    store bootstraps embeddings on first write. It only stays off via the env
    switch, the per-store config, or incapable hardware (model won't load).
    Each test simulates a capable machine by stubbing the default embedder."""

    def setUp(self):
        from fornixdb import vectors as V
        self._V = V
        self._orig_gde = V.get_default_embedder
        self._orig_env = os.environ.get("FORNIXDB_VECTORS")
        os.environ.pop("FORNIXDB_VECTORS", None)          # clear the suite-wide off
        V.get_default_embedder = lambda: FakeEmbedder()   # a capable machine

    def tearDown(self):
        self._V.get_default_embedder = self._orig_gde
        if self._orig_env is None:
            os.environ.pop("FORNIXDB_VECTORS", None)
        else:
            os.environ["FORNIXDB_VECTORS"] = self._orig_env

    def _emb(self, s, mid):
        return s.conn.execute(
            "SELECT count(*) FROM embedding WHERE memory_id = ?", (mid,)).fetchone()[0]

    def test_fresh_store_bootstraps_on_first_write(self):
        s = mem_store()                          # brand-new, zero embeddings
        mid = s.store("a vehicle on the road")
        self.assertGreater(self._emb(s, mid), 0)

    def test_existing_store_auto_backfills_on_first_use(self):
        # simulate a pre-vectors store: memories present, no embeddings, and no
        # embedder resolvable yet (model "absent").
        self._V.get_default_embedder = lambda: None
        s = mem_store()
        a = s.store("the automobile stalled")
        b = s.store("her eyes sparkled")
        self.assertEqual(self._emb(s, a), 0)     # nothing embedded yet
        # now a model becomes available and the store is used → auto-backfill
        del s._auto_embedder                     # force re-resolve
        self._V.get_default_embedder = lambda: FakeEmbedder()
        s.recall("vehicle")                      # first real vector use
        self.assertGreater(self._emb(s, a), 0)   # old memories embedded
        self.assertGreater(self._emb(s, b), 0)

    def test_cosines_for_returns_exact_best_chunk_values(self):
        s = mem_store()
        a = s.store("the automobile stalled")
        got = cosines_for(s, FakeEmbedder(), "vehicle", [a])
        self.assertIn(a, got)
        self.assertGreater(got[a], 0.5)   # synonym direction, real similarity
        self.assertEqual(cosines_for(s, FakeEmbedder(), "vehicle", []), {})

    def test_keyword_hit_outside_shortlist_keeps_true_cosine(self):
        # 30 decoys sit CLOSER to the query in vector space, so the keyword-
        # anchored answer never makes the 25-slot neighbor shortlist. Its
        # true cosine must still reach the blend and the abstention gate —
        # it used to read as 0.0 (no vector term, gate false-abstained on a
        # correct rank-1 hit, and rankings shifted with `limit`).
        s = mem_store()
        emb = FakeEmbedder()
        for i in range(30):
            s.store(f"vehicle stalled report {i}", embedder=emb)
        target = s.store("automobile stalled dock pier rope anchor mast",
                         embedder=emb)
        rows = s.recall("automobile stalled dock", limit=8, embedder=emb,
                        count_recall=False)
        hit = next((r for r in rows if r["id"] == target), None)
        self.assertIsNotNone(hit, "keyword-anchored row missing from results")
        self.assertGreaterEqual(float(hit["vec_cos"]), 0.30)  # true cosine, not 0.0
        self.assertEqual(rows[0]["id"], target)   # unique 3-token AND anchor wins
        self.assertTrue(recall_has_answer(rows))  # gate no longer false-abstains

    def test_partial_coverage_gap_heals_on_first_use(self):
        # a store that LOST coverage (a vector-dropping edit with no model in
        # the environment — the 2026-07-01 bulk distill left 250/317 rows
        # unembedded this way) must close the holes itself: the old guard
        # bailed the moment ANY embedding existed, so gaps were permanent
        # unless someone remembered to run `embed`.
        s = mem_store()
        a = s.store("the automobile stalled")
        b = s.store("her eyes sparkled")
        self.assertGreater(self._emb(s, a), 0)   # embed-on-write covered both
        s.conn.execute("DELETE FROM embedding WHERE memory_id = ?", (a,))
        s.conn.commit()
        del s._auto_embedder                     # next use re-resolves
        s.recall("vehicle")                      # first real vector use
        self.assertGreater(self._emb(s, a), 0)   # the hole closed itself
        self.assertGreater(self._emb(s, b), 0)   # untouched row still covered

    def test_env_switch_off_disables(self):
        os.environ["FORNIXDB_VECTORS"] = "off"
        s = mem_store()
        mid = s.store("a vehicle on the road")
        self.assertEqual(self._emb(s, mid), 0)
        self.assertIsNone(s._auto_embedder)

    def test_config_vectors_off_disables(self):
        s = mem_store()
        s.conn.execute("INSERT OR REPLACE INTO meta VALUES ('vectors', 'off')")
        s.conn.commit()
        mid = s.store("a vehicle on the road")
        self.assertEqual(self._emb(s, mid), 0)
        self.assertIsNone(s._auto_embedder)


if __name__ == "__main__":
    unittest.main()
