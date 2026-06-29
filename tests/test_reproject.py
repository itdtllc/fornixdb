import os
import unittest

os.environ["FORNIXDB_VECTORS"] = "off"  # default-off; vector tests pass an embedder

from fornixdb import reproject as rp
from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.multistore import get_config, set_config
from fornixdb.vectors import backfill

from test_vectors import FakeEmbedder


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestPureClassifiers(unittest.TestCase):
    def test_content_words_drops_stop_and_short(self):
        w = rp.content_words("Session: come up to speed on VIDEO rendering ok")
        self.assertIn("video", w)
        self.assertIn("rendering", w)
        self.assertNotIn("up", w)        # short
        self.assertNotIn("session", w)   # stop word
        self.assertNotIn("speed", w)     # stop word

    def test_classify_words_picks_best_with_margin(self):
        vocab = {"videos": {"video", "clip", "lipsync", "comfyui"},
                 "retire": {"transfer", "monte", "carlo", "estimator"}}
        best, margin, scores = rp.classify_words(
            {"video", "clip", "lipsync", "transfer"}, vocab)
        self.assertEqual(best, "videos")
        self.assertEqual(margin, 2)               # 3 video hits − 1 retire hit
        self.assertEqual(scores["videos"], 3.0)

    def test_classify_words_no_signal(self):
        best, margin, _ = rp.classify_words({"unrelated", "words"},
                                            {"videos": {"video"}})
        self.assertIsNone(best)
        self.assertEqual(margin, 0.0)

    def test_classify_vec_nearest_centroid(self):
        cents = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        best, margin, scores = rp.classify_vec([0.9, 0.1], cents)
        self.assertEqual(best, "a")
        self.assertGreater(margin, 0)
        self.assertGreater(scores["a"], scores["b"])

    def test_classify_vec_empty(self):
        self.assertEqual(rp.classify_vec([], {"a": [1.0]}), (None, 0.0, {}))
        self.assertEqual(rp.classify_vec([1.0], {}), (None, 0.0, {}))


class TestCanonical(unittest.TestCase):
    def test_alias_family_collapses_to_one_key(self):
        s = mem_store()
        set_config(s, "project_aliases", "fornixdb=engramdb,aimemory")
        self.assertEqual(rp._canon(s, "fornixdb"), rp._canon(s, "engramdb"))
        self.assertEqual(rp._canon(s, "AIMemory"), rp._canon(s, "fornixdb"))
        self.assertEqual(rp._canon(s, None), "")


class TestKeywordModeEndToEnd(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        # anchors: deliberately-stored, honestly-labeled, non-episodic
        self.s.store("Rendered the video clip with lipsync in comfyui render",
                     kind="reference", project="videos")
        self.s.store("Fixed the transfer conflict in monte carlo estimator income",
                     kind="reference", project="RetirementEstimator")
        # the bug: an auto-captured VIDEO session mislabeled with the launch dir
        self.bad = self.s.store(
            "Session: come up to speed on video clip lipsync comfyui render",
            kind="episodic", project="RetirementEstimator")
        # a correctly-labeled RE episodic — must be left alone
        self.ok = self.s.store(
            "Session: resume the transfer conflict monte carlo estimator work",
            kind="episodic", project="RetirementEstimator")

    def test_proposes_only_the_mislabeled_one(self):
        # the mislabel sits under the launch-dir default, so flag it suspect
        res = rp.propose(self.s, suspect=["RetirementEstimator"])
        self.assertEqual(res["mode"], "keyword")
        ids = {p["id"]: p for p in res["proposals"]}
        self.assertIn(self.bad, ids)
        self.assertEqual(ids[self.bad]["proposed"], "videos")
        self.assertEqual(ids[self.bad]["current"], "RetirementEstimator")
        self.assertNotIn(self.ok, ids)            # already content-consistent

    def test_specific_label_is_trusted_unless_suspect(self):
        # without --suspect, a labeled memory is never reconsidered (NULL only),
        # so the mislabeled-but-labeled session is left alone — no corruption.
        res = rp.propose(self.s)
        self.assertNotIn(self.bad, {p["id"] for p in res["proposals"]})

    def test_apply_then_undo_round_trips(self):
        res = rp.propose(self.s, suspect=["RetirementEstimator"])
        applied = rp.apply_proposals(self.s, res["proposals"])
        self.assertEqual(applied["applied"], len(res["proposals"]))
        row = self.s.conn.execute("SELECT project FROM memory WHERE id=?",
                                  (self.bad,)).fetchone()
        self.assertEqual(row["project"], "videos")
        # undo restores the prior label and clears the set
        un = rp.undo(self.s)
        self.assertEqual(un["restored"], applied["applied"])
        row = self.s.conn.execute("SELECT project FROM memory WHERE id=?",
                                  (self.bad,)).fetchone()
        self.assertEqual(row["project"], "RetirementEstimator")
        self.assertEqual(get_config(self.s, rp.UNDO_KEY, "[]"), "[]")

    def test_alias_equivalent_not_churned(self):
        set_config(self.s, "project_aliases", "videos=clips")
        # an episodic labeled "clips" whose content is video — same canon, skip it
        cid = self.s.store("Session: video clip lipsync comfyui render again",
                           kind="episodic", project="clips")
        # even when reconsidered, an alias of the content's project is not churned
        res = rp.propose(self.s, suspect=["clips"])
        self.assertNotIn(cid, {p["id"] for p in res["proposals"]})


class TestVectorMode(unittest.TestCase):
    def test_vector_mode_reprojects(self):
        s = mem_store()
        emb = FakeEmbedder()
        s._auto_embedder = emb  # force vector mode without the auto path
        s.store("video clip lipsync comfyui render",
                kind="reference", project="videos")
        s.store("transfer conflict monte carlo estimator income tax",
                kind="reference", project="RetirementEstimator")
        bad = s.store("session video clip lipsync comfyui render",
                      kind="episodic", project="RetirementEstimator")
        backfill(s, emb)                       # write embeddings for all rows
        res = rp.propose(s, suspect=["RetirementEstimator"])
        self.assertEqual(res["mode"], "vector")
        ids = {p["id"]: p for p in res["proposals"]}
        self.assertIn(bad, ids)
        self.assertEqual(ids[bad]["proposed"], "videos")


if __name__ == "__main__":
    unittest.main()
