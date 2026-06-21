"""The usefulness loop closing on the PUSH side (ROADMAP "per-memory usefulness
feedback"): proactive/rhythmic impressions are counted separately from genuine
recalls, and that signal nudges the per-memory relevance floor — a proven-useful
memory surfaces a touch more easily, a memory pushed over and over but never used
goes quiet. Reversible via `config usefulness_floor_adapt off`."""

import os
import unittest

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic keyword recall, no model

from fornixdb.core import (FLOOR_CAP, FLOOR_DISCOUNT_MAX, FLOOR_MIN_IMPRESSIONS,
                           FLOOR_PENALTY_MAX, PROJECT_MISMATCH_PENALTY,
                           FrozenStoreError, MemoryStore)
from fornixdb.db import connect
from fornixdb.multistore import set_config
from fornixdb.proactive import (active_project_from_cwd, proactive_recall,
                                relevant_memories, resolve_active_project)


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestRecallVsImpression(unittest.TestCase):
    """recall_count must mean a genuine PULL, never a proactive PUSH gather."""

    def setUp(self):
        self.s = mem_store()
        self.m = self.s.store("alpha fact about deploy configuration", name="m")

    def tearDown(self):
        self.s.close()

    def _row(self):
        return dict(self.s.conn.execute(
            "SELECT * FROM memory WHERE id = ?", (self.m,)).fetchone())

    def test_explicit_recall_bumps_recall_count(self):
        self.s.recall("deploy configuration")
        self.assertEqual(self._row()["recall_count"], 1)

    def test_candidate_fetch_does_not_bump_recall_count(self):
        self.s.recall("deploy configuration", count_recall=False)
        self.assertEqual(self._row()["recall_count"], 0)

    def test_proactive_path_leaves_recall_count_untouched(self):
        # relevant_memories gathers candidates via count_recall=False
        relevant_memories(self.s, "deploy configuration")
        self.assertEqual(self._row()["recall_count"], 0)


class TestRecordSurfaced(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.m = self.s.store("a fact", name="m")

    def tearDown(self):
        self.s.close()

    def _row(self):
        return dict(self.s.conn.execute(
            "SELECT * FROM memory WHERE id = ?", (self.m,)).fetchone())

    def test_bumps_and_stamps(self):
        self.assertEqual(self._row()["surfaced_count"], 0)
        self.s.record_surfaced([self.m])
        r = self._row()
        self.assertEqual(r["surfaced_count"], 1)
        self.assertIsNotNone(r["last_surfaced"])
        self.s.record_surfaced([self.m])
        self.assertEqual(self._row()["surfaced_count"], 2)

    def test_does_not_touch_recall_count_or_salience(self):
        before = self._row()
        self.s.record_surfaced([self.m])
        after = self._row()
        self.assertEqual(after["recall_count"], before["recall_count"])
        self.assertEqual(after["salience"], before["salience"])

    def test_frozen_store_skips_silently(self):
        set_config(self.s, "frozen", "1")
        self.s.__dict__.pop("_frozen_cache", None)
        self.s.record_surfaced([self.m])          # no raise
        # can't read through the frozen guard's write, but the count stayed 0
        set_config(self.s, "frozen", "0")
        self.s.__dict__.pop("_frozen_cache", None)
        self.assertEqual(self._row()["surfaced_count"], 0)


class TestEffectiveFloor(unittest.TestCase):
    BASE = 0.45

    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_fresh_memory_uses_base_floor(self):
        row = {"recall_count": 0, "helpful_count": 0, "surfaced_count": 0}
        self.assertEqual(self.s.effective_floor(row, self.BASE), self.BASE)

    def test_used_memory_gets_a_discount(self):
        row = {"recall_count": 5, "helpful_count": 0, "surfaced_count": 0}
        self.assertLess(self.s.effective_floor(row, self.BASE), self.BASE)

    def test_endorsement_discounts_more_than_a_lone_recall(self):
        recalled = {"recall_count": 1, "helpful_count": 0, "surfaced_count": 0}
        endorsed = {"recall_count": 0, "helpful_count": 1, "surfaced_count": 0}
        self.assertLess(self.s.effective_floor(endorsed, self.BASE),
                        self.s.effective_floor(recalled, self.BASE))

    def test_ignored_memory_gets_a_penalty(self):
        # pushed many times, never used → floor rises (quieter)
        row = {"recall_count": 0, "helpful_count": 0, "surfaced_count": 20}
        self.assertGreater(self.s.effective_floor(row, self.BASE), self.BASE)

    def test_no_penalty_below_min_impressions(self):
        row = {"recall_count": 0, "helpful_count": 0,
               "surfaced_count": FLOOR_MIN_IMPRESSIONS - 1}
        self.assertEqual(self.s.effective_floor(row, self.BASE), self.BASE)

    def test_use_offsets_impressions(self):
        # surfaced a lot but ALSO used a lot → not "ignored"
        ignored = {"recall_count": 0, "helpful_count": 0, "surfaced_count": 20}
        used = {"recall_count": 20, "helpful_count": 0, "surfaced_count": 20}
        self.assertLess(self.s.effective_floor(used, self.BASE),
                        self.s.effective_floor(ignored, self.BASE))

    def test_bounds(self):
        # penalty can't push past FLOOR_CAP even from a high base
        hot = {"recall_count": 0, "helpful_count": 0, "surfaced_count": 9999}
        self.assertLessEqual(self.s.effective_floor(hot, 0.95), FLOOR_CAP)
        # discount can't drop below zero from a low base
        loved = {"recall_count": 9999, "helpful_count": 9999, "surfaced_count": 0}
        self.assertGreaterEqual(self.s.effective_floor(loved, 0.02), 0.0)

    def test_discount_and_penalty_within_declared_maxima(self):
        loved = {"recall_count": 9999, "helpful_count": 9999, "surfaced_count": 0}
        ignored = {"recall_count": 0, "helpful_count": 0, "surfaced_count": 9999}
        self.assertAlmostEqual(self.BASE - self.s.effective_floor(loved, self.BASE),
                               FLOOR_DISCOUNT_MAX, places=4)
        self.assertAlmostEqual(self.s.effective_floor(ignored, self.BASE) - self.BASE,
                               FLOOR_PENALTY_MAX, places=4)

    def test_toggle_off_returns_base(self):
        set_config(self.s, "usefulness_floor_adapt", "off")
        ignored = {"recall_count": 0, "helpful_count": 0, "surfaced_count": 99}
        self.assertEqual(self.s.effective_floor(ignored, self.BASE), self.BASE)


class TestFloorInRelevantMemories(unittest.TestCase):
    """The per-memory floor actually changes what gets pushed."""

    def setUp(self):
        self.s = mem_store()
        self.s._resolve_embedder = lambda *a, **k: object()  # vectors "on"

    def tearDown(self):
        self.s.close()

    def test_ignored_memory_dropped_even_above_base_floor(self):
        # cosine 0.50 clears base 0.45, but the row has been pushed-and-ignored
        # enough that its effective floor exceeds 0.50 → it must drop out.
        self.s.recall = lambda *a, **k: [
            {"id": 1, "kind": "semantic", "gist": "noisy",
             "vec_cos": 0.50, "recall_count": 0, "helpful_count": 0,
             "surfaced_count": 200}]
        self.assertEqual(relevant_memories(self.s, "x", floor=0.45), [])

    def test_useful_memory_surfaces_just_below_base_floor(self):
        # cosine 0.44 misses base 0.45, but a proven-useful row's discount pulls
        # its floor under 0.44 → it surfaces.
        self.s.recall = lambda *a, **k: [
            {"id": 1, "kind": "semantic", "gist": "loved",
             "vec_cos": 0.44, "recall_count": 50, "helpful_count": 50,
             "surfaced_count": 0}]
        self.assertEqual([r["id"] for r in relevant_memories(self.s, "x", floor=0.45)],
                         [1])

    def test_toggle_off_uses_plain_base_floor(self):
        set_config(self.s, "usefulness_floor_adapt", "off")
        self.s.recall = lambda *a, **k: [
            {"id": 1, "kind": "semantic", "gist": "noisy",
             "vec_cos": 0.50, "recall_count": 0, "helpful_count": 0,
             "surfaced_count": 200}]
        self.assertEqual([r["id"] for r in relevant_memories(self.s, "x", floor=0.45)],
                         [1])


class TestProjectScopedFloor(unittest.TestCase):
    BASE = 0.45

    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def _r(self, project):
        return {"recall_count": 0, "helpful_count": 0, "surfaced_count": 0,
                "project": project}

    def test_off_project_memory_penalized(self):
        f = self.s.effective_floor(self._r("Videos"), self.BASE,
                                   active_project="FornixDB")
        self.assertAlmostEqual(f - self.BASE, PROJECT_MISMATCH_PENALTY, places=4)

    def test_on_project_memory_not_penalized(self):
        f = self.s.effective_floor(self._r("FornixDB"), self.BASE,
                                   active_project="FornixDB")
        self.assertEqual(f, self.BASE)

    def test_match_is_case_insensitive(self):
        f = self.s.effective_floor(self._r("fornixdb"), self.BASE,
                                   active_project="FornixDB")
        self.assertEqual(f, self.BASE)

    def test_projectless_memory_never_penalized(self):
        # general/curated facts (no project) belong everywhere
        for p in (None, "", "   "):
            f = self.s.effective_floor(self._r(p), self.BASE,
                                       active_project="FornixDB")
            self.assertEqual(f, self.BASE)

    def test_no_active_project_means_no_scoping(self):
        f = self.s.effective_floor(self._r("Videos"), self.BASE,
                                   active_project=None)
        self.assertEqual(f, self.BASE)

    def test_toggle_off_disables_only_project_scoping(self):
        set_config(self.s, "project_scoped_pulse", "off")
        f = self.s.effective_floor(self._r("Videos"), self.BASE,
                                   active_project="FornixDB")
        self.assertEqual(f, self.BASE)

    def test_independent_of_usefulness_dial(self):
        # project scoping still applies even when usefulness adaptation is off
        set_config(self.s, "usefulness_floor_adapt", "off")
        f = self.s.effective_floor(self._r("Videos"), self.BASE,
                                   active_project="FornixDB")
        self.assertAlmostEqual(f - self.BASE, PROJECT_MISMATCH_PENALTY, places=4)


class TestActiveProjectResolution(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_cwd_basename_is_the_project(self):
        self.assertEqual(active_project_from_cwd("/Users/x/dev/AIMemory"), "AIMemory")
        self.assertIsNone(active_project_from_cwd(None))
        self.assertIsNone(active_project_from_cwd(""))

    def test_pinned_config_overrides_passed(self):
        set_config(self.s, "active_project", "FornixDB")
        self.assertEqual(resolve_active_project(self.s, "RetirementEstimator"),
                         "FornixDB")

    def test_passed_used_when_unpinned(self):
        self.assertEqual(resolve_active_project(self.s, "RetirementEstimator"),
                         "RetirementEstimator")

    def test_none_when_neither(self):
        self.assertIsNone(resolve_active_project(self.s, None))


class TestProjectScopingInRelevantMemories(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.s._resolve_embedder = lambda *a, **k: object()  # vectors "on"

    def tearDown(self):
        self.s.close()

    def test_weak_off_project_hit_dropped(self):
        # cosine 0.50 clears base 0.45 but not 0.45 + mismatch penalty (0.60)
        self.s.recall = lambda *a, **k: [
            {"id": 1, "kind": "semantic", "gist": "g", "vec_cos": 0.50,
             "recall_count": 0, "helpful_count": 0, "surfaced_count": 0,
             "project": "Videos"}]
        out = relevant_memories(self.s, "x", floor=0.45, active_project="FornixDB")
        self.assertEqual(out, [])

    def test_strong_off_project_hit_still_surfaces(self):
        # a genuinely-relevant cross-project match (high cosine) clears even the
        # raised floor — scoping quiets noise, doesn't wall off real relevance
        self.s.recall = lambda *a, **k: [
            {"id": 1, "kind": "semantic", "gist": "g", "vec_cos": 0.80,
             "recall_count": 0, "helpful_count": 0, "surfaced_count": 0,
             "project": "Videos"}]
        out = relevant_memories(self.s, "x", floor=0.45, active_project="FornixDB")
        self.assertEqual([r["id"] for r in out], [1])

    def test_on_project_hit_unaffected(self):
        self.s.recall = lambda *a, **k: [
            {"id": 1, "kind": "semantic", "gist": "g", "vec_cos": 0.50,
             "recall_count": 0, "helpful_count": 0, "surfaced_count": 0,
             "project": "FornixDB"}]
        out = relevant_memories(self.s, "x", floor=0.45, active_project="FornixDB")
        self.assertEqual([r["id"] for r in out], [1])


class TestProactiveRecallRecordsImpressions(unittest.TestCase):
    """End-to-end: a pushed memory accrues surfaced_count, not recall_count."""

    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def _row(self, mid):
        return dict(self.s.conn.execute(
            "SELECT * FROM memory WHERE id = ?", (mid,)).fetchone())

    def test_injected_rows_get_an_impression_not_a_recall(self):
        mid = self.s.store("the deploy script reads configuration from env",
                           name="m")
        block = proactive_recall(self.s, "how does the deploy script read its "
                                 "configuration", session_id="s1")
        self.assertIsNotNone(block)
        r = self._row(mid)
        self.assertEqual(r["surfaced_count"], 1)   # counted as a PUSH
        self.assertEqual(r["recall_count"], 0)     # NOT counted as a PULL

    def test_per_session_dedup_counts_once(self):
        mid = self.s.store("the deploy script reads configuration from env",
                           name="m")
        q = "how does the deploy script read its configuration"
        proactive_recall(self.s, q, session_id="s1")
        proactive_recall(self.s, q, session_id="s1")  # same session → deduped
        self.assertEqual(self._row(mid)["surfaced_count"], 1)


if __name__ == "__main__":
    unittest.main()
