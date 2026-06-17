import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic: no ambient-model auto-embed

from fornixdb.consolidate import (_dream_narrative, _gist_problem, dream,
                                  propose, status, supersede_suggestion)
from fornixdb.core import FrozenStoreError, MemoryStore
from fornixdb.multistore import set_config
from fornixdb.vectors import embed_memory

from test_vectors import FakeEmbedder


def file_store(tmp):
    return MemoryStore(db_path=Path(tmp) / "t.db")


def _age(store, mem_id, days):
    old = (datetime.now() - timedelta(days=days)).isoformat()
    store.conn.execute(
        "UPDATE memory SET recorded_time=?, last_recalled=NULL, event_time=? WHERE id=?",
        (old, old, mem_id))
    store.conn.commit()


class TestPropose(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)
        self.emb = FakeEmbedder()

    def tearDown(self):
        self.s.close()  # Windows can't delete an open db file
        self.tmp.cleanup()

    def _session(self, gist="Session 2026-06-01 (9 user turns): fix the bug",
                 **kw):
        return self.s.store(gist, "transcript summary", kind="episodic",
                            source="claude-code-transcript",
                            source_ref="/tmp/x.jsonl", **kw)

    # ---------------------------------------------------------- distillation

    def test_undistilled_session_proposed_with_transcript_path(self):
        mid = self._session()
        work = propose(self.s)
        self.assertEqual([d["id"] for d in work["distill"]], [mid])
        self.assertEqual(work["distill"][0]["transcript"], "/tmp/x.jsonl")

    def test_distilled_tag_excludes_session(self):
        mid = self._session()
        self.s.tag(mid, "distilled")
        self.assertEqual(propose(self.s)["distill"], [])

    def test_fully_decayed_session_not_proposed(self):
        mid = self._session(salience=0.3)
        _age(self.s, mid, 400)  # decayed to the episodic floor
        self.assertEqual(propose(self.s)["distill"], [])

    def test_tombstoned_session_not_proposed(self):
        mid = self._session()
        self.s.tombstone(mid)
        self.assertEqual(propose(self.s)["distill"], [])

    def test_non_transcript_episodic_not_proposed(self):
        self.s.store("milestone shipped", kind="episodic", source="checkpoint")
        self.assertEqual(propose(self.s)["distill"], [])

    def test_distill_ordered_by_effective_salience(self):
        older = self._session(gist="Session A", salience=0.5)
        _age(self.s, older, 44)
        newer = self._session(gist="Session B", salience=0.5)
        work = propose(self.s)
        self.assertEqual([d["id"] for d in work["distill"]], [newer, older])

    # ------------------------------------------------------------ poor gists

    def test_gist_heuristics(self):
        self.assertIsNotNone(_gist_problem("x" * 201, None))
        self.assertIsNotNone(_gist_problem("a1b2 3f9e 77ab c0de", None))
        self.assertIsNotNone(_gist_problem("the fix", "the fix was to retry"))
        self.assertIsNone(_gist_problem("plain healthy gist", "other detail"))
        # #2: gist==detail (a normal one-liner) and detail only a little longer
        # are NOT flagged — only a gist that truncates MUCH longer detail is
        self.assertIsNone(_gist_problem("a one line memory fact",
                                        "a one line memory fact"))
        self.assertIsNone(_gist_problem("the cache size is 512mb",
                                        "the cache size is 512mb by default"))

    def test_poor_gist_proposed_with_reason(self):
        ok = self.s.store("a perfectly healthy gist", "detail")
        bad = self.s.store("x" * 250, "detail")
        work = propose(self.s)
        ids = [g["id"] for g in work["gists"]]
        self.assertIn(bad, ids)
        self.assertNotIn(ok, ids)
        self.assertIn("chars", work["gists"][0]["problem"])

    def test_pending_distill_sessions_skip_gist_check(self):
        # templated session gists get rewritten by distillation, not gist repair
        self._session(gist="Session 2026: " + "x" * 250)
        self.assertEqual(propose(self.s)["gists"], [])

    # ------------------------------------------------------- merges / contras

    def _pair(self, g1, g2, kind="semantic", kind2=None):
        a = self.s.store(g1, kind=kind)
        b = self.s.store(g2, kind=kind2 or kind)
        embed_memory(self.s, self.emb, a)
        embed_memory(self.s, self.emb, b)
        return a, b

    def test_near_duplicates_proposed_as_merge(self):
        a, b = self._pair("the deploy script reads config from env",
                          "the deploy script reads config from env always")
        work = propose(self.s)
        self.assertEqual(len(work["merges"]), 1)
        self.assertCountEqual(work["merges"][0]["ids"], [a, b])
        self.assertGreaterEqual(work["merges"][0]["cosine"], 0.88)

    def test_different_kinds_never_merge(self):
        self._pair("the deploy script reads config from env",
                   "the deploy script reads config from env always",
                   kind="semantic", kind2="reference")
        work = propose(self.s)
        self.assertEqual(work["merges"], [])
        self.assertEqual(work["contradictions"], [])

    def test_tombstoned_rows_excluded_from_pairs(self):
        a, _ = self._pair("the deploy script reads config from env",
                          "the deploy script reads config from env always")
        self.s.tombstone(a)
        self.assertEqual(propose(self.s)["merges"], [])

    def test_supersede_linked_pair_not_reproposed(self):
        a, b = self._pair("the deploy script reads config from env",
                          "the deploy script reads config from env always")
        self.s.supersede(a, b)
        self.assertEqual(propose(self.s)["merges"], [])

    def test_mid_band_pair_is_contradiction_candidate(self):
        a, b = self._pair("retry the upload three times on failure",
                          "never retry the upload three times")
        work = propose(self.s)
        all_ids = [set(m["ids"]) for m in work["merges"] + work["contradictions"]]
        self.assertIn({a, b}, all_ids)

    def test_unrelated_pair_not_proposed(self):
        self._pair("retry the upload on failure",
                   "the chart renders with a blue axis")
        work = propose(self.s)
        self.assertEqual(work["merges"], [])
        self.assertEqual(work["contradictions"], [])

    def test_episodic_rows_excluded_from_pairs(self):
        # two similar SESSION summaries are distinct timeline events, not a heal
        # candidate or an association — episodic is excluded from the pair scan
        a = self.s.store("Chat day one reviewed the deploy pipeline", kind="episodic")
        b = self.s.store("Chat day two reviewed the deploy pipeline", kind="episodic")
        embed_memory(self.s, self.emb, a)
        embed_memory(self.s, self.emb, b)
        work = propose(self.s)
        flat = {i for p in work["merges"] + work["contradictions"]
                + work["associations"] for i in p["ids"]}
        self.assertNotIn(a, flat)
        self.assertNotIn(b, flat)

    def test_no_embeddings_means_empty_pair_lists(self):
        self.s.store("alpha fact")
        self.s.store("alpha fact again")
        work = propose(self.s)
        self.assertEqual(work["merges"], [])
        self.assertEqual(work["contradictions"], [])

    # -------------------------------------------------------- new primitives

    def test_set_gist_updates_fts_and_drops_vector(self):
        mid = self.s.store("wrong words here", "stable detail")
        embed_memory(self.s, self.emb, mid)
        self.s.set_gist(mid, "rocketship launch checklist")
        hits = self.s.recall("rocketship", embedder=False)
        self.assertEqual([h["id"] for h in hits], [mid])
        self.assertEqual(self.s.recall("wrong words", embedder=False), [])
        left = self.s.conn.execute(
            "SELECT count(*) c FROM embedding WHERE memory_id=?", (mid,)).fetchone()
        self.assertEqual(left["c"], 0)

    def test_supersede_hands_off_name_handle(self):
        old = self.s.store("v1 of the fact", name="the-fact")
        new = self.s.store("v2 of the fact")
        self.s.supersede(old, new)
        self.assertEqual(self.s.show("the-fact", reinforce=False)["id"], new)

    def test_supersede_keeps_successors_own_name(self):
        old = self.s.store("v1", name="old-name")
        new = self.s.store("v2", name="new-name")
        self.s.supersede(old, new)
        self.assertEqual(self.s.show("new-name", reinforce=False)["id"], new)
        self.assertEqual(self.s.show("old-name", reinforce=False)["id"], old)

    def test_status_still_reports_due(self):
        self.assertTrue(status(self.s)["due"])

    # ----------------------------------------------------------- sleep / dream

    def test_dream_empty_store_is_tidy(self):
        rep = dream(self.s)
        self.assertEqual(rep["counts"]["total"], 0)
        self.assertIn("nothing to reconcile", rep["narrative"])
        self.assertIn("status", rep)
        self.assertIn("work", rep)

    def test_dream_counts_match_worklist_and_narrates(self):
        self.s.store("x" * 250, "detail")                     # a gist to tidy
        self._pair("the deploy script reads config from env",  # a merge pair
                   "the deploy script reads config from env always")
        rep = dream(self.s)
        c, work = rep["counts"], rep["work"]
        self.assertEqual(c["gists"], len(work["gists"]))
        self.assertEqual(c["merges"], len(work["merges"]))
        self.assertGreaterEqual(c["gists"], 1)
        self.assertEqual(c["merges"], 1)
        self.assertEqual(c["total"], c["distill"] + c["gists"] + c["merges"]
                         + c["contradictions"] + c["associations"])
        self.assertNotIn("nothing to reconcile", rep["narrative"])
        self.assertIn("dreaming", rep["narrative"])

    def test_dream_narrative_headlines_outdated_memories(self):
        # the orphaned-fix case: contradiction candidates lead the read-back
        msg = _dream_narrative({"last_consolidated": None},
                               {"contradictions": 2, "merges": 0, "associations": 0,
                                "distill": 0, "gists": 0, "total": 2})
        self.assertTrue(msg.split("surfaced ", 1)[1]
                        .startswith("2 possible outdated memories to reconcile"))

    def test_dream_finds_and_weaves_new_associations(self):
        # the generative half: a related, cross-kind, UNLINKED pair (cosine ~.67,
        # below the contradiction band) is proposed as a new connection.
        a = self.s.store("alpha beta gamma", kind="semantic")
        b = self.s.store("alpha beta delta", kind="reference")
        embed_memory(self.s, self.emb, a)
        embed_memory(self.s, self.emb, b)

        rep = dream(self.s)                          # propose only
        self.assertEqual(rep["counts"]["associations"], 1)
        self.assertEqual(rep["work"]["associations"][0]["ids"], [a, b])
        self.assertEqual(rep["counts"]["woven"], 0)
        self.assertIn("new connection", rep["narrative"])
        self.assertEqual(self.s.conn.execute(
            "SELECT count(*) c FROM memory_link").fetchone()["c"], 0)

        rep2 = dream(self.s, weave=True)             # make the link
        self.assertEqual(rep2["woven"], 1)
        self.assertIn("Wove 1 new connection", rep2["narrative"])
        woven = self.s.conn.execute(
            "SELECT related_id FROM memory_link WHERE memory_id=? AND relation='relates'",
            (a,)).fetchall()
        self.assertEqual([r["related_id"] for r in woven], [b])

        # idempotent: now linked, it is no longer re-proposed
        self.assertEqual(dream(self.s)["counts"]["associations"], 0)

    # ------------------------------------------ write-time supersede suggestion

    def test_suggestion_flags_same_kind_near_duplicate(self):
        a = self.s.store("the deploy script reads config from env", kind="semantic")
        embed_memory(self.s, self.emb, a)
        # a NEW memory under a different title that's nearly the same (not yet embedded)
        text = "the deploy script reads config from env always"
        b = self.s.store(text, kind="semantic")
        sug = supersede_suggestion(self.s, b, text, "semantic", embedder=self.emb)
        self.assertIsNotNone(sug)
        self.assertEqual(sug["id"], a)
        self.assertGreaterEqual(sug["cosine"], 0.88)

    def test_suggestion_none_when_unrelated(self):
        a = self.s.store("the chart renders with a blue axis", kind="semantic")
        embed_memory(self.s, self.emb, a)
        b = self.s.store("retry the upload on failure", kind="semantic")
        self.assertIsNone(supersede_suggestion(
            self.s, b, "retry the upload on failure", "semantic", embedder=self.emb))

    def test_suggestion_is_same_kind_only(self):
        a = self.s.store("the deploy script reads config from env", kind="reference")
        embed_memory(self.s, self.emb, a)
        text = "the deploy script reads config from env always"
        b = self.s.store(text, kind="semantic")  # near-dup but different kind
        self.assertIsNone(supersede_suggestion(
            self.s, b, text, "semantic", embedder=self.emb))

    def test_suggestion_none_without_embedder(self):
        self.assertIsNone(
            supersede_suggestion(self.s, 1, "anything", "semantic", embedder=None))

    def test_dream_refused_on_read_only_store(self):
        # consolidation is a maintenance op — a frozen (read-only) store refuses
        set_config(self.s, "frozen", "on")
        with self.assertRaises(FrozenStoreError):
            dream(self.s)

    def test_dream_done_reports_wake_summary(self):
        a = self.s.store("v1 of the fact", kind="semantic")
        b = self.s.store("v2 of the fact", kind="semantic")
        dream(self.s)               # opens the pass
        self.s.supersede(a, b)      # a reconciliation the AI applied DURING the pass
        rep = dream(self.s, done=True)
        self.assertEqual(rep["applied"]["reconciled"], 1)
        self.assertIn("woke", rep["narrative"])
        self.assertIn("reconciled", rep["narrative"])
        self.assertIsNotNone(status(self.s)["last_consolidated"])  # DUE clock reset

    def test_dream_done_nudges_remaining_heal_candidates(self):
        # closing a pass with an unreconciled pair still standing names it
        a, b = self._pair("the deploy script reads config from env",
                          "the deploy script reads config from env always")
        rep = dream(self.s, done=True)   # one-shot: nothing reconciled in-pass
        self.assertEqual(rep["applied"]["reconciled"], 0)
        self.assertIn("still need", rep["narrative"])
        self.assertIn("supersede", rep["narrative"])
        self.assertIn(f"#{a}", rep["narrative"])
        self.assertIn(f"#{b}", rep["narrative"])

    def test_dream_done_counts_only_this_pass_not_history(self):
        # supersedes from BEFORE the pass opened must NOT be counted as reconciled
        a = self.s.store("old fact", kind="semantic")
        b = self.s.store("new fact", kind="semantic")
        self.s.supersede(a, b)      # historical — happened before any dream
        rep = dream(self.s, done=True)   # one-shot pass: nothing reconciled in it
        self.assertEqual(rep["applied"], {"reconciled": 0, "woven": 0})
        self.assertIn("nothing needed changing", rep["narrative"])


# Fix A + B (2026-06-16): lifecycle-aware heal — an OLDER task memory closed by a
# NEWER closure memory. The #165->#166 case that resurfaced as still-open.
class TestResolutionHeal(unittest.TestCase):
    # subject words shared by both members so cosine clears RESOLUTION_COSINE;
    # the task/closure marker words give the supersede its direction
    TASK = ("TASKS to do: optimize the prefill performance and the token "
            "footprint report")
    CLOSE = ("optimize the prefill performance and the token footprint report "
             "shipped done")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = file_store(self.tmp.name)
        self.emb = FakeEmbedder()

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def _task_then_close(self, task_kind="semantic", close_kind="episodic"):
        task = self.s.store(self.TASK, kind=task_kind)
        _age(self.s, task, 4)                       # the task is the OLDER memory
        close = self.s.store(self.CLOSE, kind=close_kind)
        embed_memory(self.s, self.emb, task)
        embed_memory(self.s, self.emb, close)
        return task, close

    def test_resolution_proposed_with_old_to_new_direction(self):
        task, close = self._task_then_close()        # cross-kind sem -> epi (the #165/#166 shape)
        res = propose(self.s)["resolutions"]
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["ids"], [task, close])  # old -> new, direction settled

    def test_no_resolution_without_closure_language(self):
        task = self.s.store(self.TASK, kind="semantic")
        _age(self.s, task, 4)
        # a NEWER memory on the same subject but NOT phrased as closure
        other = self.s.store("optimize the prefill performance and the token "
                             "footprint report notes", kind="episodic")
        embed_memory(self.s, self.emb, task)
        embed_memory(self.s, self.emb, other)
        self.assertEqual(propose(self.s)["resolutions"], [])

    def test_no_resolution_when_subjects_unrelated(self):
        task = self.s.store(self.TASK, kind="semantic")
        _age(self.s, task, 4)
        close = self.s.store("the chart axis renders blue done shipped",
                             kind="episodic")  # closure language, different subject
        embed_memory(self.s, self.emb, task)
        embed_memory(self.s, self.emb, close)
        self.assertEqual(propose(self.s)["resolutions"], [])

    def test_resolution_supersede_linked_pair_not_reproposed(self):
        task, close = self._task_then_close()
        self.s.supersede(task, close)
        self.assertEqual(propose(self.s)["resolutions"], [])

    def test_same_kind_resolution_wins_over_merge(self):
        # a same-kind task/closure near-duplicate would also be a 'merge'; the
        # resolution carries direction, so propose() drops it from merges
        task = self.s.store("task optimize prefill token footprint report "
                            "performance latency throughput", kind="semantic")
        _age(self.s, task, 4)
        close = self.s.store("done optimize prefill token footprint report "
                             "performance latency throughput", kind="semantic")
        embed_memory(self.s, self.emb, task)
        embed_memory(self.s, self.emb, close)
        work = propose(self.s)
        self.assertEqual([r["ids"] for r in work["resolutions"]], [[task, close]])
        self.assertNotIn({task, close}, [set(m["ids"]) for m in work["merges"]])

    def test_dream_counts_and_wake_nudge_name_the_resolution(self):
        task, close = self._task_then_close()
        rep = dream(self.s)
        self.assertEqual(rep["counts"]["resolutions"], 1)
        self.assertIn("complete", rep["narrative"].lower())
        # on wake, the remaining-heal nudge names the supersede with direction
        woke = dream(self.s, done=True)
        self.assertIn(f"old=#{task} new=#{close}", woke["narrative"])

    # ----------------------------------------- Fix B: write-time resolution nudge

    def test_write_time_nudge_flags_closing_an_open_task(self):
        task = self.s.store(self.TASK, kind="semantic")
        _age(self.s, task, 4)
        embed_memory(self.s, self.emb, task)
        # a NEW closure memory (cross-kind) — should be flagged as resolving #task
        close = self.s.store(self.CLOSE, kind="episodic")
        sug = supersede_suggestion(self.s, close, self.CLOSE, "episodic",
                                   embedder=self.emb)
        self.assertIsNotNone(sug)
        self.assertEqual(sug["id"], task)
        self.assertEqual(sug["reason"], "resolves")

    def test_write_time_near_duplicate_still_reported(self):
        # the original same-kind near-dup nudge is unchanged, now tagged
        a = self.s.store("the deploy script reads config from env", kind="semantic")
        embed_memory(self.s, self.emb, a)
        text = "the deploy script reads config from env always"
        b = self.s.store(text, kind="semantic")
        sug = supersede_suggestion(self.s, b, text, "semantic", embedder=self.emb)
        self.assertEqual(sug["id"], a)
        self.assertEqual(sug["reason"], "near-duplicate")

    def test_write_time_no_nudge_for_plain_closure_text(self):
        # closure language but nothing task-like to resolve -> no suggestion
        close = self.s.store("the chart axis renders blue done shipped",
                             kind="episodic")
        embed_memory(self.s, self.emb, close)
        self.assertIsNone(supersede_suggestion(
            self.s, close, "the chart axis renders blue done shipped",
            "episodic", embedder=self.emb))


if __name__ == "__main__":
    unittest.main()
