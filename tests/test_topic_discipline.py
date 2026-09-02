"""Topics are half the glue the field clusters on.

The other half is links. A memory with neither can be recalled by its words but
can never be found BY ASSOCIATION, and a project whose memories carry no topics
gives the field no edges to settle on. Measured on a lived-in store, topic
coverage of new memories fell from 89% in one month to 32% in the next, and the
projects written during that stretch are the ones whose beats stopped settling.
"""
import contextlib
import io
import os
import tempfile
import unittest

from fornixdb.cli import main
from fornixdb.consolidate import (MAX_TOPIC_SUGGESTIONS, TOPIC_MIN_CLUSTER,
                                  TOPIC_MIN_PROJECT_DF, _is_topic_shaped,
                                  _topic_candidates, _topicless_scan, propose)
from fornixdb.core import MemoryStore
from fornixdb.db import connect


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestWritePathNudge(unittest.TestCase):
    """Say it where the memory is written, not in a report nobody reads."""

    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        main(["--db", self.db, "init"])

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(self.db + suffix)
            except OSError:
                pass

    def _store(self, *extra):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            main(["--db", self.db, "store", "--gist", "a memory", *extra])
        return err.getvalue()

    def test_storing_without_a_topic_says_so(self):
        self.assertIn("no --topic", self._store())

    def test_the_note_names_the_repair(self):
        # a nudge that doesn't say what to do is just noise
        out = self._store()
        self.assertIn("--topic", out)
        self.assertIn("tag ", out)

    def test_storing_with_a_topic_is_quiet(self):
        self.assertNotIn("no --topic", self._store("--topic", "alpha"))


class TestTopicSuggestions(unittest.TestCase):

    def _corpus(self, store):
        for i in range(8):
            store.store(f"the widget pipeline run {i} finished cleanly",
                        topics=["pipeline"], embedder=False)
        store.store("another widget note about the pipeline", topics=["pipeline"],
                    embedder=False)

    def test_existing_topic_names_are_preferred(self):
        # reusing a name the store already uses creates an EDGE to the memories
        # carrying it; coining a fresh word connects the memory to nothing
        s = mem_store()
        self._corpus(s)
        mid = s.store("a widget pipeline change nobody tagged", embedder=False)
        sug = _topic_candidates(s, [{"id": mid, "gist": s.show(mid)["gist"]}], None)
        self.assertIn("pipeline", sug[mid])
        self.assertEqual(sug[mid][0], "pipeline")

    def test_suggestions_are_capped(self):
        s = mem_store()
        self._corpus(s)
        mid = s.store("widget pipeline run finished cleanly again", embedder=False)
        sug = _topic_candidates(s, [{"id": mid, "gist": s.show(mid)["gist"]}], None)
        self.assertLessEqual(len(sug.get(mid, [])), MAX_TOPIC_SUGGESTIONS)

    def test_a_tiny_store_with_no_vocabulary_suggests_nothing(self):
        s = mem_store()
        mid = s.store("a lone untagged memory", embedder=False)
        self.assertEqual(
            _topic_candidates(s, [{"id": mid, "gist": "a lone untagged memory"}],
                              None), {})


class TestSuggestionsAreSubjects(unittest.TestCase):
    """A suggestion has to name what the memory is ABOUT.

    The first bulk tagging pass on a real store had to be done by hand-written
    rule instead of by the worklist, because the worklist offered "now",
    "never", "day", "users" and "app" — words that pass the retrieval stopword
    filter, appear all over a project, and connect everything to everything.
    """

    def _corpus(self, store, phrase):
        for i in range(8):
            store.store(f"{phrase} run {i} finished cleanly", embedder=False,
                        topics=["pipeline"])

    def test_a_time_word_is_never_suggested(self):
        s = mem_store()
        for i in range(8):
            s.store(f"the deploy never ran today, day {i}", topics=["pipeline"],
                    embedder=False)
        mid = s.store("the deploy never ran today either", embedder=False)
        picks = _topic_candidates(
            s, [{"id": mid, "gist": s.show(mid)["gist"]}], None).get(mid, [])
        for junk in ("never", "today", "day"):
            self.assertNotIn(junk, picks)

    def test_a_generic_actor_is_never_suggested(self):
        s = mem_store()
        for i in range(8):
            s.store(f"users open the app to check widget {i}", topics=["pipeline"],
                    embedder=False)
        mid = s.store("users open the app to check another widget", embedder=False)
        picks = _topic_candidates(
            s, [{"id": mid, "gist": s.show(mid)["gist"]}], None).get(mid, [])
        for junk in ("users", "app"):
            self.assertNotIn(junk, picks)

    def test_a_real_subject_still_survives_the_filter(self):
        # the filter must not be so wide that it silences the useful half
        s = mem_store()
        self._corpus(s, "the widget pipeline")
        mid = s.store("a widget pipeline change nobody tagged", embedder=False)
        picks = _topic_candidates(
            s, [{"id": mid, "gist": s.show(mid)["gist"]}], None).get(mid, [])
        self.assertIn("pipeline", picks)

    def test_a_topic_only_one_memory_carries_is_not_suggested(self):
        # a coinage is not vocabulary: suggesting it spreads a word nobody has
        # committed to, and the point of preferring known topics is that they
        # already connect something
        s = mem_store()
        self.assertGreater(TOPIC_MIN_CLUSTER, 1)
        s.store("the solitary widget note", topics=["solitary"], embedder=False)
        mid = s.store("another solitary widget remark", embedder=False)
        picks = _topic_candidates(
            s, [{"id": mid, "gist": s.show(mid)["gist"]}], None).get(mid, [])
        self.assertNotIn("solitary", picks)

    def test_a_word_in_only_two_project_memories_is_not_suggested(self):
        s = mem_store()
        self.assertGreater(TOPIC_MIN_PROJECT_DF, 2)
        for i in range(8):
            s.store(f"pipeline note {i}", topics=["pipeline"], embedder=False)
        s.store("a note mentioning quicksilver", topics=["pipeline"], embedder=False)
        mid = s.store("another note mentioning quicksilver", embedder=False)
        picks = _topic_candidates(
            s, [{"id": mid, "gist": s.show(mid)["gist"]}], None).get(mid, [])
        self.assertNotIn("quicksilver", picks)

    def test_identifier_fragments_are_not_subjects(self):
        for frag in ("e01", "v2", "72b", "2026"):
            self.assertFalse(_is_topic_shaped(frag), frag)
        for real in ("model2vec", "sqlite3", "pipeline"):
            self.assertTrue(_is_topic_shaped(real), real)

    def test_the_retrieval_stopword_list_is_untouched(self):
        # core's list protects RANKING, where a shared "day" is weak but real
        # evidence. Widening it there would change scores; this filter must
        # stay on the topic side of the fence.
        from fornixdb.core import _STOPWORDS
        for w in ("now", "never", "day", "users", "app"):
            self.assertNotIn(w, _STOPWORDS)


class TestUntag(unittest.TestCase):
    """`tag` added and nothing removed, so any bulk tagging pass was
    irreversible through the CLI — which is what made the first one timid."""

    def test_a_topic_can_be_taken_off(self):
        s = mem_store()
        mid = s.store("a memory", topics=["alpha", "beta"], embedder=False)
        self.assertTrue(s.untag(mid, "alpha"))
        self.assertEqual(s.show(mid)["topics"], ["beta"])

    def test_untagging_a_topic_it_does_not_carry_says_so(self):
        s = mem_store()
        mid = s.store("a memory", topics=["alpha"], embedder=False)
        self.assertFalse(s.untag(mid, "gamma"))
        self.assertEqual(s.show(mid)["topics"], ["alpha"])

    def test_case_and_space_are_normalized_the_way_tag_normalizes_them(self):
        s = mem_store()
        mid = s.store("a memory", topics=["alpha"], embedder=False)
        self.assertTrue(s.untag(mid, "  ALPHA "))
        self.assertEqual(s.show(mid)["topics"], [])

    def test_the_last_use_takes_the_topic_name_with_it(self):
        # a vocabulary word no memory uses connects nothing, and leaving it
        # behind would keep it in the suggestion set
        s = mem_store()
        mid = s.store("a memory", topics=["alpha"], embedder=False)
        s.untag(mid, "alpha")
        self.assertEqual(
            s.conn.execute("SELECT count(*) FROM topic WHERE name='alpha'"
                           ).fetchone()[0], 0)

    def test_a_topic_another_memory_still_uses_survives(self):
        s = mem_store()
        keep = s.store("keeper", topics=["alpha"], embedder=False)
        drop = s.store("dropper", topics=["alpha"], embedder=False)
        s.untag(drop, "alpha")
        self.assertEqual(s.show(keep)["topics"], ["alpha"])

    def test_untagging_removes_no_memory(self):
        s = mem_store()
        mid = s.store("a memory", topics=["alpha"], embedder=False)
        s.untag(mid, "alpha")
        self.assertIsNotNone(s.show(mid))

    def test_it_round_trips_through_the_cli(self):
        db = tempfile.mktemp(suffix=".db")
        try:
            main(["--db", db, "init"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                main(["--db", db, "store", "--gist", "a memory", "--topic", "alpha"])
                main(["--db", db, "tag", "1", "beta"])
                rc_ok = main(["--db", db, "untag", "1", "beta"])
                rc_miss = main(["--db", db, "untag", "1", "beta"])
                main(["--db", db, "show", "1"])
            self.assertEqual(rc_ok, None if rc_ok is None else 0)
            self.assertEqual(rc_miss, 1)
            self.assertIn("untagged 'beta'", out.getvalue())
            self.assertNotIn("beta", out.getvalue().rsplit("topics:", 1)[-1]
                             .splitlines()[0])
        finally:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(db + suffix)
                except OSError:
                    pass


class TestTopiclessWorklist(unittest.TestCase):

    def test_untagged_live_memories_are_listed(self):
        s = mem_store()
        tagged = s.store("tagged one", topics=["alpha"], embedder=False)
        untagged = s.store("untagged one", embedder=False)
        ids = {t["id"] for t in _topicless_scan(s, set())}
        self.assertIn(untagged, ids)
        self.assertNotIn(tagged, ids)

    def test_superseded_memories_are_not_listed(self):
        s = mem_store()
        old = s.store("an old untagged memory", embedder=False)
        new = s.store("its replacement", topics=["alpha"], embedder=False)
        s.supersede(old, new)
        self.assertNotIn(old, {t["id"] for t in _topicless_scan(s, set())})

    def test_rows_already_queued_for_distillation_are_skipped(self):
        s = mem_store()
        mid = s.store("an untagged memory", embedder=False)
        self.assertEqual(_topicless_scan(s, {mid}), [])

    def test_newest_first(self):
        s = mem_store()
        first = s.store("first untagged", embedder=False)
        second = s.store("second untagged", embedder=False)
        self.assertEqual([t["id"] for t in _topicless_scan(s, set())],
                         [second, first])

    def test_it_reaches_the_dream_worklist(self):
        s = mem_store()
        s.store("an untagged memory", embedder=False)
        self.assertTrue(propose(s)["topicless"])

    def test_a_fully_tagged_store_proposes_nothing(self):
        s = mem_store()
        s.store("a tagged memory", topics=["alpha"], embedder=False)
        self.assertEqual(propose(s)["topicless"], [])


class TestMcpCanWriteTopics(unittest.TestCase):
    """Before this the MCP write path had no topics parameter at all, so a
    memory stored through it could never carry one."""

    def _server(self):
        from fornixdb.adapters.mcp_server import FornixMCP
        srv = FornixMCP.__new__(FornixMCP)
        srv.store = mem_store()
        srv.stores = [("own", srv.store)]
        srv._session_writes = []
        return srv

    def test_remember_stores_topics(self):
        srv = self._server()
        srv.remember("a title", "some content", topics=["alpha", "bravo"])
        mid = srv._session_writes[-1]
        self.assertEqual(set(srv.store.topics_for([mid])[mid]), {"alpha", "bravo"})

    def test_remember_without_topics_still_works(self):
        srv = self._server()
        srv.remember("a title", "some content")
        mid = srv._session_writes[-1]
        self.assertEqual(srv.store.topics_for([mid]), {})

    def test_blank_topics_are_ignored(self):
        srv = self._server()
        srv.remember("a title", "some content", topics=["  ", "alpha"])
        mid = srv._session_writes[-1]
        self.assertEqual(srv.store.topics_for([mid])[mid], ["alpha"])

    def test_the_tool_schema_advertises_topics(self):
        from fornixdb.adapters.mcp_server import TOOLS
        remember = next(t for t in TOOLS if t["name"] == "remember")
        self.assertIn("topics", remember["inputSchema"]["properties"])


if __name__ == "__main__":
    unittest.main()


class TestAssociationBacklogIsVisible(unittest.TestCase):
    """A worklist that shows fifteen and says nothing about the rest reads as
    "there are fifteen". Weaving is the purely-additive move here and the one
    that lets the field cluster at all, so the size of the backlog matters.
    """

    def test_pair_totals_report_the_uncapped_counts(self):
        from fornixdb.consolidate import MAX_PAIR_PROPOSALS
        from test_vectors import FakeEmbedder
        s = MemoryStore(conn=connect(":memory:"))
        emb = FakeEmbedder()
        # many mutually-similar rows: far more association candidates than the cap
        for i in range(MAX_PAIR_PROPOSALS + 12):
            s.store(f"the deploy pipeline reads its config from the environment {i}",
                    embedder=emb)
        w = propose(s)
        self.assertLessEqual(len(w["associations"]), MAX_PAIR_PROPOSALS)
        self.assertIn("associations", w["pair_totals"])
        self.assertGreaterEqual(w["pair_totals"]["associations"],
                                len(w["associations"]))

    def test_totals_present_even_when_nothing_is_found(self):
        s = MemoryStore(conn=connect(":memory:"))
        s.store("a lone memory", embedder=False)
        self.assertEqual(propose(s)["pair_totals"],
                         {"merges": 0, "contradictions": 0, "associations": 0})


class TestWeavePasses(unittest.TestCase):
    """A pass proposes a reviewable number of pairs, so clearing a real backlog
    takes many. On a lived-in store that was 24 runs by hand.
    """

    def _run(self, reports, argv_extra):
        """Drive the CLI's weave loop against scripted dream() reports."""
        import fornixdb.consolidate as consolidate
        calls = []

        def fake_dream(store, weave=False, done=False):
            calls.append(weave)
            rep = reports[min(len(calls) - 1, len(reports) - 1)]
            return rep

        db = tempfile.mktemp(suffix=".db")
        main(["--db", db, "init"])
        real = consolidate.dream
        consolidate.dream = fake_dream
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                main(["--db", db, "dream", *argv_extra])
            return calls, out.getvalue()
        finally:
            consolidate.dream = real
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(db + suffix)
                except OSError:
                    pass

    def _report(self, assoc, woven):
        work = {k: [] for k in ("distill", "gists", "merges", "contradictions",
                                "associations", "resolutions", "reality",
                                "chronic", "reproject", "markdown_stale",
                                "topicless")}
        work["associations"] = [{"ids": [1, 2], "cosine": 0.7,
                                 "kinds": ["semantic", "semantic"],
                                 "gists": ["a", "b"]}] * assoc
        work["pair_totals"] = {"merges": 0, "contradictions": 0,
                               "associations": assoc}
        return {"narrative": "…", "work": work, "woven": woven,
                "counts": {}, "dials": []}

    def test_it_stops_when_the_backlog_empties(self):
        reports = [self._report(2, 2), self._report(2, 2), self._report(0, 0)]
        calls, out = self._run(reports, ["--weave", "--passes", "20"])
        self.assertEqual(len(calls), 3)          # not 20
        self.assertIn("backlog empty", out)

    def test_it_honors_the_pass_cap(self):
        calls, out = self._run([self._report(5, 5)], ["--weave", "--passes", "4"])
        self.assertEqual(len(calls), 4)
        self.assertIn("still proposed", out)

    def test_one_pass_is_the_default_and_says_nothing_extra(self):
        calls, out = self._run([self._report(5, 5)], ["--weave"])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("pass(es)", out)

    def test_passes_without_weave_does_not_loop(self):
        calls, _ = self._run([self._report(5, 0)], ["--passes", "20"])
        self.assertEqual(len(calls), 1)
