"""Unified context scoping: a memory belongs to the active project if its project
OR any non-structural topic matches the active label or an alias; the active
project can be DECLARED in a prompt ("continue the fornixdb project") and sticks
for the session, with aliases bridging messy historical names."""

import os
import unittest

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic keyword recall

from fornixdb import context
from fornixdb.core import MemoryStore, PROJECT_MISMATCH_PENALTY
from fornixdb.db import connect
from fornixdb.multistore import set_config
from fornixdb.proactive import relevant_memories, resolve_active_project


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestAliases(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        set_config(self.s, "project_aliases",
                   "fornixdb=engramdb,aimemory; videos=elira")

    def tearDown(self):
        self.s.close()

    def test_groups_parsed(self):
        groups = context.alias_groups(self.s)
        self.assertIn({"fornixdb", "engramdb", "aimemory"}, groups)
        self.assertIn({"videos", "elira"}, groups)

    def test_aliases_for_excludes_self_and_is_case_insensitive(self):
        self.assertEqual(context.aliases_for(self.s, "FornixDB"),
                         {"engramdb", "aimemory"})
        self.assertEqual(context.aliases_for(self.s, "engramdb"),
                         {"fornixdb", "aimemory"})

    def test_aliases_for_unknown_is_empty(self):
        self.assertEqual(context.aliases_for(self.s, "retirementestimator"), set())

    def test_declarable_includes_projects_and_aliases(self):
        self.s.store("a", name="a", project="RetirementEstimator")
        self.s.store("b", name="b", project="fornixdb")
        labels = context.declarable_labels(self.s)
        self.assertIn("retirementestimator", labels)
        self.assertIn("fornixdb", labels)
        self.assertIn("engramdb", labels)   # from aliases
        self.assertIn("elira", labels)


class TestPromptDetection(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.s.store("x", name="x", project="fornixdb")
        self.s.store("y", name="y", project="RetirementEstimator")
        set_config(self.s, "project_aliases", "fornixdb=engramdb,aimemory")

    def tearDown(self):
        self.s.close()

    def test_cue_plus_label_detected(self):
        self.assertEqual(
            context.detect_active_project(self.s, "let's continue the fornixdb project"),
            "fornixdb")
        self.assertEqual(
            context.detect_active_project(self.s, "switch to RetirementEstimator now"),
            "retirementestimator")

    def test_alias_name_is_declarable(self):
        self.assertEqual(
            context.detect_active_project(self.s, "working on engramdb today"),
            "engramdb")

    def test_no_cue_means_no_detection(self):
        # a bare mention mid-task must not flip context
        self.assertIsNone(
            context.detect_active_project(self.s, "the fornixdb floor code has a bug"))

    def test_cue_without_known_label_is_none(self):
        self.assertIsNone(
            context.detect_active_project(self.s, "let's continue with the thing"))

    def test_earliest_label_wins(self):
        self.assertEqual(
            context.detect_active_project(
                self.s, "switch to fornixdb, not RetirementEstimator"),
            "fornixdb")


class TestSessionStickiness(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        self.s.store("x", name="x", project="fornixdb")
        self.s.store("y", name="y", project="videos")
        set_config(self.s, "project_aliases", "videos=elira")

    def tearDown(self):
        self.s.close()

    def test_declaration_persists_for_session(self):
        self.assertIsNone(context.session_active_project(self.s, "s1"))
        context.maybe_set_session_project(self.s, "s1", "continue the fornixdb project")
        self.assertEqual(context.session_active_project(self.s, "s1"), "fornixdb")
        # a later non-declaring prompt leaves it in place
        context.maybe_set_session_project(self.s, "s1", "now fix the bug")
        self.assertEqual(context.session_active_project(self.s, "s1"), "fornixdb")

    def test_redeclaration_switches(self):
        context.maybe_set_session_project(self.s, "s1", "continue the fornixdb project")
        context.maybe_set_session_project(self.s, "s1", "switch to videos")
        self.assertEqual(context.session_active_project(self.s, "s1"), "videos")

    def test_sessions_are_independent(self):
        context.maybe_set_session_project(self.s, "s1", "continue the fornixdb project")
        self.assertIsNone(context.session_active_project(self.s, "s2"))


class TestResolvePrecedence(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_pin_beats_session_and_cwd(self):
        set_config(self.s, "active_project", "Pinned")
        set_config(self.s, context._SESSION_KEY + "s1", "Declared")
        self.assertEqual(resolve_active_project(self.s, "Cwd", session_id="s1"),
                         "Pinned")

    def test_session_beats_cwd(self):
        set_config(self.s, context._SESSION_KEY + "s1", "Declared")
        self.assertEqual(resolve_active_project(self.s, "Cwd", session_id="s1"),
                         "Declared")

    def test_cwd_when_nothing_declared(self):
        self.assertEqual(resolve_active_project(self.s, "Cwd", session_id="s1"),
                         "Cwd")


class TestBelongsByTopicAndAlias(unittest.TestCase):
    BASE = 0.45

    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def _floor(self, *, project=None, topics=(), aliases=()):
        row = {"recall_count": 0, "helpful_count": 0, "surfaced_count": 0,
               "project": project, "topics": list(topics)}
        return self.s.effective_floor(row, self.BASE, active_project="fornixdb",
                                      aliases=set(aliases))

    def test_topic_match_belongs(self):
        # off-project but a topic matches the active context → belongs
        self.assertEqual(self._floor(project="RetirementEstimator",
                                     topics=["fornixdb", "ranking"]), self.BASE)

    def test_alias_topic_match_belongs(self):
        self.assertEqual(self._floor(project=None, topics=["engramdb"],
                                     aliases=["engramdb", "aimemory"]), self.BASE)

    def test_alias_project_match_belongs(self):
        self.assertEqual(self._floor(project="engramdb",
                                     aliases=["engramdb", "aimemory"]), self.BASE)

    def test_off_context_tagged_memory_penalized(self):
        f = self._floor(project="RetirementEstimator", topics=["finance"])
        self.assertAlmostEqual(f - self.BASE, PROJECT_MISMATCH_PENALTY, places=4)

    def test_only_structural_topics_is_neutral(self):
        # a cross-cutting reference tagged only "reference"/"feedback" belongs
        # everywhere — structural topics don't make it look off-context
        self.assertEqual(self._floor(project=None,
                                     topics=["reference", "feedback"]), self.BASE)

    def test_no_tags_is_neutral(self):
        self.assertEqual(self._floor(project=None, topics=[]), self.BASE)


class TestTopicsAttachedInRelevantMemories(unittest.TestCase):
    """relevant_memories must attach real topics so a topic-only match surfaces."""

    def setUp(self):
        self.s = mem_store()
        self.s._resolve_embedder = lambda *a, **k: object()  # vectors "on"

    def tearDown(self):
        self.s.close()

    def test_topic_match_surfaces_off_project_memory(self):
        mid = self.s.store("alpha", name="alpha", project="RetirementEstimator")
        # give it the active-context topic via the real topic tables
        self.s.conn.execute("INSERT INTO topic(name) VALUES ('fornixdb')")
        tid = self.s.conn.execute("SELECT id FROM topic WHERE name='fornixdb'").fetchone()[0]
        self.s.conn.execute("INSERT INTO memory_topic(memory_id, topic_id) VALUES (?,?)",
                            (mid, tid))
        self.s.conn.commit()
        # cosine 0.50 clears base 0.45; without the topic it'd be off-project and
        # penalized to 0.60 (dropped). With the topic it belongs → surfaces.
        self.s.recall = lambda *a, **k: [
            {"id": mid, "kind": "semantic", "gist": "g", "vec_cos": 0.50,
             "recall_count": 0, "helpful_count": 0, "surfaced_count": 0,
             "project": "RetirementEstimator"}]
        out = relevant_memories(self.s, "x", floor=0.45, active_project="fornixdb")
        self.assertEqual([r["id"] for r in out], [mid])


if __name__ == "__main__":
    unittest.main()
