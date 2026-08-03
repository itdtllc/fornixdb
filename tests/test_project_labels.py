"""One project, one spelling.

Capture derives a memory's project from the host's cwd basename, so the same
project fragments the moment a session runs from a differently-cased or renamed
directory — measured live 2026-08-03, when the per-project directory split put
47 rows under `AIMemory` beside 219 `fornixdb` and 19 `FornixDB`, splitting one
thread three ways in brief and hiding 66 rows from `--project fornixdb`.

These cover the fold (write-side), the filter that must still find pre-fold rows
(read-side), and the propose-then-apply normalization of an existing store.
"""

import os
import unittest

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic keyword recall

from fornixdb import context
from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.multistore import set_config


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestAliasParsing(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_first_label_is_canonical(self):
        set_config(self.s, "project_aliases", "fornixdb=engramdb,aimemory")
        groups = context.ordered_alias_groups(self.s)
        self.assertEqual(groups, [["fornixdb", "engramdb", "aimemory"]])
        self.assertEqual(context.canonical_project(self.s, "engramdb"), "fornixdb")
        self.assertEqual(context.canonical_project(self.s, "aimemory"), "fornixdb")

    def test_multiword_label_survives(self):
        # Splitting on whitespace tore "site notes" into two junk labels,
        # so the group's real third member never aliased.
        set_config(self.s, "project_aliases",
                   "site-notes=site-notes-archive=site notes")
        self.assertEqual(context.ordered_alias_groups(self.s),
                         [["site-notes", "site-notes-archive", "site notes"]])
        self.assertEqual(context.canonical_project(self.s, "Site Notes"),
                         "site-notes")

    def test_alias_groups_still_folded_sets(self):
        # The belongs test reads membership only; order and case are irrelevant there.
        set_config(self.s, "project_aliases", "FornixDB=EngramDB; studio=archive")
        self.assertIn({"fornixdb", "engramdb"}, context.alias_groups(self.s))
        self.assertIn({"studio", "archive"}, context.alias_groups(self.s))

    def test_single_label_group_is_not_a_group(self):
        set_config(self.s, "project_aliases", "fornixdb; studio=archive")
        self.assertEqual(context.ordered_alias_groups(self.s), [["studio", "archive"]])

    def test_no_config_no_groups(self):
        self.assertEqual(context.ordered_alias_groups(self.s), [])


class TestCanonicalProject(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_dominant_spelling_wins_without_config(self):
        for _ in range(3):
            self.s.store("a fornixdb note", project="fornixdb")
        self.s.store("another one", project="FornixDB")
        self.assertEqual(context.canonical_project(self.s, "FornixDB"), "fornixdb")
        self.assertEqual(context.canonical_project(self.s, "FORNIXDB"), "fornixdb")

    def test_dominant_spelling_can_be_the_capitalized_one(self):
        # No lowercase bias: RAndDLab outnumbers randdlab, so it wins.
        for _ in range(4):
            self.s.store("a note", project="RAndDLab")
        self.assertEqual(context.canonical_project(self.s, "randdlab"), "RAndDLab")

    def test_a_misspelling_is_not_a_case_variant(self):
        # `randlab` (one D) does not fold into `RAndDLab` (two) — case is
        # the only thing we assume; a typo is a merge only the owner can declare.
        for _ in range(4):
            self.s.store("a note", project="RAndDLab")
        self.assertEqual(context.canonical_project(self.s, "randlab"), "randlab")

    def test_config_overrides_dominant_spelling(self):
        for _ in range(9):
            self.s.store("a note", project="AIMemory")
        set_config(self.s, "project_aliases", "fornixdb=AIMemory")
        self.assertEqual(context.canonical_project(self.s, "AIMemory"), "fornixdb")

    def test_unknown_label_canonicalizes_to_itself(self):
        # A project's first memory must not need config to be storable.
        self.assertEqual(context.canonical_project(self.s, "  BrandNew "), "BrandNew")

    def test_none_and_empty_pass_through(self):
        self.assertIsNone(context.canonical_project(self.s, None))
        self.assertEqual(context.canonical_project(self.s, "   "), "")

    def test_two_different_names_never_merge_on_their_own(self):
        # Only case-folding and owner-declared aliases merge. Deciding that two
        # unrelated names mean one project is the owner's call.
        self.s.store("a", project="videos")
        self.s.store("b", project="archive")
        self.assertEqual(context.canonical_project(self.s, "archive"), "archive")


class TestDeclarationResolvesToCanonical(unittest.TestCase):
    """A project has one name however the user reaches for it."""

    def setUp(self):
        self.s = mem_store()
        self.s.store("a note", project="RAndDLab")
        set_config(self.s, "project_aliases",
                   "RAndDLab=randlab,R&D,R&D Lab,RnD")

    def tearDown(self):
        self.s.close()

    def test_ampersand_spelling_resolves(self):
        # "&" is not usable in a directory name, so the project is stored
        # RAndDLab — but the owner says "R&D".
        for prompt in ("pick up the R&D world", "let's work on R&D",
                       "continue RnD", "back to randlab"):
            self.assertEqual(context.detect_active_project(self.s, prompt),
                             "RAndDLab", prompt)

    def test_longest_label_wins_at_the_same_position(self):
        # "R&D" and "R&D Lab" both match at the same offset; the more
        # specific one must win rather than set iteration order deciding.
        self.assertEqual(
            context.detect_active_project(self.s, "switch to R&D Lab"),
            "RAndDLab")

    def test_still_needs_a_cue(self):
        self.assertIsNone(
            context.detect_active_project(self.s, "the R&D rules mention dragons"))

    def test_earliest_label_still_wins_across_positions(self):
        self.s.store("b note", project="fornixdb")
        self.assertEqual(
            context.detect_active_project(self.s, "switch to fornixdb, not R&D"),
            "fornixdb")


class TestWriteSideFold(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        for _ in range(3):
            self.s.store("baseline", project="fornixdb")

    def tearDown(self):
        self.s.close()

    def test_store_folds_case_variant(self):
        mid = self.s.store("from a differently-cased cwd", project="FornixDB")
        row = self.s.conn.execute(
            "SELECT project FROM memory WHERE id = ?", (mid,)).fetchone()
        self.assertEqual(row["project"], "fornixdb")

    def test_store_folds_declared_alias(self):
        set_config(self.s, "project_aliases", "fornixdb=AIMemory")
        mid = self.s.store("captured from the AIMemory dir", project="AIMemory")
        row = self.s.conn.execute(
            "SELECT project FROM memory WHERE id = ?", (mid,)).fetchone()
        self.assertEqual(row["project"], "fornixdb")

    def test_record_session_folds_too(self):
        self.s.record_session("s1", project="FornixDB")
        row = self.s.conn.execute(
            "SELECT project FROM session WHERE id = ?", ("s1",)).fetchone()
        self.assertEqual(row["project"], "fornixdb")

    def test_no_project_is_left_alone(self):
        mid = self.s.store("unlabelled")
        row = self.s.conn.execute(
            "SELECT project FROM memory WHERE id = ?", (mid,)).fetchone()
        self.assertIsNone(row["project"])


class TestReadSideFilter(unittest.TestCase):
    """A filter must find rows written BEFORE the fold — an old store, or a
    read-only peer that will never be rewritten."""

    def setUp(self):
        self.s = mem_store()
        # Write variants directly, bypassing store()'s fold, to stand in for a
        # store that predates canonicalization.
        for proj in ("fornixdb", "fornixdb", "FornixDB", "AIMemory"):
            self.s.conn.execute(
                "INSERT INTO memory (kind, event_time, recorded_time, project, gist) "
                "VALUES ('semantic', '2026-08-01T00:00:00', '2026-08-01T00:00:00', ?, ?)",
                (proj, f"ladder note in {proj}"))
        self.s.conn.execute(
            "INSERT INTO memory (kind, event_time, recorded_time, project, gist) "
            "VALUES ('semantic', '2026-08-01T00:00:00', '2026-08-01T00:00:00', "
            "'videos', 'ladder note in videos')")
        self.s.conn.commit()
        self.s.rebuild_fts() if hasattr(self.s, "rebuild_fts") else None
        set_config(self.s, "project_aliases", "fornixdb=AIMemory")

    def tearDown(self):
        self.s.close()

    def test_clause_covers_every_spelling(self):
        sql, params = self.s._project_clause("fornixdb")
        self.assertIn("IN", sql)
        self.assertEqual(set(params), {"fornixdb", "FornixDB", "AIMemory"})

    def test_clause_is_equality_when_only_one_spelling(self):
        # The common (already-normalized) case stays index-friendly.
        sql, params = self.s._project_clause("videos")
        self.assertEqual(sql, "m.project = ?")
        self.assertEqual(params, ["videos"])

    def test_no_project_no_clause(self):
        self.assertEqual(self.s._project_clause(None), ("", []))

    def test_variant_query_finds_the_canonical_rows(self):
        # Asking by any spelling returns the whole project.
        for asked in ("fornixdb", "FornixDB", "AIMemory"):
            sql, params = self.s._project_clause(asked)
            n = self.s.conn.execute(
                f"SELECT COUNT(*) FROM memory m WHERE {sql}", params).fetchone()[0]
            self.assertEqual(n, 4, f"querying {asked!r} under-reported")

    def test_filter_does_not_leak_other_projects(self):
        sql, params = self.s._project_clause("fornixdb")
        rows = self.s.conn.execute(
            f"SELECT gist FROM memory m WHERE {sql}", params).fetchall()
        self.assertTrue(all("videos" not in r["gist"] for r in rows))

    def test_timeline_finds_variant_rows(self):
        rows = self.s.timeline("2026-08-01T00:00:00", "2026-08-02T00:00:00",
                               project="fornixdb", limit=50)
        self.assertEqual(len(rows), 4)


class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()
        for proj, n in (("fornixdb", 3), ("FornixDB", 2), ("AIMemory", 1), ("videos", 1)):
            for _ in range(n):
                self.s.conn.execute(
                    "INSERT INTO memory (kind, event_time, recorded_time, project, gist) "
                    "VALUES ('semantic', '2026-08-01T00:00:00', "
                    "'2026-08-01T00:00:00', ?, 'note')", (proj,))
        self.s.conn.execute(
            "INSERT INTO session (id, project) VALUES ('s1', 'FornixDB')")
        self.s.conn.commit()
        set_config(self.s, "project_aliases", "fornixdb=AIMemory")

    def tearDown(self):
        self.s.close()

    def _counts(self):
        return dict(self.s.conn.execute(
            "SELECT project, COUNT(*) FROM memory GROUP BY project").fetchall())

    def test_dry_run_changes_nothing(self):
        res = self.s.normalize_projects()
        self.assertFalse(res["applied"])
        self.assertEqual(res["memories"], 3)      # 2 FornixDB + 1 AIMemory
        self.assertEqual(res["sessions"], 1)
        self.assertEqual(self._counts(), {"fornixdb": 3, "FornixDB": 2,
                                          "AIMemory": 1, "videos": 1})

    def test_apply_rewrites_memory_and_session(self):
        res = self.s.normalize_projects(apply=True)
        self.assertTrue(res["applied"])
        self.assertEqual(self._counts(), {"fornixdb": 6, "videos": 1})
        self.assertEqual(self.s.conn.execute(
            "SELECT project FROM session WHERE id='s1'").fetchone()["project"],
            "fornixdb")

    def test_apply_is_idempotent(self):
        self.s.normalize_projects(apply=True)
        again = self.s.normalize_projects(apply=True)
        self.assertEqual(again["changes"], [])
        self.assertFalse(again["applied"])
        self.assertEqual(self._counts(), {"fornixdb": 6, "videos": 1})

    def test_project_labels_view_marks_variants(self):
        rows = {r["label"]: r for r in self.s.project_labels()}
        self.assertEqual(rows["FornixDB"]["canonical"], "fornixdb")
        self.assertEqual(rows["fornixdb"]["canonical"], "fornixdb")
        self.assertEqual(rows["videos"]["canonical"], "videos")

    def test_session_only_label_is_listed(self):
        self.s.conn.execute(
            "INSERT INTO session (id, project) VALUES ('s2', 'SideProject')")
        self.s.conn.commit()
        rows = {r["label"]: r["memories"] for r in self.s.project_labels()}
        self.assertEqual(rows["SideProject"], 0)

    def test_status_tips_treats_case_variants_as_one_thread(self):
        s = mem_store()
        older = s.store("ed.1 — where fornixdb stands", project="fornixdb",
                        event_time="2026-08-01T00:00:00")
        newer = s.store("ed.2 — where fornixdb stands", project="fornixdb",
                        event_time="2026-08-02T00:00:00")
        s.supersede(older, newer)
        # A second thread, written under a variant spelling before the fold.
        s.conn.execute(
            "INSERT INTO memory (kind, event_time, recorded_time, project, gist) "
            "VALUES ('semantic','2026-07-01T00:00:00','2026-07-01T00:00:00',"
            "'FornixDB','ed.0 — stale variant')")
        vid = s.conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]
        s.conn.execute(
            "INSERT INTO memory (kind, event_time, recorded_time, project, gist, "
            "superseded_by) VALUES ('semantic','2026-06-01T00:00:00',"
            "'2026-06-01T00:00:00','FornixDB','older',?)", (vid,))
        s.conn.commit()
        tips = s.status_tips()
        self.assertEqual(len(tips), 1, "one project must not show as two threads")
        self.assertEqual(tips[0]["id"], newer)
        s.close()


if __name__ == "__main__":
    unittest.main()
