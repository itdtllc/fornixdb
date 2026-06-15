"""Schema v2: name in FTS + chunked detail embeddings (eval-found gaps)."""

import tempfile
import unittest
from pathlib import Path

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.vectors import backfill, similar
from test_vectors import FakeEmbedder


class TestNameInFTS(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))

    def tearDown(self):
        self.s.close()

    def test_recall_by_title_words_keyword_only(self):
        mid = self.s.store("agreed definition for the simulation outcome",
                           "detail", name="monte-carlo-success-criteria")
        rows = self.s.recall("monte carlo success criteria", embedder=False)
        self.assertEqual(rows[0]["id"], mid)  # gist shares no query tokens

    def test_name_change_reindexes(self):
        mid = self.s.store("some gist", "detail", name="old-slug")
        self.s.set_name(mid, "fresh-handle")
        self.assertFalse(self.s.recall("old slug", embedder=False))
        hits = self.s.recall("fresh handle", embedder=False)
        self.assertEqual(hits[0]["id"], mid)

    def test_name_handoff_on_supersede_follows(self):
        old = self.s.store("v1 of the rule", "detail", name="the-rule")
        new = self.s.store("v2 of the rule", "detail")
        self.s.supersede(old, new)
        hits = self.s.recall("the rule", embedder=False)
        self.assertEqual(hits[0]["id"], new)  # live version owns the name hit


class TestDetailChunks(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))
        self.emb = FakeEmbedder()

    def tearDown(self):
        self.s.close()

    def test_fact_deep_in_detail_is_findable(self):
        filler = "lorem ipsum dolor sit amet " * 60          # ~1600 chars
        detail = filler + " the automobile parked in the harbor warehouse"
        mid = self.s.store("session log for tuesday", detail, kind="episodic")
        other = self.s.store("unrelated topic", "nothing here")
        backfill(self.s, self.emb)
        # the old single-vector scheme truncated detail at 500 chars — this
        # fact lives past 1600 and the query shares zero gist tokens
        ranked = similar(self.s, self.emb, "vehicle in the harbor warehouse")
        self.assertEqual(ranked[0][0], mid)
        self.assertGreater(ranked[0][1], 0.3)

    def test_memory_scores_by_best_chunk_not_average(self):
        relevant = self.s.store("a", "x " * 500 + "automobile harbor crane")
        diluted = self.s.store("b", "automobile " + "y " * 800)
        backfill(self.s, self.emb)
        best = dict(similar(self.s, self.emb, "automobile harbor crane"))
        self.assertGreater(best[relevant], best[diluted])

    def test_chunk_count_bounded_and_reembed_clears_stale(self):
        mid = self.s.store("g", "z" * 50000)                  # would be 70+ chunks
        backfill(self.s, self.emb)
        n = self.s.conn.execute("SELECT count(*) c FROM embedding WHERE memory_id=?",
                                (mid,)).fetchone()["c"]
        self.assertLessEqual(n, 8)
        self.s.set_gist(mid, "short now")                     # drops vectors
        self.s.conn.execute("UPDATE memory SET detail='tiny' WHERE id=?", (mid,))
        backfill(self.s, self.emb)
        n = self.s.conn.execute("SELECT count(*) c FROM embedding WHERE memory_id=?",
                                (mid,)).fetchone()["c"]
        self.assertEqual(n, 2)                                # head + one window


class TestV1Migration(unittest.TestCase):
    def test_v1_store_migrates_and_recalls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v1.db"
            s = MemoryStore(db_path=path)
            s.store("the migration survivor", "old detail",
                    name="survivor-slug")
            # devolve to the v1 shapes: 2-column FTS, chunkless embeddings
            s.conn.executescript("""
                DROP TRIGGER memory_ai; DROP TRIGGER memory_ad; DROP TRIGGER memory_au;
                DROP TABLE memory_fts;
                CREATE VIRTUAL TABLE memory_fts USING fts5(
                    gist, detail, content='memory', content_rowid='id');
                INSERT INTO memory_fts(memory_fts) VALUES('rebuild');
                DROP TABLE embedding;
                CREATE TABLE embedding (
                    memory_id INTEGER PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL);
                INSERT INTO embedding VALUES (1, 'fake', 1, x'0000803f');
                REPLACE INTO meta(key, value) VALUES ('schema_version', '1');
            """)
            s.conn.commit()
            s.close()

            m = MemoryStore(db_path=path)  # reopen → migration runs
            cols = [r[1] for r in m.conn.execute("PRAGMA table_info(memory_fts)")]
            self.assertIn("name", cols)
            self.assertIn("chunk", [r[1] for r in
                                    m.conn.execute("PRAGMA table_info(embedding)")])
            self.assertEqual(m.conn.execute(
                "SELECT count(*) c FROM embedding").fetchone()["c"], 0)  # re-embed
            from fornixdb.db import SCHEMA_VERSION
            self.assertEqual(m.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()["value"],
                str(SCHEMA_VERSION))
            # the rebuilt index serves both old content and the name column
            self.assertTrue(m.recall("migration survivor", embedder=False))
            self.assertTrue(m.recall("survivor slug", embedder=False))
            # and new writes hit the new triggers
            m.store("post migration row", "x", name="after-slug")
            self.assertTrue(m.recall("after slug", embedder=False))
            m.close()


if __name__ == "__main__":
    unittest.main()
