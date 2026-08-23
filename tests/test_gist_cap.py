"""The write-path gist ceiling (2026-08-23).

A gist is what recall returns and what a proactive push truncates to 200 chars,
so an oversized one reaches the consumer as a fragment cut mid-sentence. The cap
is enforced in core.store rather than at each call site, for the same reason the
project label is folded there: otherwise the next writer reintroduces the wall
of text. Nothing is lost — the overflow moves into detail, where `show` reads it.
"""
import unittest

from fornixdb.core import GIST_MAX_CHARS, MemoryStore, split_gist
from fornixdb.db import connect


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestSplitGist(unittest.TestCase):

    def test_short_gist_is_untouched(self):
        g, d = split_gist("a short gist", None)
        self.assertEqual(g, "a short gist")
        self.assertIsNone(d)

    def test_exactly_at_the_limit_is_untouched(self):
        gist = "x" * GIST_MAX_CHARS
        self.assertEqual(split_gist(gist), (gist, None))

    def test_overflow_moves_into_detail(self):
        head = ("The headline sentence, long enough on its own to be worth "
                "recalling rather than a stub that nobody could act on, which "
                "is what the stub floor exists to prevent. ")
        gist = head + "y" * 600
        g, d = split_gist(gist)
        self.assertEqual(g, head.strip())
        self.assertEqual(d, "y" * 600)

    def test_splits_at_the_last_sentence_boundary_before_the_limit(self):
        one = "First sentence padded out to have some length. " * 8   # ~376
        gist = one + "This tail sentence pushes it over the ceiling for sure."
        g, d = split_gist(gist)
        self.assertLessEqual(len(g), GIST_MAX_CHARS)
        self.assertTrue(g.endswith("."))
        self.assertIn("This tail sentence", d)

    def test_paragraph_break_is_a_boundary(self):
        para = ("A headline paragraph that stands on its own and runs well past "
                "the stub floor, so the split has somewhere legitimate to land "
                "without falling back to a word break")
        gist = para + "\n\n" + "z" * 500
        g, d = split_gist(gist)
        self.assertEqual(g, para)
        self.assertEqual(d, "z" * 500)

    def test_no_sentence_end_falls_back_to_a_word_boundary(self):
        gist = "word " * 200          # 1000 chars, no punctuation at all
        g, d = split_gist(gist)
        self.assertLessEqual(len(g), GIST_MAX_CHARS)
        self.assertFalse(g.endswith(" "))
        self.assertTrue(g.endswith("word"))     # never mid-token
        self.assertTrue(d.startswith("word"))

    def test_one_unbroken_token_is_hard_cut(self):
        gist = "q" * 900
        g, d = split_gist(gist)
        self.assertEqual(len(g), GIST_MAX_CHARS)
        self.assertEqual(len(d), 900 - GIST_MAX_CHARS)

    def test_early_boundary_does_not_leave_a_stub(self):
        # a sentence ends at char 12; splitting there would leave a gist too
        # short to be worth recalling, so the word-boundary fallback wins
        gist = "Tiny lead. " + "padding " * 100
        g, _ = split_gist(gist)
        self.assertGreater(len(g), 120)

    def test_existing_detail_is_preserved_after_the_overflow(self):
        head = ("A headline long enough to clear the stub floor, so that this "
                "split lands on the sentence end rather than on a word break "
                "somewhere in the middle of the padding. ")
        gist = head + "y" * 500
        g, d = split_gist(gist, "the original detail")
        self.assertEqual(g, head.strip())
        self.assertTrue(d.startswith("y" * 500))
        self.assertTrue(d.endswith("the original detail"))

    def test_last_boundary_at_or_before_the_limit_wins(self):
        # NOT the first boundary: the measured value band is 301-400 chars, so
        # the split fills the gist to the ceiling with whole sentences rather
        # than cutting at the first period and wasting the band.
        gist = "One sentence that is a reasonable length on its own. " * 12
        g, _ = split_gist(gist)
        self.assertLessEqual(len(g), GIST_MAX_CHARS)
        self.assertGreater(len(g), 300)
        self.assertEqual(g.count("."), 7)

    def test_is_idempotent(self):
        gist = "Headline. " + "y" * 500
        once = split_gist(gist)
        self.assertEqual(split_gist(*once), once)

    def test_empty_gist_survives(self):
        self.assertEqual(split_gist("", None), ("", None))

    def test_nothing_is_lost(self):
        gist = "Lead sentence here. " + "body words " * 90
        g, d = split_gist(gist)
        self.assertEqual((g + " " + d).split(), gist.split())


class TestStoreEnforcesTheCap(unittest.TestCase):

    def test_store_splits_an_oversized_gist(self):
        s = mem_store()
        long_detail_text = "detail sentence. " * 60
        mid = s.store("The headline. " + long_detail_text, embedder=False)
        row = s.conn.execute(
            "SELECT gist, detail FROM memory WHERE id = ?", (mid,)).fetchone()
        self.assertLessEqual(len(row["gist"]), GIST_MAX_CHARS)
        self.assertTrue(row["gist"].startswith("The headline."))
        self.assertTrue(row["gist"].endswith("."))
        self.assertTrue(row["detail"].startswith("detail sentence."))

    def test_store_keeps_a_short_gist_and_its_detail_intact(self):
        s = mem_store()
        mid = s.store("a fine gist", "its detail", embedder=False)
        row = s.conn.execute(
            "SELECT gist, detail FROM memory WHERE id = ?", (mid,)).fetchone()
        self.assertEqual(row["gist"], "a fine gist")
        self.assertEqual(row["detail"], "its detail")

    def test_oversized_gist_is_still_recallable_by_its_overflow(self):
        # the split must not hide content from recall — detail is in the FTS
        # index alongside gist, so a word that moved is still findable
        s = mem_store()
        s.store("Headline about nothing. "
                + "the pelican migration survey ran late. " * 20, embedder=False)
        rows = s.recall("pelican migration survey")
        self.assertTrue(rows)

    def test_capped_gist_survives_a_show_round_trip(self):
        s = mem_store()
        mid = s.store("Headline. " + "tail words " * 80, embedder=False)
        got = s.show(mid)
        self.assertLessEqual(len(got["gist"]), GIST_MAX_CHARS)
        self.assertIn("tail words", got["detail"])


if __name__ == "__main__":
    unittest.main()


class TestGistBackfill(unittest.TestCase):
    """The backfill: bring rows written BEFORE the cap into line with it.

    Mechanical and lossless, so it is a plain UPDATE rather than a supersede —
    superseding is for knowledge that contradicts, and using it here would
    tombstone rows and break lineage to record nothing new.
    """

    def _oversized(self, store, gist, detail=None):
        """Insert past core.store, which would split it on the way in."""
        cur = store.conn.execute(
            "INSERT INTO memory (kind, event_time, recorded_time, gist, detail, "
            "salience, source) VALUES ('semantic', '2026-01-01', '2026-01-01', "
            "?, ?, 0.5, 'cli')", (gist, detail))
        store.conn.commit()
        return cur.lastrowid

    def test_dry_run_changes_nothing(self):
        from fornixdb.consolidate import gist_backfill
        s = mem_store()
        long_gist = "A sentence that carries some weight. " * 20
        mid = self._oversized(s, long_gist)
        res = gist_backfill(s, apply=False)
        self.assertEqual(len(res["candidates"]), 1)
        self.assertEqual(res["applied"], 0)
        row = s.conn.execute("SELECT gist FROM memory WHERE id = ?",
                             (mid,)).fetchone()
        self.assertEqual(row["gist"], long_gist)      # untouched

    def test_apply_splits_and_loses_nothing(self):
        from fornixdb.consolidate import gist_backfill
        s = mem_store()
        long_gist = "A sentence that carries some weight. " * 20
        mid = self._oversized(s, long_gist)
        res = gist_backfill(s, apply=True, embedder=False)
        self.assertEqual(res["applied"], 1)
        row = s.conn.execute("SELECT gist, detail FROM memory WHERE id = ?",
                             (mid,)).fetchone()
        self.assertLessEqual(len(row["gist"]), GIST_MAX_CHARS)
        self.assertEqual((row["gist"] + " " + row["detail"]).split(),
                         long_gist.split())

    def test_existing_detail_is_kept_after_the_overflow(self):
        from fornixdb.consolidate import gist_backfill
        s = mem_store()
        long_gist = "Another sentence of respectable length here. " * 15
        mid = self._oversized(s, long_gist, "the original detail")
        gist_backfill(s, apply=True, embedder=False)
        row = s.conn.execute("SELECT detail FROM memory WHERE id = ?",
                             (mid,)).fetchone()
        self.assertTrue(row["detail"].endswith("the original detail"))

    def test_is_idempotent(self):
        from fornixdb.consolidate import gist_backfill
        s = mem_store()
        self._oversized(s, "A sentence that carries some weight. " * 20)
        first = gist_backfill(s, apply=True, embedder=False)
        second = gist_backfill(s, apply=True, embedder=False)
        self.assertEqual(first["applied"], 1)
        self.assertEqual(second["applied"], 0)
        self.assertEqual(second["candidates"], [])

    def test_superseded_rows_are_left_alone(self):
        # history is not rewritten: the chain stays byte-for-byte as recorded
        from fornixdb.consolidate import gist_backfill
        s = mem_store()
        long_gist = "A sentence that carries some weight. " * 20
        mid = self._oversized(s, long_gist)
        s.conn.execute("UPDATE memory SET superseded_time = '2026-02-02' "
                       "WHERE id = ?", (mid,))
        s.conn.commit()
        res = gist_backfill(s, apply=True, embedder=False)
        self.assertEqual(res["applied"], 0)
        row = s.conn.execute("SELECT gist FROM memory WHERE id = ?",
                             (mid,)).fetchone()
        self.assertEqual(row["gist"], long_gist)

    def test_split_row_stays_findable_by_moved_words(self):
        # the memory_au trigger must reindex FTS, or the backfill silently
        # hides the words it moved
        from fornixdb.consolidate import gist_backfill
        s = mem_store()
        self._oversized(s, "A lead sentence of adequate length to matter. " * 8
                        + "The kestrel ringing programme reported late.")
        gist_backfill(s, apply=True, embedder=False)
        self.assertTrue(s.recall("kestrel ringing programme"))

    def test_limit_takes_the_longest_first(self):
        from fornixdb.consolidate import gist_backfill
        s = mem_store()
        short_over = self._oversized(s, "Short over the line. " * 25)
        long_over = self._oversized(s, "Much longer over the line. " * 90)
        res = gist_backfill(s, apply=False, limit=1)
        self.assertEqual([c["id"] for c in res["candidates"]], [long_over])
        self.assertNotEqual(short_over, long_over)

    def test_frozen_store_refuses_to_apply(self):
        from fornixdb.core import FrozenStoreError
        from fornixdb.consolidate import gist_backfill
        s = mem_store()
        self._oversized(s, "A sentence that carries some weight. " * 20)
        s.conn.execute("INSERT INTO meta (key, value) VALUES ('frozen', '1')")
        s.conn.commit()
        s.__dict__.pop("_frozen_cache", None)      # lazily cached on first read
        with self.assertRaises(FrozenStoreError):
            gist_backfill(s, apply=True, embedder=False)
