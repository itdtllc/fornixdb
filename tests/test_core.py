import unittest
from datetime import datetime, timedelta

from fornixdb.core import MemoryStore, recall_has_answer, shared_term_count
from fornixdb.db import connect


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestRecallHasAnswer(unittest.TestCase):
    """The abstention gate (#191): reports presence only, never an action."""

    def test_empty_is_no_answer(self):
        self.assertFalse(recall_has_answer([]))

    def test_strong_vector_match_is_answer(self):
        self.assertTrue(recall_has_answer([{"vec_cos": 0.5, "relevance": 2.0}]))

    def test_keyword_only_recall_trusts_fts_anchor(self):
        # no vec_cos key at all = pure keyword recall; an FTS hit is a literal
        # token anchor, trusted regardless of (store-dependent) bm25 magnitude
        self.assertTrue(recall_has_answer([{"relevance": 3.0}]))

    def test_vector_store_weak_keyword_overlap_abstains(self):
        # vectors were computed (vec_cos present), the top hit is semantically
        # dissimilar AND its literal-token anchor is weak (below the calibrated
        # positive band) — tangential keyword overlap is noise, not an answer
        self.assertFalse(recall_has_answer(
            [{"vec_cos": 0.0, "kw_rel": 3.0, "relevance": 3.0}]))

    def test_hybrid_strong_keyword_anchor_is_answer(self):
        # regression for the 2026-07-16 live false-abstain: rank-1 anchored by
        # literal tokens ("qwen 72b … consumer", kw_rel 18.4) at TRUE cosine
        # 0.297 — just under the 0.30 shortlist floor, so vec_cos read 0.0.
        # Hybrid recall must trust the same anchor keyword-only mode trusts
        # when the raw cosine corroborates it.
        self.assertTrue(recall_has_answer(
            [{"vec_cos": 0.0, "raw_cos": 0.297, "kw_rel": 18.4,
              "relevance": 18.4}]))

    def test_hybrid_uncorroborated_keyword_anchor_abstains(self):
        # regression for the 'capital of France' leak: common-token overlap can
        # push bm25 over the anchor band (kw_rel 9.26) on a grown store, but
        # its raw cosine ~0 betrays it — semantically unrelated, stay abstained
        self.assertFalse(recall_has_answer(
            [{"vec_cos": 0.0, "raw_cos": 0.001, "kw_rel": 9.26,
              "relevance": 9.26}]))

    def test_hybrid_polysemous_anchor_abstains(self):
        # regression for the 2026-08-03 'sourdough' leak: a single shared word
        # used in two unrelated senses ("bake bread" vs baking geometry in a 3-D
        # tool) cleared the anchor band (kw_rel 8.99) AND dragged the raw cosine
        # to 0.167 — a static embedder cannot separate the senses, so the
        # corroboration bar has to sit above the noise the store actually
        # produces. The leg's real positives sit far higher (0.297+).
        self.assertFalse(recall_has_answer(
            [{"vec_cos": 0.0, "raw_cos": 0.167, "kw_rel": 8.99,
              "relevance": 8.99}]))

    def test_hybrid_keyword_anchor_below_band_abstains(self):
        # the calibrated negative band (clear negatives sat < 5.2) stays quiet
        # even with a fully corroborating cosine — kw_rel alone decides here
        self.assertFalse(recall_has_answer(
            [{"vec_cos": 0.11, "raw_cos": 0.30, "kw_rel": 5.1,
              "relevance": 5.2}]))

    def test_weak_vector_match_abstains(self):
        self.assertFalse(recall_has_answer([{"vec_cos": 0.1, "relevance": 2.0}]))

    def test_floor_band_cosine_without_keyword_support_abstains(self):
        # regression for the 2026-07-17 live abstain leak: best-chunk scoring
        # over a long ingested document (8 chunks, 9.5KB) let one deep chunk
        # brush cosine 0.335 against a nonsense query with almost no literal
        # overlap (kw_rel 1.73) — inside the floor band, cosine alone is a
        # lottery ticket, not an answer
        self.assertFalse(recall_has_answer(
            [{"vec_cos": 0.335, "kw_rel": 1.73, "relevance": 2.0}]))

    def test_floor_band_cosine_with_keyword_support_is_answer(self):
        # the real near-floor positives all share some query vocabulary
        # (golden-set minimum kw_rel 3.54) — a pinch of the other signal
        # keeps them answered
        self.assertTrue(recall_has_answer(
            [{"vec_cos": 0.335, "kw_rel": 3.54, "relevance": 3.6}]))

    def test_floor_band_single_shared_word_abstains(self):
        # regression for the 2026-08-03 sweep: ordinary out-of-store questions
        # leaked through the floor band on ONE accidentally-shared common word,
        # with a bm25 magnitude well over the floor. Measured on a live store,
        # real answers in this band share 3-7 content words and noise shares 0-1.
        self.assertFalse(recall_has_answer(
            [{"vec_cos": 0.354, "kw_rel": 7.46, "shared_terms": 1,
              "relevance": 7.5}]))

    def test_floor_band_several_shared_words_is_answer(self):
        # the same band, same magnitude — agreement about several distinct
        # things is what makes it an answer
        self.assertTrue(recall_has_answer(
            [{"vec_cos": 0.354, "kw_rel": 7.46, "shared_terms": 3,
              "relevance": 7.5}]))

    def test_floor_band_unmeasured_shared_terms_keeps_old_behavior(self):
        # a row built by something other than recall() has no query to measure
        # against; absent must mean unmeasured, not zero, or such a caller would
        # abstain on every floor-band hit
        self.assertTrue(recall_has_answer(
            [{"vec_cos": 0.335, "kw_rel": 3.54, "relevance": 3.6}]))

    def test_strong_cosine_needs_no_keyword_support(self):
        # at/above COS_STRONG the vector match stands alone (the zero-shared-
        # keywords recall class must keep working)
        self.assertTrue(recall_has_answer(
            [{"vec_cos": 0.40, "kw_rel": 0.0, "relevance": 1.0}]))

    def test_only_top_hit_decides(self):
        # a weak best hit abstains even if weaker rows follow
        self.assertFalse(recall_has_answer(
            [{"vec_cos": 0.1, "relevance": 1.0}, {"vec_cos": 0.9, "relevance": 9}]))


class TestStoreRecall(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def test_store_and_show(self):
        mid = self.s.store("Decided to use SQLite", "Long detail here",
                           kind="semantic", topics=["architecture"], name="sqlite-decision")
        mem = self.s.show(mid)
        self.assertEqual(mem["gist"], "Decided to use SQLite")
        self.assertEqual(mem["topics"], ["architecture"])
        mem2 = self.s.show("sqlite-decision")
        self.assertEqual(mem2["id"], mid)

    def test_native_kind_alias_maps_to_semantic(self):
        # A model reaching for the native "project"/"user" taxonomy must not
        # bounce; both are standing knowledge and map to semantic.
        for alias in ("project", "user"):
            mid = self.s.store(f"{alias} fact", kind=alias)
            self.assertEqual(self.s.show(mid)["kind"], "semantic")

    def test_unknown_kind_still_rejected_with_hint(self):
        with self.assertRaises(ValueError) as cm:
            self.s.store("bad", kind="nonsense")
        self.assertIn("nonsense", str(cm.exception))

    def test_subject_recall_ranked(self):
        self.s.store("Picked FTS5 for subject recall")
        self.s.store("Bought groceries")
        rows = self.s.recall("subject recall FTS5")
        self.assertTrue(rows)
        self.assertIn("FTS5", rows[0]["gist"])

    def test_recall_and_fallback_to_or(self):
        self.s.store("Transfer conflicts auto-fix shipped")
        rows = self.s.recall("transfer zebra")  # AND fails, OR finds transfer
        self.assertTrue(rows)

    def test_timeline(self):
        old = (datetime.now() - timedelta(days=10)).isoformat()
        self.s.store("old event", kind="episodic", event_time=old)
        self.s.store("new event", kind="episodic")
        start = (datetime.now() - timedelta(days=1)).isoformat()
        end = (datetime.now() + timedelta(days=1)).isoformat()
        rows = self.s.timeline(start, end)
        self.assertEqual([r["gist"] for r in rows], ["new event"])

    def test_timeline_includes_spans(self):
        # a session that started before the window but ended inside it
        self.s.store("long session", kind="episodic",
                     event_time=(datetime.now() - timedelta(days=5)).isoformat(),
                     event_time_end=(datetime.now() - timedelta(days=1)).isoformat())
        start = (datetime.now() - timedelta(days=2)).isoformat()
        end = datetime.now().isoformat()
        rows = self.s.timeline(start, end)
        self.assertEqual(len(rows), 1)

    def test_timeline_keeps_most_recent_on_overflow(self):
        # a window with more rows than the limit must keep the NEWEST, not the
        # oldest (a busy day dropping what just happened is the failure mode),
        # and still present them oldest-first.
        base = datetime.now().replace(microsecond=0)
        for i in range(5):  # event 0 = oldest … event 4 = newest
            self.s.store(f"event {i}", kind="episodic",
                         event_time=(base - timedelta(minutes=5 - i)).isoformat())
        start = (base - timedelta(hours=1)).isoformat()
        end = (base + timedelta(hours=1)).isoformat()
        rows = self.s.timeline(start, end, limit=3)
        self.assertEqual([r["gist"] for r in rows],
                         ["event 2", "event 3", "event 4"])

    def test_supersede_keeps_history(self):
        a = self.s.store("We use approach X for caching")
        b = self.s.store("We switched to approach Y for caching")
        self.s.supersede(a, b)
        rows = self.s.recall("caching approach")
        self.assertEqual([r["id"] for r in rows], [b])
        rows_all = self.s.recall("caching approach", include_superseded=True)
        self.assertEqual(len(rows_all), 2)
        old = self.s.show(a)
        self.assertEqual(old["superseded_by"], b)
        self.assertIsNotNone(old["superseded_time"])

    def test_tombstone_without_successor(self):
        mid = self.s.store("temporary fact to forget")
        self.s.tombstone(mid)
        self.assertEqual(self.s.recall("temporary fact"), [])
        rows = self.s.recall("temporary fact", include_superseded=True)
        self.assertEqual([r["id"] for r in rows], [mid])
        mem = self.s.show(mid, reinforce=False)
        self.assertIsNone(mem["superseded_by"])      # no successor
        self.assertIsNotNone(mem["superseded_time"])  # but retired

    def test_set_name_handoff(self):
        a = self.s.store("v1", name="handle")
        self.s.set_name(a, None)
        b = self.s.store("v2", name="handle")
        self.s.supersede(a, b)
        self.assertEqual(self.s.show("handle", reinforce=False)["id"], b)

    def test_reinforcement_on_show(self):
        mid = self.s.store("reinforce me", salience=0.5)
        before = self.s.show(mid, reinforce=False)["salience"]
        self.s.show(mid)  # reinforces
        after = self.s.show(mid, reinforce=False)["salience"]
        self.assertGreater(after, before)

    def test_recency_breaks_relevance_ties(self):
        old = (datetime.now() - timedelta(days=300)).isoformat()
        self.s.store("deploy pipeline notes", event_time=old)
        recent = self.s.store("deploy pipeline notes")
        rows = self.s.recall("deploy pipeline")
        self.assertEqual(rows[0]["id"], recent)

    def test_fts_query_injection_safe(self):
        self.s.store("safe storage")
        rows = self.s.recall('safe" OR x NEAR/ (')  # must not raise
        self.assertTrue(rows)

    def test_stats(self):
        self.s.store("one", kind="episodic")
        self.s.store("two")
        st = self.s.stats()
        self.assertEqual(st["memories"], 2)
        self.assertEqual(st["by_kind"]["episodic"], 1)


if __name__ == "__main__":
    unittest.main()


class TestSharedTermCount(unittest.TestCase):
    """Counting agreement, the signal bm25 magnitude cannot supply."""

    def test_counts_distinct_content_words_only(self):
        row = {"gist": "Baking a loaf of bread needs a hot oven",
               "detail": "Sourdough starter takes a week."}
        # 'bread' and 'baking' agree; 'how', 'do', 'the' are function words
        self.assertEqual(shared_term_count("how do I bake the bread", row), 1)
        self.assertEqual(shared_term_count("sourdough bread recipe", row), 2)

    def test_repeated_word_counts_once(self):
        row = {"gist": "harbor harbor harbor", "detail": "harbor"}
        self.assertEqual(shared_term_count("harbor harbor harbor", row), 1)

    def test_function_words_are_not_agreement(self):
        # the failure this exists to catch: two texts about nothing in common
        # still share plenty of grammar
        row = {"gist": "How do I get to the station from here", "detail": ""}
        self.assertEqual(shared_term_count("how do I get to the airport", row), 0)

    def test_short_tokens_ignored(self):
        row = {"gist": "an ox in a bin", "detail": ""}
        self.assertEqual(shared_term_count("the ox and the bin", row), 1)  # 'bin' only

    def test_detail_counts_toward_agreement(self):
        row = {"gist": "a lighthouse", "detail": "keeper polishes the lantern"}
        self.assertEqual(shared_term_count("lantern keeper", row), 2)

    def test_empty_query_or_row_is_zero(self):
        self.assertEqual(shared_term_count("", {"gist": "anything"}), 0)
        self.assertEqual(shared_term_count("anything", {}), 0)

    def test_measured_end_to_end_by_recall(self):
        # recall() must attach the count, or the gate silently degrades to its
        # pre-2026-08-03 behavior on every row
        s = mem_store()
        s.store("the lighthouse keeper polishes the lantern each evening")
        rows = s.recall("lighthouse keeper", count_recall=False)
        self.assertTrue(rows)
        self.assertEqual(rows[0].get("shared_terms"), 2)
