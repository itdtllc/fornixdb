"""Agreement has to be CONCENTRATED to mean anything.

`shared_term_count` claimed to be independent of document length and was not:
counted over a whole row, a long memory about something else collects the
query's words scattered across pages of unrelated prose, in senses that have
nothing to do with the question. Measured on a lived-in store against queries it
has no answer to, the chance of accidentally agreeing about two things was 0% up
to 800 characters and 0.47% past 1600.
"""
import unittest

from fornixdb.core import (AGREEMENT_WINDOW_CHARS, RECALL_ANSWER_MIN_TERMS,
                           recall_has_answer, shared_term_count)


def row(gist="", detail=None):
    return {"gist": gist, "detail": detail}


class TestConcentration(unittest.TestCase):

    def test_gist_agreement_counts(self):
        self.assertEqual(
            shared_term_count("lower back pain", row("lower back pain relief")), 3)

    def test_agreement_concentrated_in_detail_counts(self):
        r = row("something else", "z " * 200 + "lower back pain discussed here")
        self.assertEqual(shared_term_count("lower back pain", r), 3)

    def test_agreement_scattered_across_a_long_row_does_not(self):
        # the haystack: each word present, none of them together
        pad = " unrelated" * (AGREEMENT_WINDOW_CHARS // 5) + " "
        r = row("something else", "lower " + pad + "back " + pad + "pain")
        self.assertLess(shared_term_count("lower back pain", r),
                        RECALL_ANSWER_MIN_TERMS)

    def test_the_gist_counts_wherever_the_window_falls(self):
        # gist is the summary and is capped, so it can never be a haystack
        pad = " unrelated" * (AGREEMENT_WINDOW_CHARS // 5) + " "
        r = row("lower back", pad + "pain" + pad)
        self.assertEqual(shared_term_count("lower back pain", r), 3)

    def test_two_words_close_together_still_agree(self):
        pad = " unrelated" * (AGREEMENT_WINDOW_CHARS // 5) + " "
        r = row("x", pad + "lower back" + pad)
        self.assertGreaterEqual(shared_term_count("lower back pain", r),
                                RECALL_ANSWER_MIN_TERMS)

    def test_a_row_with_no_detail_is_just_its_gist(self):
        self.assertEqual(shared_term_count("alpha bravo", row("alpha only")), 1)

    def test_count_never_exceeds_the_querys_content_words(self):
        r = row("alpha alpha alpha", "alpha " * 500)
        self.assertEqual(shared_term_count("alpha", r), 1)

    def test_repeated_query_words_are_counted_once(self):
        self.assertEqual(shared_term_count("alpha alpha alpha", row("alpha")), 1)

    def test_empty_query_is_zero(self):
        self.assertEqual(shared_term_count("", row("anything")), 0)

    def test_query_of_only_stopwords_is_zero(self):
        self.assertEqual(shared_term_count("how do i", row("how do i")), 0)

    def test_empty_row_is_zero(self):
        self.assertEqual(shared_term_count("alpha bravo", row("", None)), 0)

    def test_short_tokens_do_not_agree(self):
        # two characters or fewer go with the stopwords
        self.assertEqual(shared_term_count("go up", row("go up")), 0)


class TestGateUsesIt(unittest.TestCase):
    """The gate's weak legs end on this count, so a haystack must abstain."""

    def _weak(self, r):
        top = dict(r, vec_cos=0.31, kw_rel=99.0, raw_cos=0.31)
        top["shared_terms"] = shared_term_count("lower back pain", top)
        return recall_has_answer([top])

    def test_a_haystack_top_hit_abstains(self):
        pad = " unrelated" * (AGREEMENT_WINDOW_CHARS // 5) + " "
        self.assertFalse(self._weak(
            row("about something else", "lower " + pad + "back " + pad + "pain")))

    def test_a_real_answer_in_the_detail_still_answers(self):
        pad = " unrelated" * (AGREEMENT_WINDOW_CHARS // 5) + " "
        self.assertTrue(self._weak(
            row("about something else", pad + "lower back pain" + pad)))

    def test_unmeasured_agreement_keeps_the_old_behavior(self):
        # a row built by something other than recall() never had a query to
        # measure against; absent means UNMEASURED, not zero
        self.assertTrue(recall_has_answer(
            [{"vec_cos": 0.31, "kw_rel": 99.0, "raw_cos": 0.31}]))


if __name__ == "__main__":
    unittest.main()
