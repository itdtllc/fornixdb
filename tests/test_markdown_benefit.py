"""The markdown bridge must do something *useful*: heading-chunked import has to
answer a section-scoped question from far fewer tokens than the whole-doc blob
it replaces, at an equal-or-better hit rate. If that ever stops being true,
this test fails — the feature would no longer be earning its place."""

import unittest
from pathlib import Path

from fornixdb.markdown_benefit import answer_cost, benefit_report, build_chunked

DOC = Path(__file__).parent.parent / "examples" / "sample_docs" / "homelab_notes.md"

QUESTIONS = [
    {"query": "nightly Time Machine offsite Backblaze backup", "answer_contains": "2:00 AM"},
    {"query": "Synology NAS web UI port", "answer_contains": "192.168.1.50"},
    {"query": "ISP Sonic fiber internet provider", "answer_contains": "Sonic"},
    {"query": "cameras retain footage Surveillance", "answer_contains": "14 days"},
]


class TestMarkdownBenefit(unittest.TestCase):
    def test_chunked_answers_from_fewer_tokens(self):
        r = benefit_report(DOC, QUESTIONS)
        # both systems hold the same text, so both should be able to answer;
        # the point is HOW MUCH the AI must read to do it.
        self.assertEqual(r["chunked_found"], r["n"])
        self.assertGreaterEqual(r["chunked_found"], r["blob_found"])
        self.assertLess(r["chunked_tokens_to_answer"], r["blob_tokens_to_answer"])
        self.assertGreater(r["token_ratio"], 1.0)

    def test_chunked_surfaces_the_right_section_top_ranked(self):
        s = build_chunked(DOC)
        try:
            c = answer_cost(s, "nightly Time Machine offsite Backblaze backup", "2:00 AM")
            self.assertTrue(c["found"])
            self.assertEqual(c["rank"], 1)          # the Backups section, ranked #1
            self.assertEqual(c["gist"], "Backups")  # gist = the heading
        finally:
            s.close()

    def test_blob_pays_for_the_whole_document_every_time(self):
        # the blob's tokens-to-answer is the same big number for every question
        # (it can only return the entire doc); the chunk is a small section.
        r = benefit_report(DOC, QUESTIONS)
        blob_costs = {c["blob"]["tokens"] for c in r["cases"] if c["blob"]["found"]}
        self.assertEqual(len(blob_costs), 1)        # identical every time
        worst_chunk = max(c["chunked"]["tokens"] for c in r["cases"]
                          if c["chunked"]["found"])
        self.assertLess(worst_chunk, blob_costs.pop())


if __name__ == "__main__":
    unittest.main()
