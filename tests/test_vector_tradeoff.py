"""Quantify the vector tradeoff so the default can be chosen from evidence.

Uses a deterministic fake embedder (synonym table → semantic match with zero
keyword overlap, no model download) so the DIRECTION of every axis is asserted
reproducibly in CI. The runnable report (examples/vector_tradeoff_report.py)
prints the real magnitudes with model2vec."""

import unittest
import zlib

from fornixdb.vector_tradeoff import format_report, measure

DIM = 1024  # wide enough to avoid crc32 hash collisions across the test corpus
SYNONYMS = {"automobile": "car", "vehicle": "car",
            "sparkled": "twinkle", "glint": "twinkle",
            "glitch": "artifact", "bug": "artifact"}


class FakeEmbedder:
    """Tokens hash to one-hot dims; synonyms share a direction (so a query and
    its target match by meaning with no shared words)."""
    name = "fake:onehot"

    def embed(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * DIM
            for tok in t.lower().split():
                tok = "".join(c for c in tok if c.isalnum())
                if tok:
                    vec[zlib.crc32(SYNONYMS.get(tok, tok).encode()) % DIM] += 1.0
            out.append(vec)
        return out


class TestVectorTradeoff(unittest.TestCase):
    def setUp(self):
        self.r = measure(embedder=FakeEmbedder(), repeats=5)

    def test_recall_ability_vectors_win_on_synonyms_no_keyword_regression(self):
        rc = self.r["recall"]
        # synonym queries share NO keyword with the target: keyword recall can't
        # find them at all; vectors find every one.
        self.assertEqual(rc["synonym_hit1_keyword"][0], 0)
        self.assertEqual(rc["synonym_hit1_vector"][0],
                         rc["synonym_hit1_vector"][1])
        # plain keyword queries: vectors must NOT regress them.
        self.assertEqual(rc["keyword_hit1_vector"][0],
                         rc["keyword_hit1_keyword"][0])
        # so overall recall is strictly better with vectors.
        self.assertGreater(rc["vector"]["mrr"], rc["keyword"]["mrr"])

    def test_db_space_costs_more_with_vectors(self):
        s = self.r["space"]
        self.assertGreater(s["vector_db_bytes"], s["keyword_db_bytes"])
        self.assertGreater(s["embedding_payload_bytes"], 0)
        self.assertGreater(s["bytes_per_memory"], 0)

    def test_prompt_tokens_not_inflated_by_vectors(self):
        pt = self.r["prompt_tokens"]
        kw, vec = pt["keyword_controls_keyword"], pt["keyword_controls_vector"]
        self.assertGreater(kw, 0)
        # vectors change WHICH rows rank first, not how many — result size stays
        # the same order of magnitude (not a multiplier on context cost).
        self.assertLessEqual(vec, kw * 1.5)

    def test_time_measured_and_finite(self):
        tm = self.r["time_ms"]
        for key in ("write_keyword", "write_vector",
                    "recall_keyword", "recall_vector"):
            self.assertGreaterEqual(tm[key], 0.0)

    def test_report_renders(self):
        out = format_report(self.r)
        self.assertIn("RECALL ABILITY", out)
        self.assertIn("DB SPACE", out)
        self.assertIn("PROMPT TOKENS", out)


if __name__ == "__main__":
    unittest.main()
