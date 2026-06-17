"""The warm-embedder seam: set_default_embedder() + $FORNIXDB_EMBEDDER.

A deployment on dedicated hardware (e.g. a humanoid robot running FornixDB in
one long-lived process) keeps the embedding model warm and injects it once,
instead of the default cold per-turn load. Both injection paths must override
the model2vec default, route through get_default_embedder() so every recall
path picks them up, and never raise on a bad spec. See INTEGRATION.md,
"Warm embedding on dedicated hardware."
"""

import os
import unittest

from fornixdb import vectors


class _StubEmbedder:
    name = "stub:test"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0] for t in texts]


def make_stub():  # module-level factory for the env-var path
    return _StubEmbedder()


class TestEmbedderInjection(unittest.TestCase):
    def setUp(self):
        self._saved = vectors._default_embedder
        self._saved_env = os.environ.get(vectors.EMBEDDER_ENV)
        vectors._default_embedder = "unset"
        os.environ.pop(vectors.EMBEDDER_ENV, None)

    def tearDown(self):
        vectors._default_embedder = self._saved
        if self._saved_env is None:
            os.environ.pop(vectors.EMBEDDER_ENV, None)
        else:
            os.environ[vectors.EMBEDDER_ENV] = self._saved_env

    def test_set_default_embedder_overrides(self):
        stub = _StubEmbedder()
        vectors.set_default_embedder(stub)
        self.assertIs(vectors.get_default_embedder(), stub)

    def test_set_none_forces_keyword_only(self):
        vectors.set_default_embedder(None)
        # None means "no embedder" (keyword-only) — must NOT fall back to model2vec.
        self.assertIsNone(vectors.get_default_embedder())

    def test_env_var_loads_factory(self):
        os.environ[vectors.EMBEDDER_ENV] = f"{__name__}:make_stub"
        emb = vectors.get_default_embedder()
        self.assertIsInstance(emb, _StubEmbedder)
        self.assertEqual(emb.name, "stub:test")

    def test_env_var_accepts_module_attribute(self):
        # a bare object attribute (not a callable) is used as-is.
        os.environ[vectors.EMBEDDER_ENV] = f"{__name__}:SHARED_STUB"
        self.assertIs(vectors.get_default_embedder(), SHARED_STUB)

    def test_injection_wins_over_env(self):
        os.environ[vectors.EMBEDDER_ENV] = f"{__name__}:make_stub"
        stub = _StubEmbedder()
        vectors.set_default_embedder(stub)
        self.assertIs(vectors.get_default_embedder(), stub)

    def test_bad_env_spec_never_raises(self):
        for bad in ("no_such_module_xyz:make", f"{__name__}:no_such_attr", "garbage"):
            with self.subTest(bad=bad):
                os.environ[vectors.EMBEDDER_ENV] = bad
                vectors._default_embedder = "unset"
                # falls through to the real default (model2vec or None) — no raise.
                vectors.get_default_embedder()


SHARED_STUB = _StubEmbedder()


if __name__ == "__main__":
    unittest.main()
