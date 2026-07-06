"""see/hear are real for single artifacts (gist lane + latent lane +
source_ref pointer); sound's non-speech lane is REQUIRED (a beep means
something); watch/feel remain honest TBD stubs."""

import tempfile
import unittest
from pathlib import Path

from fornixdb import senses
from fornixdb.core import MemoryStore
from fornixdb.db import connect


class FakeModalEmbedder:
    """Deterministic per-path vectors in a tiny fake space."""

    def __init__(self, name="fake-modal", table=None):
        self.name = name
        self.table = table or {}

    def embed_artifact(self, paths):
        return [self.table[Path(p).name] for p in paths]


class SensesBase(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def _artifact(self, name, data=b"\xff\xd8fake"):
        p = self.dir / name
        p.write_bytes(data)
        return str(p)

    def _row(self, mid):
        return self.s.conn.execute(
            "SELECT kind, gist, detail, source, source_ref FROM memory "
            "WHERE id = ?", (mid,)).fetchone()


class TestSee(SensesBase):
    def test_see_with_caption_stores_a_real_percept(self):
        path = self._artifact("door.jpg")
        mid = senses.see(self.s, path, caption="delivery person at the front door")
        kind, gist, detail, source, source_ref = self._row(mid)
        self.assertEqual(kind, "episodic")
        self.assertEqual(gist, "delivery person at the front door")
        self.assertEqual(source, "senses:sight")
        self.assertEqual(source_ref, str(Path(path).resolve()))
        self.assertIsNone(detail)
        # gist lane: ordinary recall finds the sight by meaning of its caption
        self.assertTrue(any(h["id"] == mid for h in self.s.recall("front door")))

    def test_see_uses_the_captioner_when_no_caption_given(self):
        path = self._artifact("mug.jpg")
        mid = senses.see(self.s, path,
                         captioner=lambda p: "a coffee mug on the desk")
        self.assertEqual(self._row(mid)[1], "a coffee mug on the desk")

    def test_see_without_caption_or_captioner_is_an_honest_error(self):
        path = self._artifact("x.jpg")
        with self.assertRaises(ValueError) as ctx:
            senses.see(self.s, path)
        self.assertIn("caption", str(ctx.exception))

    def test_see_missing_file_refuses_a_dangling_pointer(self):
        with self.assertRaises(FileNotFoundError):
            senses.see(self.s, str(self.dir / "absent.jpg"), caption="x")


class TestHear(SensesBase):
    def test_sound_only_is_a_complete_memory(self):
        # the owner's crosswalk case: meaning, zero words
        path = self._artifact("beep.wav")
        mid = senses.hear(self.s, path,
                          sound_caption="crosswalk signal beeping, light traffic")
        kind, gist, detail, source, _ = self._row(mid)
        self.assertEqual(gist, "crosswalk signal beeping, light traffic")
        self.assertEqual(source, "senses:sound")
        self.assertIsNone(detail)

    def test_speech_rides_with_the_sound_scene_not_instead_of_it(self):
        path = self._artifact("street.wav")
        mid = senses.hear(self.s, path,
                          sound_caption="crosswalk signal beeping",
                          transcript="ok, we can cross now")
        _, gist, detail, _, _ = self._row(mid)
        self.assertTrue(gist.startswith("crosswalk signal beeping"))
        self.assertIn('"ok, we can cross now"', gist)
        self.assertEqual(detail, "ok, we can cross now")

    def test_transcript_alone_is_not_enough(self):
        path = self._artifact("talk.wav")
        with self.assertRaises(ValueError) as ctx:
            senses.hear(self.s, path, transcript="hello there")
        self.assertIn("sound", str(ctx.exception))

    def test_tagger_and_speechless_transcriber(self):
        path = self._artifact("whistle.wav")
        mid = senses.hear(self.s, path,
                          sound_tagger=lambda p: "someone whistling a tune",
                          transcriber=lambda p: None)   # no speech present
        _, gist, detail, _, _ = self._row(mid)
        self.assertEqual(gist, "someone whistling a tune")
        self.assertIsNone(detail)

    def test_long_speech_is_quoted_short_in_gist_full_in_detail(self):
        path = self._artifact("story.wav")
        speech = "word " * 60
        mid = senses.hear(self.s, path, sound_caption="quiet room",
                          transcript=speech)
        _, gist, detail, _, _ = self._row(mid)
        self.assertIn("…", gist)
        self.assertEqual(detail, speech.strip())


class TestLatentLane(SensesBase):
    def test_modal_vector_saved_and_neighbors_stay_in_one_space(self):
        emb = FakeModalEmbedder(table={
            "a.jpg": [1.0, 0.0], "b.jpg": [0.9, 0.1], "c.jpg": [0.0, 1.0]})
        a = senses.see(self.s, self._artifact("a.jpg"), caption="scene a", embedder=emb)
        b = senses.see(self.s, self._artifact("b.jpg"), caption="scene b", embedder=emb)
        c = senses.see(self.s, self._artifact("c.jpg"), caption="scene c", embedder=emb)
        # a different model's row must never mix into the ranking
        other = FakeModalEmbedder(name="other-model", table={"d.wav": [1.0, 0.0]})
        senses.hear(self.s, self._artifact("d.wav"),
                    sound_caption="scene d", embedder=other)

        model, vec = senses.modal_vector(self.s, a)
        self.assertEqual(model, "fake-modal")
        self.assertEqual(vec, [1.0, 0.0])

        ranked = senses.modal_neighbors(self.s, a)
        self.assertEqual([mid for mid, _ in ranked], [b, c])
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_no_vector_means_no_neighbors_not_an_error(self):
        mid = senses.see(self.s, self._artifact("plain.jpg"), caption="plain")
        self.assertEqual(senses.modal_neighbors(self.s, mid), [])


class TestStreamsStillHonestStubs(unittest.TestCase):
    def test_watch_and_feel_raise_tbd(self):
        s = MemoryStore(conn=connect(":memory:"))
        for call in (lambda: senses.watch(s, "camera:0"),
                     lambda: senses.feel(s, {"force": 1.2}, sensor="gripper")):
            with self.assertRaises(NotImplementedError) as ctx:
                call()
            self.assertIn("TBD", str(ctx.exception))
        s.close()


if __name__ == "__main__":
    unittest.main()
