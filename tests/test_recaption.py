"""Dream-pass captioning (fornixdb.recaption): find the templated placeholder
gists watch() commits, run a captioner over each keyframe, rewrite the gist in
place. Pure and injectable — no camera, no screen, no VLM: real keyframe files
are written to a temp dir and the captioner is a plain callable, the same way
test_watchloop drives the loop deterministically.
"""
import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from datetime import datetime
from pathlib import Path
from unittest import mock

from fornixdb import recaption, watchloop
from fornixdb.adapters import mac_vision
from fornixdb.cli import main
from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.salience import SalienceGate

A, B, C = [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]
WALL0 = datetime(2026, 7, 8, 9, 0, 0)


class Frame(bytes):
    """Encoded frame bytes carrying their own fake embedding (test_watchloop's
    trick) — real bytes so a real .jpg keyframe is written on commit."""
    def __new__(cls, vec, payload=b"\xff\xd8fakejpeg"):
        o = super().__new__(cls, payload)
        o.vec = vec
        return o


class FakeEmbedder:
    """Text-lane embedder stub: name + embed(texts) -> vectors, per the
    vectors.Embedder protocol. Lets us prove set_gist re-embeds the row."""
    name = "fake-text"

    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


class Base(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def gist(self, mid):
        return self.s.conn.execute(
            "SELECT gist FROM memory WHERE id = ?", (mid,)).fetchone()[0]

    def watch3(self):
        """Commit three real watch keyframes with templated gists: a session
        start, then two scene changes. A quiet frame sits between each change so
        the gate re-arms (it disarms on commit until the scene settles)."""
        gate = SalienceGate(threshold=0.5, rearm_below=0.2, ema_alpha=1.0,
                            heartbeat_seconds=0)
        fr = iter([(0.0, Frame(A)), (1.0, Frame(A)),   # first, then quiet
                   (2.0, Frame(B)), (3.0, Frame(B)),   # scene change, re-arm
                   (4.0, Frame(C))])                    # scene change
        return watchloop.run_watch(
            self.s, fr, embed=lambda f: f.vec, gate=gate,
            keyframe_dir=str(self.dir), start_wall=WALL0, source_label="screen")


class TestPredicate(unittest.TestCase):
    def test_matches_the_three_placeholders_any_label(self):
        p = recaption.is_templated_watch_gist
        for label in ("screen", "camera", "/clips/door.mov"):
            self.assertTrue(p(f"watch[{label}]: session start"))
            self.assertTrue(p(f"watch[{label}]: scene change"))
            self.assertTrue(p(f"watch[{label}]: quiet interval"))
        self.assertTrue(p("  watch[screen]: scene change  "))   # trimmed

    def test_rejects_real_captions_and_near_misses(self):
        p = recaption.is_templated_watch_gist
        for g in ("a person sitting at a desk with coffee",
                  "watch[screen]: someone walked in",     # not a known reason
                  "watch: scene change",                  # no [label]
                  "scene change", "", "   "):
            self.assertFalse(p(g))


class TestPending(Base):
    def test_lists_every_placeholder_oldest_first_with_keyframe(self):
        evs = self.watch3()
        pend = recaption.pending_captions(self.s)
        self.assertEqual([mid for mid, _, _ in pend], [e.memory_id for e in evs])
        for _, g, ref in pend:
            self.assertTrue(recaption.is_templated_watch_gist(g))
            self.assertTrue(Path(ref).is_file())
        self.assertEqual(recaption.pending_count(self.s), 3)

    def test_limit_caps_the_worklist(self):
        self.watch3()
        self.assertEqual(len(recaption.pending_captions(self.s, limit=2)), 2)

    def test_missing_keyframe_is_dropped(self):
        evs = self.watch3()
        Path(evs[1].keyframe).unlink()               # keyframe decayed off disk
        pend = recaption.pending_captions(self.s)
        self.assertEqual([mid for mid, _, _ in pend],
                         [evs[0].memory_id, evs[2].memory_id])


class TestRecaption(Base):
    def test_rewrites_gists_and_is_idempotent(self):
        self.watch3()
        # key off the keyframe the captioner is actually handed (the resolved
        # source_ref), and confirm every placeholder becomes its caption
        pend = recaption.pending_captions(self.s)
        caps = {ref: f"cap-{i}" for i, (_, _, ref) in enumerate(pend)}
        applied = recaption.recaption(self.s, lambda p: caps[p])
        self.assertEqual(len(applied), 3)
        for i, (mid, _, _) in enumerate(pend):
            self.assertEqual(self.gist(mid), f"cap-{i}")
        # keyed on the placeholder -> nothing left to do the second time
        self.assertEqual(recaption.pending_count(self.s), 0)
        self.assertEqual(recaption.recaption(self.s, lambda p: "again"), [])

    def test_rewrite_is_keyword_recallable(self):
        evs = self.watch3()
        recaption.recaption(
            self.s, lambda p: "a delivery driver at the front door")
        hits = [m["id"] for m in self.s.recall("delivery driver door", limit=5)]
        self.assertIn(evs[0].memory_id, hits)        # FTS sees the new gist

    def test_empty_caption_never_clobbers_placeholder(self):
        evs = self.watch3()
        applied = recaption.recaption(self.s, lambda p: "   ")
        self.assertEqual(applied, [])
        self.assertTrue(
            recaption.is_templated_watch_gist(self.gist(evs[0].memory_id)))

    def test_one_bad_frame_does_not_abort_the_pass(self):
        self.watch3()
        pend = recaption.pending_captions(self.s)
        bad_ref, bad_mid = pend[1][2], pend[1][0]
        def cap(p):
            if p == bad_ref:
                raise RuntimeError("VLM choked on this frame")
            return "ok"
        applied = recaption.recaption(self.s, cap)
        self.assertEqual([mid for mid, _, _ in applied],
                         [pend[0][0], pend[2][0]])
        self.assertTrue(                              # the choked row is untouched
            recaption.is_templated_watch_gist(self.gist(bad_mid)))

    def test_limit_leaves_the_rest_pending(self):
        self.watch3()
        applied = recaption.recaption(self.s, lambda p: "x", limit=1)
        self.assertEqual(len(applied), 1)
        self.assertEqual(recaption.pending_count(self.s), 2)

    def test_set_gist_reembeds_the_text_lane(self):
        evs = self.watch3()
        mid = evs[0].memory_id
        recaption.recaption(self.s, lambda p: "a cat on the windowsill",
                            embedder=FakeEmbedder())
        row = self.s.conn.execute(
            "SELECT COUNT(*) FROM embedding WHERE memory_id = ? AND model = ?",
            (mid, "fake-text")).fetchone()
        self.assertGreater(row[0], 0)                 # re-embedded, not left stale


class TestRecaptionCli(unittest.TestCase):
    """`fornixdb recaption --dry-run` lists the worklist with NO VLM loaded —
    the model-free half of the command. The captioning half is exercised by the
    recaption() unit tests above; its Mac VLM adapter is wired separately."""

    def _seed(self, db_path, keyframe_dir):
        s = MemoryStore(conn=connect(db_path))
        gate = SalienceGate(threshold=0.5, rearm_below=0.2, ema_alpha=1.0,
                            heartbeat_seconds=0)
        fr = iter([(0.0, Frame(A)), (1.0, Frame(A)), (2.0, Frame(B))])
        evs = watchloop.run_watch(
            s, fr, embed=lambda f: f.vec, gate=gate,
            keyframe_dir=keyframe_dir, start_wall=WALL0, source_label="screen")
        s.close()
        return evs

    def test_dry_run_lists_pending_without_a_model(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            evs = self._seed(db, str(Path(d) / "kf"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(["--db", db, "--no-shared", "recaption", "--dry-run"])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("2 keyframe(s) await a caption", out)
            for e in evs:
                self.assertIn(f"#{e.memory_id}", out)

    def test_recaption_runs_the_injected_captioner(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            evs = self._seed(db, str(Path(d) / "kf"))
            with mock.patch.object(mac_vision, "vlm_captioner",
                                   return_value=lambda p: "a quiet room"):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = main(["--db", db, "--no-shared", "recaption"])
            self.assertEqual(rc, 0)
            self.assertIn("2 caption(s) written.", buf.getvalue())
            s = MemoryStore(conn=connect(db))
            gists = [s.conn.execute("SELECT gist FROM memory WHERE id=?",
                                    (e.memory_id,)).fetchone()[0] for e in evs]
            s.close()
            self.assertEqual(gists, ["a quiet room", "a quiet room"])


class TestVlmCaptioner(unittest.TestCase):
    """The Ollama-backed captioner is pure stdlib (urllib) — no daemon needed
    for the test: urlopen is faked. It reads the keyframe, sends the model +
    base64 image, and returns the model's trimmed response."""

    class _Resp:
        def __init__(self, payload): self._p = payload
        def read(self): return self._p
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def test_posts_image_and_returns_response(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            f.write(b"\xff\xd8jpegbytes"); f.flush()
            sent = {}
            def fake_urlopen(req, timeout=None):
                sent["url"] = req.full_url
                sent["body"] = json.loads(req.data.decode())
                return self._Resp(json.dumps({"response": "  a dog on a sofa\n"}
                                             ).encode())
            cap = mac_vision.vlm_captioner("moondream")
            with mock.patch("urllib.request.urlopen", fake_urlopen):
                out = cap(f.name)
        self.assertEqual(out, "a dog on a sofa")            # trimmed
        self.assertTrue(sent["url"].endswith("/api/generate"))
        self.assertEqual(sent["body"]["model"], "moondream")
        self.assertEqual(len(sent["body"]["images"]), 1)    # base64 image sent
        self.assertFalse(sent["body"]["stream"])

    def test_unreachable_daemon_raises_actionable_error(self):
        cap = mac_vision.vlm_captioner("llava")
        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            f.write(b"\xff\xd8x"); f.flush()
            with mock.patch("urllib.request.urlopen", boom):
                with self.assertRaises(RuntimeError) as ctx:
                    cap(f.name)
        self.assertIn("ollama", str(ctx.exception).lower())
        self.assertIn("llava", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
