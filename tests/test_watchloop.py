"""The watch-loop core: dense (t, frame) samples through the salience gate
become sparse `see` memories with event-time spans; frames that never commit
never touch disk. Pure — fake frames, fake embedder, explicit timestamps."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fornixdb import watchloop
from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.salience import SalienceGate

A, B, C = [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]
WALL0 = datetime(2026, 7, 6, 12, 0, 0)


def strict_gate(**kw):
    """ema_alpha=1.0 makes the reference the previous sample — deterministic
    distances; heartbeats off unless a test wants them."""
    kw.setdefault("threshold", 0.5)
    kw.setdefault("rearm_below", 0.2)
    kw.setdefault("ema_alpha", 1.0)
    kw.setdefault("heartbeat_seconds", 0)
    return SalienceGate(**kw)


class WatchBase(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def row(self, mid):
        return self.s.conn.execute(
            "SELECT gist, source, source_ref, event_time, event_time_end "
            "FROM memory WHERE id = ?", (mid,)).fetchone()


class Frame(bytes):
    """Encoded frame bytes that also carry their fake embedding."""
    def __new__(cls, vec, payload=b"\xff\xd8fake"):
        o = super().__new__(cls, payload)
        o.vec = vec
        return o


def frames(*spec):
    """spec = (t, vec) pairs -> (t, Frame) stream."""
    return iter((t, Frame(v)) for t, v in spec)


class TestCommits(WatchBase):
    def run_watch(self, fr, **kw):
        kw.setdefault("embed", lambda f: f.vec)
        kw.setdefault("gate", strict_gate())
        kw.setdefault("keyframe_dir", str(self.dir))
        kw.setdefault("start_wall", WALL0)
        kw.setdefault("session_id", "w1")
        return watchloop.run_watch(self.s, fr, **kw)

    def test_first_frame_commits_and_is_a_sight_memory(self):
        evs = self.run_watch(frames((0.0, A)))
        self.assertEqual([e.reason for e in evs], ["first"])
        gist, source, ref, et, ete = self.row(evs[0].memory_id)
        self.assertEqual(source, "senses:sight")
        self.assertIn("session start", gist)
        self.assertEqual(et, "2026-07-06T12:00:00")
        self.assertIsNone(ete)                    # zero-length span collapses
        self.assertTrue(Path(ref).is_file())

    def test_scene_change_commits_once_then_rearms(self):
        evs = self.run_watch(frames(
            (0.0, A), (1.0, A),                   # first, then quiet
            (2.0, B),                             # scene change -> event
            (3.0, B),                             # still B: disarmed, no commit
            (4.0, B),                             # d=0 < rearm_below -> re-arms
            (5.0, C)))                            # next change -> event
        self.assertEqual([e.reason for e in evs], ["first", "event", "event"])

    def test_event_span_runs_from_after_previous_commit(self):
        evs = self.run_watch(frames((0.0, A), (1.0, A), (2.0, A), (3.0, B)))
        ev = evs[-1]
        self.assertEqual((ev.t_start, ev.t_end), (1.0, 3.0))
        _, _, _, et, ete = self.row(ev.memory_id)
        self.assertEqual((et, ete),
                         ("2026-07-06T12:00:01", "2026-07-06T12:00:03"))

    def test_window_seconds_caps_the_span(self):
        evs = self.run_watch(
            frames((0.0, A), (1.0, A), (100.0, A), (200.0, B)),
            window_seconds=30.0)
        self.assertEqual(evs[-1].t_start, 170.0)  # 200 - 30, not 1.0

    def test_heartbeat_anchors_quiet_stretches(self):
        evs = self.run_watch(
            frames((0.0, A), (5.0, A), (11.0, A)),
            gate=strict_gate(heartbeat_seconds=10))
        self.assertEqual([e.reason for e in evs], ["first", "heartbeat"])
        self.assertIn("quiet interval", evs[-1].gist)

    def test_uncommitted_frames_never_touch_disk(self):
        self.run_watch(frames((0.0, A), (1.0, A), (2.0, A), (3.0, B)))
        written = list((self.dir / "w1").glob("*.jpg"))
        self.assertEqual(len(written), 2)         # first + event only

    def test_bytes_frames_require_keyframe_dir(self):
        with self.assertRaises(ValueError):
            watchloop.run_watch(self.s, frames((0.0, A)),
                                embed=lambda f: f.vec, keyframe_dir=None)

    def test_path_frames_pass_through(self):
        p = self.dir / "shot.jpg"
        p.write_bytes(b"\xff\xd8fake")
        evs = watchloop.run_watch(self.s, iter([(0.0, str(p))]),
                                  embed=lambda f: A, start_wall=WALL0)
        self.assertEqual(evs[0].keyframe, str(p))

    def test_captioner_writes_the_gist(self):
        evs = self.run_watch(frames((0.0, A)),
                             captioner=lambda path: "a red ball on a desk")
        self.assertEqual(self.row(evs[0].memory_id)[0],
                         "a red ball on a desk")

    def test_max_commits_and_max_seconds_stop_the_loop(self):
        evs = self.run_watch(frames((0.0, A), (1.0, B), (2.0, C)),
                             max_commits=1)
        self.assertEqual(len(evs), 1)
        evs = self.run_watch(frames((0.0, A), (1.0, A), (99.0, B)),
                             max_seconds=50.0)
        self.assertEqual([e.reason for e in evs], ["first"])

    def test_on_commit_streams_events(self):
        seen = []
        self.run_watch(frames((0.0, A), (1.0, B)), on_commit=seen.append)
        self.assertEqual([e.reason for e in seen], ["first", "event"])


class _FakeModal:
    """A modality (image) embedder stub: ignores the path, returns a fixed
    vector. Proves the latent-lane vector is written BEFORE a dropped keyframe
    is deleted, so same-modality recall survives the drop."""
    name = "fake-img"

    def embed_artifact(self, paths):
        return [[0.1, 0.2, 0.3] for _ in paths]


class TestLiveControls(WatchBase):
    """should_continue (stop a background 'eyes' loop) and
    drop_keyframe_after_commit (leave no stills on disk)."""

    def run_watch(self, fr, **kw):
        kw.setdefault("embed", lambda f: f.vec)
        kw.setdefault("gate", strict_gate())
        kw.setdefault("keyframe_dir", str(self.dir))
        kw.setdefault("start_wall", WALL0)
        kw.setdefault("session_id", "w1")
        return watchloop.run_watch(self.s, fr, **kw)

    def test_should_continue_stops_the_loop_mid_stream(self):
        seen = []
        # stop after the loop has pulled two frames
        evs = self.run_watch(
            frames((0.0, A), (1.0, B), (2.0, C)),
            should_continue=lambda: len(seen) < 2,
            on_commit=seen.append)
        self.assertEqual([e.reason for e in evs], ["first", "event"])  # C never reached

    def test_should_continue_false_from_start_commits_nothing(self):
        evs = self.run_watch(frames((0.0, A), (1.0, B)),
                             should_continue=lambda: False)
        self.assertEqual(evs, [])

    def test_drop_keyframe_deletes_file_nulls_ref_keeps_vector(self):
        evs = self.run_watch(
            frames((0.0, A)), captioner=lambda p: "a lit room with a window",
            modal_embedder=_FakeModal(), drop_keyframe_after_commit=True)
        mid = evs[0].memory_id
        self.assertFalse(Path(evs[0].keyframe).exists())     # still gone
        gist, _, ref, _, _ = self.row(mid)
        self.assertEqual(gist, "a lit room with a window")   # real caption kept
        self.assertIsNone(ref)                               # source_ref nulled
        n = self.s.conn.execute(
            "SELECT COUNT(*) FROM modal_embedding WHERE memory_id=? AND model=?",
            (mid, "fake-img")).fetchone()[0]
        self.assertEqual(n, 1)                               # latent lane survives

    def test_keyframes_kept_by_default(self):
        evs = self.run_watch(frames((0.0, A)))
        self.assertTrue(Path(evs[0].keyframe).is_file())     # default: no drop

    def test_drop_without_captioner_is_rejected(self):
        # Dropping the still while writing only a placeholder would strand a row
        # the dream pass can never caption — reject the incoherent combination.
        with self.assertRaises(ValueError):
            self.run_watch(frames((0.0, A)), drop_keyframe_after_commit=True)

    def test_stopping_closes_the_frame_generator(self):
        released = []
        def gen():
            try:
                for t, v in ((0.0, A), (1.0, B), (2.0, C)):
                    yield t, Frame(v)
            finally:
                released.append(True)      # an adapter releases the camera here
        seen = []
        self.run_watch(gen(), should_continue=lambda: len(seen) < 1,
                       on_commit=seen.append)
        self.assertEqual(released, [True])  # generator closed on stop, not at GC


if __name__ == "__main__":
    unittest.main()
