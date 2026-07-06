"""The salience gate: dense sampling in, sparse commits out — one event per
scene change, hysteresis re-arm, heartbeat anchors, deterministic and pure."""

import unittest

from fornixdb.salience import Decision, SalienceGate, cosine

E1 = [1.0, 0.0]
E2 = [0.0, 1.0]
E3 = [-1.0, 0.0]


class TestCosine(unittest.TestCase):
    def test_basic_and_zero_guard(self):
        self.assertAlmostEqual(cosine(E1, E1), 1.0)
        self.assertAlmostEqual(cosine(E1, E2), 0.0)
        self.assertAlmostEqual(cosine(E1, E3), -1.0)
        self.assertEqual(cosine([0.0, 0.0], E1), 0.0)


class TestGate(unittest.TestCase):
    def test_first_sample_always_commits(self):
        g = SalienceGate()
        d = g.observe(E1, t=0.0)
        self.assertEqual((d.commit, d.reason), (True, "first"))

    def test_stable_scene_stays_silent(self):
        g = SalienceGate(heartbeat_seconds=0)  # no heartbeats in this test
        g.observe(E1, t=0.0)
        for i in range(1, 50):
            d = g.observe(E1, t=i * 0.1)
            self.assertFalse(d.commit, f"sample {i} committed on nothing")

    def test_scene_change_commits_exactly_once(self):
        g = SalienceGate(heartbeat_seconds=0)
        g.observe(E1, t=0.0)
        commits = [g.observe(E2, t=1.0 + i * 0.1) for i in range(30)]
        events = [d for d in commits if d.commit]
        self.assertEqual(len(events), 1, "one scene change must be one commit")
        self.assertEqual(events[0].reason, "event")
        self.assertTrue(commits[0].commit, "the change itself fires, not later")

    def test_rearms_after_settling_then_fires_again(self):
        g = SalienceGate(heartbeat_seconds=0)
        g.observe(E1, t=0.0)
        for i in range(40):                      # E2 becomes the new normal
            g.observe(E2, t=1.0 + i * 0.1)
        d = g.observe(E3, t=10.0)                # next real change
        self.assertEqual((d.commit, d.reason), (True, "event"))

    def test_heartbeat_anchors_a_quiet_stream(self):
        g = SalienceGate(heartbeat_seconds=600.0)
        g.observe(E1, t=0.0)
        self.assertFalse(g.observe(E1, t=599.0).commit)
        d = g.observe(E1, t=600.0)
        self.assertEqual((d.commit, d.reason), (True, "heartbeat"))
        self.assertFalse(g.observe(E1, t=601.0).commit)  # timer reset
        self.assertEqual(g.observe(E1, t=1200.0).reason, "heartbeat")

    def test_event_resets_the_heartbeat_clock(self):
        g = SalienceGate(heartbeat_seconds=600.0)
        g.observe(E1, t=0.0)
        self.assertEqual(g.observe(E2, t=500.0).reason, "event")
        self.assertFalse(g.observe(E2, t=650.0).commit,
                         "heartbeat must count from the event, not the start")

    def test_decision_carries_distance(self):
        g = SalienceGate(heartbeat_seconds=0)
        g.observe(E1, t=0.0)
        d = g.observe(E2, t=1.0)
        self.assertIsInstance(d, Decision)
        self.assertAlmostEqual(d.distance, 1.0, places=5)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            SalienceGate(threshold=0.0)
        with self.assertRaises(ValueError):
            SalienceGate(threshold=0.3, rearm_below=0.3)
        with self.assertRaises(ValueError):
            SalienceGate(ema_alpha=0.0)


if __name__ == "__main__":
    unittest.main()
