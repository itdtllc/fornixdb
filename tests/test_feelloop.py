"""The feel-loop core: a stream of (t, reading) samples becomes sparse `feel`
memories — commit the first, commit on any watched-field change, commit a
heartbeat after quiet. Pure: fake readings, explicit timestamps, no sensor.
Plus the Mac power adapter's pure `parse_batt` (no subprocess)."""

import unittest
from datetime import datetime

from fornixdb import feelloop
from fornixdb.adapters import mac_proprioception as mp
from fornixdb.core import MemoryStore
from fornixdb.db import connect

WALL0 = datetime(2026, 7, 7, 9, 0, 0)


def readings(*spec):
    """spec = (t, reading) pairs -> iterator."""
    return iter(spec)


class FeelBase(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))

    def tearDown(self):
        self.s.close()

    def run_feel(self, rs, **kw):
        kw.setdefault("sensor", "power")
        kw.setdefault("start_wall", WALL0)
        kw.setdefault("session_id", "f1")
        kw.setdefault("heartbeat_seconds", 0)   # off unless a test asks
        return feelloop.run_feel(self.s, rs, **kw)

    def row(self, mid):
        return self.s.conn.execute(
            "SELECT gist, source, source_ref, event_time, event_time_end "
            "FROM memory WHERE id = ?", (mid,)).fetchone()


class TestCommits(FeelBase):
    AC = {"source": "AC", "state": "charged", "percent": 100}
    BATT = {"source": "battery", "state": "discharging", "percent": 90}

    def test_first_reading_commits_and_is_a_feel_memory(self):
        evs = self.run_feel(readings((0.0, self.AC)))
        self.assertEqual([e.reason for e in evs], ["first"])
        gist, source, ref, et, ete = self.row(evs[0].memory_id)
        self.assertEqual(source, "senses:feel")
        self.assertEqual(ref, "sensor:power")
        self.assertIn("source=AC", gist)
        self.assertEqual(et, "2026-07-07T09:00:00")
        self.assertIsNone(ete)                       # point event, no span

    def test_field_change_commits(self):
        evs = self.run_feel(readings(
            (0.0, self.AC),
            (60.0, self.AC),                         # unchanged -> hold
            (120.0, self.BATT)))                     # went on battery -> commit
        self.assertEqual([e.reason for e in evs], ["first", "change"])
        self.assertEqual(evs[-1].t, 120.0)
        self.assertIn("source=battery", self.row(evs[-1].memory_id)[0])

    def test_unchanged_readings_never_commit(self):
        evs = self.run_feel(readings(
            (0.0, self.AC), (60.0, self.AC), (120.0, self.AC)))
        self.assertEqual(len(evs), 1)

    def test_change_back_and_forth_commits_each_transition(self):
        evs = self.run_feel(readings(
            (0.0, self.AC), (10.0, self.BATT), (20.0, self.AC)))
        self.assertEqual([e.reason for e in evs], ["first", "change", "change"])

    def test_ignore_fields_suppresses_noisy_drift(self):
        evs = self.run_feel(readings(
            (0.0, {"source": "battery", "percent": 90}),
            (60.0, {"source": "battery", "percent": 89}),   # only percent moved
            (120.0, {"source": "battery", "percent": 88})),
            ignore_fields={"percent"})
        self.assertEqual(len(evs), 1)                # all three look identical

    def test_heartbeat_anchors_quiet_stretches(self):
        evs = self.run_feel(readings(
            (0.0, self.AC), (60.0, self.AC), (600.0, self.AC)),
            heartbeat_seconds=300)
        self.assertEqual([e.reason for e in evs], ["first", "heartbeat"])

    def test_heartbeat_clock_resets_after_each_commit(self):
        evs = self.run_feel(readings(
            (0.0, self.AC), (300.0, self.AC), (450.0, self.AC),
            (600.0, self.AC)),
            heartbeat_seconds=300)
        # commits at 0 (first) and 300 (heartbeat); 450 is <300 since 300;
        # 600 is 300 after the last commit -> heartbeat again
        self.assertEqual([e.reason for e in evs],
                         ["first", "heartbeat", "heartbeat"])

    def test_string_readings_diff_by_value(self):
        evs = self.run_feel(readings(
            (0.0, "lid open"), (10.0, "lid open"), (20.0, "lid closed")),
            sensor="lid")
        self.assertEqual([e.reason for e in evs], ["first", "change"])

    def test_max_commits_and_max_seconds_stop_the_loop(self):
        evs = self.run_feel(readings(
            (0.0, self.AC), (10.0, self.BATT), (20.0, self.AC)),
            max_commits=1)
        self.assertEqual(len(evs), 1)
        evs = self.run_feel(readings(
            (0.0, self.AC), (10.0, self.AC), (99.0, self.BATT)),
            max_seconds=50.0)
        self.assertEqual([e.reason for e in evs], ["first"])

    def test_on_commit_streams_events(self):
        seen = []
        self.run_feel(readings((0.0, self.AC), (10.0, self.BATT)),
                      on_commit=seen.append)
        self.assertEqual([e.reason for e in seen], ["first", "change"])


class TestPmsetParse(unittest.TestCase):
    def test_ac_attached_not_charging(self):
        r = mp.parse_batt(
            "Now drawing from 'AC Power'\n"
            " -InternalBattery-0 (id=23986275)\t80%; AC attached; "
            "not charging present: true")
        self.assertEqual(r["source"], "AC")
        self.assertEqual(r["percent"], 80)
        self.assertEqual(r["state"], "not charging")
        self.assertIsNone(r["remaining"])            # no estimate given

    def test_on_battery_discharging_with_estimate(self):
        r = mp.parse_batt(
            "Now drawing from 'Battery Power'\n"
            " -InternalBattery-0 (id=1)\t75%; discharging; 3:42 remaining "
            "present: true")
        self.assertEqual(
            (r["source"], r["percent"], r["state"], r["remaining"]),
            ("battery", 75, "discharging", "3:42"))

    def test_charging(self):
        r = mp.parse_batt(
            "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t"
            "95%; charging; 0:20 remaining present: true")
        self.assertEqual((r["source"], r["state"]), ("AC", "charging"))

    def test_fully_charged_zero_estimate_drops_remaining(self):
        r = mp.parse_batt(
            "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t"
            "100%; charged; 0:00 remaining present: true")
        self.assertEqual(r["state"], "charged")
        self.assertIsNone(r["remaining"])

    def test_garbage_degrades_to_nones(self):
        r = mp.parse_batt("unexpected output")
        self.assertEqual(r, {"source": None, "percent": None,
                             "state": None, "remaining": None})


class TestBatteryFrames(unittest.TestCase):
    def fake_reader(self, *seq):
        it = iter(seq)
        return lambda: next(it)

    def test_buckets_percent_and_drops_remaining(self):
        reader = self.fake_reader(
            {"source": "battery", "state": "discharging",
             "percent": 87, "remaining": "3:10"})
        (t, r), = mp.battery_frames(count=1, reader=reader,
                                    clock=lambda: 5.0, sleep=lambda s: None)
        self.assertEqual(t, 5.0)
        self.assertEqual(r, {"source": "battery", "state": "discharging",
                             "percent": 80})         # 87 -> 80, remaining gone

    def test_percent_step_one_keeps_exact_value(self):
        reader = self.fake_reader({"source": "AC", "state": "charged",
                                   "percent": 87, "remaining": None})
        (_, r), = mp.battery_frames(count=1, percent_step=1, reader=reader,
                                    clock=lambda: 0.0, sleep=lambda s: None)
        self.assertEqual(r["percent"], 87)

    def test_count_bounds_the_stream_and_sleeps_between(self):
        reader = self.fake_reader(
            {"source": "AC", "state": "charged", "percent": 100},
            {"source": "battery", "state": "discharging", "percent": 90})
        naps = []
        ticks = iter([0.0, 60.0])
        frames = list(mp.battery_frames(
            count=2, reader=reader, clock=lambda: next(ticks),
            sleep=naps.append))
        self.assertEqual([t for t, _ in frames], [0.0, 60.0])
        self.assertEqual(naps, [60.0])               # slept once, between two


if __name__ == "__main__":
    unittest.main()
