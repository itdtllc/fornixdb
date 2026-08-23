"""`--since-days` on the two log readouts.

Without a window both average across every configuration the store has ever run
under. The case that proved it: field-stats reported 90 dissent-emitted beats
while the dial that produces them was off — all 90 predated the trial's close,
and a 30-day window reports 0.
"""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from fornixdb.field_stats import load_beats
from fornixdb.floor_stats import load_records


def _ts(days_ago):
    return (datetime.now() - timedelta(days=days_ago)).isoformat() + "-05:00"


def _write(tmp, rows):
    p = Path(tmp) / "log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


class TestFieldStatsWindow(unittest.TestCase):

    def test_window_keeps_only_recent_beats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, [{"ts": _ts(60), "settled": True},
                                {"ts": _ts(40), "settled": True},
                                {"ts": _ts(3), "settled": True}])
            self.assertEqual(len(load_beats(path)), 3)
            self.assertEqual(len(load_beats(path, since_days=30)), 1)

    def test_no_window_is_all_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, [{"ts": _ts(400), "settled": True}])
            self.assertEqual(len(load_beats(path)), 1)
            self.assertEqual(len(load_beats(path, since_days=0)), 1)
            self.assertEqual(len(load_beats(path, since_days=None)), 1)

    def test_a_beat_with_no_timestamp_is_dropped_by_a_window(self):
        # undatable: it cannot be shown to belong to the window asked for
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, [{"settled": True}])
            self.assertEqual(len(load_beats(path)), 1)
            self.assertEqual(len(load_beats(path, since_days=30)), 0)

    def test_corrupt_lines_still_skipped_with_a_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "log.jsonl"
            p.write_text(json.dumps({"ts": _ts(1), "settled": True})
                         + "\nnot json at all\n", encoding="utf-8")
            self.assertEqual(len(load_beats(str(p), since_days=30)), 1)


class TestFloorStatsWindow(unittest.TestCase):

    def test_window_keeps_only_recent_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, [{"ts": _ts(90), "decision": "surfaced"},
                                {"ts": _ts(2), "decision": "surfaced"}])
            self.assertEqual(len(load_records(path)), 2)
            self.assertEqual(len(load_records(path, since_days=30)), 1)

    def test_blank_and_corrupt_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "log.jsonl"
            p.write_text("\n\n" + json.dumps({"ts": _ts(1), "decision": "x"})
                         + "\n{oops\n", encoding="utf-8")
            self.assertEqual(len(load_records(str(p), since_days=7)), 1)

    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(load_records("/nonexistent/log.jsonl", since_days=7), [])
        self.assertEqual(load_records(None, since_days=7), [])


if __name__ == "__main__":
    unittest.main()
