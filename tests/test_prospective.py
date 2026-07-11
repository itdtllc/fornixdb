"""Prospective memory — remind / due / upcoming, and parse_due (the future
twin of parse_when). All pure: in-memory store, pinned clock, no scheduler."""

import unittest
from datetime import datetime, timedelta

from fornixdb import prospective
from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.timeparse import parse_due

NOW = datetime(2026, 7, 10, 12, 0)  # a Friday, noon


class TestParseDue(unittest.TestCase):
    def d(self, text):
        return parse_due(text, now=NOW)

    def test_relative(self):
        self.assertEqual(self.d("in 20 minutes"), NOW + timedelta(minutes=20))
        self.assertEqual(self.d("in an hour"), NOW + timedelta(hours=1))
        self.assertEqual(self.d("in a few minutes"), NOW + timedelta(minutes=3))
        self.assertEqual(self.d("in a couple of hours"), NOW + timedelta(hours=2))
        self.assertEqual(self.d("in half an hour"), NOW + timedelta(minutes=30))
        self.assertEqual(self.d("in 2 days"), NOW + timedelta(days=2))
        self.assertEqual(self.d("in 1 week"), NOW + timedelta(days=7))

    def test_tomorrow_parts(self):
        self.assertEqual(self.d("tomorrow"), datetime(2026, 7, 11, 9))
        self.assertEqual(self.d("tomorrow morning"), datetime(2026, 7, 11, 9))
        self.assertEqual(self.d("tomorrow afternoon"), datetime(2026, 7, 11, 15))
        self.assertEqual(self.d("tomorrow evening"), datetime(2026, 7, 11, 19))
        self.assertEqual(self.d("tomorrow night"), datetime(2026, 7, 11, 19))
        self.assertEqual(self.d("tomorrow at 3pm"), datetime(2026, 7, 11, 15))
        self.assertEqual(self.d("tomorrow at 7:15am"), datetime(2026, 7, 11, 7, 15))

    def test_tonight_and_dayparts_roll_forward(self):
        self.assertEqual(self.d("tonight"), datetime(2026, 7, 10, 19))
        # at noon, "this morning" is gone -> tomorrow morning
        self.assertEqual(self.d("this morning"), datetime(2026, 7, 11, 9))
        self.assertEqual(self.d("this afternoon"), datetime(2026, 7, 10, 15))
        # "tonight" said late in the evening still lands in the future
        late = datetime(2026, 7, 10, 21, 0)
        self.assertEqual(parse_due("tonight", now=late), late + timedelta(hours=1))

    def test_bare_clock_rolls_if_past(self):
        self.assertEqual(self.d("at 3pm"), datetime(2026, 7, 10, 15))
        self.assertEqual(self.d("3pm"), datetime(2026, 7, 10, 15))
        self.assertEqual(self.d("15:30"), datetime(2026, 7, 10, 15, 30))
        self.assertEqual(self.d("at 9am"), datetime(2026, 7, 11, 9))   # past -> tomorrow
        self.assertEqual(self.d("noon"), datetime(2026, 7, 11, 12))    # exactly now -> tomorrow

    def test_weekdays_next_occurrence_never_today(self):
        # NOW is Friday; "friday" means NEXT friday, "monday" the coming one
        self.assertEqual(self.d("friday"), datetime(2026, 7, 17, 9))
        self.assertEqual(self.d("next friday"), datetime(2026, 7, 17, 9))
        self.assertEqual(self.d("monday"), datetime(2026, 7, 13, 9))
        self.assertEqual(self.d("monday at 2pm"), datetime(2026, 7, 13, 14))

    def test_next_week_month_and_dates(self):
        self.assertEqual(self.d("next week"), datetime(2026, 7, 13, 9))  # Monday
        self.assertEqual(self.d("next month"), datetime(2026, 8, 1, 9))
        self.assertEqual(self.d("july 20"), datetime(2026, 7, 20, 9))
        self.assertEqual(self.d("july 20 at 6pm"), datetime(2026, 7, 20, 18))
        # a month-day already past this year -> next year
        self.assertEqual(self.d("january 5"), datetime(2027, 1, 5, 9))
        self.assertEqual(self.d("2026-08-01"), datetime(2026, 8, 1, 9))
        self.assertEqual(self.d("2026-08-01T14:30"), datetime(2026, 8, 1, 14, 30))

    def test_rejections(self):
        with self.assertRaises(ValueError):
            self.d("whenever")
        with self.assertRaises(ValueError):
            self.d("2026-01-01")            # explicit past timestamp
        with self.assertRaises(ValueError):
            self.d("at 3")                  # bare hour, ambiguous


class TestProspectiveStore(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))

    def tearDown(self):
        self.s.close()

    def test_remind_stores_memory_plus_due_row(self):
        r = prospective.remind(self.s, "call the attorney", "tomorrow 9am",
                               now=NOW)
        self.assertEqual(r["due"], "2026-07-11T09:00:00")
        row = self.s.conn.execute(
            "SELECT kind, event_time, gist FROM memory WHERE id=?",
            (r["id"],)).fetchone()
        self.assertEqual(row[0], "episodic")
        self.assertEqual(row[1], "2026-07-11T09:00:00")
        self.assertIn("call the attorney", row[2])
        p = self.s.conn.execute(
            "SELECT due, delivered_at FROM prospective WHERE memory_id=?",
            (r["id"],)).fetchone()
        self.assertEqual(p[0], "2026-07-11T09:00:00")
        self.assertIsNone(p[1])

    def test_bad_phrase_raises_and_stores_nothing(self):
        with self.assertRaises(ValueError):
            prospective.remind(self.s, "x", "whenever", now=NOW)
        self.assertEqual(self.s.conn.execute(
            "SELECT COUNT(*) FROM memory").fetchone()[0], 0)

    def test_due_fires_exactly_once(self):
        prospective.remind(self.s, "stand up", "in 5 minutes", now=NOW)
        prospective.remind(self.s, "next week thing", "next week", now=NOW)
        self.assertEqual(prospective.due(self.s, now=NOW), [])   # nothing yet
        later = NOW + timedelta(minutes=6)
        rows = prospective.due(self.s, now=later)
        self.assertEqual(len(rows), 1)
        self.assertIn("stand up", rows[0]["gist"])
        # delivered: a second poll is silent
        self.assertEqual(prospective.due(self.s, now=later), [])

    def test_peek_does_not_consume(self):
        prospective.remind(self.s, "stand up", "in 5 minutes", now=NOW)
        later = NOW + timedelta(minutes=6)
        self.assertEqual(len(prospective.due(self.s, now=later,
                                             deliver=False)), 1)
        self.assertEqual(len(prospective.due(self.s, now=later)), 1)

    def test_upcoming_window_and_order(self):
        prospective.remind(self.s, "soonest", "in 2 hours", now=NOW)
        prospective.remind(self.s, "later", "tomorrow morning", now=NOW)
        prospective.remind(self.s, "far", "next week", now=NOW)
        ahead = prospective.upcoming(self.s, now=NOW, within_hours=24)
        self.assertEqual([r["gist"] for r in ahead],
                         ["Reminder: soonest", "Reminder: later"])

    def test_cancelled_reminder_never_fires(self):
        r = prospective.remind(self.s, "obsolete", "in 5 minutes", now=NOW)
        self.s.tombstone(r["id"])
        self.assertEqual(prospective.due(self.s,
                                         now=NOW + timedelta(minutes=6)), [])

    def test_superseded_reminder_never_fires(self):
        r = prospective.remind(self.s, "old time", "in 5 minutes", now=NOW)
        r2 = prospective.remind(self.s, "old time", "in 2 hours", now=NOW)
        self.s.supersede(r["id"], r2["id"])
        rows = prospective.due(self.s, now=NOW + timedelta(minutes=6))
        self.assertEqual(rows, [])                       # old one is tombstoned
        rows = prospective.due(self.s, now=NOW + timedelta(hours=3))
        self.assertEqual(len(rows), 1)                   # new one fires

    def test_reminder_rows_ride_the_timeline(self):
        prospective.remind(self.s, "dentist", "tomorrow 2pm", now=NOW)
        rows = self.s.timeline("2026-07-11T00:00:00", "2026-07-12T00:00:00")
        self.assertTrue(any("dentist" in m["gist"] for m in rows))


class TestDueReminderBlock(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))

    def tearDown(self):
        self.s.close()

    def test_block_formats_and_consumes(self):
        from fornixdb.proactive import due_reminder_block
        prospective.remind(self.s, "stretch", "in 1 minute", now=NOW)
        # simulate the clock arriving by rewriting due (block uses real now)
        self.s.conn.execute("UPDATE prospective SET due = '2000-01-01T00:00:00'")
        self.s.conn.commit()
        block = due_reminder_block(self.s)
        self.assertIn("stretch", block)
        self.assertIn("DUE", block)
        self.assertIsNone(due_reminder_block(self.s))    # consumed


if __name__ == "__main__":
    unittest.main()
