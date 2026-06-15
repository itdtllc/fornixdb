import unittest
from datetime import datetime, timedelta

from fornixdb.timeparse import parse_when

NOW = datetime(2026, 6, 10, 15, 30)  # a Wednesday


class TestParseWhen(unittest.TestCase):
    def r(self, text):
        return parse_when(text, now=NOW)

    def test_today_yesterday(self):
        s, e = self.r("today")
        self.assertEqual(s, datetime(2026, 6, 10))
        self.assertEqual(e, datetime(2026, 6, 11))
        s, _ = self.r("yesterday")
        self.assertEqual(s, datetime(2026, 6, 9))

    def test_parts_of_days(self):
        s, e = self.r("last night")  # includes the small hours of today
        self.assertEqual(s, datetime(2026, 6, 9, 17))
        self.assertEqual(e, datetime(2026, 6, 10, 6))
        s, e = self.r("this morning")
        self.assertEqual(s, datetime(2026, 6, 10))
        self.assertEqual(e, datetime(2026, 6, 10, 12))
        s, e = self.r("tonight")
        self.assertEqual(s, datetime(2026, 6, 10, 17))
        self.assertEqual(e, datetime(2026, 6, 11, 6))
        s, e = self.r("yesterday morning")
        self.assertEqual(s, datetime(2026, 6, 9))
        self.assertEqual(e, datetime(2026, 6, 9, 12))

    def test_earlier_today_runs_start_of_day_to_now(self):
        for phrase in ("earlier today", "earlier in the day", "today so far",
                       "so far today"):
            s, e = self.r(phrase)
            self.assertEqual(s, datetime(2026, 6, 10), phrase)         # start of today
            self.assertEqual(e, NOW + timedelta(seconds=1), phrase)

    def test_past_hour_and_relative_units(self):
        for phrase in ("past hour", "last hour", "the past hour", "just now"):
            s, e = self.r(phrase)
            self.assertEqual(s, NOW - timedelta(hours=1), phrase)
            self.assertEqual(e, NOW + timedelta(seconds=1), phrase)
        s, e = self.r("past 2 hours")
        self.assertEqual(s, NOW - timedelta(hours=2))
        s, e = self.r("last 30 minutes")
        self.assertEqual(s, NOW - timedelta(minutes=30))

    def test_units_ago_is_a_single_window(self):
        s, e = self.r("2 hours ago")
        self.assertEqual(s, NOW - timedelta(hours=2))
        self.assertEqual(e, NOW - timedelta(hours=1))
        s, e = self.r("30 minutes ago")
        self.assertEqual(s, NOW - timedelta(minutes=30))
        self.assertEqual(e, NOW - timedelta(minutes=29))
        s, e = self.r("an hour ago")
        self.assertEqual(s, NOW - timedelta(hours=1))

    def test_last_thursday(self):
        s, e = self.r("last thursday")
        self.assertEqual(s, datetime(2026, 6, 4))  # Thu before Wed 6/10
        self.assertEqual(e, datetime(2026, 6, 5))

    def test_bare_weekday_is_most_recent(self):
        s, _ = self.r("wednesday")  # today is Wednesday
        self.assertEqual(s, datetime(2026, 6, 10))

    def test_last_weekday_today_means_prior_week(self):
        s, _ = self.r("last wednesday")
        self.assertEqual(s, datetime(2026, 6, 3))

    def test_weeks(self):
        s, e = self.r("this week")
        self.assertEqual(s, datetime(2026, 6, 8))  # Monday
        s, e = self.r("last week")
        self.assertEqual(s, datetime(2026, 6, 1))
        self.assertEqual(e, datetime(2026, 6, 8))

    def test_n_days_ago(self):
        s, _ = self.r("3 days ago")
        self.assertEqual(s, datetime(2026, 6, 7))

    def test_last_n_days_is_range_to_now(self):
        s, e = self.r("last 3 days")
        self.assertEqual(s, datetime(2026, 6, 8))
        self.assertGreater(e, NOW)

    def test_month_name(self):
        s, _ = self.r("june 5")
        self.assertEqual(s, datetime(2026, 6, 5))
        s, _ = self.r("december 25")  # future this year → last year
        self.assertEqual(s, datetime(2025, 12, 25))

    def test_iso_forms(self):
        s, e = self.r("2026-06-05")
        self.assertEqual((s, e), (datetime(2026, 6, 5), datetime(2026, 6, 6)))
        s, e = self.r("2026-06")
        self.assertEqual((s, e), (datetime(2026, 6, 1), datetime(2026, 7, 1)))
        s, e = self.r("2026")
        self.assertEqual((s, e), (datetime(2026, 1, 1), datetime(2027, 1, 1)))

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            self.r("the day the music died")


if __name__ == "__main__":
    unittest.main()
