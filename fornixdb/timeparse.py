"""Natural-language time expressions → (start, end) datetime ranges.

Stdlib-only. Understands the phrasings a person uses when asking a memory a
time question: "yesterday", "last thursday", "this week", "3 days ago",
"last 2 weeks", "june 5", ISO dates, and ISO date ranges.
All returned ranges are half-open [start, end).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def _day(d: datetime) -> tuple[datetime, datetime]:
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _week_start(d: datetime) -> datetime:
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start - timedelta(days=start.weekday())  # Monday


def parse_when(text: str, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Parse a natural time phrase into a [start, end) datetime range.

    Raises ValueError for phrases it does not understand.
    """
    now = now or datetime.now()
    t = text.strip().lower()

    if t in ("today",):
        return _day(now)
    if t == "yesterday":
        return _day(now - timedelta(days=1))

    # parts of days — windows match how people use the words, not the clock
    # date: "last night" includes the small hours of TODAY (a 2am session
    # belongs to last night), so it runs yesterday 17:00 → today 06:00
    day0, _ = _day(now)
    if t in ("last night", "yesterday evening", "last evening"):
        return day0 - timedelta(hours=7), day0 + timedelta(hours=6)
    if t in ("tonight", "this evening"):
        return day0 + timedelta(hours=17), day0 + timedelta(hours=30)
    if t == "this morning":
        return day0, day0 + timedelta(hours=12)
    if t == "this afternoon":
        return day0 + timedelta(hours=12), day0 + timedelta(hours=17)
    if t == "yesterday morning":
        return day0 - timedelta(hours=24), day0 - timedelta(hours=12)
    if t == "yesterday afternoon":
        return day0 - timedelta(hours=12), day0 - timedelta(hours=7)

    # sub-day windows — the phrasings people use for "what just happened".
    # All run UP TO now (+1s so the half-open range includes this instant).
    soon = now + timedelta(seconds=1)
    if t in ("earlier today", "earlier in the day", "today so far",
             "so far today", "so far", "today so far now"):
        return day0, soon
    if t in ("past hour", "last hour", "the past hour", "the last hour",
             "in the past hour", "this past hour", "within the last hour",
             "just now", "a moment ago", "recently", "right now"):
        return now - timedelta(hours=1), soon
    # "past 2 hours" / "last 30 minutes" — range ending now
    m = re.fullmatch(
        r"(?:in\s+)?(?:the\s+)?(?:last|past)\s+(\d+)\s+(hour|hr|minute|min)s?", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = timedelta(hours=n) if unit in ("hour", "hr") else timedelta(minutes=n)
        return now - delta, soon
    # "2 hours ago" / "30 minutes ago" — that single hour/minute window
    m = re.fullmatch(r"(?:an?\s+|(\d+)\s+)(hour|hr|minute|min)s?\s+ago", t)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        if m.group(2) in ("hour", "hr"):
            return now - timedelta(hours=n), now - timedelta(hours=n - 1)
        return now - timedelta(minutes=n), now - timedelta(minutes=n - 1)

    if t == "this week":
        start = _week_start(now)
        return start, start + timedelta(days=7)
    if t == "last week":
        start = _week_start(now) - timedelta(days=7)
        return start, start + timedelta(days=7)
    if t == "this month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nxt = (start + timedelta(days=32)).replace(day=1)
        return start, nxt
    if t == "last month":
        this_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start = (this_start - timedelta(days=1)).replace(day=1)
        return start, this_start

    # "last thursday" / bare weekday ("thursday" = most recent one, incl. today)
    m = re.fullmatch(r"(?:last\s+)?(" + "|".join(WEEKDAYS) + r")", t)
    if m:
        target = WEEKDAYS.index(m.group(1))
        delta = (now.weekday() - target) % 7
        if t.startswith("last") and delta == 0:
            delta = 7
        return _day(now - timedelta(days=delta))

    # "3 days ago", "2 weeks ago"
    m = re.fullmatch(r"(\d+)\s+(day|week)s?\s+ago", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * (7 if unit == "week" else 1)
        if unit == "week":
            start, _ = _day(now - timedelta(days=days))
            return start, start + timedelta(days=7)
        return _day(now - timedelta(days=days))

    # "last 3 days", "past 2 weeks" — range ending now
    m = re.fullmatch(r"(?:last|past)\s+(\d+)\s+(day|week)s?", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * (7 if unit == "week" else 1)
        start, _ = _day(now - timedelta(days=days - 1))
        return start, now + timedelta(seconds=1)

    # "june 5" / "june 5 2026"
    m = re.fullmatch(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:,?\s+(\d{4}))?", t)
    if m:
        month = MONTHS.index(m.group(1)) + 1
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else now.year
        d = datetime(year, month, day)
        if not m.group(3) and d > now:  # "june 5" said in january means last year's
            d = d.replace(year=year - 1)
        return _day(d)

    # ISO date or prefix: 2026, 2026-06, 2026-06-05
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", t)
    if m:
        year = int(m.group(1))
        if m.group(3):
            return _day(datetime(year, int(m.group(2)), int(m.group(3))))
        if m.group(2):
            start = datetime(year, int(m.group(2)), 1)
            nxt = (start + timedelta(days=32)).replace(day=1)
            return start, nxt
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)

    # full ISO timestamp
    try:
        d = datetime.fromisoformat(text.strip())
        return _day(d)
    except ValueError:
        pass

    raise ValueError(f"Don't understand time expression: {text!r}")
