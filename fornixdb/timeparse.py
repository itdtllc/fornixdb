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


# Spoken-word numerals → digits, applied before any pattern matching in BOTH
# parsers. Voice hosts get their phrases from a transcriber, and Whisper
# writes small numbers as WORDS — "remind me in five minutes" reached
# parse_due exactly like that in the 2026-07-11 live demo and fell through to
# ValueError while typed "in 2 minutes" worked. Whole words only; compounds
# ("twenty-five", "twenty five") collapse first so the tens word isn't
# swallowed alone.
_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_SINGLES = {**_ONES, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19, **_TENS}
_COMPOUND_RE = re.compile(
    r"\b(" + "|".join(_TENS) + r")[-\s](" + "|".join(_ONES) + r")\b")
_SINGLE_RE = re.compile(r"\b(" + "|".join(_SINGLES) + r")\b")


def _digitize(t: str) -> str:
    t = _COMPOUND_RE.sub(
        lambda m: str(_TENS[m.group(1)] + _ONES[m.group(2)]), t)
    return _SINGLE_RE.sub(lambda m: str(_SINGLES[m.group(1)]), t)


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
    t = _digitize(text.strip().lower())

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
    # "the last minute" / "the past few minutes" — colloquial just-now, not a
    # literal 60s (asked live 2026-07-10 and it fell through to ValueError)
    if re.fullmatch(
            r"(?:in\s+|within\s+)?(?:the\s+)?(?:last|past)\s+"
            r"(?:minute|moment|(?:few|couple|several)(?:\s+of)?\s+(?:minutes|moments))",
            t):
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

    # future windows — the query side of prospective memory ("what's coming
    # up tomorrow?" over reminder rows, whose event_time is their due time)
    if t == "tomorrow":
        return _day(now + timedelta(days=1))
    if t == "tomorrow morning":
        return day0 + timedelta(hours=24), day0 + timedelta(hours=36)
    if t == "tomorrow afternoon":
        return day0 + timedelta(hours=36), day0 + timedelta(hours=41)
    if t in ("tomorrow evening", "tomorrow night"):
        return day0 + timedelta(hours=41), day0 + timedelta(hours=54)
    if t == "next week":
        start = _week_start(now) + timedelta(days=7)
        return start, start + timedelta(days=7)
    if t == "next month":
        this_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start = (this_start + timedelta(days=32)).replace(day=1)
        nxt = (start + timedelta(days=32)).replace(day=1)
        return start, nxt
    if t in ("upcoming", "coming up", "soon", "later", "the future"):
        return now, now + timedelta(days=7)

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


# ---- prospective time: when should a reminder come back? ---------------------

# Clock defaults for day-part words when a reminder names no time — chosen to
# match how people mean them ("tomorrow morning" ≈ start of the working day).
MORNING_H, AFTERNOON_H, EVENING_H = 9, 15, 19

_CLOCK = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", re.IGNORECASE)


def _parse_clock(s: str) -> tuple[int, int] | None:
    """'3pm' / '3:30 pm' / '15:00' / 'noon' / 'midnight' → (hour, minute)."""
    s = s.strip().lower()
    if s == "noon":
        return 12, 0
    if s == "midnight":
        return 0, 0
    m = _CLOCK.fullmatch(s)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    mer = (m.group(3) or "").replace(".", "")
    if mer == "pm" and hour != 12:
        hour += 12
    elif mer == "am" and hour == 12:
        hour = 0
    elif not mer and not m.group(2):
        return None    # bare "3" is ambiguous — require am/pm or a ':'
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def parse_due(text: str, now: datetime | None = None) -> datetime:
    """Parse a FUTURE natural time phrase into the single datetime it names —
    the prospective twin of `parse_when`. Understands "in 20 minutes",
    "tomorrow", "tomorrow morning", "tonight", "friday at 3pm", "at 9am",
    "next week", "june 5", and ISO stamps. A clock time already past today
    rolls forward ("at 9am" said at noon → tomorrow 9am): a reminder is
    always in the future. Raises ValueError for phrases it doesn't
    understand, or an explicit timestamp in the past.
    """
    now = now or datetime.now()
    t = _digitize(re.sub(r"\s+", " ", text.strip().lower()))
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # relative: "in 20 minutes", "in an hour", "in a few minutes",
    # "in half an hour" — never rolled, they're future by construction
    m = re.fullmatch(
        r"in (?:about )?(\d+|an?|a few|a couple(?: of)?|several|half an?) "
        r"(minute|min|hour|hr|day|week)s?", t)
    if m:
        word, unit = m.group(1), m.group(2)
        n = {"a": 1.0, "an": 1.0, "a few": 3.0, "several": 5.0,
             "a couple": 2.0, "a couple of": 2.0,
             "half an": 0.5, "half a": 0.5}.get(word)
        if n is None:
            n = float(word)
        per = {"minute": 60, "min": 60, "hour": 3600, "hr": 3600,
               "day": 86400, "week": 7 * 86400}[unit]
        return now + timedelta(seconds=n * per)

    # optional trailing clock: "<phrase> at 3pm" (or "<phrase> 3pm")
    when_part, clock = t, None
    m = re.fullmatch(r"(.*?)(?:\s+at)?\s+"
                     r"(noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)",
                     t)
    if m and m.group(1):
        maybe = _parse_clock(m.group(2))
        if maybe:
            when_part, clock = m.group(1).strip(), maybe

    def at(day: datetime, default_hour: int) -> datetime:
        h, mi = clock if clock else (default_hour, 0)
        return day.replace(hour=h, minute=mi, second=0, microsecond=0)

    when_part = re.sub(r"^(?:on|this)\s+", "", when_part)

    if when_part == "tomorrow":
        return at(day0 + timedelta(days=1), MORNING_H)
    if when_part == "tomorrow morning":
        return at(day0 + timedelta(days=1), MORNING_H)
    if when_part == "tomorrow afternoon":
        return at(day0 + timedelta(days=1), AFTERNOON_H)
    if when_part in ("tomorrow evening", "tomorrow night"):
        return at(day0 + timedelta(days=1), EVENING_H)
    if when_part in ("tonight", "evening"):
        due = at(day0, EVENING_H)
        return due if due > now else now + timedelta(hours=1)
    if when_part in ("morning", "afternoon"):
        due = at(day0, MORNING_H if when_part == "morning" else AFTERNOON_H)
        return due if due > now else due + timedelta(days=1)

    # "friday" / "next friday" [at 3pm] — the NEXT occurrence, never today
    m = re.fullmatch(r"(?:next )?(" + "|".join(WEEKDAYS) + r")", when_part)
    if m:
        ahead = (WEEKDAYS.index(m.group(1)) - now.weekday()) % 7 or 7
        return at(day0 + timedelta(days=ahead), MORNING_H)

    if when_part == "next week":
        return at(_week_start(now) + timedelta(days=7), MORNING_H)
    if when_part == "next month":
        this_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return at((this_start + timedelta(days=32)).replace(day=1), MORNING_H)

    # "june 5" [at 3pm] — the next such date
    m = re.fullmatch(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})", when_part)
    if m:
        d = datetime(now.year, MONTHS.index(m.group(1)) + 1, int(m.group(2)))
        due = at(d, MORNING_H)
        return due if due > now else due.replace(year=due.year + 1)

    # bare clock time: "at 9am" / "9am" / "15:30" — today, rolled if past
    if clock and when_part in ("", "at"):
        due = at(day0, MORNING_H)
        return due if due > now else due + timedelta(days=1)
    maybe = _parse_clock(t)
    if maybe:
        clock = maybe
        due = at(day0, MORNING_H)
        return due if due > now else due + timedelta(days=1)

    # explicit ISO date or timestamp — must actually be in the future
    try:
        d = datetime.fromisoformat(text.strip())
        if len(text.strip()) <= 10:               # bare date → default morning
            d = d.replace(hour=MORNING_H)
        if d <= now:
            raise ValueError(f"That time is in the past: {text!r}")
        return d
    except ValueError as e:
        if "in the past" in str(e):
            raise

    raise ValueError(f"Don't understand when {text!r} is — try 'in 20 minutes', "
                     "'tomorrow morning', 'friday at 3pm', or an exact time")
