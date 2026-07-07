"""macOS proprioception adapter — a Mac laptop's own power/battery state as
`feel()` readings. A reference host adapter for the feel-loop (`fornixdb.feelloop`).

Generic to any Mac laptop: it encodes the *shape* of `pmset` output, never any
one machine's state. The live values it reports (your charge, whether you are on
mains) exist only in the caller's store — nothing machine-specific is baked in
here, so this file is safe to share as an example other Mac users can fork.

Two layers, mirroring the adapter pattern elsewhere in this package:

  parse_batt(text) -> dict   PURE — parse a captured `pmset -g batt` string.
                             Testable with no subprocess.
  read_battery()   -> dict   thin wrapper that shells out to `pmset` and parses.
  battery_frames(...)        a `(monotonic_ts, reading)` generator for run_feel,
                             coarsened so field-change gating fires on real
                             transitions, not minute-to-minute drift.

`pmset -g batt` prints a source line and a battery line, e.g.

    Now drawing from 'AC Power'
     -InternalBattery-0 (id=...)\t80%; AC attached; not charging present: true

or, unplugged:

    Now drawing from 'Battery Power'
     -InternalBattery-0 (id=...)\t75%; discharging; 3:42 remaining present: true

The transition-worthy fields are `source` (AC vs battery) and `state`
(charging / discharging / charged / not charging); `percent` drifts continuously
and `remaining` is minute-level noise, so `battery_frames` buckets the percent
(default 10% steps) and drops `remaining`. Pass the raw dict straight to
`run_feel` with `ignore_fields={"percent", "remaining"}` instead if you prefer.
"""
from __future__ import annotations

import re
import subprocess
import time
from typing import Callable, Iterator

__all__ = ["parse_batt", "read_battery", "battery_frames"]

_SOURCE_RE = re.compile(r"Now drawing from '([^']+)'")
_PERCENT_RE = re.compile(r"(\d+)%")
_REMAINING_RE = re.compile(r"(\d+):(\d\d)\s+remaining")

# Canonical charge verbs, longest/most-specific first so a substring match
# ("charging" inside "discharging") never wins over the real word.
_CHARGE_WORDS = ("finishing charge", "not charging", "discharging",
                 "charging", "charged")


def parse_batt(text: str) -> dict:
    """Parse `pmset -g batt` output into a reading. Pure — no subprocess.

    Returns keys source ("AC" | "battery" | None), percent (int | None),
    state (a canonical charge word | None), remaining ("H:MM" | None; None for
    a 0:00 / no-estimate reading). Missing fields come back None rather than
    raising, so a truncated or unusual `pmset` layout degrades gracefully.
    """
    reading: dict = {"source": None, "percent": None,
                     "state": None, "remaining": None}

    m = _SOURCE_RE.search(text)
    if m:
        reading["source"] = "AC" if "AC" in m.group(1) else "battery"

    m = _PERCENT_RE.search(text)
    if m:
        reading["percent"] = int(m.group(1))

    low = text.lower()
    for word in _CHARGE_WORDS:
        if word in low:
            reading["state"] = word
            break

    m = _REMAINING_RE.search(text)
    if m and (int(m.group(1)) or int(m.group(2))):   # skip 0:00 (charged/idle)
        reading["remaining"] = f"{int(m.group(1))}:{m.group(2)}"

    return reading


def read_battery(*, timeout: float = 5.0) -> dict:
    """Run `pmset -g batt` and parse it. The thin subprocess wrapper around
    `parse_batt`; raises if `pmset` is missing (non-Mac) or times out."""
    out = subprocess.run(["pmset", "-g", "batt"],
                         capture_output=True, text=True,
                         timeout=timeout).stdout
    return parse_batt(out)


def _coarsen(reading: dict, percent_step: int) -> dict:
    """A gating-friendly view: bucket `percent`, drop `remaining`, so
    field-change gating fires on source/state transitions and ~step-sized
    charge moves rather than every minute."""
    out = {"source": reading.get("source"), "state": reading.get("state")}
    p = reading.get("percent")
    if isinstance(p, int):
        out["percent"] = (p // percent_step) * percent_step if percent_step > 1 else p
    return out


def battery_frames(*, interval_seconds: float = 60.0,
                   percent_step: int = 10,
                   count: int | None = None,
                   reader: Callable[[], dict] = read_battery,
                   clock: Callable[[], float] = time.monotonic,
                   sleep: Callable[[float], None] = time.sleep,
                   ) -> Iterator[tuple[float, dict]]:
    """Yield `(monotonic_ts, reading)` for `run_feel`, sampling every
    `interval_seconds`. Readings are coarsened (`percent` bucketed to
    `percent_step`, `remaining` dropped); pass `percent_step=1` to keep every
    point. Runs forever unless `count` is given. `reader`/`clock`/`sleep` are
    injectable so the loop is testable without a real Mac or real waiting."""
    n = 0
    while True:
        yield clock(), _coarsen(reader(), percent_step)
        n += 1
        if count is not None and n >= count:
            return
        sleep(interval_seconds)
