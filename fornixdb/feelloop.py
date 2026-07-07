"""The feel-loop core — a stream of sensor readings in, sparse `feel` memories out.

The proprioception twin of `watchloop`. Where vision samples dense frames and
an embedding-distance gate finds scene changes, machine proprioception samples a
small dict of named values (power source, charge, thermal, lid) and the moments
worth remembering are STATE CHANGES: the laptop went on battery, charge crossed
a bucket, it stopped charging. So the gate here is not cosine distance but plain
field-diff — commit the first reading, commit whenever a watched field changes,
and commit a heartbeat after a quiet stretch so the timeline keeps proof of the
steady state ("still on battery, still 40%").

Like `watchloop` this half is pure and injectable: no sensor, no clock, no sleep
lives here — an adapter owns sampling and yields `(timestamp, reading)` pairs,
and this module owns which readings become memories. No embedder, no model,
stdlib only: `feel()`'s gist lane alone makes a reading recallable ("when did
the laptop go on battery?"). Readings that never change never touch disk.

Timestamps from the adapter are monotonic seconds; `start_wall` anchors them to
calendar time for each memory's `event_time`. A reading is a discrete state
transition, so memories are point events (no `event_time_end`) — the timeline
reconstructs how long a state held from the gap to the next row.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from . import senses

__all__ = ["FeelEvent", "run_feel"]

_UNSET = object()  # "no reading committed yet" — distinct from any real value


@dataclass(frozen=True)
class FeelEvent:
    memory_id: int
    reason: str        # "first" | "change" | "heartbeat"
    t: float           # adapter (monotonic) seconds of the committed reading
    reading: object    # the reading committed (dict or str)
    gist: str


def _hashable(v):
    """A comparable, order-stable form of a value so two readings can be
    diffed. Nested dicts/lists become sorted tuples; scalars pass through."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _signature(reading, ignore: set):
    """The part of a reading that field-change gating watches: a dict minus
    the ignored (noisy) fields, or a plain string as-is."""
    if isinstance(reading, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in reading.items()
                            if k not in ignore))
    return str(reading)


def run_feel(store, readings, *, sensor: str,
             ignore_fields=(),
             heartbeat_seconds: float = 600.0,
             start_wall: datetime | None = None,
             max_seconds: float | None = None,
             max_commits: int | None = None,
             topics: list[str] | None = None,
             project: str | None = None,
             session_id: str | None = None,
             on_commit=None) -> list["FeelEvent"]:
    """Drive one proprioception session over an adapter's reading iterator.

    `readings` yields `(t, reading)` with `t` in monotonic seconds and reading
    a dict of named values (or a plain string). Commits the first reading, then
    any reading whose watched fields differ from the last committed one, then a
    heartbeat once `heartbeat_seconds` of no change have passed. `ignore_fields`
    names dict keys excluded from change detection but still recorded (e.g. a
    minute-to-minute `percent`/`remaining` you don't want a memory per tick).
    Each commit becomes a `feel` memory (gist lane only, no embedder). Returns
    the committed events in order; stops on iterator exhaustion, `max_seconds`
    of stream time, or `max_commits`.
    """
    ignore = set(ignore_fields)
    wall0 = start_wall or datetime.now()
    events: list[FeelEvent] = []
    t_first: float | None = None
    last_sig = _UNSET
    last_commit: float | None = None

    def _iso(t: float) -> str:
        return (wall0 + timedelta(seconds=t - t_first)).isoformat(
            timespec="seconds")

    for t, reading in readings:
        if t_first is None:
            t_first = t
        if max_seconds is not None and t - t_first > max_seconds:
            break

        sig = _signature(reading, ignore)
        if last_sig is _UNSET:
            reason = "first"
        elif sig != last_sig:
            reason = "change"
        elif (heartbeat_seconds and last_commit is not None
              and t - last_commit >= heartbeat_seconds):
            reason = "heartbeat"
        else:
            continue

        gist = senses.feel_gist(sensor, reading)
        mid = senses.feel(store, reading, sensor=sensor, gist=gist,
                          event_time=_iso(t), topics=topics,
                          project=project, session_id=session_id)
        ev = FeelEvent(mid, reason, t, reading, gist)
        events.append(ev)
        if on_commit is not None:
            on_commit(ev)
        last_sig = sig
        last_commit = t
        if max_commits is not None and len(events) >= max_commits:
            break
    return events
