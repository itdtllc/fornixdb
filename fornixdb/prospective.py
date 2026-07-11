"""Prospective memory — remembering to remember (Einstein & McDaniel).

Retrospective memory answers "what happened?"; prospective memory carries an
INTENTION forward to the moment it matters: "remind me tomorrow morning to
call the attorney." Humans do this constantly and badly; a store with a clock
column can do it perfectly.

The design reuses everything the store already has. A reminder is an ordinary
memory row — kind=episodic, `event_time` = when it is due, so the existing
timeline answers "what's coming up tomorrow?" with no new query path — plus
one row in the `prospective` side-table carrying the delivery state. Hosts
poll `due()` at their natural heartbeat (each chat turn, an idle tick of a
voice loop, session start) and speak whatever comes back; delivery marks the
row delivered (a tombstone, not a delete), after which it lives on as a normal
episodic memory of the intention and decays like everything else.

No scheduler lives here: FornixDB never runs threads on the host's behalf.
The host owns the clock; the store owns the intentions.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .core import MemoryStore
from .timeparse import parse_due

__all__ = ["remind", "due", "upcoming"]


def remind(store: MemoryStore, what: str, when: str, *,
           now: datetime | None = None,
           project: str | None = None,
           topics: list[str] | None = None,
           session_id: str | None = None,
           source: str = "prospective",
           detail: str | None = None) -> dict:
    """Store an intention to be surfaced at `when` (a natural phrase —
    "in 20 minutes", "tomorrow morning", "friday at 3pm" — or an ISO stamp).
    Returns {"id", "due", "gist"}. Raises ValueError when the phrase isn't
    understood or names the past, so the caller can ask the owner to rephrase
    rather than silently mis-scheduling."""
    now = now or datetime.now()
    due_at = parse_due(when, now)
    gist = f"Reminder: {what.strip()}"
    mid = store.store(
        gist, detail,
        kind="episodic",
        topics=list(dict.fromkeys(["reminder"] + (topics or []))),
        project=project,
        event_time=due_at.isoformat(timespec="seconds"),
        session_id=session_id,
        source=source,
    )
    store.conn.execute(
        "INSERT INTO prospective (memory_id, due) VALUES (?, ?)",
        (mid, due_at.isoformat(timespec="seconds")))
    store.conn.commit()
    return {"id": mid, "due": due_at.isoformat(timespec="seconds"), "gist": gist}


def _rows(store: MemoryStore, where: str, params: tuple) -> list[dict]:
    cur = store.conn.execute(
        "SELECT m.id, m.gist, m.detail, p.due, m.project "
        "FROM prospective p JOIN memory m ON m.id = p.memory_id "
        f"WHERE p.delivered_at IS NULL AND m.superseded_time IS NULL AND {where} "
        "ORDER BY p.due", params)
    return [{"id": r[0], "gist": r[1], "detail": r[2], "due": r[3],
             "project": r[4]} for r in cur.fetchall()]


def due(store: MemoryStore, now: datetime | None = None, *,
        deliver: bool = True) -> list[dict]:
    """Everything that has come due and not yet been surfaced, oldest first.
    With `deliver` (the default) the rows are marked delivered in the same
    call — the caller is about to say them, and a reminder must fire exactly
    once. Pass deliver=False to peek without consuming."""
    now = now or datetime.now()
    now_iso = now.isoformat(timespec="seconds")
    rows = _rows(store, "p.due <= ?", (now_iso,))
    if deliver and rows:
        store.conn.executemany(
            "UPDATE prospective SET delivered_at = ? WHERE memory_id = ?",
            [(now_iso, r["id"]) for r in rows])
        store.conn.commit()
    return rows


def upcoming(store: MemoryStore, now: datetime | None = None, *,
             within_hours: float = 48.0) -> list[dict]:
    """Undelivered intentions still ahead of us, soonest first — for session
    startup ("two things coming up today") and "what's on my plate?" asks.
    Never marks anything delivered."""
    now = now or datetime.now()
    horizon = now + timedelta(hours=within_hours)
    return _rows(store, "p.due > ? AND p.due <= ?",
                 (now.isoformat(timespec="seconds"),
                  horizon.isoformat(timespec="seconds")))
