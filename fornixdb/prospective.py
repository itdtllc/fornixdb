"""Prospective memory — remembering to remember (Einstein & McDaniel).

Retrospective memory answers "what happened?"; prospective memory carries an
INTENTION forward to the moment it matters: "remind me tomorrow morning to
talk to Joe about his vacation plans." Humans do this constantly and badly; a store with a clock
column can do it perfectly.

The design reuses everything the store already has. A reminder is an ordinary
memory row — kind=episodic, `event_time` = when it is due, so the existing
timeline answers "what's coming up tomorrow?" with no new query path — plus
one row in the `prospective` side-table carrying the delivery state. Hosts
poll `due()` at their natural heartbeat (each chat turn, an idle tick of a
voice loop, session start) and speak whatever comes back.

Two grades of intention (v0.8.6):

  normal — fire exactly once. `due()` marks `delivered_at` in the same call;
  after that the row is a plain episodic memory of the intention and decays
  like everything else.

  urgent — NAG until acknowledged. Delivery only increments `deliveries`;
  `due()` re-offers the reminder every `nag_interval_minutes` (config,
  default 5) up to `nag_max_attempts` (default 6), the way a person nags
  themself about medication but not about the mail. `delivered_at` here means
  ACKNOWLEDGED: the host calls `ack()` when the owner responds with anything
  at all after a delivery — the only observable evidence the reminder reached
  a person, and never something a model interprets. An urgent reminder that
  exhausts its attempts stops spamming the empty room but never silently
  dies: `unacknowledged()` surfaces it at the next session start and re-arms
  the nag cycle for the fresh chance at presence.

No scheduler lives here: FornixDB never runs threads on the host's behalf.
The host owns the clock; the store owns the intentions.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .core import MemoryStore
from .timeparse import parse_due

__all__ = ["remind", "due", "upcoming", "ack", "unacknowledged",
           "URGENT_SALIENCE", "DEFAULT_NAG_INTERVAL_MIN", "DEFAULT_NAG_MAX"]

URGENT_SALIENCE = 0.9          # an urgent intention stays sharp in recall
DEFAULT_NAG_INTERVAL_MIN = 5.0  # config: nag_interval_minutes
DEFAULT_NAG_MAX = 6             # config: nag_max_attempts (~30 min active)


def _nag_dials(store: MemoryStore) -> tuple[float, int]:
    from .multistore import get_config
    try:
        interval = float(get_config(store, "nag_interval_minutes",
                                    str(DEFAULT_NAG_INTERVAL_MIN)))
        cap = int(get_config(store, "nag_max_attempts", str(DEFAULT_NAG_MAX)))
    except (TypeError, ValueError):
        return DEFAULT_NAG_INTERVAL_MIN, DEFAULT_NAG_MAX
    return interval, cap


def remind(store: MemoryStore, what: str, when: str, *,
           urgent: bool = False,
           now: datetime | None = None,
           project: str | None = None,
           topics: list[str] | None = None,
           session_id: str | None = None,
           source: str = "prospective",
           detail: str | None = None) -> dict:
    """Store an intention to be surfaced at `when` (a natural phrase —
    "in 20 minutes", "tomorrow morning", "friday at 3pm" — or an ISO stamp).
    `urgent=True` makes it nag until acknowledged (and stores it at high
    salience). Returns {"id", "due", "gist", "urgent"}. Raises ValueError when
    the phrase isn't understood or names the past, so the caller can ask the
    owner to rephrase rather than silently mis-scheduling."""
    now = now or datetime.now()
    due_at = parse_due(when, now)
    gist = f"Reminder: {what.strip()}"
    due_iso = due_at.isoformat(timespec="seconds")
    # The prospective row rides INSIDE store()'s insert transaction: both rows
    # commit together or not at all. Two separate commits used to leave a dud
    # on a crash between them — a row that reads as a memory of the intention
    # but never fires.
    mid = store.store(
        gist, detail,
        kind="episodic",
        topics=list(dict.fromkeys(["reminder"] + (topics or []))),
        project=project,
        event_time=due_iso,
        session_id=session_id,
        salience=URGENT_SALIENCE if urgent else 0.5,
        source=source,
        _in_txn=lambda conn, mem_id: conn.execute(
            "INSERT INTO prospective (memory_id, due, urgent) VALUES (?, ?, ?)",
            (mem_id, due_iso, 1 if urgent else 0)),
    )
    return {"id": mid, "due": due_iso, "gist": gist, "urgent": urgent}


def _rows(store: MemoryStore, where: str, params: tuple) -> list[dict]:
    cur = store.conn.execute(
        "SELECT m.id, m.gist, m.detail, p.due, m.project, p.urgent, "
        "p.deliveries "
        "FROM prospective p JOIN memory m ON m.id = p.memory_id "
        f"WHERE p.delivered_at IS NULL AND m.superseded_time IS NULL AND {where} "
        "ORDER BY p.due", params)
    return [{"id": r[0], "gist": r[1], "detail": r[2], "due": r[3],
             "project": r[4], "urgent": bool(r[5]), "deliveries": r[6]}
            for r in cur.fetchall()]


def due(store: MemoryStore, now: datetime | None = None, *,
        deliver: bool = True) -> list[dict]:
    """Everything that should be said right now, oldest first. Normal
    reminders appear once (`deliver` marks them delivered in the same call —
    the caller is about to say them). Urgent reminders reappear every nag
    interval until `ack()` closes them, up to the attempt cap; each row's
    `deliveries` is the count INCLUDING this delivery ("third time now:").
    Pass deliver=False to peek without consuming or counting."""
    now = now or datetime.now()
    now_iso = now.isoformat(timespec="seconds")
    interval_min, cap = _nag_dials(store)
    renag_before = (now - timedelta(minutes=interval_min)
                    ).isoformat(timespec="seconds")
    where = ("p.due <= ? AND (p.urgent = 0 OR p.deliveries = 0 "
             "OR (p.deliveries < ? AND p.last_delivery <= ?))")
    params = (now_iso, cap, renag_before)
    if not deliver:
        return _rows(store, where, params)
    # SELECT and claim inside ONE write transaction: two hosts polling the
    # same store (a voice loop and a chat session) must not both pick up the
    # same reminder — the loser waits on the lock, re-selects, sees it claimed.
    with store.write_txn() as conn:
        rows = _rows(store, where, params)
        if rows:
            normal = [r["id"] for r in rows if not r["urgent"]]
            nagging = [r["id"] for r in rows if r["urgent"]]
            if normal:
                conn.executemany(
                    "UPDATE prospective SET delivered_at = ? WHERE memory_id = ?",
                    [(now_iso, i) for i in normal])
            if nagging:
                conn.executemany(
                    "UPDATE prospective SET deliveries = deliveries + 1, "
                    "last_delivery = ? WHERE memory_id = ?",
                    [(now_iso, i) for i in nagging])
    for r in rows:
        if r["urgent"]:
            r["deliveries"] += 1        # count includes this delivery
    return rows


def ack(store: MemoryStore, now: datetime | None = None) -> int:
    """The owner responded (any turn, any words) after urgent deliveries —
    the host's observable evidence the reminder reached them. Closes every
    urgent reminder that has been delivered at least once. Returns how many
    were acknowledged. Call on each owner turn BEFORE polling due(); it is a
    cheap no-op when nothing is nagging."""
    now_iso = (now or datetime.now()).isoformat(timespec="seconds")
    cur = store.conn.execute(
        "UPDATE prospective SET delivered_at = ? "
        "WHERE delivered_at IS NULL AND urgent = 1 AND deliveries > 0",
        (now_iso,))
    # Commit UNCONDITIONALLY: executing the UPDATE opens a write transaction
    # and takes the WAL write lock even when it matches 0 rows (the common
    # case — hosts call ack() every owner turn). Skipping the commit then
    # parks that lock on this connection forever, and every OTHER connection's
    # write dies with "database is locked" after the busy timeout (2026-07-11:
    # this took out Elira's camera/senses mid-demo once her host went to
    # per-thread connections; a shared singleton had masked it for months).
    store.conn.commit()
    return cur.rowcount


def unacknowledged(store: MemoryStore, now: datetime | None = None, *,
                   rearm: bool = True) -> list[dict]:
    """Urgent reminders that exhausted their nag attempts with no response —
    the ones that must NOT silently die. Meant for session start: report them
    ("still unacknowledged from 3pm: …") and, with `rearm` (the default),
    reset the nag cycle so the fresh session — a fresh chance the owner is
    present — nags again from attempt 1."""
    now = now or datetime.now()
    _, cap = _nag_dials(store)
    where = "p.urgent = 1 AND p.deliveries >= ? AND p.due <= ?"
    params = (cap, now.isoformat(timespec="seconds"))
    if not rearm:
        return _rows(store, where, params)
    # same select-then-claim shape as due(): serialize concurrent session
    # starts so only one host reports-and-rearms each exhausted reminder
    with store.write_txn() as conn:
        rows = _rows(store, where, params)
        if rows:
            conn.executemany(
                "UPDATE prospective SET deliveries = 0, last_delivery = NULL "
                "WHERE memory_id = ?", [(r["id"],) for r in rows])
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
