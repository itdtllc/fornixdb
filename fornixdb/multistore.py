"""Multi-store recall: merge results across a primary store and read-only peers.

This is level two of the multi-AI topology (Design §12.10): each agent owns a
per-agent store, and every agent on the machine also reads a shared tier —
owner facts and preferences that should be known to all of them. An aggregator
across per-agent stores is the future third level; this module deliberately
stays simple (merge + re-rank) so that aggregator has something clean to grow
from.

Scores are comparable across stores because every store uses the same ranking
constants, so merging is concatenate → sort → cut. Each store marks its own
recalled rows, so reinforcement stays per-store. Rows gain a `_store` key
carrying the store alias for display and disambiguation (IDs collide across
stores; "shared#12" is a different memory than "#12").
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .core import MemoryStore
from .db import DEFAULT_SHARED_PATH, SHARED_ENV, shared_db_path  # noqa: F401
                                        # (canonical in db.py; re-exported)


def open_stores(primary: MemoryStore, *, shared: bool | str | os.PathLike = True,
                primary_alias: str = "") -> list[tuple[str, MemoryStore]]:
    """[(alias, store), ...] — primary first, then the shared tier if enabled.
    `shared` may be True (default path/env), a path, or False (primary only)."""
    stores = [(primary_alias, primary)]
    if shared:
        path = shared_db_path() if shared is True else Path(shared).expanduser()
        try:
            if Path(primary.conn.execute("PRAGMA database_list").fetchone()[2]
                    ).resolve() == path.resolve():
                return stores  # primary IS the shared store; don't double-query
        except Exception:
            pass
        stores.append(("shared", MemoryStore(db_path=path)))
    return stores


def _tag(rows: list[dict], alias: str) -> list[dict]:
    for r in rows:
        r["_store"] = alias
    return rows


def _dedupe_across_stores(rows: list[dict]) -> list[dict]:
    """Drop a row whose gist is identical (normalized) to a higher-ranked row
    from a DIFFERENT store — the same fact living in both an agent store and
    the shared tier (e.g. a migrated preference) should answer once, not
    twice. Deliberately conservative: exact text only, never similarity —
    near-duplicates are consolidation's propose-not-dispose territory, and
    same-store repeats are distinct events that both belong on the record.
    The kept row names its twin in `also_in`, so nothing is hidden silently."""
    seen: dict[str, dict] = {}
    out = []
    for r in rows:  # already sorted best-first; the best copy survives
        key = re.sub(r"\W+", " ", (r.get("gist") or "").lower()).strip()
        twin = seen.get(key)
        if twin is not None and twin.get("_store") != r.get("_store"):
            ref = f"{r['_store']}:{r['id']}" if r.get("_store") else str(r["id"])
            twin.setdefault("also_in", []).append(ref)
            continue
        if twin is None:
            seen[key] = r
        out.append(r)
    return out


def multi_recall(stores, query: str, *, limit: int = 10, **kw) -> list[dict]:
    merged: list[dict] = []
    for alias, store in stores:
        merged += _tag(store.recall(query, limit=limit, **kw), alias)
    merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return _dedupe_across_stores(merged)[:limit]


def multi_timeline(stores, start: str, end: str, *, limit: int = 50, **kw) -> list[dict]:
    merged: list[dict] = []
    for alias, store in stores:
        merged += _tag(store.timeline(start, end, limit=limit, **kw), alias)
    merged.sort(key=lambda r: r["event_time"])
    return merged[:limit]


def multi_brief(stores, **kw) -> dict:
    out = {"since": None, "recent": [], "salient": [], "useful": []}
    for alias, store in stores:
        b = store.brief(**kw)
        out["since"] = out["since"] or b["since"]
        out["recent"] += _tag(b["recent"], alias)
        out["salient"] += _tag(b["salient"], alias)
        out["useful"] += _tag(b.get("useful", []), alias)
    out["recent"].sort(key=lambda r: r["event_time"], reverse=True)
    out["salient"].sort(key=lambda r: r["salience"], reverse=True)
    out["useful"].sort(key=lambda r: (r["helpful_count"], r["recall_count"]),
                       reverse=True)
    return out


def resolve_ref(stores, ref: str):
    """Resolve 'shared:12' / '12' / 'name' to (store, ref-within-store)."""
    if ":" in ref:
        alias, _, inner = ref.partition(":")
        for a, store in stores:
            if a == alias:
                return store, inner
    return stores[0][1], ref


# --------------------------------------------------- capture-mode setting

CAPTURE_MODES = ("explicit", "suggest", "auto")
CAPTURE_MODE_HELP = {
    "explicit": "remember only when the owner asks",
    "suggest": "offer to remember at natural checkpoints; store only on a yes",
    "auto": "store at the AI's own judgment; owner reviews/retires later",
}


def get_config(store: MemoryStore, key: str, default: str | None = None) -> str | None:
    row = store.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_config(store: MemoryStore, key: str, value: str) -> None:
    if key == "capture_mode" and value not in CAPTURE_MODES:
        raise ValueError(f"capture_mode must be one of {CAPTURE_MODES}")
    if key in ("budget_policy", "machine_budget_policy"):
        from .budget import POLICIES
        if value not in POLICIES:
            raise ValueError(f"{key} must be one of {POLICIES}")
    if key in ("disk_budget_mb", "machine_budget_mb") and value not in ("off", "none"):
        if float(value) <= 0:  # raises ValueError on non-numbers too
            raise ValueError(f"{key} must be a positive number (MB), or 'off'")
    if key == "frozen":
        value = "1" if value in ("1", "on", "true", "yes") else "0"
    if key == "machine_budget_mb":  # owner touched the cap = the review
        store.conn.execute("DELETE FROM meta WHERE key = 'machine_budget_defaulted'")
    if key in ("disk_budget_mb", "machine_budget_mb") and value in ("off", "none"):
        store.conn.execute("DELETE FROM meta WHERE key = ?", (key,))
    else:
        store.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
    store.conn.commit()
    # settings are cached per store instance — drop so changes apply immediately
    store.__dict__.pop("_frozen_cache", None)
    store.__dict__.pop("_decay_cache", None)


def capture_mode(store: MemoryStore) -> str:
    return get_config(store, "capture_mode", "suggest") or "suggest"
