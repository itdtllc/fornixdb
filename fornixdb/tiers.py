"""P3b retention tiers (Design §13.2) — mechanical, judgment-free, reversible.

hot          full detail in the memory row
consolidated detail zlib-compressed into detail_archive (row.detail = NULL)
cold         detail appended to <store>/<db-stem>.archive/YYYY-MM.jsonl.gz; pointer kept
             (per-store dir so stores sharing a directory never share a file)

Nothing is ever deleted (owner decision 2026-06-11): show() transparently
restores detail from either tier. Tier-down applies only to episodic rows at
the default thresholds; under disk pressure (max_store_mb config) thresholds
escalate and semantic/reference join — feedback never tiers down.
"""

from __future__ import annotations

import gzip
import json
import zlib
from datetime import datetime
from pathlib import Path

from .core import MemoryStore
from .db import _restrict_to_owner_path
from .multistore import get_config

# (tier, max_eff_salience, min_age_days) — episodic defaults
CONSOLIDATE_BELOW, CONSOLIDATE_AGE = 0.15, 60
COLD_BELOW, COLD_AGE = 0.08, 180

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detail_archive (
    memory_id   INTEGER PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE,
    compression TEXT NOT NULL,           -- 'zlib' | 'jsonl-gz'
    data        BLOB,                    -- zlib-compressed detail (consolidated)
    location    TEXT                     -- archive file path (cold)
);
"""


def _ensure(store: MemoryStore) -> None:
    store.conn.executescript(_SCHEMA)


def _db_path(store: MemoryStore) -> Path:
    return Path(store.conn.execute("PRAGMA database_list").fetchone()[2])


def _db_dir(store: MemoryStore) -> Path:
    return _db_path(store).parent


def archive_dir_for(db_path: str | Path) -> Path:
    """Cold-archive dir for a store, keyed to the db FILENAME, not just its
    directory — the single source of truth (budget.py footprint/prune use it too).

    Multiple stores often share one directory — e.g. different AIs keeping
    ~/.fornixdb/memory.db and ~/.fornixdb/artist.db. A shared `archive/` dir
    would put both stores' cold rows in the same YYYY-MM.jsonl.gz, and since
    memory_id is per-store autoincrement, ids collide and load_detail could
    return another store's detail. A per-store `<stem>.archive/` keeps each
    store's archive entirely its own."""
    p = Path(db_path)
    return p.parent / f"{p.stem}.archive"


def _archive_dir(store: MemoryStore) -> Path:
    return archive_dir_for(_db_path(store))


def load_detail(store: MemoryStore, mem: dict) -> str | None:
    """Restore detail for a tiered row — the 'give me a second' fetch."""
    _ensure(store)
    row = store.conn.execute(
        "SELECT * FROM detail_archive WHERE memory_id = ?", (mem["id"],)).fetchone()
    if row is None:
        return None
    if row["compression"] == "zlib":
        return zlib.decompress(row["data"]).decode("utf-8")
    with gzip.open(row["location"], "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["memory_id"] == mem["id"]:
                return d["detail"]
    return None


def _consolidate_row(store: MemoryStore, row) -> None:
    store.conn.execute(
        "INSERT OR REPLACE INTO detail_archive(memory_id, compression, data, location) "
        "VALUES (?,?,?,NULL)",
        (row["id"], "zlib", zlib.compress(row["detail"].encode("utf-8"), 9)))
    store.conn.execute(
        "UPDATE memory SET detail = NULL, retention_tier = 'consolidated' WHERE id = ?",
        (row["id"],))


def _cold_row(store: MemoryStore, row) -> None:
    detail = row["detail"]
    if detail is None:  # consolidated → cold: pull from zlib first
        detail = load_detail(store, dict(row))
    arc_dir = _archive_dir(store)
    new_dir = not arc_dir.exists()
    arc_dir.mkdir(parents=True, exist_ok=True)
    if new_dir:
        _restrict_to_owner_path(arc_dir, is_dir=True)
    month = (row["event_time"] or "0000-00")[:7]
    path = arc_dir / f"{month}.jsonl.gz"
    new_file = not path.exists()
    with gzip.open(path, "at", encoding="utf-8") as fh:
        fh.write(json.dumps({"memory_id": row["id"], "gist": row["gist"],
                             "detail": detail, "archived": datetime.now().isoformat()})
                 + "\n")
    if new_file:  # cold archives hold detail — same personal-data class as the db
        _restrict_to_owner_path(path)
    store.conn.execute(
        "INSERT OR REPLACE INTO detail_archive(memory_id, compression, data, location) "
        "VALUES (?,?,NULL,?)", (row["id"], "jsonl-gz", str(path)))
    store.conn.execute(
        "UPDATE memory SET detail = NULL, retention_tier = 'cold' WHERE id = ?",
        (row["id"],))


def _under_pressure(store: MemoryStore) -> bool:
    budget = get_config(store, "max_store_mb")
    if budget:
        db_path = Path(store.conn.execute("PRAGMA database_list").fetchone()[2])
        if db_path.stat().st_size / 1e6 > float(budget):
            return True
    # the hard total-footprint cap (§13.2) is also disk pressure
    from .budget import budget_bytes, footprint_bytes
    b = budget_bytes(store)
    return b is not None and footprint_bytes(store)["total"] > b


def tier_down(store: MemoryStore, dry_run: bool = False,
              force_pressure: bool = False) -> dict:
    """One mechanical pass. Returns counts; with dry_run, only counts.
    force_pressure: budget enforcement (§13.2) escalates thresholds directly."""
    _ensure(store)
    pressure = force_pressure or _under_pressure(store)
    kinds = ("episodic", "semantic", "reference") if pressure else ("episodic",)
    c_below, c_age = (CONSOLIDATE_BELOW * 2, CONSOLIDATE_AGE / 2) if pressure \
        else (CONSOLIDATE_BELOW, CONSOLIDATE_AGE)
    k_below, k_age = (COLD_BELOW * 2, COLD_AGE / 2) if pressure \
        else (COLD_BELOW, COLD_AGE)

    now = datetime.now()
    moved = {"consolidated": 0, "cold": 0, "pressure": pressure}
    ph = ",".join("?" * len(kinds))
    rows = store.conn.execute(
        f"SELECT * FROM memory WHERE kind IN ({ph}) AND detail IS NOT NULL "
        "AND retention_tier IN ('hot','consolidated')", list(kinds)).fetchall()
    # cold candidates include already-consolidated rows (detail is NULL there)
    rows += store.conn.execute(
        f"SELECT * FROM memory WHERE kind IN ({ph}) AND retention_tier = 'consolidated' "
        "AND detail IS NULL", list(kinds)).fetchall()

    for row in rows:
        try:
            age = (now - datetime.fromisoformat(row["event_time"])).days
        except (ValueError, TypeError):
            continue
        eff = store.effective_salience(dict(row), now)
        if row["retention_tier"] != "cold" and eff < k_below and age > k_age:
            if not dry_run:
                _cold_row(store, row)
            moved["cold"] += 1
        elif (row["retention_tier"] == "hot" and row["detail"]
              and eff < c_below and age > c_age):
            if not dry_run:
                _consolidate_row(store, row)
            moved["consolidated"] += 1
    if not dry_run:
        store.conn.commit()
    return moved


def tier_status(store: MemoryStore) -> dict:
    _ensure(store)
    return {r["retention_tier"]: r["c"] for r in store.conn.execute(
        "SELECT retention_tier, count(*) c FROM memory GROUP BY retention_tier")}
