"""Disk budget + boundary policy (Design §13.2, FornixDB #136).

Never-delete remains the default: with no `disk_budget_mb` set, nothing in
this module ever runs. A user-set budget caps the store's TOTAL on-disk
footprint — db file + WAL/SHM + cold archives (embeddings live inside the db)
— sized to the device: 1 TB on a workstation, MBs on a microcontroller.

When the cap is hit, mechanical tier escalation runs first (compress before
any forgetting). If that can no longer fit the budget, the per-store boundary
policy decides:

  freeze (default) — the store stops accepting new memories; everything
                     already stored stays fully recallable.
  prune            — true deletion of the lowest-effective-salience memories,
                     the way a person forgets old unused things to make room
                     for new ones. Choosing prune is the owner's explicit
                     consent to forgetting; it is the ONLY true delete in
                     FornixDB.

Prune order: tombstoned rows first, then live episodic, then semantic and
reference, feedback last (owner rules are load-bearing) — by effective
salience within each class, oldest first on ties.

`frozen` is also a standalone per-store setting independent of any cap
(`config frozen on`): a vendor may ship a curated read-only memory DB. It is
policy, not security — a vendor needing hard guarantees also ships the file
without write permission.

Everything here is algorithmic (effective salience + file sizes); no vector
model is required (§6.3 — cap enforcement works on AI-less endpoints).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from .core import DiskBudgetExceededError, MemoryStore
from .multistore import get_config
from .tiers import archive_dir_for, tier_down

POLICIES = ("freeze", "prune")
DEFAULT_POLICY = "freeze"     # prune requires the owner's explicit consent
PRUNE_HEADROOM = 0.9          # prune to 90% of budget so the boundary doesn't thrash
PRUNE_BATCH = 25

# prune priority classes (lower = forgotten first)
_KIND_CLASS = {"episodic": 1, "semantic": 2, "reference": 2, "feedback": 3}


def _store_files(store: MemoryStore) -> tuple[Path, list[Path]]:
    db = Path(store.conn.execute("PRAGMA database_list").fetchone()[2])
    files = [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")]
    files += sorted(archive_dir_for(db).glob("*.jsonl.gz"))
    return db, files


def footprint_bytes(store: MemoryStore) -> dict:
    """Total on-disk footprint: db + WAL/SHM + cold archive files."""
    db, files = _store_files(store)
    out = {"db": 0, "wal": 0, "archive": 0}
    for f in files:
        if not f.exists():
            continue
        size = f.stat().st_size
        if f == db:
            out["db"] = size
        elif f.suffix == ".gz":
            out["archive"] += size
        else:
            out["wal"] += size
    out["total"] = out["db"] + out["wal"] + out["archive"]
    return out


def budget_bytes(store: MemoryStore) -> float | None:
    raw = get_config(store, "disk_budget_mb")
    try:
        mb = float(raw)
        return mb * 1e6 if mb > 0 else None
    except (TypeError, ValueError):
        return None


def policy(store: MemoryStore) -> str:
    val = get_config(store, "budget_policy", DEFAULT_POLICY)
    return val if val in POLICIES else DEFAULT_POLICY


def status(store: MemoryStore) -> dict:
    fp = footprint_bytes(store)
    b = budget_bytes(store)
    return {
        "footprint_mb": {k: round(v / 1e6, 3) for k, v in fp.items()},
        "budget_mb": round(b / 1e6, 3) if b else None,
        "policy": policy(store),
        "frozen": store.frozen(),
        "over_budget": bool(b and fp["total"] > b),
    }


def _path_footprint(db: Path) -> float:
    """On-disk MB for one store by path alone (no db open, no migration)."""
    total = 0
    for f in [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm"),
              *sorted(archive_dir_for(db).glob("*.jsonl.gz"))]:
        if f.exists():
            total += f.stat().st_size
    return round(total / 1e6, 3)


def machine_usage() -> dict:
    """Every FornixDB store on this machine (the registry each store joins on
    open, plus the shared tier): label, size, memory count — and the total.
    Labels come from each store's `store_label` config; sizes are file stats,
    counts via a read-only peek that never migrates or locks a live store.
    Registry entries whose files are gone are pruned."""
    import json as _json
    import sqlite3
    from .db import registry_path
    from .multistore import shared_db_path

    reg = registry_path()
    paths: list[str] = []
    if reg and reg.exists():
        try:
            paths = _json.loads(reg.read_text() or "[]")
        except Exception:
            paths = []
    sp = str(shared_db_path().resolve()) if shared_db_path().exists() else None
    if sp and sp not in paths:
        paths.append(sp)

    stores, total = [], 0.0
    for p in sorted(set(paths)):
        db = Path(p)
        if not db.exists():
            continue  # moved/deleted → dropped from the registry below
        entry = {"path": p, "mb": _path_footprint(db),
                 "label": "shared tier" if p == sp else db.stem,
                 "memories": None}
        try:
            ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            row = ro.execute(
                "SELECT value FROM meta WHERE key = 'store_label'").fetchone()
            if row and p != sp:
                entry["label"] = row[0]
            entry["memories"] = ro.execute(
                "SELECT count(*) FROM memory").fetchone()[0]
            ro.close()
        except Exception:
            pass  # locked or foreign file: sizes still count
        total += entry["mb"]
        stores.append(entry)

    if reg and reg.exists():
        try:  # prune entries whose files are gone (keep the registry tidy)
            current = _json.loads(reg.read_text() or "[]")
            kept = [p for p in current if Path(p).exists()]
            if kept != current:
                reg.write_text(_json.dumps(sorted(kept), indent=1))
        except Exception:
            pass
    cap, pol, defaulted = machine_budget()
    return {"stores": stores, "total_mb": round(total, 3),
            "machine_budget_mb": round(cap / 1e6, 3) if cap else None,
            "machine_policy": pol, "machine_budget_defaulted": defaulted,
            "over_budget": bool(cap and total * 1e6 > cap)}


def machine_budget() -> tuple[float | None, str, bool]:
    """The machine-wide cap across ALL stores (set at install by default, or
    by the owner: `config machine_budget_mb <MB> --shared`; policy likewise).
    Read via a read-only peek so any store can check it without opening the
    shared tier for writing. Returns (cap_bytes|None, policy, defaulted) —
    `defaulted` means the install default has not been reviewed yet."""
    import sqlite3
    from .db import shared_db_path
    p = shared_db_path()
    if not p.exists():
        return None, DEFAULT_POLICY, False
    try:
        ro = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        rows = dict(ro.execute(
            "SELECT key, value FROM meta WHERE key IN ('machine_budget_mb', "
            "'machine_budget_policy', 'machine_budget_defaulted')"))
        ro.close()
    except Exception:
        return None, DEFAULT_POLICY, False
    try:
        mb = float(rows.get("machine_budget_mb", ""))
    except ValueError:
        mb = 0
    pol = rows.get("machine_budget_policy", DEFAULT_POLICY)
    return (mb * 1e6 if mb > 0 else None,
            pol if pol in POLICIES else DEFAULT_POLICY,
            "machine_budget_defaulted" in rows)


def _enforce_machine_cap(store: MemoryStore) -> None:
    """Hold the MACHINE-WIDE cap at write time. The writing store fixes what
    it can — compress itself, then (policy prune) forget its own least-salient
    memories toward its share of the overshoot. A store never deletes another
    AI's memories; if its own best effort can't bring the machine under the
    cap, the write is refused with the per-store breakdown so the owner can
    shrink the right store."""
    cap, pol, _ = machine_budget()
    if cap is None:
        return
    total = machine_usage()["total_mb"] * 1e6
    if total <= cap:
        return
    tier_down(store, force_pressure=True)
    _vacuum(store)
    if pol == "prune":
        # aim the MACHINE total at the usual anti-thrash headroom below the
        # cap; this store sheds the difference (its own memories only)
        needed = machine_usage()["total_mb"] * 1e6 - cap * PRUNE_HEADROOM
        if needed > 0:
            own = footprint_bytes(store)["total"]
            _prune(store, max(own - needed, 0), False)
    u = machine_usage()
    if u["total_mb"] * 1e6 > cap:
        per = "; ".join(f"{s['label']} {s['mb']} MB" for s in u["stores"])
        raise DiskBudgetExceededError(
            f"all FornixDB stores together are at the machine-wide cap "
            f"({round(cap / 1e6, 3)} MB; current {u['total_mb']} MB: {per}) and "
            "this store cannot fix that alone. Raise machine_budget_mb "
            "(--shared), or shrink the larger stores (`budget shrink`).")


def make_room(store: MemoryStore) -> None:
    """Called before accepting a new memory. No budget or under budget: free.
    Over budget: tier-escalate, then apply the boundary policy — prune makes
    room for the newcomer; freeze refuses it (DiskBudgetExceededError).
    The per-store cap is checked first, then the machine-wide cap."""
    b = budget_bytes(store)
    if b is not None and footprint_bytes(store)["total"] > b:
        result = enforce(store)
        if result["over_after"] and policy(store) == "freeze":
            raise DiskBudgetExceededError(
                f"store is at its disk budget ({result['budget_mb']} MB) with policy "
                "'freeze' — not accepting new memories. Raise disk_budget_mb, or set "
                "budget_policy to 'prune' to forget low-salience memories instead.")
    _enforce_machine_cap(store)


def enforce(store: MemoryStore, dry_run: bool = False) -> dict:
    """One enforcement pass: tier escalation → vacuum → boundary policy."""
    b = budget_bytes(store)
    fp = footprint_bytes(store)
    out = {"budget_mb": round(b / 1e6, 3) if b else None,
           "policy": policy(store),
           "before_mb": round(fp["total"] / 1e6, 3),
           "tiered": None, "pruned": None, "dry_run": dry_run}
    if b is None or fp["total"] <= b:
        out["over_after"] = False
        return out

    # 1) compress before any forgetting
    out["tiered"] = tier_down(store, dry_run=dry_run, force_pressure=True)
    if not dry_run:
        _vacuum(store)

    # 2) boundary policy
    if footprint_bytes(store)["total"] > b and policy(store) == "prune":
        out["pruned"] = _prune(store, b * PRUNE_HEADROOM, dry_run)

    total = footprint_bytes(store)["total"]
    out["after_mb"] = round(total / 1e6, 3)
    out["over_after"] = total > b
    return out


def shrink(store: MemoryStore, target_mb: float, dry_run: bool = False) -> dict:
    """One-shot shrink-to-target — the owner's "reduce this space to X MB"
    (FornixDB #164). Distinct from the standing cap: runs once against the
    named target and leaves `disk_budget_mb` and `budget_policy` untouched.
    The command itself is the owner's explicit consent to true deletion (the
    same consent `budget_policy prune` encodes), so the policy setting is not
    consulted: tier escalation first (compress before any forgetting), then
    prune straight to the target — no headroom, the target IS the ask — then
    vacuum. If even deleting everything cannot reach the target (the db file
    has a floor), `reached` is False and `after_mb` tells the truth."""
    target = float(target_mb) * 1e6
    if target <= 0:
        raise ValueError("shrink target must be a positive number of MB")
    store._check_writable()  # a frozen store refuses content mutation
    fp = footprint_bytes(store)["total"]
    out = {"target_mb": round(target / 1e6, 3),
           "before_mb": round(fp / 1e6, 3),
           "tiered": None, "pruned": None, "dry_run": dry_run}
    if fp <= target:
        out.update(after_mb=out["before_mb"], reached=True)
        return out

    out["tiered"] = tier_down(store, dry_run=dry_run, force_pressure=True)
    if not dry_run:
        _vacuum(store)
    if footprint_bytes(store)["total"] > target:
        out["pruned"] = _prune(store, target, dry_run)

    total = footprint_bytes(store)["total"]
    out["after_mb"] = round(total / 1e6, 3)
    out["reached"] = total <= target if not dry_run else None
    return out


def prune_candidates(store: MemoryStore) -> list[dict]:
    """All memories in forget-first order. Tombstoned rows go before any live
    memory; feedback goes last of all (within class: lowest effective salience,
    then oldest)."""
    rows = [dict(r) for r in store.conn.execute("SELECT * FROM memory")]
    for r in rows:
        r["_class"] = (0 if r["superseded_time"]
                       else _KIND_CLASS.get(r["kind"], 2))
        r["_eff"] = store.effective_salience(r)
    rows.sort(key=lambda r: (r["_class"], r["_eff"], r["event_time"] or ""))
    return rows


def _prune(store: MemoryStore, target_bytes: float, dry_run: bool) -> dict:
    cands = prune_candidates(store)
    if dry_run:
        return {"candidates": len(cands),
                "first": [c["id"] for c in cands[:10]], "deleted": 0}
    deleted: list[int] = []
    while cands and footprint_bytes(store)["total"] > target_bytes:
        batch = [c["id"] for c in cands[:PRUNE_BATCH]]
        cands = cands[PRUNE_BATCH:]
        ph = ",".join("?" * len(batch))
        # one transaction per batch: the archive-path lookup and the deletes
        # see one consistent state (VACUUM stays outside — it can't run in a
        # transaction)
        with store.write_txn() as conn:
            # cold-archive files that hold detail for rows about to go away
            arc_paths = {r["location"] for r in conn.execute(
                f"SELECT location FROM detail_archive WHERE memory_id IN ({ph}) "
                "AND location IS NOT NULL", batch)}
            conn.execute(
                f"UPDATE memory SET superseded_by = NULL WHERE superseded_by IN ({ph})",
                batch)
            conn.execute(f"DELETE FROM memory WHERE id IN ({ph})", batch)
        deleted += batch
        for path in arc_paths:
            _compact_archive(store, Path(path))
        _vacuum(store)
    return {"deleted": len(deleted), "ids": deleted,
            "exhausted": not cands and footprint_bytes(store)["total"] > target_bytes}


def _compact_archive(store: MemoryStore, path: Path) -> None:
    """Rewrite one cold-archive file keeping only entries whose memory still
    exists — pruning must reclaim archive bytes too."""
    if not path.exists():
        return
    keep = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            mid = json.loads(line)["memory_id"]
            if store.conn.execute("SELECT 1 FROM memory WHERE id = ?",
                                  (mid,)).fetchone():
                keep.append(line)
    if keep:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.writelines(keep)
    else:
        path.unlink()


def _vacuum(store: MemoryStore) -> None:
    store.conn.commit()
    store.conn.execute("VACUUM")
    store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
