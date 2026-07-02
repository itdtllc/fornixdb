"""The operating-levels ladder (L0–L6) as a configurable surface.

ROADMAP.md describes FornixDB's direction as a climb along one axis: *how
tightly, how often, and how much in parallel memory is fused into thinking*.
This module is that ladder turned into a setting — one place to see which rung
a store is on and to move it up or down — so the config tool mirrors the
ROADMAP table instead of making the owner set three unrelated dials by hand.

The ladder is **cumulative** (ROADMAP: L2 is "the foothold for everything above
it"): a rung is on only if every rung below it is on. So selecting a rung
enables it and everything beneath, and disabling a rung disables everything
above. The set/toggle helpers enforce that; reading reports the raw dial state
and flags any incoherent combination a hand-edited `config` may have left.

Each rung is a *view* over dials that already exist — this module owns no new
persisted state of its own:

    L0  Explicit store / retrieve        the keyed store floor — always on
    L1  Associative recall on demand     associative_recall  (on/off)
    L2  Automatic capture                capture_mode  (suggest/auto = on)
    L3  Proactive recall injection       proactive_recall  (on/off)
    L4  Rhythmic in-thought recall        rhythmic_recall   (on/off)
    L5  Parallel multi-domain activation  parallel_recall   (on/off, ships off)
    L6  Federated / distributed memory    not built — planned
"""

from __future__ import annotations

from dataclasses import dataclass

from .multistore import MemoryStore, capture_mode, get_config, set_config

_OFF = ("off", "0", "false", "no")  # mirrors doctor._OFF

# rung build-status (the ROADMAP "Status" column, distilled)
BUILT = "built"        # shipped, lived-in
DOGFOOD = "dogfood"    # built, under evaluation (not yet a published default)
PLANNED = "planned"    # not built


@dataclass(frozen=True)
class Level:
    id: str          # "L0".."L6"
    name: str
    coupling: str    # one line: how memory couples to cognition at this rung
    status: str      # BUILT | DOGFOOD | PLANNED
    locked_on: bool  # L0/L1 — capability always present; cannot be turned off
    dial: str | None  # config key this rung toggles, or None (locked/planned)
    dial_default: str = "on"  # what an unset dial means (dogfood rungs ship off)


# Single source of truth — ordered floor → top. Keep in step with ROADMAP.md.
LEVELS: tuple[Level, ...] = (
    Level("L0", "Explicit store / retrieve",
          "passive keyed put/get; memory does nothing on its own",
          BUILT, locked_on=True, dial=None),
    Level("L1", "Associative recall on demand",
          "pull-based but relevance-ranked recall (vectors + text + time)",
          BUILT, locked_on=False, dial="associative_recall"),
    Level("L2", "Automatic capture",
          "the write side runs itself — experience captured after each session",
          BUILT, locked_on=False, dial="capture_mode"),
    Level("L3", "Proactive recall injection",
          "memory pushes relevant context in unasked, once per turn",
          BUILT, locked_on=False, dial="proactive_recall"),
    Level("L4", "Rhythmic in-thought recall",
          "memory re-activates many times within one reasoning episode",
          BUILT, locked_on=False, dial="rhythmic_recall"),
    Level("L5", "Parallel multi-domain activation",
          "many domain-scoped recalls fire at once and settle into a direction",
          DOGFOOD, locked_on=False, dial="parallel_recall", dial_default="off"),
    Level("L6", "Federated / distributed memory",
          "the parallel model federated across endpoints and machines",
          PLANNED, locked_on=False, dial=None),
)

_BY_ID = {lv.id: lv for lv in LEVELS}
_INDEX = {lv.id: i for i, lv in enumerate(LEVELS)}


def level(level_id: str) -> Level:
    try:
        return _BY_ID[level_id.upper()]
    except KeyError:
        raise ValueError(
            f"unknown level {level_id!r}; expected one of "
            f"{', '.join(lv.id for lv in LEVELS)}")


# --------------------------------------------------------------- dial bridge
# A rung is "on" iff its underlying dial says so. Locked rungs are always on;
# planned rungs are always off (nothing to drive them yet).

def is_on(store: MemoryStore, level_id: str) -> bool:
    lv = level(level_id)
    if lv.locked_on:
        return True
    if lv.status == PLANNED or lv.dial is None:
        return False
    if lv.dial == "capture_mode":
        return capture_mode(store) in ("suggest", "auto")
    val = (get_config(store, lv.dial, lv.dial_default)
           or lv.dial_default).strip().lower()
    return val not in _OFF


def _set_one(store: MemoryStore, level_id: str, on: bool) -> None:
    """Write a single rung's dial. No cascade — callers enforce cumulativity."""
    lv = level(level_id)
    if lv.locked_on:
        if not on:
            raise ValueError(f"{lv.id} ({lv.name}) is always on and cannot be "
                             f"turned off — it is the floor of the ladder")
        return  # already on, nothing to write
    if lv.status == PLANNED:
        raise ValueError(f"{lv.id} ({lv.name}) is not built yet (planned) — "
                         f"it cannot be turned on")
    if lv.dial == "capture_mode":
        if on:
            # don't clobber a richer choice (auto) — only lift an off state
            if capture_mode(store) not in ("suggest", "auto"):
                set_config(store, "capture_mode", "suggest")
        else:
            set_config(store, "capture_mode", "explicit")
        return
    set_config(store, lv.dial, "on" if on else "off")


# --------------------------------------------------------------- read state

def ladder_state(store: MemoryStore) -> list[dict]:
    """Each rung with its live on/off and status, floor → top."""
    rung, incoherent = current_rung(store)
    rows = []
    for lv in LEVELS:
        rows.append({
            "id": lv.id,
            "name": lv.name,
            "coupling": lv.coupling,
            "status": lv.status,
            "locked_on": lv.locked_on,
            "on": is_on(store, lv.id),
            "is_rung": lv.id == rung,
        })
    return rows


def current_rung(store: MemoryStore) -> tuple[str, bool]:
    """The highest *contiguously*-enabled rung, and whether the ladder is
    incoherent (some rung above an off rung is on — only reachable by setting a
    dial directly via `config`, never via the `level` command)."""
    on = [is_on(store, lv.id) for lv in LEVELS]
    # first gap: highest k such that L0..Lk are all on
    rung_idx = 0
    for i, _ in enumerate(LEVELS):
        if on[i]:
            rung_idx = i
        else:
            break
    incoherent = any(on[j] for j in range(rung_idx + 1, len(LEVELS)))
    return LEVELS[rung_idx].id, incoherent


# --------------------------------------------------------------- mutate

def set_rung(store: MemoryStore, level_id: str) -> str:
    """Move the store to a rung: enable it and everything below, disable
    everything above (cumulative). Returns a human-readable summary."""
    target = level(level_id)
    if target.status == PLANNED:
        raise ValueError(f"{target.id} ({target.name}) is not built yet "
                         f"(planned) — cannot select it as a rung")
    ti = _INDEX[target.id]
    for lv in LEVELS:
        if lv.locked_on or lv.status == PLANNED:
            continue
        _set_one(store, lv.id, _INDEX[lv.id] <= ti)
    return f"operating level set to {target.id} — {target.name}"


def toggle(store: MemoryStore, level_id: str, on: bool) -> str:
    """Flip one rung, cascading to keep the ladder cumulative: turning a rung
    ON enables every rung below it; turning a rung OFF disables every rung
    above it."""
    lv = level(level_id)
    if lv.locked_on:
        raise ValueError(f"{lv.id} ({lv.name}) is always on and cannot be "
                         f"toggled — it is part of the ladder's floor")
    if lv.status == PLANNED:
        raise ValueError(f"{lv.id} ({lv.name}) is not built yet (planned)")
    idx = _INDEX[lv.id]
    if on:
        # enable this rung and every built dial below it
        for other in LEVELS:
            if other.locked_on or other.status == PLANNED:
                continue
            if _INDEX[other.id] <= idx:
                _set_one(store, other.id, True)
    else:
        # disable this rung and every rung above it
        for other in LEVELS:
            if other.locked_on or other.status == PLANNED:
                continue
            if _INDEX[other.id] >= idx:
                _set_one(store, other.id, False)
    return f"{lv.id} ({lv.name}) turned {'on' if on else 'off'}"


# --------------------------------------------------------------- format

_STATUS_NOTE = {
    BUILT: "",
    DOGFOOD: "  (under evaluation)",
    PLANNED: "  (not built yet)",
}


def format_ladder(store: MemoryStore) -> str:
    rows = ladder_state(store)
    rung, incoherent = current_rung(store)
    lines = []
    for r in rows:
        if r["locked_on"]:
            mark = "on (locked)"
        elif r["status"] == PLANNED:
            mark = "--"
        else:
            mark = "on " if r["on"] else "off"
        cursor = " <- current rung" if r["is_rung"] else ""
        note = _STATUS_NOTE[r["status"]]
        lines.append(f"  {r['id']}  {mark:<11}  {r['name']}{note}{cursor}")
    out = "\n".join(lines)
    if incoherent:
        out += ("\n\n[!] ladder is incoherent — a rung above an off rung is on "
                "(set directly via `config`). `doctor` explains; `level "
                f"{rung}` re-normalizes.")
    return out
