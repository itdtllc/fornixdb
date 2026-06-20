"""Interactive configuration wizard — `fornixdb configure`.

A single command that walks the user through every user-selectable setting,
showing the current value (and letting Enter keep it), collects the desired
changes, then shows a diff and writes only after an explicit confirmation.
Nothing is written until the final "Apply?" yes.

The wizard is I/O-injected (``ask`` / ``out``) so it can be driven by a script
in tests without a real terminal.
"""

from __future__ import annotations

from . import levels
from .adapters.native_memory import (MODES as INGEST_MODES, ingest_mode,
                                      set_ingest_mode)
from .multistore import CAPTURE_MODES, capture_mode, get_config, set_config

_OFF = ("off", "0", "false", "no")
_RUNG_CHOICES = ("L0", "L1", "L2", "L3", "L4")  # L5/L6 not built — not offered


def _on(store, key: str, default: str = "on") -> str:
    return "off" if (get_config(store, key, default) or default).strip().lower() in _OFF else "on"


def _ask_keep(ask, out, label: str, current: str, choices: tuple[str, ...]) -> str:
    """Prompt once; Enter keeps `current`; re-prompt until a valid choice."""
    opts = "/".join(choices)
    while True:
        raw = ask(f"  {label} ({opts}) [{current}]: ").strip()
        if raw == "":
            return current
        low = raw.lower()
        for c in choices:
            if low == c.lower():
                return c
        out(f"    '{raw}' is not one of {opts} — try again")


def _ask_budget(ask, out, current: str) -> str:
    while True:
        raw = ask(f"  disk budget in MB (a number, or 'off') [{current}]: ").strip()
        if raw == "":
            return current
        if raw.lower() in ("off", "none"):
            return "off"
        try:
            if float(raw) > 0:
                return raw
        except ValueError:
            pass
        out(f"    '{raw}' is not a positive number or 'off' — try again")


def build_plan(store, ask, out) -> list[dict]:
    """Collect intended changes as {label, old, new, apply}. Pure prompting —
    no writes happen here."""
    plan: list[dict] = []

    # 1) operating level — the ladder rung controls auto-capture(L2)/proactive
    #    (L3)/rhythmic(L4) together, so we don't ask those three separately.
    out("\nOperating level (memory↔thinking coupling):")
    out(levels.format_ladder(store))
    cur_rung, _ = levels.current_rung(store)
    rung = _ask_keep(ask, out, "set rung", cur_rung, _RUNG_CHOICES)
    if rung != cur_rung:
        plan.append({"label": "operating_level", "old": cur_rung, "new": rung,
                     "apply": lambda r=rung: levels.set_rung(store, r)})
    rung_idx = _RUNG_CHOICES.index(rung)

    # 2) capture flavor — only meaningful once auto-capture (L2) is on
    if rung_idx >= _RUNG_CHOICES.index("L2"):
        cur_cap = capture_mode(store)
        flavor_now = cur_cap if cur_cap in ("suggest", "auto") else "suggest"
        flavor = _ask_keep(ask, out, "capture style", flavor_now, ("suggest", "auto"))
        if flavor != cur_cap:
            plan.append({"label": "capture_mode", "old": cur_cap, "new": flavor,
                         "apply": lambda v=flavor: set_config(store, "capture_mode", v)})

    # 3) session capture
    cur = _on(store, "session_capture")
    new = _ask_keep(ask, out, "session capture", cur, ("on", "off"))
    if new != cur:
        plan.append({"label": "session_capture", "old": cur, "new": new,
                     "apply": lambda v=new: set_config(store, "session_capture", v)})

    # 4) vectors (semantic recall vs keyword-only)
    cur = _on(store, "vectors")
    new = _ask_keep(ask, out, "vectors (semantic recall)", cur, ("on", "off"))
    if new != cur:
        plan.append({"label": "vectors", "old": cur, "new": new,
                     "apply": lambda v=new: set_config(store, "vectors", v)})

    # 5) ingest mode (following a host AI's native memory dir)
    cur = ingest_mode(store)
    new = _ask_keep(ask, out, "ingest mode", cur, INGEST_MODES)
    if new != cur:
        plan.append({"label": "ingest_mode", "old": cur, "new": new,
                     "apply": lambda v=new: set_ingest_mode(store, v)})

    # 6) disk budget + policy
    cur_budget = get_config(store, "disk_budget_mb") or "off"
    new_budget = _ask_budget(ask, out, cur_budget)
    if new_budget != cur_budget:
        plan.append({"label": "disk_budget_mb", "old": cur_budget, "new": new_budget,
                     "apply": lambda v=new_budget: set_config(store, "disk_budget_mb", v)})
    if new_budget != "off":
        cur_pol = get_config(store, "budget_policy", "prune") or "prune"
        new_pol = _ask_keep(ask, out, "at the cap", cur_pol, ("prune", "freeze"))
        if new_pol != cur_pol:
            plan.append({"label": "budget_policy", "old": cur_pol, "new": new_pol,
                         "apply": lambda v=new_pol: set_config(store, "budget_policy", v)})

    # 7) MCP tools — which tools this store advertises to an AI. Core tools are
    #    always on; optional ones can be trimmed (smaller per-turn prompt).
    from .adapters.mcp_server import (TOOLS, set_tool_enabled, tool_tier,
                                      tools_disabled)
    disabled = set(tools_disabled(store))
    optional = [t for t in TOOLS if tool_tier(t["name"]) != "core"]
    n_on = sum(1 for t in TOOLS if t["name"] not in disabled)
    out(f"\nMCP tools: {n_on}/{len(TOOLS)} advertised "
        f"({len(optional)} optional, core always on).")
    mode = _ask_keep(ask, out, "MCP tools", "keep", ("keep", "minimal", "custom"))
    if mode == "minimal":
        for t in optional:
            if t["name"] not in disabled:
                plan.append({"label": f"tool:{t['name']}", "old": "on", "new": "off",
                             "apply": lambda n=t["name"]: set_tool_enabled(store, n, False)})
    elif mode == "custom":
        out("  (core tools are always on and not shown; each option's full "
            "explanation is printed below it)")
        for t in optional:
            nm, cur = t["name"], ("off" if t["name"] in disabled else "on")
            out(f"\n  {nm}")
            out(f"    {t['description']}")
            new = _ask_keep(ask, out, "enable", cur, ("on", "off"))
            if new != cur:
                plan.append({"label": f"tool:{nm}", "old": cur, "new": new,
                             "apply": lambda n=nm, v=new: set_tool_enabled(store, n, v == "on")})

    return plan


def run_configure(store, *, ask=input, out=print, db_label: str = "") -> dict:
    """Drive the wizard. Returns {applied: [labels], aborted: bool}."""
    out("FornixDB configuration wizard")
    if db_label:
        out(f"Store: {db_label}")
        out("(not this one? Ctrl-C and re-run with --db <path>)")
    out("Press Enter to keep the value in [brackets]; Ctrl-C aborts.")

    if store.frozen():
        ans = ask("\nStore is frozen (read-only). Unfreeze to make changes? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            out("Left frozen — no changes made. (`config frozen off` to unlock later)")
            return {"applied": [], "aborted": True}
        set_config(store, "frozen", "off")
        out("Unfrozen.")

    try:
        plan = build_plan(store, ask, out)
    except (EOFError, KeyboardInterrupt):
        out("\nAborted — no changes made.")
        return {"applied": [], "aborted": True}

    if not plan:
        out("\nNo changes — every setting left as it was.")
        return {"applied": [], "aborted": False}

    out("\nAbout to change:")
    for c in plan:
        out(f"  {c['label']:<16} {c['old']} -> {c['new']}")
    confirm = ask(f"\nApply {len(plan)} change(s)? [y/N] ").strip().lower()
    if confirm not in ("y", "yes"):
        out("Aborted — no changes made.")
        return {"applied": [], "aborted": True}

    applied = []
    for c in plan:
        c["apply"]()
        applied.append(c["label"])
        out(f"  ✓ {c['label']} = {c['new']}")
    out(f"\nDone — {len(applied)} change(s) saved.")
    return {"applied": applied, "aborted": False}
