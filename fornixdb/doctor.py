"""Consolidated configuration view + health check (the single place to see
"how is this FornixDB set up?"). Two surfaces:

  config_overview(store) -> the current value of every store-level setting in
      one list, so `fornixdb config` (no args) answers "what is configured?"
      instead of erroring. Read-only; setting a value stays `config <key> <val>`.

  diagnose(store) -> a `doctor` pass: schema currency, the host-side hooks that
      make capture + proactive recall actually fire (these live in the HOST's
      settings, not in FornixDB, so they are the most common silent gap), and
      config smells worth flagging. Returns structured rows so the CLI and any
      MCP/host can render them; nothing here mutates the store.

Host-hook detection is best-effort and host-aware: the SessionEnd capture and
UserPromptSubmit recall hooks are wired as commands in a host config file. We
look for FornixDB's own adapter module names in the known Claude Code settings
locations; absence is reported as "not detected," never as an error (another
host may wire them differently, or the user may not use hooks at all)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .adapters.mcp_server import TOOLS, active_tools
from .adapters.native_memory import auto_background_enabled, ingest_mode, native_dir
from .budget import DEFAULT_POLICY, status as budget_status
from .db import (DEFAULT_MACHINE_CAP_DISK_FRACTION, DEFAULT_MACHINE_CAP_MAX_MB,
                 SCHEMA_VERSION, default_db_path)
from .multistore import capture_mode, get_config, set_config
from .tokens import estimate_tokens

_OFF = ("off", "0", "false", "no")

# the two host hooks that turn passive automation on, and the adapter module
# each is wired to — the string we scan a host settings file for.
HOST_HOOKS = (
    ("SessionEnd capture", "fornixdb.adapters.claude_code_session_end"),
    ("UserPromptSubmit recall", "fornixdb.adapters.claude_code_recall"),
)

# Claude Code's settings files, most-global first. Others hosts differ; this is
# a best-effort default, overridable by passing explicit paths to diagnose().
DEFAULT_HOST_SETTINGS = (
    "~/.claude/settings.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
)


def _vectors_setting(store) -> str:
    """The vectors SETTING (not a model load): env override wins, else the
    per-store config, else on by default."""
    env = os.environ.get("FORNIXDB_VECTORS")
    if env is not None:
        return "off (env FORNIXDB_VECTORS)" if env.strip().lower() in _OFF else "on"
    val = (get_config(store, "vectors", "on") or "on").strip().lower()
    return "off" if val in _OFF else "on"


def config_overview(store) -> list[tuple[str, str]]:
    """Every store-level setting and its current value, in display order."""
    st = budget_status(store)
    budget = (f"{st['budget_mb']} MB cap, policy '{st['policy']}'"
              if st.get("budget_mb") else "no cap (never-delete)")
    nd = native_dir(store)
    ingest = ingest_mode(store)
    bg = "on" if (auto_background_enabled(store) and nd) else "off"
    active = active_tools(store)
    tools = (f"{len(active)}/{len(TOOLS)} advertised "
             f"(~{estimate_tokens(json.dumps(active))} tok)")
    proactive = (get_config(store, "proactive_recall", "on") or "on")
    floor_adapt = (get_config(store, "usefulness_floor_adapt", "on") or "on")
    proj_scope = (get_config(store, "project_scoped_pulse", "on") or "on")
    dedup = (get_config(store, "cross_pulse_dedup", "on") or "on")
    pinned_proj = (get_config(store, "active_project", "") or "").strip()
    session_cap = (get_config(store, "session_capture", "on") or "on")
    from .levels import current_rung, level
    rung, incoherent = current_rung(store)
    rung_str = (f"{rung} — {level(rung).name}"
                + ("  (incoherent — see doctor)" if incoherent else ""))
    return [
        ("operating_level", rung_str),
        ("capture_mode", capture_mode(store)),
        ("ingest_mode", f"{ingest} (background ingest: {bg}, dir: {nd or 'unset'})"),
        ("session_capture", "off" if session_cap in _OFF else "on"),
        ("proactive_recall", "off" if proactive in _OFF else "on"),
        ("usefulness_floor_adapt", "off" if floor_adapt in _OFF else "on"),
        ("project_scoped_pulse", "off" if proj_scope in _OFF else "on"),
        ("cross_pulse_dedup", "off" if dedup in _OFF else "on"),
        ("active_project", pinned_proj or "(auto: prompt / cwd)"),
        ("vectors", _vectors_setting(store)),
        ("disk_budget", budget),
        ("frozen", "yes (read-only)" if store.frozen() else "no"),
        ("MCP tools", tools),
    ]


# The out-of-the-box value of every config_overview row, so a read-only `config`
# run shows what each option *would* be if never touched. Keys MUST stay in step
# with config_overview labels (test_doctor enforces full coverage).
CONFIG_DEFAULTS: dict[str, str] = {
    "operating_level": "L4 (every built rung on)",
    "capture_mode": "suggest",
    "ingest_mode": "passive",
    "session_capture": "on",
    "proactive_recall": "on",
    "usefulness_floor_adapt": "on",
    "project_scoped_pulse": "on",
    "cross_pulse_dedup": "on",
    "active_project": "(auto: prompt / cwd)",
    "vectors": "on",
    "disk_budget": "no cap (never-delete)",
    "frozen": "no",
    "MCP tools": "all advertised",
}


def host_hook_status(paths=DEFAULT_HOST_SETTINGS) -> list[dict]:
    """For each host hook: is FornixDB's adapter wired in any known settings
    file? Returns one row per hook with wired/where, plus which files existed."""
    blobs: list[tuple[str, str]] = []
    for p in paths:
        fp = Path(p).expanduser()
        if fp.is_file():
            try:
                blobs.append((str(fp), fp.read_text()))
            except OSError:
                pass
    rows = []
    for label, module in HOST_HOOKS:
        where = [path for path, text in blobs if module in text]
        rows.append({"hook": label, "module": module,
                     "wired": bool(where), "where": where})
    return {"hooks": rows, "files_seen": [path for path, _ in blobs]}


# --- config integrity: a config key with no runtime reader does nothing -------
# A user can set ANY key (`config <key> <value>` is generic). A key nothing reads
# is dead weight at best and a silent footgun at worst — the user thinks they
# changed behavior and nothing happened (the "L1 was declarative-only" class of
# bug). We catch it by comparing the keys SET in a store against the keys the code
# actually READS, scanned from source so the reader set self-updates.

# Read indirectly, so the literal source scan misses them: via a method
# (`store.frozen()`), a named constant (`mcp_tools_disabled`), or multistore's
# set_config side-effect handlers (the machine-budget family).
_INDIRECT_CONFIG_READERS = frozenset({
    "frozen", "mcp_tools_disabled",
    "machine_budget_mb", "machine_budget_policy", "machine_budget_defaulted",
})
# Per-entity dynamic keys (prefix + session-id / kind), read via helper-built
# names or a `LIKE` scan — recognized by prefix, never flagged.
_CONFIG_READ_PREFIXES = ("decay_", "active_project_session_", "cadence_turn_",
                         "cadence_episode_", "proactive_injected_")
_READER_RE = re.compile(
    r'(?:get_config\([^,]+,\s*|_setting_off\()"([a-z_][a-z0-9_]*)"')


def _literal_config_readers() -> set[str]:
    """Every config key read via a literal get_config/_setting_off call, scanned
    from the package source so it stays current as readers are added."""
    keys: set[str] = set()
    for p in Path(__file__).parent.rglob("*.py"):
        try:
            keys |= set(_READER_RE.findall(p.read_text(encoding="utf-8")))
        except OSError:
            pass
    return keys


def config_readers() -> set[str]:
    """All config keys the code reads (literal scan + the indirectly-read ones)."""
    return _literal_config_readers() | _INDIRECT_CONFIG_READERS


def _config_key_is_read(key: str, readers: set[str]) -> bool:
    return key in readers or any(key.startswith(p) for p in _CONFIG_READ_PREFIXES)


def config_integrity(store) -> list[dict]:
    """Health rows for the store's config: each meta key SET here that NO code
    reads — a typo, a stale/removed setting, or a key set expecting an effect it
    can't have. Pure read; never mutates the store."""
    readers = config_readers()
    out: list[dict] = []
    for (key,) in store.conn.execute("SELECT key FROM meta ORDER BY key"):
        if not _config_key_is_read(key, readers):
            out.append({"level": "warn",
                        "msg": f"config '{key}' is set but no code reads it — "
                               "a typo, a stale setting, or one with no effect; "
                               "remove it or check the name"})
    return out


def diagnose(store, *, host_paths=DEFAULT_HOST_SETTINGS) -> list[dict]:
    """Health rows: {level: ok|warn|info, msg}. Ordered: schema, host hooks,
    config smells, then config-integrity (keys set but unread). Pure read —
    never mutates the store."""
    out: list[dict] = []

    stored = get_config(store, "schema_version")
    if stored and str(stored) == str(SCHEMA_VERSION):
        out.append({"level": "ok", "msg": f"store schema v{SCHEMA_VERSION} (current)"})
    elif stored:
        out.append({"level": "warn",
                    "msg": f"store schema v{stored}, code expects v{SCHEMA_VERSION} "
                           "— open the store once to migrate"})

    hs = host_hook_status(host_paths)
    for row in hs["hooks"]:
        if row["wired"]:
            out.append({"level": "ok",
                        "msg": f"{row['hook']} hook wired ({row['where'][0]})"})
        else:
            seen = (", ".join(hs["files_seen"]) if hs["files_seen"]
                    else "no host settings file found")
            out.append({"level": "warn",
                        "msg": f"{row['hook']} hook NOT detected "
                               f"(looked in: {seen}) — that automation won't fire"})

    st = budget_status(store)
    if not st.get("budget_mb"):
        out.append({"level": "info",
                    "msg": "no disk budget cap set (never-delete; set "
                           "`config disk_budget_mb <MB>` to bound growth)"})
    elif st.get("over_budget"):
        out.append({"level": "warn",
                    "msg": f"store is OVER its {st['budget_mb']} MB cap "
                           f"(policy '{st['policy']}')"})
    if _vectors_setting(store).startswith("off"):
        out.append({"level": "info",
                    "msg": "vectors OFF — recall is keyword + time only "
                           "(quality, not function)"})
    if capture_mode(store) == "explicit":
        out.append({"level": "info",
                    "msg": "capture_mode=explicit — no automatic capture; the AI "
                           "remembers only when told"})
    if store.frozen():
        out.append({"level": "info",
                    "msg": "store is frozen (read-only) — `config frozen off` to write"})
    from .levels import current_rung
    rung, incoherent = current_rung(store)
    if incoherent:
        out.append({"level": "warn",
                    "msg": f"operating-levels ladder is incoherent — a level above "
                           f"an off level is on (set directly via `config`). "
                           f"`level {rung}` re-normalizes to a clean rung"})
    out.extend(config_integrity(store))
    return out


def _store_dir(store) -> Path:
    """The directory holding the store's db file (for free-space math). Reads
    the connection's main-db path; falls back to the default location for an
    in-memory or path-less store."""
    try:
        for seq, name, file in store.conn.execute("PRAGMA database_list"):
            if name == "main" and file:
                return Path(file).expanduser().parent
    except Exception:
        pass
    return Path(default_db_path()).expanduser().parent


def suggested_disk_budget_mb(store=None) -> int:
    """A sensible per-store disk cap, same heuristic as the machine-tier
    default: 20% of free disk, capped at the 2 GB ceiling — generous on a
    workstation, protective on a small device. Falls back to the ceiling if
    free space can't be read."""
    import shutil
    try:
        free_mb = shutil.disk_usage(_store_dir(store)).free / 1e6
        return max(1, min(DEFAULT_MACHINE_CAP_MAX_MB,
                          int(free_mb * DEFAULT_MACHINE_CAP_DISK_FRACTION)))
    except (OSError, ValueError):
        return DEFAULT_MACHINE_CAP_MAX_MB


def suggested_settings(store) -> list[dict]:
    """Recommended defaults with rationale, each flagged satisfied/not against
    the store's current value. Most defaults are already applied by the code;
    the one that is NOT set out of the box — and the whole reason to surface
    this — is a disk cap (never-delete by default), so we suggest a concrete MB
    figure scaled to the device rather than a vague 'set one'."""
    st = budget_status(store)
    cap_set = bool(st.get("budget_mb"))
    proactive = (get_config(store, "proactive_recall", "on") or "on")
    session_cap = (get_config(store, "session_capture", "on") or "on")
    rows = [
        {"key": "disk_budget_mb", "suggested": str(suggested_disk_budget_mb(store)),
         "current": str(st["budget_mb"]) if cap_set else "(none — never-delete)",
         "satisfied": cap_set,
         "why": "bound growth: 20% of free disk, max 2 GB; never-delete until set"},
        {"key": "budget_policy", "suggested": DEFAULT_POLICY,
         "current": st.get("policy", DEFAULT_POLICY),
         "satisfied": st.get("policy", DEFAULT_POLICY) == DEFAULT_POLICY,
         "why": "at the cap, refuse new memories rather than delete old ones"},
        {"key": "capture_mode", "suggested": "suggest",
         "current": capture_mode(store),
         "satisfied": capture_mode(store) == "suggest",
         "why": "offer to remember at checkpoints (not silent, not manual-only)"},
        # an env override (FORNIXDB_VECTORS=off) is a deliberate machine-wide
        # choice a per-store `config vectors on` can't beat — so don't suggest
        # flipping it (that would also make apply_suggested non-idempotent).
        {"key": "vectors", "suggested": "on", "current": _vectors_setting(store),
         "satisfied": ("off" not in _vectors_setting(store)
                       or "env" in _vectors_setting(store)),
         "why": "associative recall; switch off only for a deliberately lean build"},
        {"key": "proactive_recall", "suggested": "on",
         "current": "off" if proactive in _OFF else "on",
         "satisfied": proactive not in _OFF,
         "why": "surface relevant past once per turn (host UserPromptSubmit hook)"},
        {"key": "session_capture", "suggested": "on",
         "current": "off" if session_cap in _OFF else "on",
         "satisfied": session_cap not in _OFF,
         "why": "remember each session like a day (host SessionEnd hook)"},
    ]
    return rows


def apply_suggested(store) -> list[str]:
    """Set only the UNSATISFIED suggestions (so we never churn settings already
    at the recommended value). Returns a line per change applied."""
    store._check_writable()
    applied = []
    for r in suggested_settings(store):
        if not r["satisfied"]:
            set_config(store, r["key"], r["suggested"])
            applied.append(f"{r['key']} = {r['suggested']}")
    return applied


def format_suggested(rows: list[dict]) -> str:
    w = max(len(r["key"]) for r in rows)
    out = []
    for r in rows:
        mark = "ok " if r["satisfied"] else "SET"
        line = f"[{mark}] {r['key'].ljust(w)} suggested={r['suggested']}"
        if not r["satisfied"]:
            line += f"  (now: {r['current']}) — {r['why']}"
        out.append(line)
    return "\n".join(out)


def format_config(rows: list[tuple[str, str]],
                  defaults: dict[str, str] | None = None) -> str:
    w = max(len(k) for k, _ in rows)
    out = []
    for k, v in rows:
        line = f"{k.ljust(w)} = {v}"
        if defaults is not None and k in defaults:
            line += f"   [default: {defaults[k]}]"
        out.append(line)
    return "\n".join(out)


def format_diagnose(rows: list[dict]) -> str:
    tag = {"ok": "[ok]  ", "warn": "[warn]", "info": "[info]"}
    return "\n".join(f"{tag.get(r['level'], '[?]   ')} {r['msg']}" for r in rows)
