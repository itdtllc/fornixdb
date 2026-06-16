"""Follow a host AI's native file-memory — additive, never a takeover (§15.2 #2).

Owner directive 2026-06-16: FornixDB must NEVER short-circuit a host AI's own
memory mechanism. For a consumer that HAS native memory (e.g. Claude Code, whose
memory Anthropic controls and may improve), the flow is **native → FornixDB**:
FornixDB *follows* the native memory directory downstream via this adapter,
adding what native can't do (time/episodic axis, ranked recall, supersede
history) — it never owns the directory, generates those files, or sits in the
write path. The native mechanism stays authoritative, evolvable, and removable;
delete FornixDB and native memory is untouched.

Mode control — the user chooses and is always shown what runs (`ingest_mode`):
  explicit  no background automation at all (native ingest AND passive session
            capture suppressed); FornixDB is used only via deliberate tool/CLI
            calls. The user's "leave my background alone" switch.
  passive   (default) background automation runs: native auto-ingest on the
            session hook + passive session capture. Native ingest additionally
            requires a configured `native_dir`, so it is still opt-in.
  both      background automation runs AND the AI is encouraged to also capture/
            recall explicitly.
"""

from __future__ import annotations

from ..core import MemoryStore
from ..multistore import get_config, set_config

MODES = ("explicit", "passive", "both")
SOURCE = "claude-code-native"


def ingest_mode(store: MemoryStore) -> str:
    m = (get_config(store, "ingest_mode", "passive") or "passive").lower()
    return m if m in MODES else "passive"


def set_ingest_mode(store: MemoryStore, mode: str) -> str:
    mode = mode.lower()
    if mode not in MODES:
        return f"mode must be one of {', '.join(MODES)}"
    set_config(store, "ingest_mode", mode)
    return f"ingest_mode = {mode}"


def auto_background_enabled(store: MemoryStore) -> bool:
    """True when background automation may run (passive/both); False in explicit
    mode. The session hook checks this before capturing OR ingesting."""
    return ingest_mode(store) in ("passive", "both")


def native_dir(store: MemoryStore) -> str | None:
    return get_config(store, "native_dir", None)


def set_native_dir(store: MemoryStore, path: str) -> str:
    set_config(store, "native_dir", path)
    return f"native_dir = {path}"


def ingest(store: MemoryStore, path: str | None = None) -> dict:
    """Follow the native memory directory into this store: additive, idempotent
    (skips memories already present by name), with content-level dedup so the
    same fact re-slugged under a new name is not double-stored. Returns the
    import counts plus the directory used. Caller decides WHEN to run this
    (the session hook in passive/both mode, or an explicit `fornixdb ingest`)."""
    directory = path or native_dir(store)
    if not directory:
        return {"ok": False, "reason": "no native_dir configured "
                "(`fornixdb ingest --dir <path>`)"}
    from .markdown_import import import_directory
    r = import_directory(store, directory, source=SOURCE, dedup_gist=True)
    r.update({"ok": True, "dir": str(directory)})
    return r
