"""Markdown↔store staleness correlation — the dream-time cross-check.

A hand-maintained markdown memory file carries forward-looking claims
(PICKUP / NEXT / TODO / pending blocks) that rot silently: events overtake
them, nobody rewrites the file, and the stale claim is PUSHED into the host's
context every session start, where it wins over the store's passively-captured
truth by default. Measured live 2026-07-17: a demo filmed and shipped 7/12–7/14
was still presented as pending by a PICKUP block written 7/10, while the
store's auto-captured episodic row had recorded the truth on the day it
happened. The store has machinery for noticing it is stale (passive capture,
supersede chains, the reality check); the flat file has none. This scan is
that missing half for the markdown side.

At DREAM time, each file in the bridge's markdown directory (`native_dir`)
whose body carries a forward-looking marker is checked against the store's
episodic timeline: an episodic row RECORDED AFTER the file's last edit, whose
headline carries closure language (consolidate's `_CLOSURE_RE`) and whose
stored vector sits close to the file's imported row's stored vector, reads as
the file's open item having been overtaken. Model-free and billed-token-free
by construction — both sides of the cosine are vectors already in the
embedding table (the native-memory ingest keeps the directory imported), and
the scan runs offline in the dream. The only in-session surface is ONE brief
line when an unreviewed flag exists (~30–60 tokens; silence otherwise).

Propose-not-dispose (§6.5): a flag is a question, never an edit. The natural
fix is rewriting the markdown file — its fresh mtime clears the flag on its
own. A false positive (the item genuinely still open despite a similar closure
row) is accepted forever with `link <file-row> <session-row> --relation
distinct`, the same pair-level accept the merge/contradiction lists use.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from .core import MemoryStore
from .multistore import get_config, set_config
from .vectors import cosine, from_blob

# Forward-looking markers a maintained project file uses for its open work.
# Deliberately NOT consolidate's _TASK_RE: prose words like "investigate" or
# "needs to" appear in nearly every project note and would flag the whole
# directory; these are the section-marker forms ("NEXT:" with punctuation, not
# every "next" in a sentence). The block they open is display context only —
# the MATCH runs on stored vectors, so marker recall matters more than marker
# precision inside an already-flagged file.
_FORWARD_RE = re.compile(
    r"\b(?:pickup|pick\s+up\s+here|resume\s+here|next\s+steps?|next\s+up|"
    r"to-?dos?|open\s+items?|still\s+open|awaiting|remaining\s+gate|"
    r"not\s+yet\s+\w+|pending)\b|\bnext\s*[:=]|\btodo\b",
    re.I)

_SKIP_NAMES = ("MEMORY.md", "FornixDB.md", "README.md")
_BLOCK_MAX_CHARS = 400
FLAGS_KEY = "markdown_stale_flags"

# Closure vocabulary beyond consolidate's _CLOSURE_RE, for THIS scan only.
# The motivating live miss ("demo already filmed + posted to X") carries none
# of the code-lifecycle words the resolution scan was tuned on — real-world
# closures also arrive as published/released/posted/filmed. Kept separate so
# widening it never changes the supersede-direction scan's behavior.
_CLOSURE_EXTRA_RE = re.compile(
    r"\b(published|released|delivered|posted|filmed|launched|submitted)\b", re.I)


_WIKILINK_RE = re.compile(r"\[\[[^\]\n]+\]\]")


def forward_blocks(body: str) -> list[str]:
    """The forward-looking blocks in a markdown body: each line carrying a
    marker plus its continuation lines, up to a blank line or the next heading.
    A line already inside the previous block never opens a new one. Wikilinks
    are stripped before the marker test — a slug like
    [[project-video-series-pickup]] is a NAME, not an open-work claim
    (measured false positive on the live store's first run, 2026-07-17)."""
    lines = (body or "").splitlines()
    blocks, i = [], 0
    while i < len(lines):
        if not _FORWARD_RE.search(_WIKILINK_RE.sub("", lines[i])):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith("#"):
            j += 1
        block = "\n".join(lines[i:j]).strip()
        if block:
            blocks.append(block[:_BLOCK_MAX_CHARS])
        i = j
    return blocks


def _name_and_body(path: Path, text: str) -> tuple[str, str]:
    """The identity the bridge imported this file under (frontmatter `name`,
    falling back to the file stem — markdown_import's rule) plus the BODY with
    frontmatter stripped: a `name: Session Pickup — …` header line is a title,
    not an open-work claim, and must never trip the marker test (measured
    false positive on the live store's first run, 2026-07-17)."""
    from .adapters.markdown_import import parse_frontmatter
    meta, body = parse_frontmatter(text)
    return (meta.get("name") or path.stem), body


def scan(store: MemoryStore, directory: str | Path) -> list[dict]:
    """Flags, best (highest cosine) first, one per file, capped like the other
    dream lists. Read-only. A file with no imported row is skipped — the
    session hook's native ingest brings it in by the next pass."""
    from .consolidate import (MAX_PAIR_PROPOSALS, RESOLUTION_COSINE,
                              _CLOSURE_RE, _distinct_pairs, _headline)
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        return []
    model_row = store.conn.execute(
        "SELECT model, count(*) c FROM embedding GROUP BY model "
        "ORDER BY c DESC LIMIT 1").fetchone()
    if model_row is None:
        return []
    model = model_row["model"]
    # closure-carrying episodic rows, vectors attached — the overtaking side
    epi = [r for r in store.conn.execute(
        """SELECT m.id, m.gist, m.detail, m.recorded_time, e.vector
           FROM memory m
           JOIN embedding e ON e.memory_id = m.id AND e.model = ? AND e.chunk = 0
           WHERE m.superseded_time IS NULL AND m.kind = 'episodic'""",
        (model,)).fetchall()
        if _CLOSURE_RE.search(h := _headline(r["gist"], r["detail"]))
        or _CLOSURE_EXTRA_RE.search(h)]
    if not epi:
        return []
    evecs = {r["id"]: from_blob(r["vector"]) for r in epi}
    distinct = _distinct_pairs(store)

    flags = []
    for path in sorted(directory.glob("*.md")):
        if path.name in _SKIP_NAMES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        name, body = _name_and_body(path, text)
        blocks = forward_blocks(body)
        if not blocks:
            continue
        row = store.conn.execute(
            """SELECT m.id, e.vector FROM memory m
               JOIN embedding e ON e.memory_id = m.id AND e.chunk = 0 AND e.model = ?
               WHERE m.name = ? AND m.superseded_time IS NULL""",
            (model, name)).fetchone()
        if row is None:
            continue
        edited = datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0)
        fvec = from_blob(row["vector"])
        best = None
        for r in epi:
            if (r["recorded_time"] or "") <= edited.isoformat():
                continue
            if frozenset((row["id"], r["id"])) in distinct:
                continue
            cos = cosine(fvec, evecs[r["id"]])
            if cos < RESOLUTION_COSINE:
                continue
            if best is None or cos > best["cosine"]:
                best = {"overtaken_by": r["id"], "cosine": round(cos, 3),
                        "epi_gist": r["gist"],
                        "epi_time": (r["recorded_time"] or "")[:10]}
        if best:
            flags.append({"file": path.name, "path": str(path),
                          "name": name, "file_id": row["id"],
                          "edited": edited.isoformat(),
                          "marker": blocks[0].splitlines()[0].strip(),
                          "markers": len(blocks), **best})
    flags.sort(key=lambda f: f["cosine"], reverse=True)
    return flags[:MAX_PAIR_PROPOSALS]


# ------------------------------------------------------- the brief-time line
# The dream persists its flags; brief only READS them back — one line when
# something is standing, nothing otherwise. Each flag is cheaply re-validated
# at read time (mtime stat + one distinct-pair query, no file parsing, no
# vectors) so a file the user already rewrote never nags again even before the
# next dream.

def persist_flags(store: MemoryStore, flags: list[dict]) -> None:
    """Called by dream() after each scan: the current flags become the standing
    set brief reads. An empty scan clears it."""
    set_config(store, FLAGS_KEY, json.dumps(flags) if flags else "")


def standing_flags(store: MemoryStore) -> list[dict]:
    """The persisted flags still valid right now: the file still exists, has
    not been edited since the flag was raised, BOTH rows are still live, and
    the pair has not been accepted as distinct. Read-only (safe on frozen
    stores). The liveness check matters because flags persist BETWEEN dreams:
    a row superseded or forgotten after the scan must never be re-cited by the
    brief line — retired memories stay retired on every surface."""
    raw = (get_config(store, FLAGS_KEY, "") or "").strip()
    if not raw:
        return []
    try:
        flags = json.loads(raw)
    except (ValueError, TypeError):
        return []
    from .consolidate import _distinct_pairs
    distinct = _distinct_pairs(store)
    ids = {i for f in flags for i in (f.get("file_id"), f.get("overtaken_by"))
           if isinstance(i, int)}
    live = {r["id"] for r in store.conn.execute(
        f"SELECT id FROM memory WHERE superseded_time IS NULL "
        f"AND id IN ({','.join('?' * len(ids))})", sorted(ids))} if ids else set()
    out = []
    for f in flags:
        if not (f.get("file_id") in live and f.get("overtaken_by") in live):
            continue  # either side superseded/forgotten since the dream
        try:
            edited = datetime.fromtimestamp(
                os.stat(f["path"]).st_mtime).replace(microsecond=0)
        except (OSError, KeyError):
            continue  # file gone (or malformed flag) — nothing to nag about
        if edited.isoformat() > f.get("edited", ""):
            continue  # rewritten since the flag — the natural fix happened
        if frozenset((f.get("file_id"), f.get("overtaken_by"))) in distinct:
            continue  # reviewed and accepted as legitimately distinct
        out.append(f)
    return out


def brief_line(store: MemoryStore) -> str | None:
    """The one-line pointer brief prints when unreviewed flags stand, or None.
    Never raises — brief must not die on a malformed flag set."""
    try:
        flags = standing_flags(store)
    except Exception:
        return None
    if not flags:
        return None
    f0 = flags[0]
    more = f" (+{len(flags) - 1} more)" if len(flags) > 1 else ""
    return (f"--- markdown may be stale: {f0['file']} still says "
            f"\"{f0['marker'][:50]}\" but #{f0['overtaken_by']} "
            f"({f0['epi_time']}) reads as its closure{more} — "
            "review with `fornixdb dream` ---")
