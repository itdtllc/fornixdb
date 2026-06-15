"""Marginal-recall benefit harness (FornixDB #190): the quantifiable value of
FornixDB *over the flat markdown memory the AI already had* — not over "no
memory".

The golden eval (evals.py) asks "can recall find a stored fact?". This asks the
owner's real question: "does FornixDB hold / answer things the flat MEMORY.md
has already LOST?" — the surprise the project exists to prevent.

The honest baseline is layered, and FornixDB was *seeded* from these same files,
so overlap is expected and is NOT counted as benefit. Every live memory is
classified against the flat system:

  in_flat       name is in the LOADED MEMORY.md index (free at session start)
  on_disk_only  in the files / full index but past the truncation cap —
                lost at session start, recoverable only by grepping on disk
  fornix_only   no markdown file at all — the flat system can't produce it

Deltas (the benefit):
  marginal_at_startup = on_disk_only + fornix_only   (not free at session start)
  marginal_content    = fornix_only                  (absent from the flat system)

Caveats the report states plainly: (1) MEMORY.md being over its cap is itself
the proof the flat system is shedding facts; (2) `on_disk_only` is recoverable
IF the model greps — effort it may not spend; (3) `fornix_only` partly reflects
where a fact happened to be written, except for episodic/timeline rows, which
the flat markdown system has no structure to hold at all (a categorical win).
"""

from __future__ import annotations

from pathlib import Path

# Claude Code loads MEMORY.md up to a session-start cap; past it, entries are
# silently dropped (the "Only part of it was loaded" truncation).
DEFAULT_CAP_CHARS = 24_400


def scan_flat_baseline(memory_md_path: str | Path, memory_dir: str | Path,
                       cap_chars: int = DEFAULT_CAP_CHARS) -> dict:
    """Read the flat markdown memory system the way the AI actually sees it:
    the loaded slice of the index, the full index, and the topic-file names."""
    md = Path(memory_md_path)
    full = md.read_text(encoding="utf-8") if md.exists() else ""
    files = {p.stem for p in Path(memory_dir).glob("*.md")
             if p.name != md.name} if Path(memory_dir).exists() else set()
    return {
        "loaded_index": full[:cap_chars],
        "full_index": full,
        "truncated": len(full) > cap_chars,
        "index_chars": len(full),
        "cap_chars": cap_chars,
        "file_names": files,
    }


def classify(memory: dict, baseline: dict) -> str:
    """Where a single memory lives relative to the flat system."""
    name = memory.get("name")
    if not name:                                   # episodic/native: no slug
        return "fornix_only"
    on_disk = name in baseline["file_names"] or name in baseline["full_index"]
    if not on_disk:
        return "fornix_only"
    return "in_flat" if name in baseline["loaded_index"] else "on_disk_only"


def coverage(store, baseline: dict) -> dict:
    """Classify every live memory; tally overall and per kind."""
    rows = [dict(r) for r in store.conn.execute(
        "SELECT id, kind, name, gist FROM memory WHERE superseded_time IS NULL")]
    buckets = {"in_flat": 0, "on_disk_only": 0, "fornix_only": 0}
    by_kind: dict[str, dict] = {}
    for r in rows:
        loc = classify(r, baseline)
        buckets[loc] += 1
        by_kind.setdefault(r["kind"], {"in_flat": 0, "on_disk_only": 0,
                                       "fornix_only": 0})[loc] += 1
    total = len(rows) or 1
    return {
        "total": len(rows),
        "buckets": buckets,
        "by_kind": by_kind,
        "marginal_at_startup": buckets["on_disk_only"] + buckets["fornix_only"],
        "marginal_content": buckets["fornix_only"],
        "pct_marginal_at_startup": round(
            100 * (buckets["on_disk_only"] + buckets["fornix_only"]) / total, 1),
        "pct_marginal_content": round(100 * buckets["fornix_only"] / total, 1),
        "truncated": baseline["truncated"],
        "index_chars": baseline["index_chars"],
        "cap_chars": baseline["cap_chars"],
    }


def golden_marginal(store, baseline: dict, golden_path: str | Path,
                    embedder=None) -> dict:
    """On the real golden questions: how many are ANSWERED by a memory the flat
    system would not freely surface? This ties the store-wide coverage number
    to questions a user actually asks — a directly-counted prevented surprise."""
    from .evals import load_golden, run_case
    cases = load_golden(golden_path)
    answered = startup_marginal = content_marginal = 0
    detail = []
    for c in cases:
        res = run_case(store, c, embedder=embedder)
        if res["rank"] is None:                    # FornixDB couldn't answer
            continue
        answered += 1
        # the expected memory FornixDB actually surfaced (best-ranked hit)
        got = res["got"]
        hit_id = next((mid for mid in got if mid in set(res["expected"])), None)
        row = store.conn.execute(
            "SELECT id, kind, name, gist FROM memory WHERE id = ?",
            (hit_id,)).fetchone()
        loc = classify(dict(row), baseline) if row else "fornix_only"
        if loc != "in_flat":
            startup_marginal += 1
        if loc == "fornix_only":
            content_marginal += 1
        detail.append({"query": c["query"], "hit": hit_id, "loc": loc})
    return {
        "answered": answered,
        "startup_marginal": startup_marginal,   # not free at session start
        "content_marginal": content_marginal,   # absent from flat system
        "detail": detail,
    }


def format_report(cov: dict, gold: dict | None = None) -> str:
    b = cov["buckets"]
    lines = [
        "FornixDB marginal value over the flat markdown memory",
        "(overlap is seeded and NOT counted — only what the flat system lacks)",
        "",
        f"Live memories: {cov['total']}",
        f"  in_flat       {b['in_flat']:>4}  (in the loaded MEMORY.md — free at startup)",
        f"  on_disk_only  {b['on_disk_only']:>4}  (past the truncation cap — grep-only)",
        f"  fornix_only   {b['fornix_only']:>4}  (no markdown file — flat system can't produce it)",
        "",
        f"Marginal at session start: {cov['marginal_at_startup']} "
        f"({cov['pct_marginal_at_startup']}%) not freely available in the loaded index",
        f"Marginal content:          {cov['marginal_content']} "
        f"({cov['pct_marginal_content']}%) absent from the flat system entirely",
    ]
    if cov["truncated"]:
        lines.append(f"NOTE: MEMORY.md is {cov['index_chars']} chars > the "
                     f"{cov['cap_chars']} cap — the flat index IS truncating now.")
    lines.append("")
    lines.append("Per kind (in_flat / on_disk_only / fornix_only):")
    for kind, k in sorted(cov["by_kind"].items()):
        lines.append(f"  {kind:<10} {k['in_flat']:>4} / {k['on_disk_only']:>4} "
                     f"/ {k['fornix_only']:>4}")
    if gold is not None:
        lines += [
            "",
            f"On {gold['answered']} real golden questions FornixDB answers:",
            f"  {gold['startup_marginal']} answered by a memory NOT free at "
            "session start (prevented surprise)",
            f"  {gold['content_marginal']} answered by a fornix_only memory "
            "(unanswerable from the flat system)",
        ]
    return "\n".join(lines)
