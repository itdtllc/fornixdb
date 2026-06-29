"""Honest push-usefulness from session transcripts.

The per-memory usefulness loop credits a memory as "used" only on an explicit
PULL (recall_count) or endorsement (helpful_count). But a PROACTIVELY PUSHED
memory is already in context — the model references it in its reasoning without
ever pulling it — so a useful push and an ignored push look identical to the
counters, and any outcome join keyed on recall_count measures "is this a
frequently-pulled memory", not "was THIS push used". (Live: floor-stats called
143/148 surfaced rows "useful" purely on lifetime pulls of 100-200 on memories
pushed 3-11 times.)

This recovers the true signal from the host's own session transcripts, where both
sides are visible: FornixDB's injected block (an `attachment` carrying the
"possibly-relevant past" header and the pushed `#id` rows) and the assistant's
later messages (which cite memories by `#id`). Walk a session in order; a push of
#id is REFERENCED if the assistant cites #id after it AND before the same id is
pushed again — so each injection is credited only by a use that actually followed
it, and a re-push with no citation between counts as ignored.

Portable-pure where it can be: `attribute` is a function over ordered events; the
only host-specific edge is the transcript JSONL shape (`iter_events`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BLOCK_MARKER = "possibly-relevant past"   # stable substring of proactive.HEADER
_ID = re.compile(r"#(\d{1,6})")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return ""


def iter_events(path: str | Path):
    """Yield this transcript's ordered events as ("push", ids, channel) for each
    injected block and ("cite", ids, None) for each assistant message that cites
    memory ids. Order is file order (chronological append). Robust to malformed
    lines."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(d, dict):
            continue
        t = d.get("type")
        if t == "attachment":
            att = d.get("attachment") or {}
            content = att.get("content") or ""
            if isinstance(content, str) and BLOCK_MARKER in content:
                ids = {int(m) for m in _ID.findall(content)}
                if ids:
                    yield ("push", ids, att.get("hookEvent"))
        elif t == "assistant" and not d.get("isSidechain"):
            txt = _text_of((d.get("message") or {}).get("content"))
            ids = {int(m) for m in _ID.findall(txt)}
            if ids:
                yield ("cite", ids, None)


def attribute(events) -> dict:
    """Per-memory push/reference tally from one session's ordered events.

    Returns {id: {"impressions": n, "referenced": n}}. Each push is one
    impression; it is `referenced` iff a later assistant citation of that id
    occurs before the id is pushed again (precise per-injection attribution)."""
    tally: dict[int, dict[str, int]] = {}
    pending: dict[int, bool] = {}     # id -> an injection awaiting a citation

    def slot(i):
        return tally.setdefault(i, {"impressions": 0, "referenced": 0})

    for kind, ids, _chan in events:
        if kind == "push":
            for i in ids:
                slot(i)["impressions"] += 1
                pending[i] = True       # a prior un-cited push (if any) stays ignored
        elif kind == "cite":
            for i in ids:
                if pending.get(i):
                    slot(i)["referenced"] += 1
                    pending[i] = False
    return tally


def _merge(into: dict, more: dict) -> None:
    for i, c in more.items():
        s = into.setdefault(i, {"impressions": 0, "referenced": 0})
        s["impressions"] += c["impressions"]
        s["referenced"] += c["referenced"]


def transcript_paths(source: str | Path) -> list[Path]:
    """A single .jsonl, or every *.jsonl under a directory (one file = one
    session, so attribution never crosses sessions)."""
    p = Path(source).expanduser()
    if p.is_dir():
        # recurse: the host keeps one subdir per project, each holding session
        # files (~/.claude/projects/<project>/<session>.jsonl)
        return sorted(p.rglob("*.jsonl"))
    return [p] if p.exists() else []


def scan(source: str | Path) -> dict:
    """Aggregate push-usefulness across all sessions under `source`."""
    per_memory: dict[int, dict[str, int]] = {}
    sessions = 0
    for path in transcript_paths(source):
        evs = list(iter_events(path))
        if not evs:
            continue
        sessions += 1
        _merge(per_memory, attribute(evs))
    impressions = sum(c["impressions"] for c in per_memory.values())
    referenced = sum(c["referenced"] for c in per_memory.values())
    return {
        "source": str(source),
        "sessions": sessions,
        "memories_pushed": len(per_memory),
        "impressions": impressions,
        "referenced": referenced,
        "reference_rate": round(referenced / impressions, 4) if impressions else 0.0,
        "per_memory": per_memory,
    }


def outcomes_from_scan(scan_result: dict) -> dict:
    """Map each pushed id to a push-OUTCOME for the floor-stats join: "useful" if
    any of its pushes were referenced, "noise" if it was pushed but never
    referenced, else (not pushed) absent. This replaces the lifetime-recall_count
    proxy with what actually happened to the pushes."""
    out: dict[int, str] = {}
    for i, c in scan_result.get("per_memory", {}).items():
        if c["impressions"] <= 0:
            continue
        out[i] = "useful" if c["referenced"] > 0 else "noise"
    return out


def format_report(s: dict) -> str:
    out = [f"usefulness scan: {s['source']}",
           f"sessions: {s['sessions']}  memories pushed: {s['memories_pushed']}"]
    if not s["impressions"]:
        out.append("  (no injected blocks found — point --transcripts at the host's "
                   "session JSONL dir, e.g. ~/.claude/projects/<project>)")
        return "\n".join(out)
    out.append(f"push impressions: {s['impressions']}  referenced downstream: "
               f"{s['referenced']}  ({s['reference_rate']:.0%})")
    pm = s["per_memory"]
    chronic = sorted(((i, c) for i, c in pm.items()
                      if c["referenced"] == 0 and c["impressions"] >= 3),
                     key=lambda kv: -kv[1]["impressions"])[:12]
    if chronic:
        out.append("chronically pushed but NEVER referenced (noise — floor should rise):")
        for i, c in chronic:
            out.append(f"  #{i:<5} pushed {c['impressions']}, used 0")
    proven = sorted(((i, c) for i, c in pm.items() if c["referenced"] > 0),
                    key=lambda kv: -kv[1]["referenced"])[:8]
    if proven:
        out.append("most-referenced pushes (proven-useful):")
        for i, c in proven:
            out.append(f"  #{i:<5} pushed {c['impressions']}, used {c['referenced']}")
    return "\n".join(out)
