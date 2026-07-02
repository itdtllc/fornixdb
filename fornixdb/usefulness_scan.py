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
            # The injected block lands in different fields per channel: L3
            # (UserPromptSubmit) puts it in `content`; L4 (PostToolUse) puts it in
            # `stdout` as a hookSpecificOutput JSON string. Read both.
            att = d.get("attachment") or {}
            text = "\n".join(att.get(k) for k in ("content", "stdout")
                             if isinstance(att.get(k), str))
            if BLOCK_MARKER in text:
                ids = {int(m) for m in _ID.findall(text)}
                if ids:
                    # An L5 SETTLED block carries its direction line; a degraded
                    # field block is L4 behavior and is fairly counted as L4.
                    ev = "L5" if "\nsettled: " in text else att.get("hookEvent")
                    yield ("push", ids, ev)
        elif t == "assistant" and not d.get("isSidechain"):
            txt = _text_of((d.get("message") or {}).get("content"))
            # An assistant message that REPRODUCES the block (quoting/summarizing
            # it) is not citing memories — skip it so its ids aren't miscounted.
            if BLOCK_MARKER in txt:
                continue
            ids = {int(m) for m in _ID.findall(txt)}
            if ids:
                yield ("cite", ids, None)


def _channel(raw) -> str:
    """Normalize a push's hookEvent to a rung label: UserPromptSubmit = L3 (one
    pulse per turn), any tool-call seam = L4 (rhythmic in-thought). "L5" arrives
    pre-labeled from the settled-block marker (iter_events) — the gate measures
    whether SETTLING earns references, so only settled blocks count as L5."""
    if raw == "L5":
        return "L5"
    return "L3" if raw == "UserPromptSubmit" else "L4"


def attribute(events) -> tuple[dict, dict]:
    """Per-memory and per-CHANNEL push/reference tallies from one session's
    ordered events.

    Returns (per_memory, per_channel), each {key: {"impressions", "referenced"}}.
    Each push is one impression; it is `referenced` iff a later assistant citation
    of that id occurs before the id is pushed again (precise per-injection
    attribution). A citation is credited to the CHANNEL of the injection it
    satisfies, so L3 and L4 each get a fair reference rate."""
    per_memory: dict[int, dict[str, int]] = {}
    per_channel: dict[str, dict[str, int]] = {}
    pending: dict[int, str] = {}      # id -> channel of an injection awaiting a cite

    def slot(d, k):
        return d.setdefault(k, {"impressions": 0, "referenced": 0})

    for kind, ids, chan in events:
        if kind == "push":
            ch = _channel(chan)
            for i in ids:
                slot(per_memory, i)["impressions"] += 1
                slot(per_channel, ch)["impressions"] += 1
                pending[i] = ch         # a prior un-cited push (if any) stays ignored
        elif kind == "cite":
            for i in ids:
                ch = pending.get(i)
                if ch is not None:
                    slot(per_memory, i)["referenced"] += 1
                    slot(per_channel, ch)["referenced"] += 1
                    pending[i] = None
    return per_memory, per_channel


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
    per_channel: dict[str, dict[str, int]] = {}
    sessions = 0
    for path in transcript_paths(source):
        evs = list(iter_events(path))
        if not evs:
            continue
        sessions += 1
        pm, pc = attribute(evs)
        _merge(per_memory, pm)
        _merge(per_channel, pc)
    impressions = sum(c["impressions"] for c in per_memory.values())
    referenced = sum(c["referenced"] for c in per_memory.values())
    for c in per_channel.values():
        c["reference_rate"] = (round(c["referenced"] / c["impressions"], 4)
                               if c["impressions"] else 0.0)
    return {
        "source": str(source),
        "sessions": sessions,
        "memories_pushed": len(per_memory),
        "impressions": impressions,
        "referenced": referenced,
        "reference_rate": round(referenced / impressions, 4) if impressions else 0.0,
        "by_channel": per_channel,
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


def referenced_counts_from_scan(scan_result: dict) -> dict[int, int]:
    """Map each pushed id to how many of its pushes were referenced downstream —
    the use-credit `MemoryStore.record_referenced` materializes into the store so
    `effective_floor` stops treating proven-useful pushes as ignored noise. Every
    pushed id is included (0 for never-referenced) so an `--apply` pass also resets
    the credit of a memory that has since gone quiet (idempotent absolute set)."""
    return {int(i): int(c["referenced"])
            for i, c in scan_result.get("per_memory", {}).items()}


def format_report(s: dict) -> str:
    out = [f"usefulness scan: {s['source']}",
           f"sessions: {s['sessions']}  memories pushed: {s['memories_pushed']}"]
    if not s["impressions"]:
        out.append("  (no injected blocks found — point --transcripts at the host's "
                   "session JSONL dir, e.g. ~/.claude/projects/<project>)")
        return "\n".join(out)
    out.append(f"push impressions: {s['impressions']}  referenced downstream: "
               f"{s['referenced']}  ({s['reference_rate']:.0%})")
    bc = s.get("by_channel") or {}
    if bc:
        out.append("by channel (L3 = per-turn, L4 = rhythmic in-thought, "
                   "L5 = settled field):")
        for ch in sorted(bc):
            c = bc[ch]
            out.append(f"  {ch}  pushed {c['impressions']:<5} referenced "
                       f"{c['referenced']:<4} ({c['reference_rate']:.0%})")
        if {"L3", "L4"} <= set(bc):
            out.append("  (note: a citation credits the most-recent injection, so "
                       "when L3 and L4 push the same id the split leans toward L4.)")
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
