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

It also measures the OTHER channel, and that turned out to be the bigger one.
A memory can reach the model two ways: pushed (a hook injects it, paid for on
every session whether used or not) or PULLED (the agent runs recall/show/
timeline/brief itself, paid for only when asked for). Measuring only pushes and
calling the result "did memory help" credits the most expensive, least-referenced
channel with the whole question — on this store 74% of all citations were of
memories no push ever surfaced. Pulls are attributed the same way pushes are, on
one shared pending map, so a citation is credited to whichever delivery actually
preceded it and the channels can be compared on equal terms.

Portable-pure where it can be: `attribute` is a function over ordered events; the
only host-specific edge is the transcript JSONL shape (`iter_events`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BLOCK_MARKER = "possibly-relevant past"   # stable substring of proactive.HEADER
_ID = re.compile(r"#(\d{1,6})")
# A pull is recognized by the SHAPE OF ITS RESULT, not by parsing the command
# that produced it: the command arrives as free-form shell (variables, aliases,
# pipes, `-m fornixdb` vs the console script), and every reader verb —
# recall/timeline/show/brief/lineage — renders its rows the same way. Anchored
# at line start with an id then an ISO date, which "stored #123" and
# "#12 superseded by #13" cannot match, so writes are never counted as reads.
_PULL_ROW = re.compile(r"^#(\d{1,6})\s+\d{4}-\d{2}-\d{2}\s", re.M)


def _tool_result_text(block) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content
                        if isinstance(x, dict) and isinstance(x.get("text"), str))
    return ""


def _unwrap_hook_stdout(s: str) -> tuple[str, str | None]:
    """A tool-seam hook's stdout arrives as a JSON `hookSpecificOutput` wrapper
    whose `additionalContext` string IS the injected block — but ESCAPED, so a
    newline is the two characters `\\` + `n` and the `"\\nsettled: "` marker can
    never match the raw field. Unescape before any marker/size test (this bug
    silently credited every L5 settled push to L4 from v0.5.0 until 2026-07-18).
    Returns (block text, hookEventName) or (raw text, None) when not a wrapper."""
    t = s.strip()
    start = t.find("{")
    if start != -1:
        try:
            d = json.loads(t[start:])
        except (ValueError, TypeError):
            d = None
        if isinstance(d, dict):
            hso = d.get("hookSpecificOutput")
            if isinstance(hso, dict) and isinstance(hso.get("additionalContext"), str):
                ev = hso.get("hookEventName")
                return hso["additionalContext"], ev if isinstance(ev, str) else None
    return s, None


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
            # (UserPromptSubmit) puts it in `content` as plain text; L4/L5
            # (PostToolUse) put it in `stdout` as a hookSpecificOutput JSON
            # wrapper that must be UNWRAPPED before marker tests.
            att = d.get("attachment") or {}
            # ONE injection, recorded twice. On the UserPromptSubmit seam the
            # host echoes the hook's stdout back into `content`, so the same
            # block sits in both fields — but the model is shown it once.
            # Joining them made every L3 push cost double, which is how the
            # per-turn pulse came to look like the most expensive channel in the
            # ladder when it is close to the cheapest. The PostToolUse seams
            # carry the block in `stdout` alone and were never affected.
            wrapper_event = None
            stdout_block = None
            if isinstance(att.get("stdout"), str):
                stdout_block, wrapper_event = _unwrap_hook_stdout(att["stdout"])
            content = att.get("content") if isinstance(att.get("content"), str) else None
            if stdout_block and BLOCK_MARKER in stdout_block:
                # prefer the unwrapped stdout: it is the unescaped text, so its
                # length is the honest one
                text = stdout_block
            elif content and BLOCK_MARKER in content:
                text = content
            else:
                text = "\n".join(x for x in (content, stdout_block) if x)
            if BLOCK_MARKER in text:
                ids = {int(m) for m in _ID.findall(text)}
                if ids:
                    # An L5 SETTLED block carries its direction line; a degraded
                    # field block is L4 behavior and is fairly counted as L4.
                    ev = ("L5" if "\nsettled: " in text
                          else att.get("hookEvent") or wrapper_event)
                    # 4th field = the block's size in chars: the MEASURED context
                    # cost of this push (cite events stay 3-tuples — a citation
                    # costs nothing). Unescaped, so the cost is honest.
                    yield ("push", ids, ev, len(text))
        elif t == "user" and not d.get("isSidechain"):
            # An explicit pull: the agent ran a reader verb and its rows came
            # back as a tool result. Counted at the RESULT, so it does not
            # matter how the command was spelled.
            content = (d.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if not (isinstance(blk, dict) and blk.get("type") == "tool_result"):
                    continue
                text = _tool_result_text(blk)
                # A hook can append its injected block to a tool result (24 of
                # them in the live transcripts). Those ids are PUSHES and are
                # already counted from the attachment, so drop the whole result
                # rather than risk crediting a push to the pull channel — an
                # undercount is the honest direction for the channel being
                # newly measured.
                if BLOCK_MARKER in text:
                    continue
                ids = {int(m) for m in _PULL_ROW.findall(text)}
                if ids:
                    # 4th field = measured context cost, same contract as a push
                    yield ("pull", ids, "L1", len(text))
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


PULL_CHANNEL = "L1"     # the explicit-pull rung, in the ladder's own vocabulary


def attribute(events) -> tuple[dict, dict]:
    """Per-memory and per-CHANNEL delivery/reference tallies from one session's
    ordered events.

    Returns (per_memory, per_channel). Each push is one impression; it is
    `referenced` iff a later assistant citation of that id occurs before the id
    is DELIVERED again (precise per-injection attribution). A citation is
    credited to the CHANNEL of the delivery it satisfies, so L3, L4 and L5 each
    get a fair reference rate.

    Pulls run through the SAME pending map under the L1 channel, which is what
    makes the comparison honest: if a memory was pushed and the agent then
    pulled it anyway, the citation belongs to the pull, because the pull is what
    put it in front of the model at the moment it was used. Per-memory pull
    counts are kept in their own keys so that the push-only figures the
    suppression and floor joins depend on are unchanged by this."""
    per_memory: dict[int, dict[str, int]] = {}
    per_channel: dict[str, dict[str, int]] = {}
    # id -> ("push"|"pull", channel) of a delivery awaiting a citation
    pending: dict[int, tuple[str, str] | None] = {}

    def slot(d, k):        # per-memory: push and pull tallied separately
        return d.setdefault(k, {"impressions": 0, "referenced": 0,
                                "pull_impressions": 0, "pull_referenced": 0})

    def cslot(k):          # per-channel: one delivery is one impression
        return per_channel.setdefault(k, {"impressions": 0, "referenced": 0})

    for ev in events:
        kind, ids, chan = ev[0], ev[1], ev[2]   # a delivery carries a 4th field (chars)
        if kind == "push":
            ch = _channel(chan)
            for i in ids:
                slot(per_memory, i)["impressions"] += 1
                cslot(ch)["impressions"] += 1
                pending[i] = ("push", ch)   # a prior un-cited delivery stays ignored
        elif kind == "pull":
            for i in ids:
                slot(per_memory, i)["pull_impressions"] += 1
                cslot(PULL_CHANNEL)["impressions"] += 1
                pending[i] = ("pull", PULL_CHANNEL)
        elif kind == "cite":
            for i in ids:
                got = pending.get(i)
                if got is None:
                    continue
                how, ch = got
                if how == "push":
                    slot(per_memory, i)["referenced"] += 1
                else:
                    slot(per_memory, i)["pull_referenced"] += 1
                cslot(ch)["referenced"] += 1
                pending[i] = None
    return per_memory, per_channel


def _merge(into: dict, more: dict) -> None:
    """Sum tallies key by key. Key-agnostic on purpose: per-memory rows carry
    push AND pull counts, per-channel rows carry only the two, and neither
    should acquire the other's fields by being merged."""
    for i, c in more.items():
        s = into.setdefault(i, {})
        for k, v in c.items():
            s[k] = s.get(k, 0) + v


def transcript_paths(source: str | Path) -> list[Path]:
    """A single .jsonl, or every *.jsonl under a directory (one file = one
    session, so attribution never crosses sessions)."""
    p = Path(source).expanduser()
    if p.is_dir():
        # recurse: the host keeps one subdir per project, each holding session
        # files (~/.claude/projects/<project>/<session>.jsonl)
        return sorted(p.rglob("*.jsonl"))
    return [p] if p.exists() else []


def _session_start(path: Path):
    """First parseable line timestamp (UTC datetime) — the session's start.
    None when the file carries no timestamps (can't be dated)."""
    from datetime import datetime
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines[:200]:
        try:
            d = json.loads(line.strip())
        except (ValueError, TypeError):
            continue
        ts = d.get("timestamp") if isinstance(d, dict) else None
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def scan(source: str | Path, since_days: int | None = None) -> dict:
    """Aggregate push-usefulness across all sessions under `source`.

    `since_days` windows at SESSION granularity (a transcript counts iff its
    first timestamped line is inside the window) so push→cite attribution never
    splits a session; undatable files are excluded from a windowed scan."""
    per_memory: dict[int, dict[str, int]] = {}
    per_channel: dict[str, dict[str, int]] = {}
    chars_by_channel: dict[str, int] = {}
    sessions = 0
    cutoff = None
    if since_days is not None:
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    for path in transcript_paths(source):
        if cutoff is not None:
            start = _session_start(path)
            if start is None or start < cutoff:
                continue
        evs = list(iter_events(path))
        if not evs:
            continue
        sessions += 1
        pm, pc = attribute(evs)
        _merge(per_memory, pm)
        _merge(per_channel, pc)
        for ev in evs:
            if len(ev) > 3 and ev[0] in ("push", "pull"):
                ch = PULL_CHANNEL if ev[0] == "pull" else _channel(ev[2])
                chars_by_channel[ch] = chars_by_channel.get(ch, 0) + ev[3]
    impressions = sum(c["impressions"] for c in per_memory.values())
    referenced = sum(c["referenced"] for c in per_memory.values())
    # Kept SEPARATE from the push totals on purpose. The referenced-PUSH rate is
    # the number this project has tracked release over release, and folding
    # pulls into it would silently redefine the health metric mid-history.
    pull_impressions = sum(c.get("pull_impressions", 0) for c in per_memory.values())
    pull_referenced = sum(c.get("pull_referenced", 0) for c in per_memory.values())
    from .tokens import EST_CHARS_PER_TOKEN
    for name, c in per_channel.items():
        c["reference_rate"] = (round(c["referenced"] / c["impressions"], 4)
                               if c["impressions"] else 0.0)
        c["injected_tokens"] = round(
            chars_by_channel.get(name, 0) / EST_CHARS_PER_TOKEN)
        # The figure that lets a push rung and the pull channel be compared at
        # all: what a downstream reference COST on this channel. None when the
        # channel earned no reference — a rate of zero has no cost-per-use.
        c["tokens_per_reference"] = (
            round(c["injected_tokens"] / c["referenced"]) if c["referenced"]
            else None)
    pulled_chars = chars_by_channel.get(PULL_CHANNEL, 0)
    injected_chars = sum(v for k, v in chars_by_channel.items()
                         if k != PULL_CHANNEL)
    return {
        "source": str(source),
        "since_days": since_days,
        "sessions": sessions,
        "memories_pushed": len(per_memory),
        "impressions": impressions,
        "referenced": referenced,
        "reference_rate": round(referenced / impressions, 4) if impressions else 0.0,
        # MEASURED context cost of every injected block found (not an estimate;
        # chars→tokens is the only approximation)
        "injected_chars": injected_chars,
        "injected_tokens": round(injected_chars / EST_CHARS_PER_TOKEN),
        # The pull channel, measured the same way and reported alongside — never
        # merged into the push figures above.
        "memories_pulled": sum(1 for c in per_memory.values()
                               if c.get("pull_impressions")),
        "pull_impressions": pull_impressions,
        "pull_referenced": pull_referenced,
        "pull_rate": (round(pull_referenced / pull_impressions, 4)
                      if pull_impressions else 0.0),
        "pulled_chars": pulled_chars,
        "pulled_tokens": round(pulled_chars / EST_CHARS_PER_TOKEN),
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
    # PUSHED ids only. per_memory also carries memories that were merely pulled;
    # writing them a 0 here would clear use-credit a push had legitimately
    # earned, on the strength of a window in which nothing pushed them.
    return {int(i): int(c["referenced"])
            for i, c in scan_result.get("per_memory", {}).items()
            if c.get("impressions", 0) > 0}


def format_report(s: dict) -> str:
    window = (f"  (window: last {s['since_days']} days, session-granularity)"
              if s.get("since_days") is not None else "")
    out = [f"usefulness scan: {s['source']}{window}",
           f"sessions: {s['sessions']}  memories pushed: {s['memories_pushed']}"]
    if not s["impressions"]:
        out.append("  (no injected blocks found — point --transcripts at the host's "
                   "session JSONL dir, e.g. ~/.claude/projects/<project>)")
        return "\n".join(out)
    out.append(f"push impressions: {s['impressions']}  referenced downstream: "
               f"{s['referenced']}  ({s['reference_rate']:.0%})")
    if s.get("pull_impressions"):
        out.append(f"pull  deliveries: {s['pull_impressions']}  referenced "
                   f"downstream: {s['pull_referenced']}  ({s['pull_rate']:.0%})"
                   f"   [{s['memories_pulled']} memories]")
    out.append(f"context cost: pushes {s['injected_tokens']:,} tok "
               f"(paid every session, used or not), pulls "
               f"{s.get('pulled_tokens', 0):,} tok (paid only when asked for)")
    bc = s.get("by_channel") or {}
    if bc:
        out.append("by channel (L1 = explicit pull, L3 = per-turn, "
                   "L4 = rhythmic in-thought, L5 = settled field):")
        for ch in sorted(bc):
            c = bc[ch]
            verb = "pulled" if ch == PULL_CHANNEL else "pushed"
            out.append(f"  {ch}  {verb} {c['impressions']:<5} referenced "
                       f"{c['referenced']:<4} ({c['reference_rate']:.0%})")
        if {"L3", "L4"} <= set(bc):
            out.append("  (note: a citation credits the most-recent DELIVERY, so "
                       "when L3 and L4 push the same id the split leans toward "
                       "L4 — and a pull of an already-pushed id takes the credit, "
                       "because the pull is what put it in front of the model.)")
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
    # The memories the push channels never surfaced but the agent went and got
    # anyway — the clearest statement of what the push side is missing.
    missed = sorted(((i, c) for i, c in pm.items()
                     if c.get("pull_referenced", 0) > 0 and not c["impressions"]),
                    key=lambda kv: -kv[1]["pull_referenced"])[:8]
    if missed:
        out.append("referenced after a PULL, never pushed at all "
                   "(the push channels did not surface these):")
        for i, c in missed:
            out.append(f"  #{i:<5} pulled {c['pull_impressions']}, "
                       f"used {c['pull_referenced']}")
    return "\n".join(out)
