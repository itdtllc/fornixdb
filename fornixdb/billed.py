"""Billed-share report: what FornixDB costs in the host's BILLED tokens,
measured from the host's own session transcripts (FornixDB #488).

`fornixdb tokens` reports CONTEXT-SPACE — each surface counted once. But a
Claude-style host re-sends the whole conversation on EVERY API request, and
every tool call is its own request, so anything FornixDB puts in context is
billed again on each later request of that session. The honest unit is the
TOKEN-TURN: block tokens × requests that re-read it. That multiplier runs
30–150+ per session, which is why a host's per-server usage panel shows a
number 60–90× larger than the once-counted footprint — the two are different
units, not a disagreement.

This module measures token-turns directly against the per-request `usage`
fields the host writes into its transcripts (input + cache_read +
cache_creation — the same numbers its own usage panel aggregates), so the
share it prints is comparable to what the host attributes.

What counts as FornixDB content (entering at request i, re-read by requests
i..N of that session):
  * proactive push blocks — hook `attachment` records carrying the block
    header (same detection as usefulness_scan),
  * mcp__<server>__* tool calls (the model's tool_use input), and
  * their tool_results, matched by tool_use id.

The RESIDENT surface (tool schemas + server instructions) is MODELED, not
measured: transcripts don't record whether the host loaded schemas eagerly or
deferred them behind tool-search, so it is reported as a 0…full band on top
of the measured content share.

Keep perspective in the output: nearly all token-turns are prompt-cache reads
(~0.1× input price), and a resident block's share of a session is simply
its size ÷ the average context size — small in dollars even when the
percentage looks noticeable.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from .tokens import estimate_tokens
from .usefulness_scan import BLOCK_MARKER, transcript_paths

DEFAULT_TRANSCRIPTS = "~/.claude/projects"
MCP_SERVER = "fornixdb"
TOOLUSE_OVERHEAD_TOKENS = 50   # per tool_use block: name + envelope around the input


def _blob(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, default=str)


def analyze_transcript(path: str | Path, server: str = MCP_SERVER) -> dict | None:
    """One session's billed totals and FornixDB content events.

    Returns {"requests": N, "billed": total, "events": [(req_index, tokens,
    kind)]} — req_index is how many requests had completed when the content
    entered context (it is re-read by requests req_index..N) — or None for a
    transcript with no usage records (nothing was billed).
    """
    prefix = f"mcp__{server}"
    reqs = 0
    billed = 0
    events: list[tuple[int, int, str]] = []
    tool_ids: set[str] = set()
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
        msg = d.get("message") or {}
        if t == "assistant" and isinstance(msg.get("usage"), dict):
            u = msg["usage"]
            reqs += 1
            billed += (u.get("input_tokens") or 0) \
                + (u.get("cache_read_input_tokens") or 0) \
                + (u.get("cache_creation_input_tokens") or 0)
            for c in msg.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use" \
                        and str(c.get("name", "")).startswith(prefix):
                    tool_ids.add(c.get("id"))
                    events.append((reqs, estimate_tokens(_blob(c.get("input", {})))
                                   + TOOLUSE_OVERHEAD_TOKENS, "call"))
        elif t == "user":
            content = msg.get("content")
            for c in (content if isinstance(content, list) else []):
                if isinstance(c, dict) and c.get("type") == "tool_result" \
                        and c.get("tool_use_id") in tool_ids:
                    events.append((reqs, estimate_tokens(_blob(c.get("content", ""))),
                                   "result"))
        elif t == "attachment":
            att = d.get("attachment") or {}
            # push blocks land in `content` (UserPromptSubmit) or `stdout`
            # (PostToolUse) — same shape usefulness_scan reads
            text = "\n".join(att.get(k) for k in ("content", "stdout")
                             if isinstance(att.get(k), str))
            if BLOCK_MARKER in text:
                # same channel labels as usefulness_scan: L5 = settled field
                # block, L3 = prompt pulse, L4 = tool-seam pulse
                ch = ("push:L5" if "\nsettled: " in text else
                      "push:L3" if att.get("hookEvent") == "UserPromptSubmit"
                      else "push:L4")
                events.append((reqs, estimate_tokens(text), ch))
    if not billed:
        return None
    return {"requests": reqs, "billed": billed, "events": events}


def token_turns(events: list[tuple[int, int, str]], requests: int) -> int:
    """Content entering after request i sits in context for requests i+1..N:
    N−i re-reads (a pre-session push has i=0 and is read by all N)."""
    return sum(tok * max(0, requests - idx) for idx, tok, _ in events)


def report(store, source: str | Path | None = None,
           since_days: float | None = None, server: str = MCP_SERVER) -> dict:
    """Billed-share across the host's transcripts, plus the modeled resident
    band from THIS store's advertised surface."""
    src = Path(source or DEFAULT_TRANSCRIPTS).expanduser()
    cutoff = (datetime.now() - timedelta(days=since_days)) if since_days else None

    sessions = []
    tot_billed = tot_fnx = tot_reqs = 0
    kinds: dict[str, int] = {}
    for p in transcript_paths(src):
        if cutoff and datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
            continue
        r = analyze_transcript(p, server=server)
        if not r:
            continue
        fnx = token_turns(r["events"], r["requests"])
        for _, tok, kind in r["events"]:
            kinds[kind] = kinds.get(kind, 0) + tok
        sessions.append({"session": p.stem[:12], "requests": r["requests"],
                         "billed_tokens": r["billed"], "fornixdb_token_turns": fnx,
                         "share_pct": round(100 * fnx / r["billed"], 2)})
        tot_billed += r["billed"]
        tot_fnx += fnx
        tot_reqs += r["requests"]

    out = {
        "source": str(src),
        "since_days": since_days,
        "sessions": sessions,
        "content": {
            "token_turns": tot_fnx,
            "one_time_tokens_by_kind": kinds,
            "billed_tokens_total": tot_billed,
            "share_pct": round(100 * tot_fnx / tot_billed, 2) if tot_billed else None,
        },
    }
    if tot_reqs and tot_billed:
        # resident band: transcripts don't say whether the host kept schemas
        # loaded (eager) or deferred them, so give both ends
        from .tokens import report as tokens_report
        fixed = tokens_report(store)["fixed_per_session"]
        resident_full = (fixed["mcp_tool_schemas"]["tokens"]
                         + fixed["mcp_instructions"]["tokens"])
        instr_only = fixed["mcp_instructions"]["tokens"]
        out["resident_band"] = {
            "per_request_tokens": {"deferred_schemas": instr_only,
                                   "eager_schemas": resident_full},
            "token_turns": {"deferred_schemas": instr_only * tot_reqs,
                            "eager_schemas": resident_full * tot_reqs},
            "share_pct": {
                "deferred_schemas": round(100 * instr_only * tot_reqs / tot_billed, 2),
                "eager_schemas": round(100 * resident_full * tot_reqs / tot_billed, 2)},
        }
        out["avg_context_tokens"] = round(tot_billed / tot_reqs)
    return out


def format_report(r: dict) -> str:
    lines = [f"FornixDB billed share, measured from host transcripts "
             f"({r['source']}"
             + (f", last {r['since_days']:g} days" if r.get("since_days") else "")
             + ")", ""]
    if not r["sessions"]:
        return lines[0] + "\n  no transcripts with billed usage found"
    hdr = f"  {'session':14} {'requests':>8} {'billed tok':>13} {'fornixdb':>10} {'share':>7}"
    lines += [hdr, "  " + "-" * (len(hdr) - 2)]
    for s in r["sessions"]:
        lines.append(f"  {s['session']:14} {s['requests']:8d} "
                     f"{s['billed_tokens']:13,d} {s['fornixdb_token_turns']:10,d} "
                     f"{s['share_pct']:6.2f}%")
    c = r["content"]
    lines += ["",
              f"CONTENT (measured): pushes + tool calls + results, re-read on every "
              f"later request",
              f"  {c['token_turns']:,} of {c['billed_tokens_total']:,} billed tokens "
              f"= {c['share_pct']}%   (entered once as "
              + " + ".join(f"{v:,} {k}" for k, v in
                           sorted(c["one_time_tokens_by_kind"].items())) + " tokens)"]
    band = r.get("resident_band")
    if band:
        s = band["share_pct"]
        p = band["per_request_tokens"]
        lines += ["",
                  "RESIDENT (modeled — transcripts don't record schema loading):",
                  f"  instructions only (host defers schemas)  ~{p['deferred_schemas']:>5} "
                  f"tok/request = {s['deferred_schemas']}%",
                  f"  instructions + all schemas (eager host)  ~{p['eager_schemas']:>5} "
                  f"tok/request = {s['eager_schemas']}%",
                  "",
                  f"avg context/request ~{r['avg_context_tokens']:,} tokens — a resident "
                  "block's share is simply its size ÷ this number, every request.",
                  "Nearly all of these token-turns are prompt-cache READS (~0.1× input "
                  "price): the % is a share of tokens processed, not of dollars."]
    return "\n".join(lines)
