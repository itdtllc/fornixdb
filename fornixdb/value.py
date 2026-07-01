"""One-shot "how useful has FornixDB been?" — composes the three existing
signals into a single summary so any session can answer it in one call:

  COST  — tokens.report: fixed per-session + per-call footprint.
  REACH — benefit.coverage: how much of the store the flat markdown can't give
          (optional; needs the host's memory files).
  USED  — usefulness_scan: referenced-push rate from the host's transcripts —
          the honest "did pushed memory actually get used" signal (optional).

Read-only; wraps existing functions, adds no schema or ranking behavior. REACH
and USED are optional so it still answers on any store / air-gapped endpoint.
"""
from __future__ import annotations

DEFAULT_TRANSCRIPTS = "~/.claude/projects"


def report(store, *, transcripts: str | None = None,
           memory_md: str | None = None, memory_dir: str | None = None) -> dict:
    from . import tokens
    out: dict = {"memories": store.stats().get("memories"),
                 "cost": tokens.report(store)}
    if memory_md and memory_dir:
        from . import benefit
        base = benefit.scan_flat_baseline(memory_md, memory_dir)
        out["reach"] = benefit.coverage(store, base)
    if transcripts:
        from . import usefulness_scan
        s = usefulness_scan.scan(transcripts)
        out["used"] = {"sessions": s["sessions"], "impressions": s["impressions"],
                       "referenced": s["referenced"],
                       "reference_rate": s["reference_rate"],
                       "by_channel": s.get("by_channel", {})}
    return out


def format_report(r: dict) -> str:
    c = r.get("cost", {})
    fixed = (c.get("fixed_per_session", {}) or {}).get("total_tokens")
    per = c.get("per_call", {}) or {}
    recall_t = (per.get("recall_default_limit_5", {}) or {}).get("tokens")
    brief_t = (per.get("brief", {}) or {}).get("tokens")
    schemas = (c.get("fixed_per_session", {}) or {}).get("mcp_tool_schemas", {})
    out = ["How useful has FornixDB been?",
           f"  Store: {r.get('memories')} memories", ""]

    out.append(f"  COST  ~ {fixed} tokens fixed/session"
               + (f" ({schemas.get('tools')} MCP tool schemas)" if schemas else "")
               + (f"; ~{recall_t}/recall" if recall_t is not None else "")
               + (f", ~{brief_t}/brief" if brief_t is not None else "")
               + " — paid only when used.")

    reach = r.get("reach")
    if reach:
        b = reach.get("buckets", {})
        out.append(f"  REACH ~ {reach.get('pct_marginal_content')}% "
                   f"({b.get('fornix_only')} of {reach.get('total')}) absent from "
                   f"the flat memory index — incl. all episodic (no timeline axis "
                   f"in flat markdown).")
    else:
        out.append("  REACH   (not measured — pass --memory-md/--memory-dir)")

    used = r.get("used")
    if used and used.get("impressions"):
        bc = used.get("by_channel", {}) or {}
        chans = " ".join(f"{k} {v.get('reference_rate', 0):.0%}"
                         for k, v in sorted(bc.items()))
        out.append(f"  USED  ~ {used['reference_rate']:.0%} of proactive pushes "
                   f"referenced downstream ({chans}) over {used['sessions']} "
                   f"sessions — the honest 'did memory help' signal.")
    else:
        out.append("  USED    (no injected blocks found in transcripts)")
    return "\n".join(out)
