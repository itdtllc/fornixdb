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

# The savings side of NET cannot be measured (the session-without-FornixDB
# counterfactual doesn't exist), so it is an EXPLICIT assumption band, printed
# in the report: what one referenced push replaces — the user re-explaining
# history, or the AI re-reading files / re-deriving a past decision.
REDERIVE_TOKENS = {"low": 300, "mid": 1500, "high": 5000}


def report(store, *, transcripts: str | None = None,
           memory_md: str | None = None, memory_dir: str | None = None) -> dict:
    from . import tokens
    out: dict = {"memories": store.stats().get("memories"),
                 "cost": tokens.report(store)}
    if memory_md and memory_dir:
        from . import benefit
        base = benefit.scan_flat_baseline(memory_md, memory_dir)
        out["reach"] = benefit.coverage(store, base)
    from .doctor import _OFF
    from .multistore import get_config
    out["floor_log"] = ("off" if (get_config(store, "floor_log", "off") or
                                  "off").strip().lower() in _OFF else "on")
    if transcripts:
        from . import usefulness_scan
        s = usefulness_scan.scan(transcripts)
        out["used"] = {"sessions": s["sessions"], "impressions": s["impressions"],
                       "referenced": s["referenced"],
                       "reference_rate": s["reference_rate"],
                       "injected_tokens": s.get("injected_tokens", 0),
                       "by_channel": s.get("by_channel", {})}
        if s["sessions"]:
            out["net"] = _net(out["cost"], s)
    return out


def _net(cost: dict, scan: dict) -> dict:
    """Net tokens/session = assumed savings − measured cost.

    Cost side is measured: the fixed integration surfaces plus the actual
    injected push blocks found in the transcripts. Savings side is the
    REDERIVE_TOKENS assumption band applied to the measured count of pushes
    that were actually referenced downstream. Both context-space figures;
    per-turn re-sends are mostly prompt-cached by the host."""
    sess = scan["sessions"]
    fixed = cost["fixed_per_session"]["total_tokens"]
    push_ps = round(scan.get("injected_tokens", 0) / sess)
    refs_ps = scan["referenced"] / sess
    return {
        "sessions_scanned": sess,
        "measured_cost_per_session": {"fixed_surfaces": fixed,
                                      "injected_pushes": push_ps,
                                      "total": fixed + push_ps},
        "referenced_pushes_per_session": round(refs_ps, 2),
        "assumed_tokens_saved_per_referenced_push": dict(REDERIVE_TOKENS),
        "net_tokens_per_session": {
            k: round(refs_ps * v) - (fixed + push_ps)
            for k, v in REDERIVE_TOKENS.items()},
        "not_counted": ("explicit pull results (recall/brief/show — see per_call "
                        "cost); session-end auto-capture costs 0 prompt tokens "
                        "(post-session OS process); timeline answers have no "
                        "re-derivation path, so their value exceeds any token "
                        "count"),
    }


def format_report(r: dict) -> str:
    c = r.get("cost", {})
    fixed = (c.get("fixed_per_session", {}) or {}).get("total_tokens")
    per = c.get("per_call", {}) or {}
    recall_t = (per.get("recall_default_limit_5", {}) or {}).get("tokens")
    brief_t = (per.get("brief", {}) or {}).get("tokens")
    schemas = (c.get("fixed_per_session", {}) or {}).get("mcp_tool_schemas", {})

    # NET verdict first — the owner's question is "is memory saving me tokens
    # or costing me tokens, and how much".
    net = r.get("net")
    if net:
        n = net["net_tokens_per_session"]
        cps = net["measured_cost_per_session"]
        band = net["assumed_tokens_saved_per_referenced_push"]
        mid = n["mid"]
        head = (f"Estimated tokens SAVED: ~{mid:,}/session"
                if mid >= 0 else
                f"Estimated EXTRA tokens: ~{-mid:,}/session")
        out = [head + f" (mid assumption; low {n['low']:+,} … high {n['high']:+,})",
               "",
               "  Supporting data "
               f"(measured over {net['sessions_scanned']} sessions):",
               f"    cost/session (measured)   ~{cps['total']:,} = "
               f"{cps['fixed_surfaces']:,} fixed surfaces"
               + (f" ({schemas.get('tools')} tool schemas + instructions + "
                  f"startup)" if schemas else "")
               + f" + {cps['injected_pushes']:,} injected push blocks",
               f"    use/session (measured)     {net['referenced_pushes_per_session']} "
               f"pushes referenced downstream",
               f"    saving/reference (ASSUMED) {band['low']:,} / {band['mid']:,} / "
               f"{band['high']:,} tokens (low/mid/high) — the re-derivation or "
               "re-explaining one referenced push replaces; printed, not measured",
               f"    net = use × assumption − cost; a true measured savings "
               "number is impossible (no without-memory session to compare).",
               f"    not counted: {net['not_counted']}",
               ""]
    else:
        out = [f"Estimated net tokens: unknown — no sessions scanned; measured "
               f"fixed cost is ~{fixed if fixed is not None else '?'} "
               "tokens/session.", ""]

    out += ["How useful has FornixDB been?",
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

    if r.get("floor_log") == "off":
        out.append("")
        out.append("  Logging is OFF — `fornixdb config floor_log on` records "
                   "per-push floor decisions and per-beat field telemetry "
                   "(floor-stats / field-stats), adding push-suppression and "
                   "per-beat detail this readout can't see from transcripts "
                   "alone.")
    elif r.get("floor_log") == "on":
        out.append("")
        out.append("  Logging is ON — `fornixdb floor-stats` / `field-stats` "
                   "break down the push pipeline behind these numbers.")
    return "\n".join(out)
