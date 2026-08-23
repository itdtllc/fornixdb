"""One-shot "how useful has FornixDB been?" — composes the three existing
signals into a single summary so any session can answer it in one call:

  COST  — tokens.report: fixed per-session + per-call footprint.
  REACH — benefit.coverage: how much of the store the flat markdown can't give
          (optional; needs the host's memory files).
  USED  — usefulness_scan: reference rate from the host's transcripts, for BOTH
          channels — memories FornixDB pushed and memories the agent pulled
          for itself — the honest "did memory actually get used" signal, with
          the cost each channel paid per use (optional).

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
                       "pull_impressions": s.get("pull_impressions", 0),
                       "pull_referenced": s.get("pull_referenced", 0),
                       "pull_rate": s.get("pull_rate", 0.0),
                       "pulled_tokens": s.get("pulled_tokens", 0),
                       "by_channel": s.get("by_channel", {})}
        if s["sessions"]:
            out["net"] = _net(out["cost"], s)
    return out


def _net(cost: dict, scan: dict) -> dict:
    """Net tokens/session = assumed savings − measured cost.

    Cost side is measured: the fixed integration surfaces, the injected push
    blocks, and the results of the agent's own pulls — all three found in the
    transcripts. Savings side is the REDERIVE_TOKENS assumption band applied to
    the measured count of deliveries actually referenced downstream, from EITHER
    channel.

    Both channels or neither. Counting push cost but not pull cost, and push
    benefit but not pull benefit, half-counted both sides and let the verdict
    rest on the more expensive, less-referenced channel — while three quarters
    of the demonstrated value sat outside the frame. A pull is cheaper per use
    (paid only when asked for) but it is not free, and it earns references at
    roughly twice the rate, so both belong in the same sum.

    Both sides are CONTEXT-SPACE figures — each token counted once. The host
    re-reads everything on every API request (token-turns), so a usage panel
    will show numbers 30-150x larger than these; but that multiplier applies
    to BOTH sides (re-derived content would sit in context and be re-read the
    same way), so the net verdict's direction survives the unit change. For
    the billed view itself, see `fornixdb tokens --billed`."""
    sess = scan["sessions"]
    fixed = cost["fixed_per_session"]["total_tokens"]
    push_ps = round(scan.get("injected_tokens", 0) / sess)
    pull_ps = round(scan.get("pulled_tokens", 0) / sess)
    refs_ps = scan["referenced"] / sess
    pull_refs_ps = scan.get("pull_referenced", 0) / sess
    used_ps = refs_ps + pull_refs_ps
    total_cost = fixed + push_ps + pull_ps
    bc = scan.get("by_channel") or {}
    return {
        "sessions_scanned": sess,
        "measured_cost_per_session": {"fixed_surfaces": fixed,
                                      "injected_pushes": push_ps,
                                      "pull_results": pull_ps,
                                      "total": total_cost},
        "referenced_pushes_per_session": round(refs_ps, 2),
        "referenced_pulls_per_session": round(pull_refs_ps, 2),
        "referenced_per_session": round(used_ps, 2),
        # what a downstream reference cost on each channel — the like-for-like
        # comparison between paying up front (push) and paying on demand (pull)
        "tokens_per_reference_by_channel": {
            ch: c.get("tokens_per_reference") for ch, c in sorted(bc.items())},
        "assumed_tokens_saved_per_referenced_push": dict(REDERIVE_TOKENS),
        "net_tokens_per_session": {
            k: round(used_ps * v) - total_cost
            for k, v in REDERIVE_TOKENS.items()},
        "not_counted": ("session-end auto-capture costs 0 prompt tokens "
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
               + f" + {cps['injected_pushes']:,} injected push blocks"
               + f" + {cps.get('pull_results', 0):,} pull results",
               f"    use/session (measured)     "
               f"{net.get('referenced_per_session', 0)} deliveries referenced "
               f"downstream ({net['referenced_pushes_per_session']} pushed, "
               f"{net.get('referenced_pulls_per_session', 0)} pulled)",
               f"    saving/reference (ASSUMED) {band['low']:,} / {band['mid']:,} / "
               f"{band['high']:,} tokens (low/mid/high) — the re-derivation or "
               "re-explaining one referenced push replaces; printed, not measured",
               f"    net = use × assumption − cost; a true measured savings "
               "number is impossible (no without-memory session to compare)."]
        tpr = net.get("tokens_per_reference_by_channel") or {}
        priced = {c: v for c, v in tpr.items() if v}
        if priced:
            out.append("    cost per reference, by channel (lower is better — "
                       "L1 is the agent asking, L3/L4/L5 are FornixDB offering):")
            for ch, v in sorted(priced.items(), key=lambda kv: kv[1]):
                how = "pulled" if ch == "L1" else "pushed"
                out.append(f"      {ch}  ~{v:,} tokens per downstream "
                           f"reference ({how})")
        out += [
               "    units: context-space, each token counted ONCE. Hosts re-read "
               "context every API request, so usage panels show ~30-150x these "
               "figures — on both sides equally (`fornixdb tokens --billed` "
               "measures that view).",
               f"    not counted: {net['not_counted']}",
               ""]
    else:
        out = [f"Estimated net tokens: unknown — no sessions scanned; measured "
               f"fixed cost is ~{fixed if fixed is not None else '?'} "
               "tokens/session.", ""]

    out += ["How useful has FornixDB been?",
            f"  Store: {r.get('memories')} memories", ""]

    out.append(f"  COST  ~ {fixed} tokens resident"
               + (f" ({schemas.get('tools')} MCP tool schemas)" if schemas else "")
               + " — re-read by the host every request"
               + (f"; ~{recall_t}/recall" if recall_t is not None else "")
               + (f", ~{brief_t}/brief" if brief_t is not None else "")
               + " paid only when used.")

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
                   f"and {used.get('pull_rate', 0):.0%} of explicit pulls "
                   f"referenced downstream over {used['sessions']} sessions — "
                   f"the honest 'did memory help' signal.")
        out.append(f"          by channel: {chans}   "
                   "(L1 = the agent asking; L3/L4/L5 = FornixDB offering)")
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
