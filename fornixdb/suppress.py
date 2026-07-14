"""Per-memory proactive-push suppression from OUTCOME history.

Push noise concentrates in a handful of chronic memories that get pushed every
session and are never once referenced downstream. The relevance floor provably
cannot filter them: on a lived-in store the useful and noise push-cosines fully
overlap (measured 2026-07-12: useful med 0.648, noise med 0.662 — a global floor
raise can't separate them). But each memory's own push OUTCOME history does: a
row pushed many times with zero downstream citations is noise by its own record.

This module owns the RULE (which ids qualify); `core.MemoryStore` owns the
mechanical write (suppress/clear/list + audit log), exactly as `usefulness_scan`
owns the citation-crediting policy and `record_referenced` owns its store write.
The push stats come straight from `usefulness_scan.scan` — the same transcript
join that credits references — so suppression and use-credit read one source of
truth and never disagree about what "referenced" means.

Suppression only ever removes a memory from the L3/L4/L5 PUSH channels. Explicit
recall/show/timeline always still return it (core enforces the invariant), and it
is redeemable: show/mark_helpful/supersede/set-gist clear it, and this scan itself
un-suppresses any row that has since earned a downstream reference.
"""
from __future__ import annotations

from .multistore import get_config
from .usefulness_scan import scan

DEFAULT_MIN_PUSHED = 8      # pushed at least this many times ...
DEFAULT_MAX_REFERENCED = 0  # ... and referenced at most this many → suppress


def thresholds(store) -> tuple[int, int]:
    """(min_pushed, max_referenced), config-overridable per store. Defaults 8 / 0:
    the chronic-offender line the honing measurement drew."""
    lo = int(get_config(store, "suppress_min_pushed", str(DEFAULT_MIN_PUSHED)))
    hi = int(get_config(store, "suppress_max_referenced", str(DEFAULT_MAX_REFERENCED)))
    return lo, hi


def classify(scan_result: dict, min_pushed: int,
             max_referenced: int) -> tuple[dict, set]:
    """Split a `usefulness_scan.scan` result into (to_suppress, earned_reference).

    to_suppress: {id: (pushed, referenced)} for rows pushed >= min_pushed and
    referenced <= max_referenced — the noise population.
    earned_reference: {id} for any row referenced > max_referenced — proven-useful,
    and the redemption set for a row that was suppressed before it earned a cite."""
    to_suppress: dict[int, tuple] = {}
    earned_reference: set[int] = set()
    for i, c in (scan_result.get("per_memory") or {}).items():
        pushed = int(c.get("impressions", 0))
        ref = int(c.get("referenced", 0))
        if ref > max_referenced:
            earned_reference.add(int(i))
        elif pushed >= min_pushed:
            to_suppress[int(i)] = (pushed, ref)
    return to_suppress, earned_reference


def scan_and_apply(store, transcripts, *, apply: bool = True) -> dict:
    """Scan the host transcripts, decide, and (unless `apply` is False) update the
    store: suppress fresh qualifiers, and self-correct by un-suppressing any
    currently-suppressed row that has since earned a downstream reference. Returns
    a report dict (JSON-friendly) describing what would change / did change."""
    result = scan(transcripts)
    lo, hi = thresholds(store)
    to_suppress, earned = classify(result, lo, hi)
    report = {
        "transcripts": str(transcripts),
        "sessions": result.get("sessions", 0),
        "min_pushed": lo,
        "max_referenced": hi,
        "candidates": {i: {"pushed": p, "referenced": r}
                       for i, (p, r) in sorted(to_suppress.items())},
        "candidate_count": len(to_suppress),
    }
    if apply:
        currently = {row["id"] for row in store.proactive_suppressed()}
        redeem = sorted(currently & earned)
        redeemed = (store.clear_proactive_suppression(redeem, "scan_earned_references")
                    if redeem else 0)
        newly = store.suppress_proactive(to_suppress)
        report["applied"] = {"newly_suppressed": newly,
                             "redeemed": redeemed,
                             "redeemed_ids": redeem}
        report["total_suppressed"] = len(store.proactive_suppressed())
    return report


def format_report(report: dict) -> str:
    """Human-readable summary for the CLI (dry-run and applied)."""
    lo, hi = report.get("min_pushed"), report.get("max_referenced")
    out = [f"proactive suppression scan: {report.get('transcripts')}",
           f"sessions: {report.get('sessions', 0)}  "
           f"rule: pushed >= {lo} AND referenced <= {hi}",
           f"candidates (chronic push-noise): {report.get('candidate_count', 0)}"]
    for i, c in list(report.get("candidates", {}).items())[:20]:
        out.append(f"  #{i:<5} pushed {c['pushed']}, referenced {c['referenced']}")
    ap = report.get("applied")
    if ap is not None:
        out.append(f"applied: {ap['newly_suppressed']} newly suppressed, "
                   f"{ap['redeemed']} redeemed"
                   + (f" ({ap['redeemed_ids']})" if ap["redeemed_ids"] else "")
                   + f"; {report.get('total_suppressed', 0)} suppressed total.")
    else:
        out.append("(dry run — pass --apply to write. recall/show/timeline "
                   "always still return suppressed rows.)")
    return "\n".join(out)
