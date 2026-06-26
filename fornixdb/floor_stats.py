"""Analyzer for the opt-in floor log (`floor_log.jsonl`; see
`proactive._log_floor_decision`). Turns the per-decision cosine records into
distributions, dial activity, and — when joined to a store's USE outcomes — an
evidence-based floor recommendation, so the relevance floor is chosen from data
instead of hand-set.

Portable by the same principle as the rest of the engine (#276/#332): the math is
pure functions over parsed records + an optional `{id: outcome}` map, testable with
no store. The single store-touching function (`outcomes_from_store`) is the thin
edge the CLI wires in.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median


def load_records(path: str | Path | None) -> list[dict]:
    """Parse a floor_log.jsonl into records, skipping blank/corrupt lines."""
    out: list[dict] = []
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _spread(xs: list[float]) -> dict:
    """Distribution summary for a list of cosines/margins (empty -> {})."""
    if not xs:
        return {}
    s = sorted(xs)

    def q(frac: float) -> float:        # nearest-rank percentile
        i = min(len(s) - 1, max(0, round(frac * (len(s) - 1))))
        return round(s[i], 4)

    return {"n": len(s), "min": round(s[0], 4), "p25": q(0.25),
            "median": round(median(s), 4), "p75": q(0.75),
            "max": round(s[-1], 4), "mean": round(mean(s), 4)}


def _cos(r: dict):
    v = r.get("vec_cos")
    return None if v is None else float(v)


def summarize(records: list[dict], outcomes: dict | None = None) -> dict:
    """Plain (json-able) stats over floor-log records. `outcomes` optionally maps a
    surfaced memory id to "useful" | "noise" | "unknown" (see outcomes_from_store),
    which unlocks the outcome split and the floor recommendation."""
    by_decision: dict[str, int] = {}
    by_channel: dict[str, int] = {}
    surfaced: list[dict] = []
    below: list[dict] = []
    raised = lowered = unchanged = 0

    for r in records:
        by_decision[r.get("decision", "?")] = by_decision.get(r.get("decision", "?"), 0) + 1
        by_channel[r.get("channel", "?")] = by_channel.get(r.get("channel", "?"), 0) + 1
        if r.get("decision") == "surfaced":
            surfaced.append(r)
        elif r.get("decision") == "below_floor":
            below.append(r)
        ef, bf = r.get("eff_floor"), r.get("base_floor")
        if ef is not None and bf is not None:
            delta = round(float(ef) - float(bf), 4)
            raised += delta > 0
            lowered += delta < 0
            unchanged += delta == 0

    freq: dict = {}
    for r in surfaced:
        if r.get("id") is not None:
            freq[r["id"]] = freq.get(r["id"], 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]

    summary: dict = {
        "records": len(records),
        "by_decision": by_decision,
        "by_channel": by_channel,
        "surfaced_cosine": _spread([c for r in surfaced if (c := _cos(r)) is not None]),
        "below_floor_cosine": _spread([c for r in below if (c := _cos(r)) is not None]),
        "below_floor_margin": _spread([float(r["margin"]) for r in below
                                       if r.get("margin") is not None]),
        "dial_activity": {"raised": raised, "lowered": lowered, "unchanged": unchanged},
        "top_surfaced_ids": [{"id": i, "times": n} for i, n in top],
    }

    if outcomes:
        useful = [c for r in surfaced if outcomes.get(r.get("id")) == "useful"
                  and (c := _cos(r)) is not None]
        noise = [c for r in surfaced if outcomes.get(r.get("id")) == "noise"
                 and (c := _cos(r)) is not None]
        summary["outcome"] = {
            "useful": _spread(useful),
            "noise": _spread(noise),
            "unknown_surfaced": sum(1 for r in surfaced
                                    if outcomes.get(r.get("id")) not in ("useful", "noise")),
        }
        summary["recommendation"] = recommend_floor(useful, noise)
    return summary


def recommend_floor(useful_cos: list[float], noise_cos: list[float]) -> dict:
    """Suggest a floor from outcome-labeled SURFACED cosines. The goal: keep rows
    that proved useful, drop rows that were pushed-but-never-used. A clean gap means
    a lossless floor exists; overlap means raising the floor costs real useful rows."""
    if not noise_cos:
        return {"verdict": "insufficient_evidence",
                "detail": "no pushed-but-unused (noise) rows labeled yet — keep "
                          "collecting; cannot justify raising the floor."}
    if not useful_cos:
        return {"verdict": "raise_safe",
                "suggested_floor": round(max(noise_cos) + 0.0001, 4),
                "detail": f"only noise labeled; a floor above {round(max(noise_cos), 4)} "
                          "drops it with no measured loss of useful rows."}
    hi_noise = max(noise_cos)
    lo_useful = min(useful_cos)
    if lo_useful > hi_noise:
        return {"verdict": "clean_separation",
                "suggested_floor": round((lo_useful + hi_noise) / 2, 4),
                "keeps_useful": len(useful_cos), "drops_noise": len(noise_cos),
                "useful_min": round(lo_useful, 4), "noise_max": round(hi_noise, 4)}
    cost = sum(1 for c in useful_cos if c <= hi_noise)
    return {"verdict": "overlap_no_lossless_floor",
            "noise_max": round(hi_noise, 4), "useful_min": round(lo_useful, 4),
            "floor_at_noise_max_drops_useful": cost, "of_useful": len(useful_cos),
            "detail": "useful and noise cosines overlap; tightening the floor here "
                      "trades recall for precision — decide which matters more."}


def outcomes_from_store(store, ids) -> dict:
    """Map each surfaced id to its USE outcome from the store's own counters:
    "useful" if it was ever pulled or endorsed (recall_count/helpful_count > 0),
    "noise" if it was pushed but never used (surfaced_count > 0, no use — the
    db.py:76 noise signal), else "unknown". The store-touching edge."""
    ids = [i for i in {*ids} if i is not None]
    if not ids:
        return {}
    rows = store.conn.execute(
        "SELECT id, helpful_count, recall_count, surfaced_count FROM memory "
        f"WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
    out: dict = {}
    for r in rows:
        if r["helpful_count"] > 0 or r["recall_count"] > 0:
            out[r["id"]] = "useful"
        elif r["surfaced_count"] > 0:
            out[r["id"]] = "noise"
        else:
            out[r["id"]] = "unknown"
    return out


def format_report(s: dict) -> str:
    """Human-readable rendering of summarize()'s dict."""
    def line(label, sp):
        if not sp:
            return f"  {label:<16} (none)"
        return (f"  {label:<16} n={sp['n']:<4} min={sp['min']:.3f} "
                f"med={sp['median']:.3f} max={sp['max']:.3f} mean={sp['mean']:.3f}")

    out = [f"floor log: {s.get('log_path', '(in-memory)')}",
           f"records: {s['records']}"]
    if not s["records"]:
        out.append("  (empty — enable with `config floor_log on` and let pulses run)")
        return "\n".join(out)
    out.append("by decision: " + ", ".join(f"{k}={v}" for k, v in
                                            sorted(s["by_decision"].items())))
    out.append("by channel:  " + ", ".join(f"{k}={v}" for k, v in
                                            sorted(s["by_channel"].items())))
    d = s["dial_activity"]
    out.append(f"floor dial:  raised={d['raised']} lowered={d['lowered']} "
               f"unchanged={d['unchanged']}  (eff_floor vs base_floor)")
    out.append("cosine spread:")
    out.append(line("surfaced", s["surfaced_cosine"]))
    out.append(line("below-floor", s["below_floor_cosine"]))
    if s.get("top_surfaced_ids"):
        top = ", ".join(f"#{t['id']}×{t['times']}" for t in s["top_surfaced_ids"][:5])
        out.append(f"top pushers:  {top}")
    if "outcome" in s:
        out.append("outcome (surfaced rows joined to store use):")
        out.append(line("useful", s["outcome"]["useful"]))
        out.append(line("noise", s["outcome"]["noise"]))
        out.append(f"  unknown          {s['outcome']['unknown_surfaced']}")
        rec = s["recommendation"]
        out.append(f"recommendation: {rec['verdict']}"
                   + (f" → set floor ≈ {rec['suggested_floor']}"
                      if "suggested_floor" in rec else ""))
        if rec.get("detail"):
            out.append(f"  {rec['detail']}")
    else:
        out.append("(no outcome join — pass a store to label useful vs noise)")
    return "\n".join(out)
