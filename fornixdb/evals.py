"""Recall-quality eval harness — makes ranking tuning measurable.

The unit tests prove the machinery; this proves the *quality*: given a golden
file of queries with the memories a good recall should surface, score the
store's actual ranking. Run it before and after touching any ranking constant
(core.py weights, decay half-lives, vector blend) — the numbers move or the
change reverts. It is also the regression fence for the owner's
extensive-testing gate: a golden set over a real store catches recall
regressions that synthetic unit fixtures never see.

Golden file: JSONL, one case per line:

    {"query": "capping disk space", "expect": [136, 141], "k": 5,
     "kind": null, "note": "paraphrase, no shared keywords"}

`expect` entries are memory ids or name slugs (names follow supersession, so
a golden case keeps tracking the live version of a named memory). A case
passes when ANY expected memory ranks in the top k (the golden question is
"would the AI have found it?", not "is the ordering exactly so").

Scoring per case: rank of the best-ranked expected memory → hit@1, hit@k,
reciprocal rank. Aggregate: hit@1, hit@k, MRR (mean reciprocal rank, 0 for
misses). Golden sets over personal stores are personal data — keep them next
to the store (gitignored), not in the repo.

Drift guard: a case may set `"rank1": true` to assert it *should* rank first.
hit@k is generous (top-k is enough), so it stays at 100% while a correct hit
quietly slides from rank 1 to rank 3 as the store grows and newer rows crowd
the embedding space. A `rank1`-flagged case whose live rank is no longer 1 is
reported as DRIFTED — the early warning for that re-ranking decay, separate
from an outright miss. The CLI's `--max-drift N` turns it into a fence.
"""

from __future__ import annotations

import json
from pathlib import Path

from .core import MemoryStore

DEFAULT_K = 5


def load_golden(path: str | Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case = json.loads(line)
            # a case is either a positive (expect a memory in top-k) or a
            # negative (expect_abstain: recall should report nothing relevant)
            if "query" not in case or not (case.get("expect") or case.get("expect_abstain")):
                raise ValueError(
                    f"{path}:{n}: golden case needs 'query' and 'expect' (or 'expect_abstain')")
            cases.append(case)
    return cases


def _resolve(store: MemoryStore, ref) -> int | None:
    """Golden expectation (id or name slug) → memory id, None if absent."""
    if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
        row = store.conn.execute("SELECT id FROM memory WHERE id = ?",
                                 (int(ref),)).fetchone()
    else:
        row = store.conn.execute("SELECT id FROM memory WHERE name = ?",
                                 (ref,)).fetchone()
    return row["id"] if row else None


def run_case(store: MemoryStore, case: dict, embedder=None) -> dict:
    from .core import recall_has_answer
    k = int(case.get("k", DEFAULT_K))
    expected = {mid for mid in (_resolve(store, r) for r in case.get("expect", []))
                if mid is not None}
    since, until = case.get("since"), case.get("until")
    if case.get("when"):  # combined subject+time golden cases
        from .timeparse import parse_when
        s, e = parse_when(case["when"])
        since, until = s.isoformat(), e.isoformat()
    # count_recall=False: the fence must not perturb what it measures —
    # recall_count is a genuine-use signal (it feeds _usefulness ranking and
    # the push floor), and an eval sweep is not use.
    rows = store.recall(case["query"], limit=k, kind=case.get("kind"),
                        since=since, until=until, embedder=embedder,
                        count_recall=False)
    if case.get("expect_abstain"):
        # negative case: recall SHOULD report nothing relevant (abstention gate)
        has = recall_has_answer(rows)
        return {
            "query": case["query"], "note": case.get("note", ""), "k": k,
            "expect_abstain": True, "abstained": not has,
            "got": [r["id"] for r in rows],
            "expected": [], "unresolved": [], "rank": None,
            "hit1": False, "hitk": False, "rr": 0.0,
            "rank1_expected": False, "drifted": False,
        }
    ranked = [r["id"] for r in rows]
    rank = next((i + 1 for i, mid in enumerate(ranked) if mid in expected), None)
    rank1_expected = bool(case.get("rank1"))
    return {
        "query": case["query"],
        "note": case.get("note", ""),
        "k": k,
        "expected": sorted(expected),
        "unresolved": [r for r in case.get("expect", []) if _resolve(store, r) is None],
        "rank": rank,                      # 1-based rank of best expected hit
        "hit1": rank == 1,
        "hitk": rank is not None,
        "rr": (1.0 / rank) if rank else 0.0,
        "got": ranked,                     # what actually ranked, for misses
        "rank1_expected": rank1_expected,
        # drift = asserted #1, still found in top-k, but quietly slid below #1.
        # (a rank1 case that fell out of top-k entirely is a miss — hit@k catches it.)
        "drifted": rank1_expected and rank is not None and rank != 1,
        # would the #191 abstention gate wrongly suppress this REAL question?
        "gate_abstained": not recall_has_answer(rows),
    }


def run(store: MemoryStore, golden_path: str | Path, embedder=None) -> dict:
    cases = load_golden(golden_path)
    results = [run_case(store, c, embedder=embedder) for c in cases]
    pos = [r for r in results if not r.get("expect_abstain")]
    neg = [r for r in results if r.get("expect_abstain")]
    n = len(pos) or 1
    return {
        "cases": len(pos),                 # positive (recall-quality) cases
        "hit@1": round(sum(r["hit1"] for r in pos) / n, 3),
        "hit@k": round(sum(r["hitk"] for r in pos) / n, 3),
        "mrr": round(sum(r["rr"] for r in pos) / n, 3),
        "misses": [r for r in pos if not r["hitk"]],
        "drift": [r for r in pos if r["drifted"]],
        # the abstention gate must cut both ways: abstain on negatives (no leak)
        # and NOT abstain on real questions (no false-abstain)
        "abstain_cases": len(neg),
        "abstain_correct": sum(r["abstained"] for r in neg),
        "abstain_leaks": [r for r in neg if not r["abstained"]],
        "false_abstain": [r for r in pos if r["gate_abstained"]],
        "results": results,
    }


def record_run(report: dict, path: str | Path, *, store: MemoryStore | None = None,
               when: str | None = None) -> dict:
    """Append one eval run to a JSONL history so recall precision over a
    GROWING live store is visible across sessions. This is distinct from the
    CI fence (`--min-hitk`/`--max-drift`), which compares on identical store
    content: aging is a real signal here, not a regression. The store keeps
    growing, so a slow hit@1 decline is the cue to refresh the golden set or
    tune ranking. Personal data — the history lives by the store (gitignored)."""
    from datetime import datetime
    rec = {
        "when": when or datetime.now().replace(microsecond=0).isoformat(),
        "cases": report["cases"],
        "hit@1": report["hit@1"],
        "hit@k": report["hit@k"],
        "mrr": report["mrr"],
        "drift": len(report.get("drift", [])),
    }
    if store is not None:
        rec["store_memories"] = store.stats()["memories"]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def load_history(path: str | Path) -> list[dict]:
    """The recorded eval runs oldest-first ([] if never recorded)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]
    except FileNotFoundError:
        return []


def format_report(report: dict, verbose: bool = False) -> str:
    header = (f"golden cases: {report['cases']}   "
              f"hit@1: {report['hit@1']:.0%}   hit@k: {report['hit@k']:.0%}   "
              f"MRR: {report['mrr']:.3f}")
    if report.get("drift"):
        header += f"   DRIFT: {len(report['drift'])}"
    if report.get("abstain_cases"):
        header += (f"   abstain: {report['abstain_correct']}/"
                   f"{report['abstain_cases']}")
    if report.get("false_abstain"):
        header += f"   FALSE-ABSTAIN: {len(report['false_abstain'])}"
    lines = [header]
    for r in report.get("false_abstain", []):
        lines.append(f"FALSE-ABSTAIN  {r['query']!r} — gate suppressed a real "
                     f"question (top hit {r['got'][:1]})")
    pos = [r for r in report["results"] if not r.get("expect_abstain")]
    if pos and all(not r["expected"] for r in pos):
        lines.append("WARNING: no golden expectation resolved to a memory — "
                     "wrong store? (check --db / $FORNIXDB_DB)")
    for r in report["results"]:
        if r.get("expect_abstain"):
            if r["abstained"] and not verbose:
                continue
            mark = "ok  " if r["abstained"] else "LEAK"
            note = f"  ({r['note']})" if r["note"] else ""
            lines.append(f"{mark} abstain   {r['query']!r}{note}")
            if not r["abstained"]:
                lines.append(f"     expected nothing relevant, got {r['got']}")
            continue
        drifted = r["drifted"]
        if r["hitk"] and not drifted and not verbose:
            continue
        mark = "MISS" if not r["hitk"] else ("DRIFT" if drifted else "ok  ")
        note = f"  ({r['note']})" if r["note"] else ""
        lines.append(f"{mark} rank={r['rank'] or '-'}/{r['k']}  "
                     f"{r['query']!r}{note}")
        if not r["hitk"]:
            lines.append(f"     expected {r['expected']}, got {r['got']}")
        elif drifted:
            lines.append(f"     asserted rank 1, now rank {r['rank']} — "
                         f"expected {r['expected']}, got {r['got']}")
        if r["unresolved"]:
            lines.append(f"     WARNING unresolved expectations: {r['unresolved']}")
    return "\n".join(lines)
