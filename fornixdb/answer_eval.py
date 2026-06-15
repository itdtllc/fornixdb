"""End-to-end A/B answer harness (FornixDB #4) — closes the #190 caveat.

`evals.py` proves recall *ranks* the right memory; `benefit.py` proves the
memory is *absent* from the flat MEMORY.md. Neither proves that recall actually
improves the AI's *answer*. This harness does: for each golden case carrying a
known answer-fact that lives only in memory, the answerer is asked the question
twice —

  condition A (no memory)   : answer from the model's own knowledge only
  condition B (with recall) : the recalled memories are supplied as context

— and we score whether the fact appears in each answer.

  LIFT = fact present WITH recall AND absent WITHOUT it.

That delta is the end-to-end value of recall, distinct from retrieval quality.
The honest counter-cases are scored too: A already knew (no marginal value —
parametric), recall present but answer still missed (model didn't use it / recall
didn't surface it), and the regression case where recall made a right answer
wrong.

The answerer is a pluggable callable `answerer(question, context) -> str`
(`context` is None for condition A). `default_answerer()` calls the Claude API —
the autonomous tier #197 identifies as where FornixDB's value is realized — but
tests inject a fake answerer so scoring is verified without the network.

Golden file: JSONL, one case per line:

    {"query": "what port does the relay listen on?",
     "answer_contains": ["8188"], "k": 5, "kind": null, "when": null,
     "match": "all", "note": "fact lives only in a fornix_only memory"}

`answer_contains` is a string or list of strings; `match` is "all" (default) or
"any". `expect` (optional, list of ids/name slugs as in evals.py) lets the
report distinguish "recall surfaced nothing" from "model ignored the context".
"""

from __future__ import annotations

import json
from pathlib import Path

from .core import MemoryStore

DEFAULT_K = 5
DEFAULT_MODEL = "claude-opus-4-8"


def load_answer_golden(path: str | Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case = json.loads(line)
            if "query" not in case or not case.get("answer_contains"):
                raise ValueError(
                    f"{path}:{n}: answer-golden case needs 'query' and "
                    "'answer_contains'")
            cases.append(case)
    return cases


def _facts(case: dict) -> list[str]:
    raw = case["answer_contains"]
    return [raw] if isinstance(raw, str) else list(raw)


def fact_present(answer: str, facts: list[str], match: str = "all") -> bool:
    """Case-insensitive substring check; `match` is 'all' or 'any'."""
    low = (answer or "").lower()
    hits = (f.lower() in low for f in facts)
    return all(hits) if match == "all" else any(hits)


def recall_context(store: MemoryStore, case: dict, embedder=None):
    """Recall for condition B; return (rows, formatted-context-or-None)."""
    k = int(case.get("k", DEFAULT_K))
    since, until = case.get("since"), case.get("until")
    if case.get("when"):
        from .timeparse import parse_when
        s, e = parse_when(case["when"])
        since, until = s.isoformat(), e.isoformat()
    rows = store.recall(case["query"], limit=k, kind=case.get("kind"),
                        since=since, until=until, embedder=embedder)
    if not rows:
        return rows, None
    blocks = []
    for r in rows:
        block = f"- {r['gist']}"
        if r.get("detail"):
            block += f"\n  {r['detail']}"
        blocks.append(block)
    return rows, "\n".join(blocks)


def default_answerer(model: str = DEFAULT_MODEL):
    """A Claude-API answerer (the autonomous tier). Lazy-imports the SDK so this
    module loads without `anthropic` installed — tests use a fake answerer."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - exercised only without the SDK
        raise RuntimeError(
            "default_answerer needs the Claude SDK: pip install anthropic and "
            "set ANTHROPIC_API_KEY (or run `ant auth login`)") from e
    client = anthropic.Anthropic()
    system = (
        "Answer the question in one or two sentences. If notes are provided, "
        "rely on them; otherwise answer from your own knowledge. If you do not "
        "know the answer, say so plainly rather than guessing.")

    def answer(question: str, context: str | None) -> str:
        user = (f"Notes:\n{context}\n\nQuestion: {question}" if context
                else question)
        resp = client.messages.create(
            model=model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if b.type == "text")

    return answer


def run_case(store: MemoryStore, case: dict, answerer, embedder=None) -> dict:
    facts = _facts(case)
    match = case.get("match", "all")
    a_answer = answerer(case["query"], None)                 # parametric only
    rows, context = recall_context(store, case, embedder=embedder)
    b_answer = answerer(case["query"], context)              # with recall
    a_has = fact_present(a_answer, facts, match)
    b_has = fact_present(b_answer, facts, match)
    return {
        "query": case["query"],
        "note": case.get("note", ""),
        "facts": facts,
        "a_answer": a_answer,
        "b_answer": b_answer,
        "a_has": a_has,
        "b_has": b_has,
        "lift": b_has and not a_has,        # recall earned the answer
        "regressed": a_has and not b_has,   # recall broke a right answer
        "recalled": bool(rows),             # did recall surface anything?
        "recalled_ids": [r["id"] for r in rows],
    }


def run(store: MemoryStore, golden_path: str | Path, answerer,
        embedder=None) -> dict:
    cases = load_answer_golden(golden_path)
    results = [run_case(store, c, answerer, embedder=embedder) for c in cases]
    n = len(results) or 1
    lifts = [r for r in results if r["lift"]]
    # both_miss: recall context was present but the fact still didn't land —
    # the model ignored it, or what recall surfaced didn't carry the fact.
    both_miss = [r for r in results
                 if not r["a_has"] and not r["b_has"] and r["recalled"]]
    no_recall = [r for r in results
                 if not r["a_has"] and not r["b_has"] and not r["recalled"]]
    return {
        "cases": len(results),
        "lift": round(len(lifts) / n, 3),          # the headline: recall helped
        "lift_count": len(lifts),
        "a_correct": sum(r["a_has"] for r in results),   # parametric already knew
        "b_correct": sum(r["b_has"] for r in results),
        "regressions": [r for r in results if r["regressed"]],
        "both_miss": both_miss,                    # recall present, unused/insufficient
        "no_recall": no_recall,                    # recall surfaced nothing
        "results": results,
    }


def record_run(report: dict, path: str | Path, *, store: MemoryStore | None = None,
               when: str | None = None) -> dict:
    """Append one A/B run to a JSONL history so end-to-end lift over a GROWING
    live store is visible across sessions (personal data — keep by the store)."""
    from datetime import datetime
    rec = {
        "when": when or datetime.now().replace(microsecond=0).isoformat(),
        "cases": report["cases"],
        "lift": report["lift"],
        "a_correct": report["a_correct"],
        "b_correct": report["b_correct"],
        "regressions": len(report["regressions"]),
    }
    if store is not None:
        rec["store_memories"] = store.stats()["memories"]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def load_history(path: str | Path) -> list[dict]:
    """The recorded A/B runs oldest-first ([] if never recorded)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]
    except FileNotFoundError:
        return []


def format_report(report: dict, verbose: bool = False) -> str:
    header = (f"A/B answer cases: {report['cases']}   "
              f"lift: {report['lift']:.0%} ({report['lift_count']})   "
              f"A-correct: {report['a_correct']}   "
              f"B-correct: {report['b_correct']}")
    if report["regressions"]:
        header += f"   REGRESSIONS: {len(report['regressions'])}"
    lines = [header]
    for r in report["regressions"]:
        lines.append(f"REGRESS  {r['query']!r} — recall broke a right answer")
    for r in report["no_recall"]:
        lines.append(f"NO-RECALL {r['query']!r} — recall surfaced nothing "
                     f"(expected {r['facts']})")
    for r in report["both_miss"]:
        lines.append(f"BOTH-MISS {r['query']!r} — recall present but fact "
                     f"absent from answer (model didn't use it)")
    for r in report["results"]:
        if not verbose:
            continue
        mark = ("LIFT" if r["lift"] else "regr" if r["regressed"]
                else "both" if r["a_has"] and r["b_has"]
                else "para" if r["a_has"] else "miss")
        note = f"  ({r['note']})" if r["note"] else ""
        lines.append(f"{mark}  A={int(r['a_has'])} B={int(r['b_has'])}  "
                     f"{r['query']!r}{note}")
    return "\n".join(lines)
