"""Does heading-chunked Markdown import actually *help*? (markdown-bridge eval)

In the spirit of benefit.py / answer_eval.py: measure the bridge's value
honestly, against the realistic baseline it replaces — not against "no memory".

THE BASELINE. Before the bridge, the only way to put an arbitrary Markdown doc
into FornixDB was as ONE memory: the whole document as a single gist + one big
detail blob. (You can still do exactly that, so it's a fair "before".)

THE TREATMENT. `import_document` splits the same doc along its headings into one
memory per section.

THE QUESTION a user actually has is answerable from ONE section ("when does the
backup run?"). So the measurable benefit is:

  tokens_to_answer  the size (in tokens) of the top-ranked recalled memory that
                    contains the answer — i.e. how much text the AI must read /
                    re-prefill to get the answer. The blob forces the WHOLE doc;
                    a chunk is just the relevant section.
  found / rank      did recall surface an answer-bearing memory at all, and how
                    near the top — precision, not just cost.

BENEFIT = the chunked store answers from far fewer tokens at an equal-or-better
hit rate. If it doesn't, this harness will say so plainly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .adapters.markdown_doc import import_document, slugify
from .adapters.markdown_import import parse_frontmatter
from .core import MemoryStore
from .db import connect
from .tokens import estimate_tokens


def _mem_store() -> MemoryStore:
    return MemoryStore(conn=connect(":memory:"))


def _contains(text: str, needles) -> bool:
    if isinstance(needles, str):
        needles = [needles]
    low = (text or "").lower()
    return all(n.lower() in low for n in needles)


def build_chunked(doc_path: str | Path) -> MemoryStore:
    s = _mem_store()
    import_document(s, doc_path)
    return s


def build_blob(doc_path: str | Path) -> MemoryStore:
    """The honest 'before': the whole document as a single memory."""
    s = _mem_store()
    path = Path(doc_path)
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    gist = meta.get("title") or meta.get("description") or path.stem
    s.store(gist, body.strip(), name=slugify(gist), source="markdown-blob")
    return s


def answer_cost(store: MemoryStore, query: str, answer_contains, k: int = 5) -> dict:
    """Recall top-k; report the highest-ranked memory that contains the answer
    and how many tokens reading it costs."""
    rows = store.recall(query, limit=k)
    for rank, r in enumerate(rows, 1):
        blob = (r.get("gist") or "") + "\n" + (r.get("detail") or "")
        if _contains(blob, answer_contains):
            return {"found": True, "rank": rank,
                    "tokens": estimate_tokens(blob), "gist": r["gist"]}
    return {"found": False, "rank": None, "tokens": 0, "gist": None}


def benefit_report(doc_path: str | Path, questions: list[dict], k: int = 5) -> dict:
    """questions: [{'query': str, 'answer_contains': str|list}, ...]."""
    chunked, blob = build_chunked(doc_path), build_blob(doc_path)
    try:
        cases = []
        for q in questions:
            c = answer_cost(chunked, q["query"], q["answer_contains"], k)
            b = answer_cost(blob, q["query"], q["answer_contains"], k)
            cases.append({"query": q["query"], "chunked": c, "blob": b})
        cf = [c for c in cases if c["chunked"]["found"]]
        bf = [c for c in cases if c["blob"]["found"]]
        chunk_tok = sum(c["chunked"]["tokens"] for c in cf)
        blob_tok = sum(c["blob"]["tokens"] for c in bf)
        return {
            "doc": str(doc_path),
            "n": len(cases),
            "chunked_found": len(cf),
            "blob_found": len(bf),
            "chunked_tokens_to_answer": chunk_tok,
            "blob_tokens_to_answer": blob_tok,
            "token_ratio": round(blob_tok / chunk_tok, 1) if chunk_tok else None,
            "chunked_sections": chunked.stats()["memories"],
            "cases": cases,
        }
    finally:
        chunked.close(); blob.close()


def format_benefit(r: dict) -> str:
    out = [
        f"Markdown-bridge benefit on {Path(r['doc']).name}",
        f"  doc imported as {r['chunked_sections']} section memories "
        f"(vs 1 whole-doc blob)",
        f"  questions: {r['n']}   answer found — chunked {r['chunked_found']}/"
        f"{r['n']}, blob {r['blob_found']}/{r['n']}",
        "",
        f"  {'question':<42} {'chunked':>16} {'blob':>16}",
    ]
    for c in r["cases"]:
        def cell(x):
            return f"#{x['rank']} {x['tokens']}tok" if x["found"] else "— miss"
        out.append(f"  {c['query'][:42]:<42} {cell(c['chunked']):>16} "
                   f"{cell(c['blob']):>16}")
    out += [
        "",
        f"  tokens to read the answer:  chunked {r['chunked_tokens_to_answer']}"
        f"  vs  blob {r['blob_tokens_to_answer']}",
    ]
    if r["token_ratio"]:
        out.append(f"  => the whole-doc blob makes the AI read {r['token_ratio']}x "
                   "more text to answer the same questions.")
    return "\n".join(out)
