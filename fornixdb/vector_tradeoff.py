"""What do vectors actually cost and buy? — the four-axis tradeoff harness.

End users won't know what "vectors" are, so the default has to be chosen from
evidence. This measures, on one controlled corpus, vector-on vs vector-off
across the four axes that decide it:

  1. recall quality   hit@1 / hit@k / MRR — can recall FIND the right memory,
                       especially when the query shares no keywords with it.
  2. db space         extra bytes on disk for the embedding payload.
  3. write time       added latency to store() (embedding on write).
  4. recall time      added latency to recall() (cosine over stored vectors).
  5. prompt tokens    what recall RESULTS add to the AI's context — vectors
                       change WHICH rows come back, not how many, so at a fixed
                       limit the token cost is ~unchanged (the model itself is
                       local compute, never in the prompt).

It is embedder-agnostic: tests pass a deterministic fake embedder (no model
download, reproducible direction); the runnable report (examples/) passes the
real default embedder for true numbers. Keyword-only is always recall with
embedder=False, so the comparison is the SAME data and FTS index either way.
"""

from __future__ import annotations

import time

from .core import MemoryStore
from .db import connect
from .tokens import estimate_tokens


# Corpus: half the queries are SYNONYM queries that share no keyword with their
# target (only meaning links them — where vectors should win); half are plain
# keyword queries both modes should answer (the no-regression control).
MEMORIES = [
    ("commute", "Morning commute", "The automobile stalled."),
    ("render",  "Render note",     "Her eyes sparkled."),
    ("bugfix",  "Bug fix",         "Fixed the glitch."),
    ("pool",    "Pool service",    "The pool guy came on Tuesday afternoon."),
    ("backup",  "Backups",         "Backups run to the NAS at 2 AM nightly."),
    ("storage", "Storage",         "The Synology NAS holds the family photos."),
    ("router",  "Router",          "The router admin page is at 192.168.1.1."),
    ("isp",     "Internet",        "The ISP is Sonic on a gigabit fiber line."),
    ("garden",  "Garden",          "Tomatoes were planted along the south fence."),
    ("recipe",  "Recipe",          "The bread needs a long cold proof overnight."),
]

QUERIES = [
    # synonym / paraphrase — zero keyword overlap with the target
    {"query": "vehicle",  "expect": "commute", "type": "synonym"},
    {"query": "twinkle",  "expect": "render",  "type": "synonym"},
    {"query": "artifact", "expect": "bugfix",  "type": "synonym"},
    # plain keyword — both modes should answer (no-regression control)
    {"query": "pool guy",        "expect": "pool",    "type": "keyword"},
    {"query": "nightly backups", "expect": "backup",  "type": "keyword"},
    {"query": "Synology photos", "expect": "storage", "type": "keyword"},
]


def _build(memories, embedder) -> MemoryStore:
    s = MemoryStore(conn=connect(":memory:"))
    for name, gist, detail in memories:
        s.store(gist, detail, name=name, embedder=embedder)
    return s


def _db_bytes(store: MemoryStore) -> int:
    pc = store.conn.execute("PRAGMA page_count").fetchone()[0]
    ps = store.conn.execute("PRAGMA page_size").fetchone()[0]
    return pc * ps


def _embedding_payload_bytes(store: MemoryStore) -> int:
    return store.conn.execute(
        "SELECT COALESCE(SUM(LENGTH(vector)), 0) FROM embedding").fetchone()[0]


def _eval(store: MemoryStore, queries, embedder, k: int) -> dict:
    per, hit1, hitk, rr, tok = [], 0, 0, 0.0, 0
    for q in queries:
        rows = store.recall(q["query"], limit=k, embedder=embedder)
        rank = next((i for i, r in enumerate(rows, 1)
                     if r.get("name") == q["expect"]), None)
        if rank == 1:
            hit1 += 1
        if rank is not None:
            hitk += 1
            rr += 1.0 / rank
        tok += estimate_tokens("\n".join(r["gist"] for r in rows))
        per.append({**q, "rank": rank, "n_returned": len(rows)})
    n = len(queries)
    return {"hit1": hit1, "hitk": hitk, "n": n, "mrr": round(rr / n, 3),
            "result_tokens": tok, "cases": per}


def _avg_recall_ms(store, queries, embedder, repeats: int) -> float:
    t = time.perf_counter()
    for _ in range(repeats):
        for q in queries:
            store.recall(q["query"], limit=3, embedder=embedder)
    return (time.perf_counter() - t) * 1000 / (repeats * len(queries))


def measure(memories=MEMORIES, queries=QUERIES, *, embedder, k: int = 3,
            repeats: int = 30) -> dict:
    """Compare keyword-only vs vector-on across all four axes."""
    kw = _build(memories, embedder=False)        # never embeds
    t0 = time.perf_counter()
    vec = _build(memories, embedder=embedder)    # embeds on write
    write_ms_vec = (time.perf_counter() - t0) * 1000 / len(memories)
    t0 = time.perf_counter()
    _build(memories, embedder=False)
    write_ms_kw = (time.perf_counter() - t0) * 1000 / len(memories)

    # recall quality + result-token cost: SAME store (vec), embedder toggled,
    # so only the ranking signal differs — not the data or the FTS index.
    kw_q = _eval(vec, queries, embedder=False, k=k)
    vec_q = _eval(vec, queries, embedder=embedder, k=k)

    def subset(d, t):  # hit@1 over just the synonym (or keyword) queries
        ids = [c for c in d["cases"] if c["type"] == t]
        return sum(1 for c in ids if c["rank"] == 1), len(ids)

    report = {
        "embedder": getattr(embedder, "name", str(embedder)),
        "n_memories": len(memories),
        "space": {
            "keyword_db_bytes": _db_bytes(kw),
            "vector_db_bytes": _db_bytes(vec),
            "embedding_payload_bytes": _embedding_payload_bytes(vec),
            "bytes_per_memory": round(_embedding_payload_bytes(vec) / len(memories)),
        },
        "recall": {
            "keyword": kw_q, "vector": vec_q,
            "synonym_hit1_keyword": subset(kw_q, "synonym"),
            "synonym_hit1_vector": subset(vec_q, "synonym"),
            "keyword_hit1_keyword": subset(kw_q, "keyword"),
            "keyword_hit1_vector": subset(vec_q, "keyword"),
        },
        "time_ms": {
            "write_keyword": round(write_ms_kw, 3),
            "write_vector": round(write_ms_vec, 3),
            "recall_keyword": round(_avg_recall_ms(vec, queries, False, repeats), 3),
            "recall_vector": round(_avg_recall_ms(vec, queries, embedder, repeats), 3),
        },
        "prompt_tokens": {
            # result tokens on the KEYWORD-control queries only, where both modes
            # return rows — isolates "does ranking-by-vector change token size?"
            "keyword_controls_keyword": sum(
                estimate_tokens("\n".join(r["gist"] for r in
                    vec.recall(q["query"], limit=k, embedder=False)))
                for q in queries if q["type"] == "keyword"),
            "keyword_controls_vector": sum(
                estimate_tokens("\n".join(r["gist"] for r in
                    vec.recall(q["query"], limit=k, embedder=embedder)))
                for q in queries if q["type"] == "keyword"),
        },
    }
    kw.close(); vec.close()
    return report


def format_report(r: dict) -> str:
    s, rc, tm, pt = r["space"], r["recall"], r["time_ms"], r["prompt_tokens"]
    extra = s["vector_db_bytes"] - s["keyword_db_bytes"]
    out = [
        f"Vector tradeoff  (embedder: {r['embedder']}, {r['n_memories']} memories)",
        "",
        "RECALL ABILITY (can it find the right memory?)",
        f"  overall hit@1:   keyword {rc['keyword']['hit1']}/{rc['keyword']['n']}"
        f"   vector {rc['vector']['hit1']}/{rc['vector']['n']}",
        f"  overall MRR:     keyword {rc['keyword']['mrr']}"
        f"   vector {rc['vector']['mrr']}",
        f"  synonym queries (no shared keywords): keyword "
        f"{rc['synonym_hit1_keyword'][0]}/{rc['synonym_hit1_keyword'][1]}"
        f"   vector {rc['synonym_hit1_vector'][0]}/{rc['synonym_hit1_vector'][1]}",
        f"  keyword queries (control):            keyword "
        f"{rc['keyword_hit1_keyword'][0]}/{rc['keyword_hit1_keyword'][1]}"
        f"   vector {rc['keyword_hit1_vector'][0]}/{rc['keyword_hit1_vector'][1]}",
        "",
        "DB SPACE",
        f"  keyword-only db: {s['keyword_db_bytes']:,} bytes",
        f"  vector db:       {s['vector_db_bytes']:,} bytes  "
        f"(+{extra:,} = +{extra/max(s['keyword_db_bytes'],1)*100:.0f}%)",
        f"  embedding payload: {s['embedding_payload_bytes']:,} bytes  "
        f"(~{s['bytes_per_memory']} bytes/memory)",
        "",
        "EXECUTION TIME (per operation)",
        f"  store():  keyword {tm['write_keyword']} ms   vector {tm['write_vector']} ms",
        f"  recall(): keyword {tm['recall_keyword']} ms   vector {tm['recall_vector']} ms",
        "",
        "PROMPT TOKENS (recall result the AI must read, keyword-control queries)",
        f"  keyword ranking: {pt['keyword_controls_keyword']} tok   "
        f"vector ranking: {pt['keyword_controls_vector']} tok",
        "  (vectors change WHICH rows rank first, not how many — result size is ~unchanged;",
        "   the embedding model is local compute, never added to the prompt.)",
    ]
    return "\n".join(out)
