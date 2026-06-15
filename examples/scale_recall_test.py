#!/usr/bin/env python3
"""Scale test: load a large synthetic corpus into a THROWAWAY FornixDB store and
measure recall quality, latency, and on-disk footprint at size.

Two questions it answers:
  1. Does recall still find the needle in a big haystack? (hit@1 / hit@k / MRR
     over planted "needle" memories whose distinctive fact is buried among N
     filler memories.)
  2. How much disk does N memories actually cost? (bytes/memory, and how many
     memories fit in a given budget — e.g. 500 MB vs 2 GB.)

It NEVER touches the real store — it builds its own temp db (or --db path).

    python examples/scale_recall_test.py --n 5000           # with vectors
    python examples/scale_recall_test.py --n 20000 --no-embed
    python examples/scale_recall_test.py --n 5000 --db /tmp/scale.db --keep
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fornixdb.core import MemoryStore

TOPICS = ["the deploy pipeline", "the billing service", "the mobile client",
          "the data warehouse", "the auth layer", "the search index",
          "the recommendation model", "the notification queue", "the CDN",
          "the analytics dashboard", "the payment gateway", "the cache tier"]
ACTIONS = ["was refactored", "got a new owner", "hit a latency regression",
           "shipped a feature flag", "was rolled back", "added a retry policy",
           "migrated regions", "dropped a dependency", "fixed a memory leak",
           "added rate limiting", "split into two services", "got new metrics"]
PEOPLE = ["Dana", "Priya", "Marcus", "Lena", "Tomás", "Aiko", "Omar", "Wei",
          "Sofia", "Idris", "Hana", "Diego"]
DETAILS = ["The change landed after review and a staged rollout.",
           "Follow-up work was tracked in a ticket for next sprint.",
           "Metrics returned to baseline within an hour of the change.",
           "A postmortem captured the timeline and the corrective actions.",
           "The on-call engineer verified the dashboards before signing off."]

# Needles: a distinctive fact + the query that should retrieve it. The facts are
# deliberately weird so they can't be answered by generic filler.
NEEDLES = [
    ("The Zephyr-9 turbopump must be primed to 4.7 bar before ignition.",
     "what pressure must the Zephyr-9 turbopump be primed to"),
    ("Project Marigold's archive key rotates every 38 days, not 30.",
     "how often does Project Marigold's archive key rotate"),
    ("The Quokka build agent only runs on rack B17, slot 4.",
     "where does the Quokka build agent run"),
    ("Customer Hollowell's SLA credit threshold is 99.93 percent uptime.",
     "what is Customer Hollowell's SLA credit threshold"),
    ("The Saffron migration cutover is scheduled for the 2027 leap day.",
     "when is the Saffron migration cutover scheduled"),
    ("Invoice prefix QX7 is reserved for the Brisbane reseller channel.",
     "what is invoice prefix QX7 reserved for"),
    ("The Nimbus cache evicts entries older than 11 minutes, not the default 5.",
     "how old before the Nimbus cache evicts entries"),
    ("Falcon-tier accounts are capped at 1450 webhooks per minute.",
     "what is the webhook cap for Falcon-tier accounts"),
    ("The Verdigris secret lives in vault path kv/teams/atlas/verdigris.",
     "where does the Verdigris secret live"),
    ("Runbook RB-204 requires two approvers from the Helios group.",
     "how many approvers does runbook RB-204 require"),
    ("The Cobalt export job writes to bucket s3://exports-cobalt-eu-3.",
     "which bucket does the Cobalt export job write to"),
    ("Sensor array T-12 reports in kelvin, every other array in celsius.",
     "what unit does sensor array T-12 report in"),
]


def filler(rng: random.Random, i: int) -> tuple[str, str]:
    gist = (f"{rng.choice(PEOPLE)} noted that {rng.choice(TOPICS)} "
            f"{rng.choice(ACTIONS)} on day {i}.")
    detail = " ".join(rng.sample(DETAILS, k=rng.randint(1, 3)))
    return gist, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="number of filler memories")
    ap.add_argument("--db", help="store path (default: a temp file, deleted after)")
    ap.add_argument("--no-embed", action="store_true", help="skip vector embedding")
    ap.add_argument("--k", type=int, default=5, help="top-k for hit@k")
    ap.add_argument("--keep", action="store_true", help="keep the db (with --db)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tmp = None
    if args.db:
        db_path = args.db
    else:
        tmp = tempfile.TemporaryDirectory()
        db_path = str(Path(tmp.name) / "scale.db")

    store = MemoryStore(db_path=db_path)
    print(f"loading {args.n} filler + {len(NEEDLES)} needle memories into {db_path}")

    t0 = time.perf_counter()
    needle_ids = []
    # interleave needles uniformly through the corpus so they aren't all newest
    needle_at = {int((j + 0.5) * args.n / len(NEEDLES)): k
                 for j, k in enumerate(NEEDLES)}
    for i in range(args.n):
        if i in needle_at:
            fact, _q = needle_at[i]
            needle_ids.append(store.store(fact, kind="semantic"))
        g, d = filler(rng, i)
        store.store(g + " " + d, kind=rng.choice(["semantic", "episodic"]))
    load_s = time.perf_counter() - t0
    total = store.stats()["memories"]
    print(f"  stored {total} memories in {load_s:.1f}s "
          f"({total / load_s:.0f}/s)")

    embedder = None
    if not args.no_embed:
        from fornixdb.vectors import backfill, get_default_embedder
        embedder = get_default_embedder()
        if embedder is None:
            print("  (no embedder available — pip install model2vec; "
                  "running keyword-only)")
        else:
            t0 = time.perf_counter()
            n = backfill(store, embedder)
            print(f"  embedded {n} chunks in {time.perf_counter() - t0:.1f}s")

    # --- recall quality + latency over the needles ---
    hits1 = hitsk = 0
    rr = 0.0
    lats = []
    for nid, (fact, query) in zip(needle_ids, NEEDLES):
        t0 = time.perf_counter()
        rows = store.recall(query, limit=args.k, embedder=embedder)
        lats.append((time.perf_counter() - t0) * 1000)
        ranked = [r["id"] for r in rows]
        rank = ranked.index(nid) + 1 if nid in ranked else None
        if rank == 1:
            hits1 += 1
        if rank is not None:
            hitsk += 1
            rr += 1.0 / rank
        else:
            print(f"  MISS: {query!r} -> got {ranked}")
    n_needles = len(NEEDLES)
    lats.sort()
    p50 = lats[len(lats) // 2]
    p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]

    # --- on-disk footprint ---
    def sz(p):
        return os.path.getsize(p) if os.path.exists(p) else 0
    store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # fold WAL into the db
    db_b = sz(db_path) + sz(db_path + "-wal") + sz(db_path + "-shm")
    per = db_b / max(total, 1)

    print("\n=== recall quality (needle in haystack) ===")
    print(f"  needles: {n_needles}   hit@1: {hits1/n_needles:.0%}   "
          f"hit@{args.k}: {hitsk/n_needles:.0%}   MRR: {rr/n_needles:.3f}")
    print(f"  recall latency: p50 {p50:.1f} ms   p95 {p95:.1f} ms   "
          f"({'vectors' if embedder else 'keyword-only'})")
    print("\n=== on-disk footprint ===")
    print(f"  {total} memories = {db_b/1e6:.1f} MB on disk "
          f"({per:.0f} bytes/memory{'  incl. vectors' if embedder else ''})")
    print(f"  -> ~{int(500e6/per):,} memories fit in 500 MB")
    print(f"  -> ~{int(2_000e6/per):,} memories fit in 2 GB")

    store.close()
    if tmp and not args.keep:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
