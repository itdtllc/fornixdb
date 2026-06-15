#!/usr/bin/env python3
"""Does Sleep/Dream actually do anything useful? — a real-embedder benefit test.

The unit tests prove the MECHANICS with a deterministic bag-of-words embedder.
This proves the VALUE with the real model2vec embedder on a realistic corpus:
it plants four genuine consolidation opportunities among distractors that must
NOT be flagged, runs `dream`, and scores recall (found the planted ones?) and
precision (avoided the noise?). Then it shows the concrete payoff: applying the
dream's proposals removes a stale memory from recall and surfaces a connection
that did not exist before.

Designed to be able to FAIL: if the thresholds are noisy or miss the orphan,
the numbers say so. Run:  .venv/bin/python examples/dream_eval.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fornixdb.consolidate import dream, supersede_suggestion
from fornixdb.core import MemoryStore
from fornixdb.vectors import embed_memory, get_default_embedder


def _ids(pairs):
    return {frozenset(p["ids"]) for p in pairs}


def main() -> int:
    emb = get_default_embedder()
    if emb is None:
        print("model2vec not installed — this benefit test needs the real "
              "embedder (pip install model2vec). Skipping.")
        return 0

    tmp = tempfile.TemporaryDirectory()
    s = MemoryStore(db_path=Path(tmp.name) / "eval.db")
    M = {}  # label -> id

    def add(label, gist, *, kind="semantic", detail=None):
        M[label] = s.store(gist, detail or gist, kind=kind)

    # ---- planted opportunities (ground truth) -----------------------------
    # 1) ORPHANED FIX: a fact corrected later under a different title. Same kind.
    add("rate_old", "The production API rate limit is 1000 requests per minute")
    add("rate_new", "The production API rate limit is now 5000 requests per minute")
    # 2) NEAR-DUPLICATE: two phrasings of one fact (redundant, not a correction)
    add("dup_a", "Database backups run nightly at 2am UTC")
    add("dup_b", "The database backup job runs every night at 2am UTC")
    # 3) CROSS-KIND ASSOCIATION: related but different kinds, should be linked
    add("commit_fb", "Prefer terse, factual commit messages", kind="feedback")
    add("commit_sem", "Commit message subject lines are capped at 50 characters")
    # 4) MESSY GIST: an over-long gist that should be flagged for tidying
    add("messy", "x" * 230)

    # ---- distractors: diverse, must NOT be flagged together ----------------
    distractors = [
        "The office coffee machine is on the third floor",
        "Quarterly estimated taxes are due April 15",
        "The team standup is at 9:30 each morning",
        "Kubernetes evicts pods when a node runs out of memory",
        "The conference room projector uses an HDMI cable",
        "Annual performance reviews happen in December",
        "The parking garage closes at 11pm on weekdays",
        "Sales tax in the home state is 8.25 percent",
    ]
    for i, d in enumerate(distractors):
        add(f"distract_{i}", d)

    for mid in M.values():
        embed_memory(s, emb, mid)

    planted_heal = [frozenset((M["rate_old"], M["rate_new"])),   # find as merge OR contradiction
                    frozenset((M["dup_a"], M["dup_b"]))]
    planted_assoc = frozenset((M["commit_fb"], M["commit_sem"]))

    rep = dream(s)
    work = rep["counts"]
    heal_found = _ids(rep["work"]["merges"]) | _ids(rep["work"]["contradictions"])
    assoc_found = _ids(rep["work"]["associations"])
    gist_found = {g["id"] for g in rep["work"]["gists"]}

    print("=" * 64)
    print("SLEEP/DREAM BENEFIT TEST  (real model2vec embedder)")
    print(f"corpus: {len(M)} memories ({len(distractors)} distractors)")
    print("=" * 64)
    print(rep["narrative"])
    print()

    # ---- RECALL: did it find the planted opportunities? -------------------
    def hit(label, present):
        print(f"  [{'FOUND' if present else 'MISS '}] {label}")
        return present

    print("RECALL (planted opportunities it should surface):")
    r1 = hit("orphaned fix  rate_old~rate_new (heal candidate)",
             planted_heal[0] in heal_found)
    r2 = hit("near-duplicate dup_a~dup_b (merge)", planted_heal[1] in heal_found)
    r3 = hit("cross-kind association commit_fb<->commit_sem",
             planted_assoc in assoc_found)
    r4 = hit("messy gist flagged for tidy", M["messy"] in gist_found)
    recall_n = sum([r1, r2, r3, r4])

    # ---- PRECISION: how much of what it proposed was NOT planted? ---------
    intended_heal = set(planted_heal)
    intended_assoc = {planted_assoc}
    spurious_heal = heal_found - intended_heal
    spurious_assoc = assoc_found - intended_assoc
    print("\nPRECISION (proposals that were NOT planted = noise):")
    print(f"  heal candidates: {len(heal_found)} total, "
          f"{len(spurious_heal)} unplanned")
    print(f"  associations:    {len(assoc_found)} total, "
          f"{len(spurious_assoc)} unplanned")
    if spurious_assoc:
        id2gist = {m["id"]: m["gist"] for m in
                   (dict(r) for r in s.conn.execute("SELECT id, gist FROM memory"))}
        for pair in list(spurious_assoc)[:6]:
            a, b = tuple(pair)
            print(f"      ? #{a} <-> #{b}: {id2gist[a][:38]} | {id2gist[b][:38]}")

    # ---- WRITE-TIME NUDGE on a real re-store ------------------------------
    nudge = supersede_suggestion(
        s, -1, "The production API rate limit is 1000 req/min", "semantic",
        embedder=emb)
    unrelated_nudge = supersede_suggestion(
        s, -1, "The fire extinguisher is by the kitchen exit", "semantic",
        embedder=emb)
    print("\nWRITE-TIME NUDGE:")
    print(f"  re-stating the rate-limit fact -> "
          f"{'nudged #' + str(nudge['id']) if nudge else 'no nudge (MISS)'}")
    print(f"  storing an unrelated fact      -> "
          f"{'nudged (FALSE POSITIVE)' if unrelated_nudge else 'silent (correct)'}")

    # ---- PAYOFF: before/after the dream's proposals -----------------------
    print("\nPAYOFF (the concrete win):")
    before = [m["id"] for m in s.recall("production API rate limit", embedder=emb)]
    stale_before = M["rate_old"] in before
    s.supersede(M["rate_old"], M["rate_new"])           # apply the heal
    after = [m["id"] for m in s.recall("production API rate limit", embedder=emb)]
    stale_after = M["rate_old"] in after
    print(f"  recall 'API rate limit' before heal: stale#{M['rate_old']} "
          f"present={stale_before}, current#{M['rate_new']} present="
          f"{M['rate_new'] in before}")
    print(f"  recall 'API rate limit' after  heal: stale#{M['rate_old']} "
          f"present={stale_after}, current#{M['rate_new']} present="
          f"{M['rate_new'] in after}")
    heal_win = stale_before and not stale_after

    before_links = len(s.show(M["commit_fb"], reinforce=False)["links"])
    dream(s, weave=True)                                # weave associations
    after_links = len(s.show(M["commit_fb"], reinforce=False)["links"])
    weave_win = after_links > before_links
    print(f"  commit-message memory links: before weave={before_links}, "
          f"after weave={after_links}")

    # ---- VERDICT ----------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"RECALL  : {recall_n}/4 planted opportunities found")
    print(f"PRECISION: {len(spurious_assoc)} spurious associations, "
          f"{len(spurious_heal)} spurious heal candidates")
    print(f"NUDGE   : fires on real re-store={bool(nudge)}, "
          f"silent on unrelated={not unrelated_nudge}")
    print(f"PAYOFF  : stale removed from recall={heal_win}, "
          f"new connection created={weave_win}")
    useful = (recall_n >= 3 and heal_win and weave_win
              and bool(nudge) and not unrelated_nudge)
    print(f"\nVERDICT : {'USEFUL ✅' if useful else 'NEEDS REVIEW ⚠️'}")
    print("=" * 64)
    s.close()
    tmp.cleanup()
    return 0 if useful else 1


if __name__ == "__main__":
    sys.exit(main())
