#!/usr/bin/env python3
"""Association-threshold sweep — informs the ASSOC_COSINE choice (#3).

Builds a clustered corpus with the REAL model2vec embedder: memories grouped by
topic (varied wording, mixed kinds) are the connections dreaming SHOULD make;
every cross-cluster / distractor pair is noise it should avoid. Sweeps the
association floor and reports, at each, how many intended links it would weave
(recall) vs how many spurious ones (precision) — so we can see whether any floor
catches conceptual associations without flooding.

Run:  .venv/bin/python examples/assoc_threshold_sweep.py
"""
from __future__ import annotations

import statistics
import sys
from itertools import combinations

from fornixdb.consolidate import ASSOC_COSINE, CONTRA_COSINE, MERGE_COSINE
from fornixdb.vectors import cosine, get_default_embedder

# (kind, text) grouped into topical clusters; within a cluster = intended assoc
CLUSTERS = {
    "backups": [
        ("semantic", "Database backups run nightly at 2am UTC"),
        ("feedback", "Keep thirty days of database backup retention"),
        ("reference", "Database restore drills happen each quarter"),
    ],
    "code_review": [
        ("semantic", "Every pull request needs two approving reviews before merge"),
        ("feedback", "Prefer small, focused pull requests"),
        ("semantic", "CI must be green before a pull request can merge"),
    ],
    "incidents": [
        ("semantic", "Page the on-call engineer for any production outage"),
        ("feedback", "Write a blameless postmortem after every incident"),
        ("reference", "Sev1 incidents need an executive status update hourly"),
    ],
}
DISTRACTORS = [
    ("semantic", "The office coffee machine is on the third floor"),
    ("semantic", "Quarterly estimated taxes are due April 15"),
    ("semantic", "The parking garage closes at 11pm on weekdays"),
    ("semantic", "The team standup is at 9:30 each morning"),
    ("reference", "Home-state sales tax is 8.25 percent"),
    ("feedback", "The conference projector uses an HDMI cable"),
]


def proposed(kind_a, kind_b, cos, floor):
    """Mirror _pair_scan's association rule at a given floor."""
    if cos < floor:
        return False
    if kind_a == kind_b and cos >= CONTRA_COSINE:
        return False   # merge/contradiction band, not an association
    return True


def main() -> int:
    emb = get_default_embedder()
    if emb is None:
        print("model2vec required for the sweep."); return 0

    items = []  # (cluster_or_None, kind, text, vec)
    for cluster, mems in CLUSTERS.items():
        for kind, text in mems:
            items.append((cluster, kind, text))
    for kind, text in DISTRACTORS:
        items.append((None, kind, text))
    vecs = emb.embed([t for _, _, t in items])

    intended_cos, pairs = [], []
    for (i, a), (j, b) in combinations(enumerate(items), 2):
        ca, ka, _ = a
        cb, kb, _ = b
        cos = cosine(vecs[i], vecs[j])
        same_cluster = ca is not None and ca == cb
        pairs.append((ka, kb, cos, same_cluster))
        if same_cluster:
            intended_cos.append(cos)

    n_intended = sum(1 for *_, sc in pairs if sc)
    print("=" * 70)
    print("ASSOCIATION THRESHOLD SWEEP  (real model2vec)")
    print(f"{len(items)} memories, {n_intended} intended within-cluster pairs, "
          f"{len(pairs) - n_intended} noise pairs")
    print(f"current floor ASSOC_COSINE={ASSOC_COSINE}  (CONTRA={CONTRA_COSINE}, "
          f"MERGE={MERGE_COSINE})")
    print("=" * 70)
    print(f"intended within-cluster cosines: min={min(intended_cos):.3f} "
          f"median={statistics.median(intended_cos):.3f} "
          f"max={max(intended_cos):.3f}")
    noise_cos = sorted((c for _, _, c, sc in pairs if not sc), reverse=True)[:5]
    print("top-5 noise-pair cosines:    ",
          ", ".join(f"{c:.3f}" for c in noise_cos))
    print()
    print(f"{'floor':>6} {'proposed':>9} {'intended':>9} {'spurious':>9} "
          f"{'recall':>7} {'precision':>10}")
    for floor in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        prop = [(c, sc) for ka, kb, c, sc in pairs if proposed(ka, kb, c, floor)]
        n_prop = len(prop)
        n_found = sum(1 for _, sc in prop if sc)
        n_spur = n_prop - n_found
        recall = n_found / n_intended if n_intended else 0
        prec = n_found / n_prop if n_prop else 0
        mark = "  <- current" if abs(floor - ASSOC_COSINE) < 1e-9 else ""
        print(f"{floor:>6.2f} {n_prop:>9} {n_found:>9} {n_spur:>9} "
              f"{recall:>6.0%} {prec:>9.0%}{mark}")
    print()
    print("Read: a useful floor has HIGH recall (catches intended links) and "
          "HIGH precision (few spurious). If recall only rises once precision "
          "collapses, model2vec can't separate conceptual links from noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
