"""Re-derive a memory's project label from its CONTENT.

For a store whose auto-captured history was mislabeled — the pre-0.3.1 launch-dir
bug, or any single-home setup where the working directory carries no project
signal — the cwd is the wrong thing to scope by. The portable signal is the
memory's own content. This re-projects suspect memories by what they are ABOUT.

Two modes, chosen by what the store has:
  - VECTOR mode (store has embeddings): build a centroid per project from its
    reliably-labeled ANCHOR memories — the deliberately-stored non-episodic ones,
    which carry their project honestly — then assign each suspect memory to its
    nearest centroid. Confidence = the cosine margin between best and runner-up.
  - KEYWORD mode (no embedder): score a suspect's content words against each
    project's anchor vocabulary; confidence = the margin in the overlap score.

Portable like the rest of the engine (#276/#332): `classify_vec` / `classify_words`
are pure functions over numbers, testable with no store; the store-touching edges
(`anchors_and_candidates`, `apply_proposals`, `undo`) are thin.

Propose-not-dispose (§6.5): `propose` only reports. `apply_proposals` mutates and
records an undo set in meta so `reproject --undo` restores prior labels. Nothing
is relabeled within an alias family — those already unify at recall time.
"""
from __future__ import annotations

import json
import math

from . import context
from .multistore import get_config, set_config

UNDO_KEY = "reproject_undo"
DEFAULT_MIN_MARGIN = 0.06   # mean-centered cosine gap between top-2 centroids
DEFAULT_MIN_WORD_MARGIN = 1.0  # idf-weighted overlap-score gap to be "confident"

# Structural / stop words that carry no project signal in keyword mode.
_STOP = frozenset("""
a an the and or but of to in on for with from by at as is are was were be been
this that these those it its session branch user turns come up speed pick let
lets resume continue work working next where left off about into over run ran new
""".split())


def _alias_head(store, label_lc: str) -> str | None:
    """The HEAD of `label_lc`'s alias group as DECLARED in config — the label
    before '=' in "head=alias1,alias2" — so a project keeps its real name
    (videos, not the alphabetically-first 'elira'). None if it has no group."""
    raw = get_config(store, "project_aliases", "") or ""
    import re
    for chunk in re.split(r"[;\n]+", raw):
        labels = [x.strip().lower() for x in re.split(r"[=,\s]+", chunk) if x.strip()]
        if len(labels) > 1 and label_lc in labels:
            return labels[0]
    return None


def _canon(store, label: str | None) -> str:
    """The canonical name for a project: its alias group's declared head, else the
    label itself (lowercased). So every name of one project maps to a single key
    while keeping the name the owner actually uses. None/'' -> ''."""
    l = (label or "").strip().lower()
    if not l:
        return ""
    return _alias_head(store, l) or l


def _mean(vecs: list[list[float]]) -> list[float]:
    n = len(vecs)
    if not n:
        return []
    dim = len(vecs[0])
    acc = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            acc[i] += x
    return [x / n for x in acc]


def _sub(a: list[float], b: list[float]) -> list[float]:
    """a − b (element-wise); b empty -> a unchanged."""
    return a if not b else [x - y for x, y in zip(a, b)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def classify_vec(vec: list[float], centroids: dict[str, list[float]]
                 ) -> tuple[str | None, float, dict[str, float]]:
    """Nearest project centroid for `vec`. Returns (best_project, margin, scores)
    where margin = best cosine − runner-up cosine (best's confidence). Pure."""
    if not vec or not centroids:
        return None, 0.0, {}
    scores = {p: round(cosine(vec, c), 4) for p, c in centroids.items() if c}
    if not scores:
        return None, 0.0, {}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, top = ranked[0]
    margin = round(top - (ranked[1][1] if len(ranked) > 1 else 0.0), 4)
    return best, margin, scores


def content_words(text: str) -> set[str]:
    """Lowercased alpha words >2 chars, minus stop words — the keyword signal."""
    out = set()
    for raw in (text or "").lower().split():
        w = "".join(ch for ch in raw if ch.isalpha())
        if len(w) > 2 and w not in _STOP:
            out.add(w)
    return out


def classify_words(words: set[str], vocab: dict[str, set[str]]
                   ) -> tuple[str | None, float, dict[str, float]]:
    """Nearest project by content-word overlap with each project's vocabulary,
    each shared term weighted by inverse class-frequency so a generic word in many
    projects' vocabularies counts for little and a DISTINCTIVE word counts for a
    lot (otherwise a broad catch-all project, whose vocab is a superset, wins
    everything). Returns (best, weighted_score_gap, scores). Pure."""
    if not words or not vocab:
        return None, 0.0, {}
    df: dict[str, int] = {}
    for terms in vocab.values():
        for t in terms:
            df[t] = df.get(t, 0) + 1
    scores: dict[str, float] = {}
    for p, terms in vocab.items():
        scores[p] = round(sum(1.0 / df[t] for t in (words & terms)), 4)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, top = ranked[0]
    if top <= 0:
        return None, 0.0, scores
    margin = round(top - (ranked[1][1] if len(ranked) > 1 else 0.0), 4)
    return best, margin, scores


# ------------------------------------------------------------------ store edges

def _is_candidate(store, row, suspect: set[str]) -> bool:
    """Worth re-projecting only if its label is UNRELIABLE: unscoped (NULL/''),
    or a label the caller flagged `--suspect` (the launch-dir default the 0.3.1
    bug stamps). A specific, non-suspect label is EVIDENCE — never overridden,
    even for an episodic — so a correctly-labeled session isn't corrupted because
    its transcript happens to mention another project."""
    proj = (row["project"] or "").strip()
    return (not proj) or (_canon(store, proj) in suspect)


def anchors_and_candidates(store, suspect: set[str]):
    """Split live memories into ANCHORS (reliably-labeled: non-episodic with a
    project) and CANDIDATES (unscoped or suspect-labeled). Anchors keep their
    full set regardless of suspect so the centroids stay well-attested."""
    rows = store.conn.execute(
        "SELECT id, kind, project, name, gist, detail FROM memory "
        "WHERE superseded_by IS NULL").fetchall()
    anchors, candidates = [], []
    for r in rows:
        if (r["project"] or "").strip() and r["kind"] != "episodic":
            anchors.append(r)
        if _is_candidate(store, r, suspect):
            candidates.append(r)
    return anchors, candidates


def _row_vectors(store, model: str, ids):
    """{memory_id: mean-of-chunks vector} for the given ids and embedding model."""
    ids = list(ids)
    if not ids:
        return {}
    from .vectors import from_blob
    by_id: dict[int, list[list[float]]] = {}
    qmarks = ",".join("?" * len(ids))
    for r in store.conn.execute(
            f"SELECT memory_id, vector FROM embedding WHERE model = ? "
            f"AND memory_id IN ({qmarks})", [model, *ids]):
        by_id.setdefault(r["memory_id"], []).append(from_blob(r["vector"]))
    return {mid: _mean(vs) for mid, vs in by_id.items()}


def propose(store, *, min_margin: float | None = None, suspect=()) -> dict:
    """Re-projection proposals: for each candidate, the project its content points
    to, the confidence, and whether that differs from its current label. Reports
    only — see apply_proposals. min_margin defaults per mode. `suspect` names extra
    labels (beyond NULL) to reconsider — the launch-dir default the bug stamps."""
    suspect = {_canon(store, s) for s in suspect}
    anchors, candidates = anchors_and_candidates(store, suspect)
    embedder = store._resolve_embedder(None)
    mode = "vector" if embedder is not None else "keyword"

    proposals: list[dict] = []
    if mode == "vector":
        mm = DEFAULT_MIN_MARGIN if min_margin is None else min_margin
        avecs = _row_vectors(store, embedder.name, [a["id"] for a in anchors])
        grouped: dict[str, list[list[float]]] = {}
        for a in anchors:
            v = avecs.get(a["id"])
            if v:
                grouped.setdefault(_canon(store, a["project"]), []).append(v)
        # Mean-center: subtract the global anchor mean so each centroid captures
        # what is DISTINCTIVE about its project, not raw proximity. Without this a
        # broad catch-all project (its centroid ≈ the global mean) has high cosine
        # with everything and attracts every candidate.
        gmean = _mean([v for vs in grouped.values() for v in vs])
        centroids = {p: _sub(_mean(vs), gmean) for p, vs in grouped.items()}
        cvecs = _row_vectors(store, embedder.name, [c["id"] for c in candidates])
        for c in candidates:
            v = cvecs.get(c["id"])
            if not v:
                continue
            best, margin, scores = classify_vec(_sub(v, gmean), centroids)
            proposals.append(_proposal(store, c, best, margin, scores, mm))
    else:
        mm = DEFAULT_MIN_WORD_MARGIN if min_margin is None else min_margin
        vocab: dict[str, set[str]] = {}
        for a in anchors:
            vocab.setdefault(_canon(store, a["project"]), set()).update(
                content_words(f"{a['name'] or ''} {a['gist'] or ''} {a['detail'] or ''}"))
        for c in candidates:
            words = content_words(f"{c['name'] or ''} {c['gist'] or ''} {c['detail'] or ''}")
            best, margin, scores = classify_words(words, vocab)
            proposals.append(_proposal(store, c, best, margin, scores, mm))

    proposals = [p for p in proposals if p is not None]
    confident = [p for p in proposals if p["confident"]]
    return {
        "mode": mode,
        "min_margin": mm,
        "suspect": sorted(suspect),
        "anchors": len(anchors),
        "candidates": len(candidates),
        "projects": sorted({_canon(store, a["project"]) for a in anchors}),
        "proposals": confident,
        "ambiguous": [p for p in proposals if not p["confident"]],
    }


def _proposal(store, row, best, margin, scores, min_margin) -> dict | None:
    """Build one proposal record, or None when the content points at the project
    the memory already belongs to (canonically — alias families count as same)."""
    if best is None:
        return None
    cur = (row["project"] or "").strip()
    if cur and _canon(store, cur) == best:
        return None       # already correctly scoped (or alias-equivalent)
    return {
        "id": row["id"],
        "current": cur or None,
        "proposed": best,
        "margin": round(float(margin), 4),
        "confident": margin >= min_margin,
        "scores": scores,
        "gist": (row["gist"] or "")[:80],
    }


def apply_proposals(store, proposals: list[dict]) -> dict:
    """Write the proposed project onto each row, recording (id, old_project) in
    meta so `undo` can restore. Returns counts. Caller passes the rows to apply
    (typically the confident set)."""
    undo = json.loads(get_config(store, UNDO_KEY, "[]") or "[]")
    applied = 0
    for p in proposals:
        cur = store.conn.execute("SELECT project FROM memory WHERE id = ? "
                                 "AND superseded_by IS NULL", (p["id"],)).fetchone()
        if cur is None:
            continue
        undo.append([p["id"], cur["project"]])
        store.conn.execute("UPDATE memory SET project = ? WHERE id = ?",
                           (p["proposed"], p["id"]))
        applied += 1
    store.conn.commit()
    set_config(store, UNDO_KEY, json.dumps(undo))
    return {"applied": applied, "undo_size": len(undo)}


def undo(store) -> dict:
    """Restore every project label changed by apply_proposals, then clear the
    undo set. A no-op (restored=0) when there is nothing to undo."""
    undo_list = json.loads(get_config(store, UNDO_KEY, "[]") or "[]")
    restored = 0
    for mid, old in reversed(undo_list):
        store.conn.execute("UPDATE memory SET project = ? WHERE id = ?", (old, mid))
        restored += 1
    store.conn.commit()
    set_config(store, UNDO_KEY, "[]")
    return {"restored": restored}


def format_report(result: dict, *, applied: dict | None = None) -> str:
    suspect = result.get("suspect") or []
    out = [f"reproject ({result['mode']} mode, min_margin {result['min_margin']})",
           f"anchors: {result['anchors']} across {len(result['projects'])} projects "
           f"({', '.join(result['projects']) or 'none'})",
           f"candidates scanned: {result['candidates']} "
           f"(unscoped{' + suspect: ' + ', '.join(suspect) if suspect else ''})"]
    props = result["proposals"]
    if not props:
        out.append("no confident re-projections — labels look content-consistent.")
    else:
        out.append(f"--- confident re-projections ({len(props)}) ---")
        for p in sorted(props, key=lambda x: -x["margin"]):
            out.append(f"  #{p['id']:>4}  {str(p['current']):>20} -> {p['proposed']:<16} "
                       f"(margin {p['margin']})  {p['gist']}")
    if result["ambiguous"]:
        out.append(f"--- ambiguous, left as-is ({len(result['ambiguous'])}) "
                   f"(lower --min-margin to include) ---")
        for p in sorted(result["ambiguous"], key=lambda x: -x["margin"])[:20]:
            out.append(f"  #{p['id']:>4}  {str(p['current']):>20} ?> {p['proposed']:<16} "
                       f"(margin {p['margin']})  {p['gist']}")
    if applied is not None:
        out.append(f"APPLIED {applied['applied']} (undo with `reproject --undo`; "
                   f"{applied['undo_size']} change(s) recorded).")
    elif props:
        out.append("dry run — nothing changed. `reproject --apply` to write, "
                   "`--undo` to revert the last apply.")
    return "\n".join(out)
