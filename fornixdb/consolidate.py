"""Consolidation — the "sleep step" (Design §13).

P3a ships the mechanical, judgment-free part: decay is live in ranking
(core.effective_salience) and this module reports when a consolidation pass is
DUE. P3c adds `propose()`: a mechanical worklist of the four judgment-requiring
moves (session distillation, gist repair, near-duplicate merges, contradiction
candidates) which a thinking AI reviews and applies via existing primitives
(store/supersede/link/tag/set-gist) — FornixDB never rewrites memories on its
own (§6.5). propose() is read-only and model-free: pair similarity comes from
vectors already in the embedding table, so it runs on AI-less endpoints too
(the lists are simply emptier there).

Cadence (owner-approved 2026-06-11): due after 7 days or 10 new sessions since
the last pass; per-store overrides in meta (consolidate_days / consolidate_sessions).
"""

from __future__ import annotations

import re
from datetime import datetime

from .core import MemoryStore, now_iso
from .multistore import get_config, set_config
from .vectors import cosine, from_blob

DEFAULT_DAYS = 7
DEFAULT_SESSIONS = 10

# §13.3 thresholds (provisional, same spirit as core's ranking constants)
GIST_MAX_CHARS = 200          # longer gists are summaries that failed
GIST_NONALPHA_FRAC = 0.30     # hash/jargon density that hurts embeddings + scanning
MERGE_COSINE = 0.88           # near-duplicate: propose a merge
CONTRA_COSINE = 0.70          # same-topic band: AI checks for contradiction
ASSOC_COSINE = 0.60           # related-but-distinct: a NEW connection to weave
NEAR_DUP_COSINE = 0.85        # write-time "you may have just re-stored #N" nudge —
                              # below MERGE so abbreviation-level restatements
                              # ("1000 req/min" vs "1000 requests per minute" ≈ 0.88)
                              # still nudge; corrections (~0.96) and unrelated
                              # (<0.5) stay clearly on either side
MAX_PAIR_PROPOSALS = 15       # per list, best first — a pass is incremental

# Lifecycle-aware heal (Fix A, 2026-06-16): a memory recording an OPEN task
# ("(5) performance — investigate the lag") stays live and recallable even after
# a LATER memory records its closure ("tasks 5+6 shipped"), because closure is
# usually logged as a NEW sibling rather than a supersede. Similarity alone can't
# fix this: "X is a task" and "X is done" are OPPOSITE in status yet near-identical
# in vocabulary, so embeddings group them as merge/relates, never supersede WITH
# DIRECTION. The missing signal is LIFECYCLE LANGUAGE — an older memory phrased as
# a task, resolved by a newer one phrased as closure. Similarity then only confirms
# the two are about the same subject; the language carries the direction.
RESOLUTION_COSINE = 0.50      # "same subject" gate; the language gates carry precision
_CLOSURE_RE = re.compile(
    r"\b(shipped|done|resolved|closed|completed?|finished|fixed|implemented|"
    r"merged|landed)\b|\bcommit\s+[0-9a-f]{7,}", re.I)
_TASK_RE = re.compile(
    r"\b(tasks?|to-?dos?|backlog|remaining|next steps?|to build|to add|"
    r"should build|needs? to|planned|plan to|open items?|wip|in progress|"
    r"pending|investigate)\b", re.I)

_UNSET = object()


def status(store: MemoryStore) -> dict:
    last = get_config(store, "last_consolidated")
    days_cfg = int(get_config(store, "consolidate_days", str(DEFAULT_DAYS)))
    sessions_cfg = int(get_config(store, "consolidate_sessions", str(DEFAULT_SESSIONS)))

    if last:
        days_since = (datetime.now() - datetime.fromisoformat(last)).days
        new_sessions = store.conn.execute(
            "SELECT count(*) c FROM session WHERE started > ?", (last,)
        ).fetchone()["c"]
        new_memories = store.conn.execute(
            "SELECT count(*) c FROM memory WHERE recorded_time > ?", (last,)
        ).fetchone()["c"]
    else:
        days_since, new_sessions = None, store.conn.execute(
            "SELECT count(*) c FROM session").fetchone()["c"]
        new_memories = store.conn.execute(
            "SELECT count(*) c FROM memory").fetchone()["c"]

    due = last is None or days_since >= days_cfg or new_sessions >= sessions_cfg
    reason = ("never consolidated" if last is None else
              f"{days_since}d since last (threshold {days_cfg}d)" if days_since >= days_cfg else
              f"{new_sessions} new sessions (threshold {sessions_cfg})" if new_sessions >= sessions_cfg
              else "not due")
    return {
        "last_consolidated": last,
        "days_since": days_since,
        "new_sessions": new_sessions,
        "new_memories": new_memories,
        "due": due,
        "reason": reason,
    }


def mark_done(store: MemoryStore) -> str:
    """Record that a consolidation pass completed (called by the AI that ran
    it — P3c — or manually after a hand-run pass)."""
    ts = now_iso()
    set_config(store, "last_consolidated", ts)
    return ts


# ---------------------------------------------------------- the AI worklist

def _gist_problem(gist: str, detail: str | None) -> str | None:
    if len(gist) > GIST_MAX_CHARS:
        return f"{len(gist)} chars (max {GIST_MAX_CHARS})"
    tokens = gist.split()
    if tokens:
        # half-alphabetic counts as noise: hex ids ("d61c069") sit near 50%
        nonalpha = sum(1 for t in tokens
                       if sum(c.isalpha() for c in t) <= len(t) / 2)
        if nonalpha / len(tokens) > GIST_NONALPHA_FRAC:
            return f"{nonalpha}/{len(tokens)} tokens are non-alpha"
    # A gist that is a raw truncation of a MUCH longer detail wants a real
    # summary. But gist==detail (or detail only a little longer) is a normal
    # one-line memory — and remember() stores gist=content[:120], so flagging
    # every short/truncated memory is noise. Only flag a genuine summary gap.
    if detail and len(detail) > 2 * len(gist) and detail.startswith(gist):
        return "gist is a raw truncation of much longer detail"
    return None


def _pair_scan(store: MemoryStore, exclude_ids: set[int]) -> tuple[list, list, list]:
    """Cosine pairs from STORED vectors (dominant model only), best first:
      merges       — same-kind near-duplicates (>= MERGE_COSINE)
      contras      — same-kind same-topic band (CONTRA..MERGE): check contradiction
      associations — related-but-UNLINKED pairs to connect with a NEW 'relates'
                     link: cross-kind at any qualifying cosine, or same-kind in
                     [ASSOC, CONTRA). This is dreaming's generative half — making
                     connections that didn't exist, not just pruning."""
    model_row = store.conn.execute(
        "SELECT model, count(*) c FROM embedding GROUP BY model "
        "ORDER BY c DESC LIMIT 1").fetchone()
    if model_row is None:
        return [], [], []
    # Episodic rows are TIMELINE events, not standing knowledge: two similar
    # session/diary summaries ("Chat <date>: reviewed the morning" vs "…last
    # night") are distinct events, not an outdated fact or a duplicate to merge —
    # and they flood the pair lists with templated false neighbors. Consolidation
    # pairs only operate on facts/lessons/refs; the timeline axis handles episodic.
    rows = store.conn.execute(
        """SELECT m.id, m.kind, m.gist, e.vector FROM memory m
           JOIN embedding e ON e.memory_id = m.id AND e.model = ? AND e.chunk = 0
           WHERE m.superseded_time IS NULL AND m.kind != 'episodic'""",
        (model_row["model"],)).fetchall()
    rows = [r for r in rows if r["id"] not in exclude_ids]
    vecs = {r["id"]: from_blob(r["vector"]) for r in rows}
    supersede_linked = {(r["memory_id"], r["related_id"]) for r in store.conn.execute(
        "SELECT memory_id, related_id FROM memory_link WHERE relation='supersedes'")}
    # an association must be NEW — skip any pair already connected by any relation
    any_linked = {(r["memory_id"], r["related_id"]) for r in store.conn.execute(
        "SELECT memory_id, related_id FROM memory_link")}

    merges, contras, assocs = [], [], []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            cos = cosine(vecs[a["id"]], vecs[b["id"]])
            if cos < ASSOC_COSINE:
                continue
            ab, ba = (a["id"], b["id"]), (b["id"], a["id"])
            if a["kind"] == b["kind"] and cos >= CONTRA_COSINE:
                if ab in supersede_linked or ba in supersede_linked:
                    continue
                entry = {"ids": [a["id"], b["id"]], "cosine": round(cos, 3),
                         "kind": a["kind"], "gists": [a["gist"], b["gist"]]}
                (merges if cos >= MERGE_COSINE else contras).append(entry)
            else:  # cross-kind, or same-kind below the contradiction band
                if ab in any_linked or ba in any_linked:
                    continue
                assocs.append({"ids": [a["id"], b["id"]], "cosine": round(cos, 3),
                               "kinds": [a["kind"], b["kind"]],
                               "gists": [a["gist"], b["gist"]]})
    for lst in (merges, contras, assocs):
        lst.sort(key=lambda e: e["cosine"], reverse=True)
    return (merges[:MAX_PAIR_PROPOSALS], contras[:MAX_PAIR_PROPOSALS],
            assocs[:MAX_PAIR_PROPOSALS])


def _resolution_scan(store: MemoryStore, exclude_ids: set[int]) -> list:
    """Lifecycle-aware heal (Fix A): an OLDER task memory resolved by a NEWER one
    carrying closure language — propose supersede(old -> new) WITH direction.

    Differs from _pair_scan in two ways that matter: it INCLUDES episodic rows
    (the "shipped/done" note usually lands in a session/diary memory, the kind
    _pair_scan deliberately skips) and it gates on lifecycle LANGUAGE, not
    similarity band. Cosine here only confirms the pair is about the same
    subject (>= RESOLUTION_COSINE); the task/closure regexes give the supersede
    its direction — the one thing a status-flip's near-identical embeddings
    cannot. Propose-only (§6.5): the reviewing AI applies the supersede."""
    model_row = store.conn.execute(
        "SELECT model, count(*) c FROM embedding GROUP BY model "
        "ORDER BY c DESC LIMIT 1").fetchone()
    if model_row is None:
        return []
    rows = store.conn.execute(
        """SELECT m.id, m.kind, m.gist, m.detail, m.recorded_time, e.vector
           FROM memory m
           JOIN embedding e ON e.memory_id = m.id AND e.model = ? AND e.chunk = 0
           WHERE m.superseded_time IS NULL""",
        (model_row["model"],)).fetchall()
    rows = [r for r in rows if r["id"] not in exclude_ids]
    vecs = {r["id"]: from_blob(r["vector"]) for r in rows}
    text = {r["id"]: f"{r['detail'] or ''} {r['gist'] or ''}" for r in rows}
    supersede_linked = {(r["memory_id"], r["related_id"]) for r in store.conn.execute(
        "SELECT memory_id, related_id FROM memory_link WHERE relation='supersedes'")}

    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            cos = cosine(vecs[a["id"]], vecs[b["id"]])
            if cos < RESOLUTION_COSINE:
                continue
            # the NEWER memory is the resolution; order by recorded_time so the
            # supersede direction is unambiguous (the closure entry wins)
            older, newer = (a, b) if (a["recorded_time"] or "") <= (b["recorded_time"] or "") \
                else (b, a)
            if not (_TASK_RE.search(text[older["id"]])
                    and _CLOSURE_RE.search(text[newer["id"]])):
                continue
            if (older["id"], newer["id"]) in supersede_linked \
                    or (newer["id"], older["id"]) in supersede_linked:
                continue
            out.append({"ids": [older["id"], newer["id"]], "cosine": round(cos, 3),
                        "kinds": [older["kind"], newer["kind"]],
                        "gists": [older["gist"], newer["gist"]]})
    out.sort(key=lambda e: e["cosine"], reverse=True)
    return out[:MAX_PAIR_PROPOSALS]


def propose(store: MemoryStore) -> dict:
    """Emit the §13.3 worklist. Read-only: nothing is tagged, rewritten, or
    embedded here — the reviewing AI applies what survives its judgment."""
    now = datetime.now()
    _, floors = store._decay_cfg()

    distill = []
    for r in store.conn.execute(
            """SELECT m.* FROM memory m
               WHERE m.kind = 'episodic' AND m.source = 'claude-code-transcript'
                 AND m.superseded_time IS NULL
                 AND m.id NOT IN (SELECT mt.memory_id FROM memory_topic mt
                                  JOIN topic t ON t.id = mt.topic_id
                                  WHERE t.name = 'distilled')"""):
        row = dict(r)
        eff = store.effective_salience(row, now)
        if eff > floors.get("episodic", 0.05):  # fully-decayed sessions can wait
            distill.append({"id": row["id"], "eff_salience": round(eff, 3),
                            "gist": row["gist"], "transcript": row["source_ref"]})
    distill.sort(key=lambda d: d["eff_salience"], reverse=True)
    pending_distill = {d["id"] for d in distill}

    gists = []
    for r in store.conn.execute(
            "SELECT id, gist, detail FROM memory WHERE superseded_time IS NULL"):
        if r["id"] in pending_distill:  # distillation rewrites these anyway
            continue
        problem = _gist_problem(r["gist"] or "", r["detail"])
        if problem:
            gists.append({"id": r["id"], "problem": problem, "gist": r["gist"]})

    # undistilled session gists are templated ("Session <date> (N user turns…")
    # and would flood the pair lists with false neighbors — scan without them
    merges, contradictions, associations = _pair_scan(store, pending_distill)
    resolutions = _resolution_scan(store, pending_distill)

    # a same-kind task/closure pair can surface in both scans; resolutions carry
    # direction, so they win — drop those pairs from the fuzzier lists
    res_pairs = {frozenset(r["ids"]) for r in resolutions}
    merges = [m for m in merges if frozenset(m["ids"]) not in res_pairs]
    contradictions = [c for c in contradictions if frozenset(c["ids"]) not in res_pairs]

    return {"distill": distill, "gists": gists, "merges": merges,
            "contradictions": contradictions, "associations": associations,
            "resolutions": resolutions}


# ------------------------------------------------------------- sleep / dream

def _dream_narrative(st: dict, counts: dict, woven: int = 0) -> str:
    """The user-facing 'dreaming' read-back (owner's framing 2026-06-14)."""
    when = "first dream" if st.get("last_consolidated") is None \
        else f"last dream {st['last_consolidated'][:10]}"
    if counts["total"] == 0 and not woven:
        return ("\U0001f4a4 FornixDB drifted off and found nothing to "
                f"reconcile — memories are tidy ({when}).")

    def plural(n, one, many):
        return f"{n} {one if n == 1 else many}"

    parts = []
    if counts.get("resolutions"):
        # the strongest signal: an older task memory a newer entry has closed —
        # supersede direction is already known (close the open one)
        parts.append(plural(counts["resolutions"],
                            "completed task to close",
                            "completed tasks to close"))
    if counts["contradictions"]:
        # the headline: a later fix stored under a different title leaves the
        # stale one live; contradiction pairs surface exactly those
        parts.append(plural(counts["contradictions"],
                            "possible outdated memory to reconcile",
                            "possible outdated memories to reconcile"))
    if counts["associations"] and not woven:
        # the generative half: connections that didn't exist before the dream
        # (when woven, the woke-clause below reports them instead)
        parts.append(plural(counts["associations"],
                            "new connection to weave", "new connections to weave"))
    if counts["merges"]:
        parts.append(plural(counts["merges"], "near-duplicate to merge",
                            "near-duplicates to merge"))
    if counts["distill"]:
        parts.append(plural(counts["distill"], "session to distill",
                            "sessions to distill"))
    if counts["gists"]:
        parts.append(plural(counts["gists"], "gist to tidy", "gists to tidy"))

    woke = (f" Wove {plural(woven, 'new connection', 'new connections')} while "
            "dreaming." if woven else "")
    tail = (" Review and apply — nothing else is changed on its own (§6.5)."
            if parts else "")
    lead = ("\U0001f4a4 FornixDB is dreaming… surfaced " + "; ".join(parts) + "."
            if parts else "\U0001f4a4 FornixDB dreamed.")
    return lead + woke + tail


def _superseded_count(store: MemoryStore) -> int:
    return store.conn.execute(
        "SELECT count(*) c FROM memory WHERE superseded_time IS NOT NULL"
    ).fetchone()["c"]


def wake_summary(store: MemoryStore, baseline: int) -> dict:
    """What the consolidation pass reconciled: the NET new superseded rows since
    the pass opened (`baseline` = the superseded count at open). A count delta,
    not a timestamp comparison, so it is exact even when many ops land in the
    same second (now_iso is second-granular). Supersede is the one healing move
    tracked — it covers outdated-fix supersedes AND near-duplicate merges; links,
    gist rewrites, and tags aren't counted (weaving is reported by the pass that
    did it)."""
    return {"reconciled": max(0, _superseded_count(store) - int(baseline))}


def _wake_narrative(applied: dict) -> str:
    def plural(n, one, many):
        return f"{n} {one if n == 1 else many}"
    parts = []
    if applied["reconciled"]:
        parts.append(plural(applied["reconciled"],
                            "outdated/duplicate memory reconciled",
                            "outdated/duplicate memories reconciled"))
    if applied["woven"]:
        parts.append(plural(applied["woven"], "new connection woven",
                            "new connections woven"))
    if not parts:
        return "\U0001f4a4 FornixDB woke — pass complete; nothing needed changing."
    return "\U0001f4a4 FornixDB woke — " + "; ".join(parts) + ". Pass complete."


def dream(store: MemoryStore, weave: bool = False, done: bool = False) -> dict:
    """Sleep/Dream mode (Design §13): a single user-visible consolidation pass.
    Wraps status() + propose() into a 'dream report' — cadence, counts, the
    worklist, and a narrated read-back.

    Two halves: PRUNING/healing (the AI reviews and applies the judgment moves —
    supersede an outdated memory, merge near-duplicates, set-gist a messy gist,
    distill a session — propose-not-dispose, §6.5) and the GENERATIVE half,
    ASSOCIATIONS: related-but-unlinked pairs the dream can connect with NEW
    'relates' links. Linking is non-destructive (it adds a relationship, never
    rewrites or removes a memory), so `weave=True` opts into making those links
    in the same pass and reports how many; default stays read-only/suggest.

    Headline of the healing half: a later fix stored under a different title
    leaves the stale original live and recallable — the contradiction pairs are
    exactly those candidates.

    A pass opens on the first dream() call and CLOSES on done=True ("wake"):
    the wake reports what was reconciled DURING the pass (wake_summary —
    supersedes/merges the AI applied between open and close) plus what it wove
    this call, resets the DUE clock (mark_done), and clears the pass marker. The
    narrative becomes the wake read-back instead of the entering one.

    Refused on a read-only store: consolidation is a maintenance operation, so a
    frozen (vendor-shipped read-only) store raises FrozenStoreError rather than
    proposing work that can never be applied."""
    store._check_writable()
    # A pass opens on the first dream() call and closes on done=True. Snapshot the
    # superseded-row count at open; the wake reports the NET delta — what was
    # reconciled DURING the pass — robust to same-second timestamp collisions and
    # never counting the store's prior supersede history.
    marker = get_config(store, "dream_pass_super0")
    if marker is None:
        marker = str(_superseded_count(store))
        set_config(store, "dream_pass_super0", marker)
    st = status(store)
    work = propose(store)
    woven = 0
    if weave:
        for a in work["associations"]:
            store.link(a["ids"][0], a["ids"][1], relation="relates")
            woven += 1
    counts = {k: len(work[k]) for k in ("distill", "gists", "merges",
                                        "contradictions", "associations",
                                        "resolutions")}
    counts["total"] = sum(counts.values())
    counts["woven"] = woven

    applied = None
    if done:  # "wake": report what was reconciled DURING the pass, then close it
        applied = wake_summary(store, marker)
        applied["woven"] = woven
        mark_done(store)
        store.conn.execute("DELETE FROM meta WHERE key = 'dream_pass_super0'")
        store.conn.commit()

    if done:
        narrative = _wake_narrative(applied)
        # nudge the (user-driven) healing: name the heal candidates still
        # standing so they don't silently slip past a closed pass. Never forces
        # a reconcile — a contradiction pair can be two legitimately distinct
        # memories; the decision is the AI's / owner's.
        # resolutions first: their supersede direction is already settled
        remaining = work["resolutions"] + work["contradictions"] + work["merges"]
        if remaining:
            one = len(remaining) == 1
            lines = [narrative,
                     f"{len(remaining)} outdated/duplicate pair{'' if one else 's'} "
                     f"still need{'s' if one else ''} a decision — supersede the "
                     "stale one (default: keep the newer):"]
            for mm in remaining[:5]:
                ids = mm["ids"]
                gmap = dict(zip(ids, mm["gists"]))
                # order older -> newer by recorded_time so the supersede
                # direction is unambiguous (default: the newer entry wins)
                rt = {r["id"]: (r["recorded_time"] or "") for r in
                      store.conn.execute(
                          "SELECT id, recorded_time FROM memory WHERE id IN "
                          f"({','.join('?' * len(ids))})", ids)}
                older, newer = sorted(ids, key=lambda i: rt.get(i, ""))
                lines.append(f"  supersede old=#{older} new=#{newer}: "
                             f"{gmap[older][:40]} -> {gmap[newer][:40]}")
            narrative = "\n".join(lines)
    else:
        narrative = _dream_narrative(st, counts, woven)
    return {"status": st, "counts": counts, "work": work, "woven": woven,
            "applied": applied, "narrative": narrative}


def supersede_suggestion(store: MemoryStore, new_id: int, text: str,
                         kind: str, embedder=_UNSET) -> dict | None:
    """Write-time near-duplicate nudge: the proactive half of orphaned-fix
    prevention. After a NEW memory is stored under a fresh title, check whether
    it closely matches an EXISTING (non-superseded) memory of the SAME kind —
    i.e. it may be an UPDATE that should supersede the old one rather than
    silently create an orphan that leaves stale info recallable. Returns
    {id, gist, cosine} for the best such match, or None.

    Two nudges, returned with a `reason`:
      "near-duplicate" — same kind, near-duplicate band (NEAR_DUP_COSINE): may be
                         an UPDATE that should supersede rather than orphan.
      "resolves"       — this memory carries CLOSURE language (shipped/done/…) and
                         loosely matches an OLDER memory phrased as a TASK: it may
                         CLOSE that open task (Fix B, the write-time twin of the
                         dream pass's _resolution_scan). Cross-kind, looser band —
                         the language carries the precision the cosine can't.

    SUGGEST ONLY — never auto-supersedes: a high-similarity pair can still be
    two legitimately distinct memories, and only the writer knows. Model-free
    safe: with no embedder (or no stored vectors) it returns None, like the
    dream pass."""
    if embedder is _UNSET:
        from .vectors import get_default_embedder
        embedder = get_default_embedder()
    if embedder is None or not text:
        return None
    try:
        from .vectors import similar
        matches = similar(store, embedder, text, limit=8)
        # 1) same-kind near-duplicate (the original nudge)
        for mid, cos in matches:
            if cos < NEAR_DUP_COSINE:
                break  # sorted best-first: nothing else qualifies for THIS band
            if mid == new_id:
                continue
            row = store.conn.execute(
                "SELECT id, gist, kind FROM memory WHERE id = ?", (mid,)).fetchone()
            if row and row["kind"] == kind:
                return {"id": row["id"], "gist": row["gist"],
                        "cosine": round(cos, 3), "reason": "near-duplicate"}
        # 2) lifecycle resolution: if THIS memory reads as closure, does it close
        #    an older task memory? cross-kind, RESOLUTION_COSINE band.
        if _CLOSURE_RE.search(text):
            for mid, cos in matches:
                if cos < RESOLUTION_COSINE:
                    break
                if mid == new_id:
                    continue
                row = store.conn.execute(
                    "SELECT id, gist, detail FROM memory WHERE id = ?", (mid,)).fetchone()
                if row and _TASK_RE.search(f"{row['detail'] or ''} {row['gist'] or ''}"):
                    return {"id": row["id"], "gist": row["gist"],
                            "cosine": round(cos, 3), "reason": "resolves"}
        return None
    except Exception:
        return None
