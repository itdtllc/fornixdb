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

from .core import (HELPFUL_USE_WEIGHT, REFERENCED_USE_WEIGHT, MemoryStore,
                   now_iso)
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
CHRONIC_MIN_PUSHES = 6        # pushes with zero downstream use before a row is
                              # a DISPOSITION question; the per-memory floor
                              # penalty already quiets it from 3 (core.
                              # FLOOR_MIN_IMPRESSIONS) and saturates by ~7, so
                              # by 6 the mechanical remedy has fully applied
                              # and the row is still being pushed

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


def _headline(gist: str | None, detail: str | None, lede_chars: int = 160) -> str:
    """The text the lifecycle gates should read: a memory's GIST plus the first
    line of its detail — its announced subject — NOT the whole body.

    Why this matters (the precision the 0.50 cosine cannot give): a long
    status/resume memory incidentally contains the project's entire vocabulary —
    task words AND closure words alike. Matching _TASK_RE / _CLOSURE_RE over its
    full text then misfires: a design note that merely MENTIONS "shipped" once,
    deep in a table, reads as a closure and gets proposed as resolving some
    unrelated task it happens to share nouns with. A genuine task or closure note
    states its status in its headline (the gist is the human-written title; for a
    short single-purpose memory the gist *is* the whole memory). Scoping the gates
    to the headline keeps the real lifecycle pairs and drops the incidental ones."""
    g = (gist or "").strip()
    lede = (detail or "").strip().split("\n", 1)[0][:lede_chars]
    return f"{g}\n{lede}"


_UNSET = object()


def _distinct_pairs(store: MemoryStore) -> set:
    """Pairs the reviewer has accepted as legitimately distinct — a 'distinct'
    link in either direction. Unordered (frozenset) so callers never care which
    side the accept was written from."""
    return {frozenset((r["memory_id"], r["related_id"])) for r in store.conn.execute(
        "SELECT memory_id, related_id FROM memory_link WHERE relation='distinct'")}


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
    # a reviewed pair accepted as legitimately distinct (the pair-level
    # reality-ok/noise-ok) — never re-proposed as merge or contradiction
    distinct_linked = _distinct_pairs(store)
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
                if frozenset(ab) in distinct_linked:
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
    # cosine runs on the full-text embeddings (above); the lifecycle gates read
    # only the HEADLINE (see _headline) — incidental "shipped"/"backlog" keywords
    # buried in a long status memory must not pass as task/closure direction.
    head = {r["id"]: _headline(r["gist"], r["detail"]) for r in rows}
    supersede_linked = {(r["memory_id"], r["related_id"]) for r in store.conn.execute(
        "SELECT memory_id, related_id FROM memory_link WHERE relation='supersedes'")}
    distinct_linked = _distinct_pairs(store)

    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            cos = cosine(vecs[a["id"]], vecs[b["id"]])
            if cos < RESOLUTION_COSINE:
                continue
            if frozenset((a["id"], b["id"])) in distinct_linked:
                continue
            # the NEWER memory is the resolution; order by recorded_time so the
            # supersede direction is unambiguous (the closure entry wins)
            older, newer = (a, b) if (a["recorded_time"] or "") <= (b["recorded_time"] or "") \
                else (b, a)
            if not (_TASK_RE.search(head[older["id"]])
                    and _CLOSURE_RE.search(head[newer["id"]])):
                continue
            if (older["id"], newer["id"]) in supersede_linked \
                    or (newer["id"], older["id"]) in supersede_linked:
                continue
            out.append({"ids": [older["id"], newer["id"]], "cosine": round(cos, 3),
                        "kinds": [older["kind"], newer["kind"]],
                        "gists": [older["gist"], newer["gist"]]})
    out.sort(key=lambda e: e["cosine"], reverse=True)
    return out[:MAX_PAIR_PROPOSALS]


# ------------------------------------------------------------ reality check
# A memory that points at the filesystem can silently rot: the file moves or
# is deleted and the pointer stays live and recallable. (The motivating case,
# 2026-07-01: the project's own design doc vanished in a disk reorg and its
# pointer memories sat stale for two weeks until someone happened to reach for
# it.) Human memory is embedded in perception — you notice the gap the next
# time you look at the shelf. This scan is that primitive sense organ: during
# a dream, verify file-path claims against the world and surface what is
# missing. Propose-not-dispose (§6.5): a missing path may be an unmounted
# volume, a moved file, or a memory worth superseding — judgment stays with
# the reviewing AI/owner. Episodic rows are exempt for the same reason they
# never carry a staleness flag: they are history, not claims about the
# present. Only paths under THIS machine's home are checked, so pointers to
# other machines (the PC, network shares) never false-positive.

_FS_PATH_RE = re.compile(r"(?:~|/Users)/[^\s`'\"()\[\]{}<>*,;|]+")
_PATH_STRIP = ".,;:!?…"          # sentence punctuation that rides a path's tail
MAX_REALITY_PER_MEMORY = 5       # a pathological detail can't flood the list
# Paths that are missing by NATURE, not by rot — flagging them is noise
# (measured on the live store's first run, 2026-07-01):
_EPHEMERAL_SEGMENTS = ("Library/Developer/CoreSimulator/",  # sim containers
                       "Library/Caches/")                    # caches


def _extract_paths(text: str) -> list[list[str]]:
    """Candidate filesystem paths in prose: `~/...` or `/Users/...`, trailing
    sentence punctuation stripped. Each entry is a CANDIDATE LIST — the match
    plus space-extended variants, because a path with a space in a segment
    (`Test Cases/…`, `v1.4.0 Data/…`) truncates at the space and reads as
    missing when the real thing exists; if ANY candidate exists the pointer
    is fine. Excluded as unjudgeable (every pattern measured on live runs):
    matches truncated by a placeholder (`AppStore/v<X.Y.Z>` → `…/v`), elided
    paths (`/Users/dad/.../x`), template prefixes whose last segment ends
    `_`/`-`, and ephemeral OS containers."""
    out = []
    text = text or ""
    for m in _FS_PATH_RE.finditer(text):
        p = m.group().rstrip(_PATH_STRIP)
        nxt = text[m.end():m.end() + 1]
        if nxt in "<*{…":                     # placeholder cut the match short
            continue
        if p.rstrip("/").count("/") < 2:      # bare "~/x" is too generic to judge
            continue
        if "..." in p:                        # prose elision, never a real name
            continue
        if p.rstrip("/").rsplit("/", 1)[-1].endswith(("_", "-")):
            continue
        if any(seg in p for seg in _EPHEMERAL_SEGMENTS):
            continue
        cands = [p]
        if nxt == " ":                        # maybe a space inside a segment
            tail = re.split(r"[`'\"()\[\]{}<>,;|\n]", text[m.end() + 1:], maxsplit=1)[0]
            word = tail.split(" ", 1)[0].rstrip(_PATH_STRIP)
            if word:
                joined = f"{p} {word}"
                cands.append(joined)          # …/Test Cases/P20/TestPlan.md
                cut = joined.find("/", len(p) + 1)
                if cut != -1:
                    cands.append(joined[:cut])  # …/Test Cases
        out.append(cands)
    return out


def _reality_scan(store: MemoryStore) -> list:
    """Live non-episodic memories whose gist/detail names a path under this
    machine's home that no longer exists. Rows tagged `reality-ok` are the
    reviewed-and-accepted ones (a historical mention, a documented default,
    a described absence) — skipped so an accepted flag stays accepted."""
    import os
    home = os.path.expanduser("~")
    out = []
    for r in store.conn.execute(
            """SELECT id, gist, detail FROM memory
               WHERE superseded_time IS NULL AND kind != 'episodic'
                 AND id NOT IN (SELECT mt.memory_id FROM memory_topic mt
                                JOIN topic t ON t.id = mt.topic_id
                                WHERE t.name = 'reality-ok')
               ORDER BY id"""):
        missing, seen = [], set()
        for cands in _extract_paths(f"{r['gist'] or ''}\n{r['detail'] or ''}"):
            full = os.path.expanduser(cands[0])
            if not full.startswith(home + os.sep) or full in seen:
                continue
            seen.add(full)
            if not any(os.path.exists(os.path.expanduser(c)) for c in cands):
                missing.append(cands[0])
            if len(missing) >= MAX_REALITY_PER_MEMORY:
                break
        for p in missing:
            out.append({"id": r["id"], "path": p, "gist": r["gist"]})
    return out


# ------------------------------------------------------------ chronic noise
# The push-noise loop has a mechanical half and a judgment half. Mechanical:
# `effective_floor` already quiets a memory pushed repeatedly but never used
# (bounded penalty, reversible, explicit recall unaffected). Judgment: whether
# such a row should keep LIVING — it may be obsolete (forget/supersede it),
# mis-scoped (`reproject` it), or legitimate-but-rarely-relevant (keep it, quiet
# under the penalty). The floor cannot and should not decide that; the dream
# worklist surfaces the question. Propose-not-dispose (§6.5).

def _chronic_scan(store: MemoryStore) -> list:
    """Live memories pushed >= CHRONIC_MIN_PUSHES times with ZERO push-uses —
    `uses` mirrors the floor math (endorsements weigh HELPFUL_USE_WEIGHT,
    referenced pushes REFERENCED_USE_WEIGHT). Lifetime pulls (recall_count) are
    REPORTED but never exempt a row: a frequently-pulled memory whose pushes
    are all ignored is exactly the "keep, but leave it quiet" case the reviewer
    should see and settle. Rows tagged `noise-ok` are the reviewed-and-accepted
    ones (the reality-ok analogue) — skipped so an accepted row stays accepted."""
    out = []
    for r in store.conn.execute(
            """SELECT id, kind, project, gist, surfaced_count, recall_count,
                      helpful_count, referenced_count
               FROM memory
               WHERE superseded_time IS NULL AND surfaced_count >= ?
                 AND id NOT IN (SELECT mt.memory_id FROM memory_topic mt
                                JOIN topic t ON t.id = mt.topic_id
                                WHERE t.name = 'noise-ok')
               ORDER BY surfaced_count DESC""", (CHRONIC_MIN_PUSHES,)):
        uses = (HELPFUL_USE_WEIGHT * (r["helpful_count"] or 0)
                + REFERENCED_USE_WEIGHT * (r["referenced_count"] or 0))
        if uses > 0:
            continue
        out.append({"id": r["id"], "kind": r["kind"], "project": r["project"],
                    "pushed": r["surfaced_count"], "pulls": r["recall_count"],
                    "gist": r["gist"]})
        if len(out) >= MAX_PAIR_PROPOSALS:
            break
    return out


def use_credit_refresh(store: MemoryStore) -> dict | None:
    """The mechanical half of push-noise housekeeping, run once per dream pass
    (at open, before the worklist): rescan the host's session transcripts
    (usefulness_scan) and materialize each pushed memory's downstream-reference
    count (`record_referenced`) — the same closing-of-the-loop as
    `usefulness-scan --apply`, no judgment involved. Without a periodic refresh
    the floor's penalty side keeps accruing at push time while its credit side
    goes stale, slowly over-penalizing recently-useful memories — and the
    chronic-noise scan above would misfire on exactly those rows.

    The pairing must be EXPLICIT: the refresh runs only when this store's
    `transcripts_path` config is set (or env FORNIXDB_TRANSCRIPTS overrides it;
    `off` skips — the machine-wide/test switch, like FORNIXDB_VECTORS). A
    transcript's `#id`s belong to the store the host's hooks inject from, and
    ids collide across stores — crediting any OTHER store on the machine writes
    phantom counts onto whatever rows share those numbers (measured live on the
    second store's rows, 2026-07-03, first cross-store dream). No config, no
    scan: an Elira-style consumer dreams exactly as before. Wire the one store
    the host injects from with e.g.
    `fornixdb config transcripts_path ~/.claude/projects`.
    `dream_use_credit off` hard-disables regardless. Returns None whenever
    skipped, including a configured path that doesn't exist."""
    if store._setting_off("dream_use_credit"):
        return None
    import os
    src = (os.environ.get("FORNIXDB_TRANSCRIPTS")
           or get_config(store, "transcripts_path") or "")
    if not src or src.strip().lower() in ("off", "none", "no", "false", "0"):
        return None
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        return None
    from .usefulness_scan import (outcomes_from_scan,
                                  referenced_counts_from_scan, scan)
    result = scan(src)
    counts = referenced_counts_from_scan(result)
    credited = store.record_referenced(counts)
    # by_channel + outcomes ride along for the dial report: the scan is the one
    # honest push-outcome source (outcomes_from_store's recall proxy is inflated
    # on lived-in stores), and it was just computed — never recomputed for dials
    return {"source": src, "sessions": result["sessions"],
            "memories_scanned": len(counts), "credited": credited,
            "by_channel": result.get("by_channel") or {},
            "outcomes": outcomes_from_scan(result)}


def suppress_refresh(store: MemoryStore) -> dict | None:
    """The judgment-free push-SUPPRESSION refresh, run once per dream pass right
    after the use-credit refresh (they read the same transcript scan): mute the
    memories chronically pushed but never referenced, and un-suppress any that have
    since earned a reference. Same closing-of-the-loop as `suppress --scan --apply`.

    Gated EXACTLY like use_credit_refresh — the id-collision hazard is identical: a
    transcript's `#id`s belong to the one store the host injects from, so the scan
    runs only against this store's `transcripts_path` (env FORNIXDB_TRANSCRIPTS
    overrides; `off` skips). An Elira-style consumer with no configured path never
    suppresses from a foreign transcript. Skipped (returns None) when the feature is
    off (`proactive_suppression off`), the dream hook is off (`dream_suppress off`),
    or there is no existing transcripts path. Never raises: a scan failure must not
    kill the dream."""
    if store._setting_off("dream_suppress") or store._setting_off("proactive_suppression"):
        return None
    import os
    src = (os.environ.get("FORNIXDB_TRANSCRIPTS")
           or get_config(store, "transcripts_path") or "")
    if not src or src.strip().lower() in ("off", "none", "no", "false", "0"):
        return None
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        return None
    try:
        from .suppress import scan_and_apply
        return scan_and_apply(store, src, apply=True)
    except Exception:
        return None


def _reproject_scan(store: MemoryStore) -> list:
    """Mis-scoped rows are the OTHER root of cross-project push noise (the floor
    penalty and project-scoped pulse only treat symptoms of a wrong/missing
    label). Fold reproject's confident proposals into the worklist: unscoped
    (or --suspect-labeled) rows whose CONTENT points at a project. Best margin
    first. The reviewer applies via `fornixdb reproject --apply` (undo-able) or
    relabels the rows it accepts. Never raises: a dream must not die because a
    model failed to load — reproject falls back to keyword mode on its own, and
    anything harder is reported as an empty list."""
    try:
        from .reproject import propose as reproject_propose
        res = reproject_propose(store)
    except Exception:
        return []
    props = sorted(res["proposals"], key=lambda p: -p["margin"])
    return [{"id": p["id"], "current": p["current"], "proposed": p["proposed"],
             "margin": p["margin"], "gist": p["gist"]}
            for p in props[:MAX_PAIR_PROPOSALS]]


# --------------------------------------------------------------- dial report
# Sleep as self-review of the DIALS: the telemetry the store accrues (floor
# log, field log, the pass-open scan's per-channel rates) exists to answer
# config questions, but nothing was reading it back at decision moments. The
# dream is that moment. Propose-not-dispose applied to configuration itself:
# each entry names the dial, the evidence, and a suggestion — nothing is ever
# flipped here; the owner/AI weighs each line. Every rule has an evidence
# minimum so a thin log can't produce a confident-sounding lie.

DIAL_MIN_SHADOW = 10          # settled beats carrying an unemitted minority
                              # report before dissent-on is worth weighing
DIAL_MIN_IMPRESSIONS = 20     # scanned push impressions before a channel's
                              # reference rate counts as gate evidence


def dial_report(store: MemoryStore, scan_channels: dict | None = None,
                scan_outcomes: dict | None = None) -> list:
    """Evidence-backed config proposals from the accrued telemetry. Read-only.
    `scan_channels` / `scan_outcomes` are the pass-open usefulness scan's
    by-channel rates and per-memory outcomes (use_credit_refresh) — the honest
    push-outcome source; the floor and gate rules stay silent without them
    rather than fall back to the inflated lifetime-recall proxy. Empty list
    when the logs are off or too thin to say anything honest."""
    out = []

    # parallel_dissent: does the tension line have real content? (field log)
    try:
        from .field import field_log_path_for
        from .field_stats import load_beats
        from .field_stats import summarize as summarize_beats
        fs = summarize_beats(load_beats(field_log_path_for(store)))
    except Exception:
        fs = None
    if fs and fs["settled"]:
        dissent_off = store._setting_off("parallel_dissent", default="off")
        if (dissent_off and fs["dissent_shadow"] >= DIAL_MIN_SHADOW
                and fs["dissent_emitted"] == 0):
            out.append({
                "dial": "parallel_dissent", "current": "off",
                "evidence": (f"a minority report existed on {fs['dissent_shadow']} "
                             f"of {fs['settled']} settled beats and was never "
                             "shown (shadow only)"),
                "suggestion": "weigh `config parallel_dissent on` — the tension "
                              "line has real content"})

    # parallel_recall (the L5 gate, default-on since 0.5.0): settled-push
    # reference rate vs L4's — after the flip this readout is the REVERT signal
    if scan_channels:
        l4, l5 = scan_channels.get("L4"), scan_channels.get("L5")
        recall_on = not store._setting_off("parallel_recall", default="on")
        if (l4 and l5 and l4["impressions"] >= DIAL_MIN_IMPRESSIONS
                and l5["impressions"] >= DIAL_MIN_IMPRESSIONS):
            r4 = l4["referenced"] / l4["impressions"]
            r5 = l5["referenced"] / l5["impressions"]
            ev = (f"L5 settled pushes referenced at {r5:.0%} "
                  f"({l5['referenced']}/{l5['impressions']}) vs L4 {r4:.0%} "
                  f"({l4['referenced']}/{l4['impressions']})")
            out.append({
                "dial": "parallel_recall",
                "current": "on (default)" if recall_on else "off",
                "evidence": ev,
                "suggestion": ("gate evidence FOR default-on — settling beats "
                               "the L4 baseline" if r5 > r4 else
                               "gate evidence AGAINST default-on so far — weigh "
                               "`config parallel_recall off` or revisit the "
                               "settle thresholds")})
        elif recall_on:
            n = l5["impressions"] if l5 else 0
            out.append({
                "dial": "parallel_recall", "current": "on (default)",
                "evidence": (f"only {n} settled-block impressions in the scanned "
                             f"window (need {DIAL_MIN_IMPRESSIONS}+ alongside L4)"),
                "suggestion": "gate still accruing — no readout yet"})

    # push floor: is there a lossless floor? (floor log × honest scan outcomes)
    if scan_outcomes:
        try:
            from .floor_stats import load_records, recommend_floor
            from .floor_stats import _cos as floor_cos
            from .proactive import floor_log_path_for
            records = load_records(floor_log_path_for(store))
        except Exception:
            records = []
        surfaced = [r for r in records if r.get("decision") == "surfaced"]
        useful = [c for r in surfaced if scan_outcomes.get(r.get("id")) == "useful"
                  and (c := floor_cos(r)) is not None]
        noise = [c for r in surfaced if scan_outcomes.get(r.get("id")) == "noise"
                 and (c := floor_cos(r)) is not None]
        if surfaced:
            rec = recommend_floor(useful, noise)
            if rec.get("verdict") in ("raise_safe", "clean_separation"):
                out.append({
                    "dial": "push floor", "current": "adaptive per-memory",
                    "evidence": (f"scan-labeled surfaced cosines separate: "
                                 f"useful n={len(useful)}, noise n={len(noise)} "
                                 f"(verdict: {rec['verdict']})"),
                    "suggestion": (f"a floor at {rec['suggested_floor']} drops the "
                                   "measured noise with no measured useful loss")})
    return out


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
            "resolutions": resolutions, "reality": _reality_scan(store),
            "chronic": _chronic_scan(store),
            "reproject": _reproject_scan(store)}


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
    if counts.get("reality"):
        # memory vs world: a live memory points at a file that isn't there
        parts.append(plural(counts["reality"],
                            "pointer to a missing file to verify",
                            "pointers to missing files to verify"))
    if counts["associations"] and not woven:
        # the generative half: connections that didn't exist before the dream
        # (when woven, the woke-clause below reports them instead)
        parts.append(plural(counts["associations"],
                            "new connection to weave", "new connections to weave"))
    if counts["merges"]:
        parts.append(plural(counts["merges"], "near-duplicate to merge",
                            "near-duplicates to merge"))
    if counts.get("chronic"):
        # the judgment half of push-noise: the floor already quiets these; the
        # dream asks whether they should keep living at all
        parts.append(plural(counts["chronic"],
                            "chronically ignored push to judge",
                            "chronically ignored pushes to judge"))
    if counts.get("reproject"):
        # the other root of cross-project noise: the label is wrong/missing
        parts.append(plural(counts["reproject"],
                            "mis-scoped memory to re-project",
                            "mis-scoped memories to re-project"))
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

    Opening a pass also runs the one judgment-free housekeeping move: the push
    use-credit refresh (use_credit_refresh — `usefulness-scan --apply` in dream
    clothing), so the worklist's chronic-noise question is asked over current
    counts. Reported as `use_credit`; None when skipped (off, or no transcripts).

    Every dream also reads the telemetry back as a DIAL REPORT (dial_report):
    evidence-backed config suggestions — dissent shadow, the L5 gate readout,
    a lossless-floor verdict — reported as `dials`, never applied.

    Refused on a read-only store: consolidation is a maintenance operation, so a
    frozen (vendor-shipped read-only) store raises FrozenStoreError rather than
    proposing work that can never be applied."""
    store._check_writable()
    # A pass opens on the first dream() call and closes on done=True. Snapshot the
    # superseded-row count at open; the wake reports the NET delta — what was
    # reconciled DURING the pass — robust to same-second timestamp collisions and
    # never counting the store's prior supersede history.
    marker = get_config(store, "dream_pass_super0")
    credit = None
    suppression = None
    if marker is None:
        marker = str(_superseded_count(store))
        set_config(store, "dream_pass_super0", marker)
        # opening a pass refreshes the push use-credit BEFORE proposing, so the
        # chronic-noise list below runs on current counts, not stale ones
        credit = use_credit_refresh(store)
        # ...then re-classify chronic push-noise on those fresh counts: suppress
        # the never-referenced, redeem any that just earned a reference
        suppression = suppress_refresh(store)
    st = status(store)
    work = propose(store)
    woven = 0
    if weave:
        for a in work["associations"]:
            store.link(a["ids"][0], a["ids"][1], relation="relates")
            woven += 1
    counts = {k: len(work[k]) for k in ("distill", "gists", "merges",
                                        "contradictions", "associations",
                                        "resolutions", "reality", "chronic",
                                        "reproject")}
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
                     "stale one (default: keep the newer), or accept a reviewed "
                     "pair as legitimately distinct with: link <a> <b> "
                     "--relation distinct:"]
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
    if credit:
        narrative += (f"\n(pass open: refreshed push use-credit from "
                      f"{credit['sessions']} transcript session"
                      f"{'' if credit['sessions'] == 1 else 's'} — "
                      f"{credit['credited']} of {credit['memories_scanned']} "
                      "pushed memories proven used downstream)")
    ap = (suppression or {}).get("applied") or {}
    if ap.get("newly_suppressed") or ap.get("redeemed"):
        narrative += (f"\n(pass open: push-noise scan — {ap['newly_suppressed']} "
                      f"memory(ies) muted, {ap['redeemed']} redeemed; "
                      f"{suppression.get('total_suppressed', 0)} suppressed total)")
    # dial report: read the accrued telemetry back at the decision moment. The
    # scan-derived rules only have their honest inputs at pass open (credit);
    # the field-log rule reads on every call.
    dials = dial_report(store,
                        scan_channels=(credit or {}).get("by_channel"),
                        scan_outcomes=(credit or {}).get("outcomes"))
    if dials and not done:
        narrative += (f"\n🎛 {len(dials)} dial suggestion"
                      f"{'' if len(dials) == 1 else 's'} from the accrued "
                      "telemetry — evidence attached; nothing flipped (§6.5).")
    # perceptual worklist: watch() keeps the hot path model-free, so committed
    # keyframes land under a templated placeholder gist. Surface how many still
    # await a real caption — the model-bearing pass (`fornixdb recaption`, a
    # local VLM) is deliberately separate from this stdlib-only dream.
    from . import recaption
    awaiting_captions = recaption.pending_count(store)
    if awaiting_captions and not done:
        one = awaiting_captions == 1
        narrative += (f"\n👁 {awaiting_captions} watch keyframe{'' if one else 's'} "
                      f"still hold{'s' if one else ''} a templated placeholder — "
                      "run `recaption` with a local VLM to fill real captions.")
    return {"status": st, "counts": counts, "work": work, "woven": woven,
            "applied": applied, "use_credit": credit, "suppression": suppression,
            "dials": dials, "awaiting_captions": awaiting_captions,
            "narrative": narrative}


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
        #    an older task memory? cross-kind, RESOLUTION_COSINE band. Both
        #    lifecycle gates read the HEADLINE (gist + lede), not the full body
        #    (see _headline) — a long status memory that merely MENTIONS "shipped"
        #    deep in its detail no longer reads as a closure of some noun-sharing
        #    task, and an older memory that merely mentions "backlog" no longer
        #    reads as the open task being closed.
        new_row = store.conn.execute(
            "SELECT gist, detail FROM memory WHERE id = ?", (new_id,)).fetchone()
        new_head = _headline(new_row["gist"], new_row["detail"]) if new_row else text
        if _CLOSURE_RE.search(new_head):
            for mid, cos in matches:
                if cos < RESOLUTION_COSINE:
                    break
                if mid == new_id:
                    continue
                row = store.conn.execute(
                    "SELECT id, gist, detail FROM memory WHERE id = ?", (mid,)).fetchone()
                if row and _TASK_RE.search(_headline(row["gist"], row["detail"])):
                    return {"id": row["id"], "gist": row["gist"],
                            "cosine": round(cos, 3), "reason": "resolves"}
        return None
    except Exception:
        return None
