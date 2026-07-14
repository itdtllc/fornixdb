"""Core memory operations: store, recall (subject + time), supersede, reinforce.

The store has no agency: it never decides or acts (Design §6.5). Every function
here is bookkeeping about memories; the connected thinking AI owns judgment.

P1 recall ranking is a transparent, provisional blend (no vectors yet — P2):
    score = text_relevance * (1 + SALIENCE_WEIGHT * salience) + RECENCY_WEIGHT * recency
where text_relevance = -bm25 (FTS5; more negative bm25 = better match) and
recency decays exponentially with the age of event_time. Constants below are
deliberately simple and documented so they can be tuned against real use.
"""

from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime, timedelta

from .db import KIND_ALIASES, KINDS, RELATIONS, connect

SALIENCE_WEIGHT = 1.0     # how much a salient memory outranks an equally-relevant one
RECENCY_WEIGHT = 2.0      # max score bonus for a memory from "right now"
RECENCY_HALFLIFE_DAYS = 90.0
REINFORCE_BUMP = 0.05     # salience bump each time detail is recalled
HELPFUL_BUMP = 0.15       # salience bump when a memory is explicitly marked
                          # helpful — larger than passive reinforce because an
                          # endorsement is stronger evidence than a mere read
USEFULNESS_WEIGHT = 0.5   # max ranking bonus from "this helped" endorsements;
                          # saturating so the first endorsement matters and a
                          # popular memory can't drown a more relevant one
USEFULNESS_SATURATION = 2.0  # endorsements for ~half the max bonus (1-e^-1)
REFERENCED_WEIGHT = 0.2   # max ranking bonus from scan-verified downstream use
                          # (referenced_count) — weaker evidence than an explicit
                          # endorsement, so it tops out well below USEFULNESS_WEIGHT.
                          # recall_count deliberately does NOT rank: listing surfaces
                          # (brief/timeline) inflated historic counts far past any
                          # honest use, freezing old rows at the top (rich-get-richer,
                          # measured 2026-07-02); referenced_count is the same honest
                          # use currency the push floor already runs on.
REFERENCED_SATURATION = 5.0  # referenced uses for ~half the max bonus

# Per-memory relevance-floor adaptation (the usefulness loop closing on the PUSH
# side). The proactive (L3) / rhythmic (L4) push uses one cosine floor for every
# memory; this nudges that floor PER MEMORY by proven usefulness so the ambient
# stream learns what to keep surfacing. A memory that has been USED (explicitly
# recalled or endorsed) earns a small DISCOUNT — easier to surface; one PUSHED
# many times but never used earns a PENALTY — quieter. That penalty is the
# implicit attack on cross-project noise (a wrong-project memory pulsed every
# session, never used, fades from the stream). Bounded and additive: it never
# hides a memory — explicit recall ignores the push floor entirely — and is fully
# reversible via `config usefulness_floor_adapt off`.
FLOOR_DISCOUNT_MAX = 0.05      # most a proven-useful memory lowers its push floor
FLOOR_PENALTY_MAX = 0.15       # most a chronically-ignored memory raises it
FLOOR_USE_SATURATION = 2.0     # uses for ~half the max discount
FLOOR_IGNORE_SATURATION = 4.0  # ignored impressions for ~half the max penalty
FLOOR_MIN_IMPRESSIONS = 3      # don't penalize until pushed at least this often —
                               # a brand-new memory hasn't been "ignored" yet
FLOOR_CAP = 0.95               # never raise a floor so high a memory can't surface
HELPFUL_USE_WEIGHT = 2.0       # one endorsement counts as this many referenced
                               # uses when tallying "uses" for the floor math
REFERENCED_USE_WEIGHT = 1.0    # the unit of floor "uses": one PUSH that was
                               # actually used downstream (cited in reasoning).
                               # Closes the loop: a pushed memory is used WITHOUT a
                               # pull, so without this a proven-useful push looks
                               # identical to ignored noise to the floor math.
                               # (Pull counts don't tally — see effective_floor.)

# Project-scoped pulse (the other half of the cross-project noise fix). When a
# pulse knows its active context, a memory that BELONGS to a different context
# clears a higher push floor — a tangential off-context hit is dropped, but a
# strongly-relevant one (high cosine) still surfaces. "Belongs" unifies both
# axes: a memory is on-context if its project OR any of its topics matches the
# active label (or an alias). Memories with NO scoping tags (general principles,
# curated cross-cutting facts) are never penalized — they belong everywhere. Like
# usefulness adaptation this only touches the PUSH floor, never explicit recall,
# and is reversible via `config project_scoped_pulse off`.
PROJECT_MISMATCH_PENALTY = 0.15  # floor bump for an off-context memory on a pulse
# Topics that tag a memory's SHAPE, not its project — they must not make a memory
# look "off-context" (a cross-cutting reference tagged only "reference" belongs
# everywhere). Excluded from the belongs test; domain topics (fornixdb, elira,
# security, …) still count.
STRUCTURAL_TOPICS = frozenset({
    "reference", "feedback", "project", "milestone", "distilled", "pickup",
    "publication", "documentation", "roadmap", "naming",
})
SALIENCE_CAP = 1.0
VECTOR_WEIGHT = 15.0      # scales cosine into the -bm25 range. Tuned 2026-06-11
                          # via the eval fence: at 6.0, OR-mode keyword noise
                          # (bm25 ≈ 7-9 from common tokens) buried the clearly
                          # best vector hit (cos .57 vs .41 → eval miss #17).
                          # Pure rank fusion (RRF) was tried first: fixed the
                          # miss but flattened hit@1 78→56% — margins matter.
VECTOR_MIN_COS = 0.30     # noise floor: weaker similarity is no evidence at all

# Abstention gate (FornixDB #191, owner observation 2026-06-13): recall used to
# return its top-k even when nothing was actually relevant, so a consumer (esp.
# a small model) treated noise as "the answer found in memory" and STOPPED —
# leaving the user a dead-end instead of acting / answering from its own
# knowledge. recall_has_answer() reports, tool-agnostically, whether the best
# hit is a real match. Substrate-not-actor: it only reports PRESENCE; it never
# tells the agent what to do next (use a tool, answer from knowledge, abstain)
# — that routing is the consumer's prompt, since memory can't know the agent's
# capabilities. Calibrated on the live store 2026-06-13: abstains on clearly
# out-of-store queries with ZERO false-abstention on the 28 golden positives
# (every positive clears cos>=0.30 OR relevance>=7.1; clear negatives sit at
# cos<0.12 AND relevance<5.2). The ambiguous middle is left to the consumer on
# purpose — no single threshold separates weak-but-relevant from weak-noise.
RECALL_ANSWER_COS = 0.30  # a real vector match (== the include floor)
# Unsolicited PUSH needs a higher bar than an explicit PULL. When the user asks
# (recall_memory), surfacing a weak 0.30 match is acceptable — they invited it.
# When memory injects itself every turn (proactive recall), that same 0.30 floor
# lets tangential matches drift in and erodes trust, so proactive recall gates at
# this higher cosine by default. Per-store override: meta proactive_recall_floor.
PROACTIVE_RECALL_COS = 0.45

# Rhythmic (L4) recall fires MANY times inside one reasoning episode, so an
# unwanted pulse interrupts mid-thought — more intrusive than the once-per-turn
# L3 push. It therefore gates a notch HIGHER than L3 (above PROACTIVE_RECALL_COS).
# The earlier 0.60 was set to suppress a bland "Chat: Hello" episodic leak onto
# action queries; that leak is now handled independently by the _is_low_information
# filter (proactive.py), so the floor no longer has to carry it. Re-measured
# 2026-06-20 on a live Claude-Code store (#351): pure-noise queries return cosine
# ~0.0 (no vector neighbor) while GENUINE hits span 0.42–0.92 — so 0.60 was
# silencing a wide band of real signal (e.g. an L4-design query at 0.51) for no
# noise benefit. 0.50 admits that signal, stays clear of the ~0 noise floor, and
# remains stricter than L3. Per-store override: meta rhythmic_recall_floor.
RHYTHMIC_RECALL_COS = 0.50


def recall_has_answer(rows: list[dict]) -> bool:
    """True if recall's best hit is a real match; False if the store has
    nothing relevant (the consumer should then act / use its own knowledge /
    say it doesn't know — recall must NOT pose as the answer). Reports presence
    only; never prescribes the next action.

    The gate lives in the VECTOR regime: cosine is a normalized, store-
    independent signal, so a weak top cosine means weak-noise-as-answer (the
    failure that strands a small model). In pure keyword-only recall there is
    no cosine on the rows — an FTS hit there is a literal token anchor by
    definition, so it is trusted (this is also the pre-gate behavior; bm25
    magnitudes are store-dependent and make no portable threshold)."""
    if not rows:
        return False
    top = rows[0]
    if top.get("vec_cos") is None:        # keyword-only recall: trust the FTS anchor
        return True
    return float(top["vec_cos"]) >= RECALL_ANSWER_COS

# Negative feedback (explicit mark_irrelevant, query-conditional penalty; shipped
# 2026-06-12). When the current query is similar to a query a
# memory was explicitly marked irrelevant for, that memory's score is cut to
# a quarter — feedback is an explicit "not that one", so it must displace
# even a strongly-dominant wrong hit (a 0.5 cut survived a salient
# vector-heavy hit in testing), yet the memory is never hidden and stays
# fully ranked for every other question. Provisional, like the other
# ranking constants; tune against the eval fence.
NEG_FEEDBACK_PENALTY = 0.75  # fraction of the score removed when triggered
NEG_FEEDBACK_COS = 0.60      # query↔query cosine: "similar enough to count"
NEG_FEEDBACK_OVERLAP = 0.5   # token-Jaccard fallback when vectors are absent

# P3a decay (Design §13.1): ranking uses EFFECTIVE salience — stored salience
# decaying since last recall, with per-kind half-lives and floors. Lazy: it is
# computed at read time, never written back; nothing is ever deleted by decay.
# Per-store overrides live in meta as decay_halflife_<kind> / decay_floor_<kind>.
DECAY_HALFLIFE = {"episodic": 45.0, "semantic": 120.0, "reference": 180.0,
                  "feedback": 365.0}
DECAY_FLOOR = {"episodic": 0.05, "semantic": 0.15, "reference": 0.15,
               "feedback": 0.35}

# B4 (security assessment 2026-06-12): sources whose content was ingested by
# machinery with no owner review — transcript back-fill and SessionEnd capture
# gist whole sessions, including tool results carrying third-party text (web
# pages, emails), so injected instructions can land verbatim. Recall output
# flags these rows [auto-captured]; consumers must treat recalled content as
# data about the past, never as instructions (INTEGRATION.md). cli/mcp/
# markdown-import stay unflagged: an owner or owner-gated agent wrote those.
AUTO_CAPTURE_SOURCES = frozenset({"claude-code-transcript"})


class FrozenStoreError(RuntimeError):
    """The store does not accept content changes (Design §13.2): either the
    standalone `frozen` setting (vendor-shipped read-only store) or, via the
    subclass below, the disk budget with policy 'freeze'."""


class DiskBudgetExceededError(FrozenStoreError):
    """The store is at its disk budget with policy 'freeze' — new memories
    are refused until the budget is raised or the policy changed."""


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _fts_query(text: str, mode: str = "AND") -> str:
    """Sanitize free text into an FTS5 query (quoted tokens, AND/OR joined)."""
    tokens = re.findall(r"[A-Za-z0-9_]+", text)
    if not tokens:
        return '""'
    joiner = " " if mode == "AND" else " OR "
    return joiner.join(f'"{tok}"' for tok in tokens)


class MemoryStore:
    def __init__(self, db_path=None, conn: sqlite3.Connection | None = None):
        self.conn = conn or connect(db_path)

    def close(self) -> None:
        """Release the SQLite connection. Required on Windows before the db
        file can be moved or deleted (open files can't be unlinked there)."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------ freeze (§13.2)

    def frozen(self) -> bool:
        """Standalone read-only flag (`config frozen on`) — vendor-shipped
        stores. Blocks all content mutation; recall still works (without
        reinforcement writes, so the file itself may be read-only)."""
        if not hasattr(self, "_frozen_cache"):
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key = 'frozen'").fetchone()
            self._frozen_cache = bool(
                row and row["value"] not in ("0", "off", "false", ""))
        return self._frozen_cache

    def _check_writable(self) -> None:
        if self.frozen():
            raise FrozenStoreError(
                "store is frozen (read-only) — `config frozen off` to unfreeze")

    # ---------------------------------------------------------------- store

    def store(
        self,
        gist: str,
        detail: str | None = None,
        *,
        kind: str = "semantic",
        name: str | None = None,
        topics: list[str] | None = None,
        project: str | None = None,
        event_time: str | None = None,
        event_time_end: str | None = None,
        session_id: str | None = None,
        salience: float = 0.5,
        source: str = "cli",
        source_ref: str | None = None,
        recorded_time: str | None = None,
        writer: str | None = None,
        embedder=None,  # None = auto (embed when this store uses vectors); False = skip
    ) -> int:
        kind = KIND_ALIASES.get(kind, kind)
        if kind not in KINDS:
            raise ValueError(
                f"kind must be one of {KINDS} (got {kind!r}); "
                f"or a known alias {tuple(KIND_ALIASES)}")
        self._check_writable()
        # Resolve the embedder BEFORE inserting: first resolution runs the
        # missing-vector backfill, and with the new row already committed the
        # backfill would count it as a gap — embedding it a first time and
        # announcing a heal on every write from a fresh store handle (seen
        # live 2026-07-10: every Elira sense call printed "embedded 1").
        emb = self._resolve_embedder(embedder)
        from .budget import make_room  # lazy: avoids import cycle, free when no budget set
        make_room(self)
        cur = self.conn.execute(
            """INSERT INTO memory (name, kind, event_time, event_time_end,
                                   recorded_time, session_id, project, gist, detail,
                                   salience, source, source_ref, writer)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                name, kind,
                event_time or now_iso(), event_time_end,
                recorded_time or now_iso(),
                session_id, project, gist, detail,
                min(max(salience, 0.0), SALIENCE_CAP), source, source_ref,
                writer,
            ),
        )
        mem_id = cur.lastrowid
        for topic in topics or []:
            self.tag(mem_id, topic)
        self.conn.commit()
        # Embed on write so a vector-using store can recall this memory by
        # meaning immediately. Auto-resolution only loads a model when the
        # store already uses vectors (see _resolve_embedder), so keyword-only
        # deployments never pay for it; pass embedder=False to skip explicitly.
        # Embedding must never block a write, so a failure here is swallowed —
        # `embed` backfill remains the safety net.
        if emb is not None:
            try:
                from .vectors import embed_memory
                embed_memory(self, emb, mem_id)
            except Exception:
                pass
        # Auto-link: [[name]] wikilinks in the content become real 'relates'
        # edges so the graph the author already wrote in prose actually exists
        # (the markdown directory-importer always did this; a plain store()
        # used to drop them). Unresolved targets are left alone on purpose — a
        # [[name]] to a not-yet-written memory marks intent, not an error.
        self.link_wikilinks(mem_id, " ".join(t for t in (gist, detail) if t))
        return mem_id

    _WIKILINK = re.compile(r"\[\[([^\[\]\n]+?)\]\]")

    def link_wikilinks(self, memory_id: int, text: str) -> list[int]:
        """Resolve [[name]] mentions in `text` to live memories by name and add
        a 'relates' edge from `memory_id` to each. Skips self, unknown names,
        and duplicates (link() is INSERT OR IGNORE). Returns the ids linked.
        Reusable for back-filling stores written before auto-link existed."""
        linked: list[int] = []
        for name in dict.fromkeys(m.strip() for m in self._WIKILINK.findall(text)):
            if not name:
                continue
            row = self.conn.execute(
                "SELECT id FROM memory WHERE name = ? "
                "ORDER BY superseded_time IS NULL DESC, recorded_time DESC LIMIT 1",
                (name,)).fetchone()
            if row is None or row["id"] == memory_id:
                continue
            self.link(memory_id, row["id"], "relates")
            linked.append(row["id"])
        return linked

    def tag(self, memory_id: int, topic: str) -> None:
        self._check_writable()
        topic = topic.strip().lower()
        self.conn.execute("INSERT OR IGNORE INTO topic(name) VALUES (?)", (topic,))
        self.conn.execute(
            """INSERT OR IGNORE INTO memory_topic(memory_id, topic_id)
               SELECT ?, id FROM topic WHERE name = ?""",
            (memory_id, topic),
        )
        self.conn.commit()

    def link(self, memory_id: int, related_id: int, relation: str = "relates") -> None:
        if relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}")
        self._check_writable()
        self.conn.execute(
            "INSERT OR IGNORE INTO memory_link(memory_id, related_id, relation) VALUES (?,?,?)",
            (memory_id, related_id, relation),
        )
        self.conn.commit()

    # --------------------------------------------- candidate staging (§15.2 #1)

    def jot(self, note: str, session_id: str | None = None) -> int:
        """Stage a raw thought for later review — cheap mid-work capture with no
        title/kind/embedding cost. Not a memory; never recalled until promoted."""
        self._check_writable()
        cur = self.conn.execute(
            "INSERT INTO candidate(note, session_id, created) VALUES (?,?,?)",
            (note, session_id, now_iso()))
        self.conn.commit()
        return cur.lastrowid

    def candidates(self, session_id: str | None = None) -> list[dict]:
        """Pending (un-promoted) candidates, oldest first. `session_id` narrows
        to one session's jots; None returns all pending."""
        sql = ("SELECT id, note, session_id, created FROM candidate "
               "WHERE promoted IS NULL")
        args: tuple = ()
        if session_id is not None:
            sql += " AND session_id = ?"
            args = (session_id,)
        sql += " ORDER BY created"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def discard_candidates(self, ids=None, session_id: str | None = None) -> int:
        """Drop pending candidates: a list of ids, or all pending (optionally
        scoped to a session). Returns how many were removed."""
        self._check_writable()
        if ids:
            qs = ",".join("?" * len(ids))
            cur = self.conn.execute(
                f"DELETE FROM candidate WHERE promoted IS NULL AND id IN ({qs})",
                tuple(ids))
        elif session_id is not None:
            cur = self.conn.execute(
                "DELETE FROM candidate WHERE promoted IS NULL AND session_id = ?",
                (session_id,))
        else:
            cur = self.conn.execute("DELETE FROM candidate WHERE promoted IS NULL")
        self.conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------ supersede

    def supersede(self, old_id: int, new_id: int) -> None:
        """Tombstone old_id as superseded by new_id. Old row is kept — the
        trail is the record of learning (Design §2.5)."""
        if old_id == new_id:
            raise ValueError("a memory cannot supersede itself")
        self._check_writable()
        # the successor inherits the unique name handle unless it has its own,
        # so `show <name>` keeps resolving to the live version
        names = {r["id"]: r["name"] for r in self.conn.execute(
            "SELECT id, name FROM memory WHERE id IN (?, ?)", (old_id, new_id))}
        if names.get(old_id) and not names.get(new_id):
            self.conn.execute("UPDATE memory SET name = NULL WHERE id = ?", (old_id,))
            self.conn.execute("UPDATE memory SET name = ? WHERE id = ?",
                              (names[old_id], new_id))
        self.conn.execute(
            "UPDATE memory SET superseded_by = ?, superseded_time = ? WHERE id = ?",
            (new_id, now_iso(), old_id),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO memory_link(memory_id, related_id, relation) "
            "VALUES (?,?, 'supersedes')",
            (new_id, old_id),
        )
        self.conn.commit()
        # content changed: the successor is a fresh (unsuppressed) row, and the
        # old row is tombstoned — clear any suppression on it so the audit trail
        # doesn't carry a stale flag on a row that no longer participates.
        self.clear_proactive_suppression([old_id], "superseded")

    def tombstone(self, memory_id: int) -> None:
        """Retire a memory with no successor ("forget"). The row is kept and
        the tombstone is the record that it was deliberately retired —
        FornixDB never deletes. Tombstoned = superseded_time set; a successor
        (superseded_by) is optional."""
        self._check_writable()
        self.conn.execute(
            "UPDATE memory SET superseded_time = ? WHERE id = ? AND superseded_time IS NULL",
            (now_iso(), memory_id),
        )
        self.conn.commit()

    def set_name(self, memory_id: int, name: str | None) -> None:
        """Reassign a memory's unique name handle (e.g. when a named memory is
        superseded and the successor inherits the handle)."""
        self._check_writable()
        self.conn.execute("UPDATE memory SET name = ? WHERE id = ?", (name, memory_id))
        self.conn.commit()

    def set_gist(self, memory_id: int, gist: str, embedder=None) -> None:
        """In-place gist rewrite (consolidation, Design §13.5 decision 2): the
        gist is derived presentation, the detail/source is the record, so no
        supersession. A meaning change is a new memory + supersede, not this.
        The FTS index updates via trigger; the vector is re-embedded in place
        (embed-on-write parity with store() — a bulk consolidation pass must
        not leave rows semantically invisible until someone remembers to run
        `embed`: a 2026-07-01 distill pass dropped 250/317 live rows' vectors
        that way). With no embedder the stale vector is still dropped so
        backfill re-embeds the row later."""
        self._check_writable()
        self.conn.execute("UPDATE memory SET gist = ? WHERE id = ?",
                          (gist, memory_id))
        self.conn.execute("DELETE FROM embedding WHERE memory_id = ?", (memory_id,))
        self.conn.commit()
        # a rewritten gist is a content change — the old push-outcome history no
        # longer describes this text, so re-evaluate: clear suppression and let a
        # future scan re-classify it on the new gist.
        self.clear_proactive_suppression([memory_id], "gist_rewritten")
        emb = self._resolve_embedder(embedder)
        if emb is not None:
            try:
                from .vectors import embed_memory
                embed_memory(self, emb, memory_id)
            except Exception:
                pass  # embedding never blocks the rewrite; backfill heals

    # ---------------------------------------------------- negative feedback

    def mark_irrelevant(self, memory_id: int, query: str,
                        embedder=None) -> int:
        """Explicit negative feedback: this memory was irrelevant to this
        query. Future recalls downweight the memory only for similar queries
        (query-conditional — it stays fully ranked elsewhere). The query is
        embedded now if a model is available, so similarity is associative.
        Re-marking a retracted pair reactivates it; nothing is ever deleted."""
        self._check_writable()
        if not self.conn.execute("SELECT 1 FROM memory WHERE id = ?",
                                 (memory_id,)).fetchone():
            raise ValueError(f"no memory #{memory_id}")
        query = query.strip()
        if not query:
            raise ValueError("feedback needs the query the memory was wrong for")
        model = vector = None
        emb = self._resolve_embedder(embedder)
        if emb is not None:
            from .vectors import to_blob
            try:
                model, vector = emb.name, to_blob(emb.embed([query])[0])
            except Exception:
                pass  # keyword-only feedback still works
        cur = self.conn.execute(
            """INSERT INTO recall_feedback (memory_id, query, model, vector, created)
               VALUES (?,?,?,?,?)
               ON CONFLICT(memory_id, query) DO UPDATE SET retracted = NULL""",
            (memory_id, query, model, vector, now_iso()),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM recall_feedback WHERE memory_id = ? AND query = ?",
            (memory_id, query)).fetchone()
        return row["id"] if row else cur.lastrowid

    def retract_feedback(self, feedback_id: int) -> None:
        """Tombstone one feedback row — the memory ranks normally again for
        that query. The row is kept (never delete); re-marking reactivates."""
        self._check_writable()
        self.conn.execute(
            "UPDATE recall_feedback SET retracted = ? WHERE id = ? AND retracted IS NULL",
            (now_iso(), feedback_id),
        )
        self.conn.commit()

    def list_feedback(self, memory_id: int | None = None) -> list[dict]:
        where, params = ("WHERE f.memory_id = ?", [memory_id]) if memory_id else ("", [])
        return [dict(r) for r in self.conn.execute(
            f"""SELECT f.id, f.memory_id, f.query, f.created, f.retracted,
                       m.gist
                FROM recall_feedback f JOIN memory m ON m.id = f.memory_id
                {where} ORDER BY f.id""", params)]

    def mark_helpful(self, ref: int | str) -> dict:
        """Explicit POSITIVE feedback: this memory actually helped. Unlike
        `mark_irrelevant` (negative, query-conditional), an endorsement is a
        durable, query-independent statement that the memory itself is worth
        surfacing — so it raises ranking everywhere (via `_usefulness`), bumps
        salience, and reinforces (a helped memory was just confirmed current,
        so it should not read as stale). Idempotent only in spirit: each call
        counts, letting repeated value accumulate."""
        self._check_writable()
        if isinstance(ref, str) and not ref.isdigit():
            row = self.conn.execute(
                "SELECT id FROM memory WHERE name = ?", (ref,)).fetchone()
        else:
            row = self.conn.execute(
                "SELECT id FROM memory WHERE id = ?", (int(ref),)).fetchone()
        if row is None:
            raise ValueError(f"no memory {ref!r}")
        ts = now_iso()
        self.conn.execute(
            """UPDATE memory
               SET helpful_count = helpful_count + 1, last_helpful = ?,
                   last_recalled = ?, last_reinforced = ?,
                   salience = min(salience + ?, ?)
               WHERE id = ?""",
            (ts, ts, ts, HELPFUL_BUMP, SALIENCE_CAP, row["id"]))
        self.conn.commit()
        self.clear_proactive_suppression([row["id"]], "marked_helpful")
        out = self.conn.execute(
            "SELECT id, gist, kind, helpful_count, last_helpful, recall_count, "
            "salience FROM memory WHERE id = ?", (row["id"],)).fetchone()
        return dict(out)

    def top_useful(self, limit: int = 5) -> list[dict]:
        """The startup rollup: live memories ranked by endorsements, then by
        passive recall hits — what has actually proven worth surfacing. Empty
        until something is marked helpful or recalled, so a fresh store shows
        nothing rather than noise."""
        return [dict(r) for r in self.conn.execute(
            """SELECT id, gist, kind, event_time, helpful_count, recall_count,
                      last_helpful
               FROM memory
               WHERE superseded_time IS NULL
                 AND (helpful_count > 0 OR recall_count > 0)
               ORDER BY helpful_count DESC, recall_count DESC, event_time DESC
               LIMIT ?""", (limit,))]

    def _negative_penalties(self, query: str, emb) -> dict[int, float]:
        """{memory_id: score factor} for memories whose active feedback
        queries are similar to this query. Vector similarity when both sides
        have it; token-overlap otherwise — so keyword-only stores get the
        feature too, just with stricter matching."""
        rows = self.conn.execute(
            "SELECT memory_id, query, model, vector FROM recall_feedback "
            "WHERE retracted IS NULL").fetchall()
        if not rows:
            return {}
        qtokens = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
        qvec = None
        if emb is not None:
            try:
                qvec = emb.embed([query])[0]
            except Exception:
                qvec = None
        penalties: dict[int, float] = {}
        for r in rows:
            triggered = False
            if qvec is not None and r["vector"] is not None and r["model"] == emb.name:
                from .vectors import cosine, from_blob
                triggered = cosine(qvec, from_blob(r["vector"])) >= NEG_FEEDBACK_COS
            if not triggered and qtokens:
                ftokens = set(re.findall(r"[A-Za-z0-9_]+", r["query"].lower()))
                union = qtokens | ftokens
                if union and len(qtokens & ftokens) / len(union) >= NEG_FEEDBACK_OVERLAP:
                    triggered = True
            if triggered:
                penalties[r["memory_id"]] = 1.0 - NEG_FEEDBACK_PENALTY
        return penalties

    # --------------------------------------------------------------- recall

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: str | None = None,
        project: str | None = None,
        since: str | None = None,   # ISO bound: combined subject+time recall —
        until: str | None = None,   # "that bug we fixed last month"
        related: bool = False,      # spreading activation: attach 1-hop links
        include_superseded: bool = False,
        embedder=None,  # None = auto-detect; False = keyword-only
        count_recall: bool = True,  # False = a candidate fetch (e.g. proactive
                                    # PUSH gathering), which must NOT inflate
                                    # recall_count — that count is reserved for an
                                    # explicit PULL so it stays a real "use" signal
    ) -> list[dict]:
        """Subject-axis recall: ranked gists. Keyword matching (strict AND,
        falling back to OR — people loosen, not give up), blended with vector
        similarity when embeddings are available (P2). With no embedder
        installed this is pure FTS, identical to P1. Rows carry `stale_days`
        when an un-reinforced fact has outlived its decay half-life (verify
        before trusting), and `related` link neighbors when requested.
        Memories marked irrelevant for a similar query (mark_irrelevant) are
        downweighted — flagged `neg_feedback`, still present, never hidden."""
        # When vectors will blend, keep the OVERFETCHED keyword rows (not just
        # the top `limit`) so a row with weak keyword + strong semantic match
        # carries its bm25 relevance into the blend instead of being re-added
        # later with relevance 0 — that erasure sank eval #17 (rank 1 -> 4).
        if self._setting_off("associative_recall"):
            # L0/L1 boundary (ROADMAP: L0 = "exact lookups, no ranking"). When
            # associative recall is disabled the store behaves as a plain keyed
            # get/put: exact name lookup only, no FTS/vector ranking. (Keyed
            # access via show_memory by id/name still works regardless.)
            rows = self._recall_exact_name(query, limit, kind, project,
                                           include_superseded, since, until)
        else:
            emb = self._resolve_embedder(embedder)
            keep = max(limit * 5, 25) if emb is not None else limit
            rows = self._recall_fts(query, "AND", limit, kind, project,
                                    include_superseded, since, until, keep=keep)
            if not rows:
                rows = self._recall_fts(query, "OR", limit, kind, project,
                                        include_superseded, since, until, keep=keep)

            if emb is not None:
                rows = self._blend_vectors(rows, query, emb, limit, kind, project,
                                           include_superseded, since, until)
            penalties = self._negative_penalties(query, emb)
            if penalties:
                for r in rows:
                    factor = penalties.get(r["id"])
                    if factor is not None:
                        r["score"] = float(r.get("score") or 0.0) * factor
                        r["neg_feedback"] = True
                rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        now = datetime.now()
        for r in rows:
            r["stale_days"] = self.stale_days(r, now)
        if related:
            self._attach_neighbors(rows)
        if count_recall:
            self._mark_recalled([r["id"] for r in rows], reinforce=False)
        return rows

    def _setting_off(self, key: str, default: str = "on") -> bool:
        """Read a boolean meta setting directly (core can't import multistore)."""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        val = (row["value"] if row else default) or default
        return str(val).strip().lower() in ("off", "0", "false", "no")

    def _recall_exact_name(self, query, limit, kind, project,
                           include_superseded, since, until) -> list[dict]:
        """Keyed get: rows whose name matches `query` exactly (case-folded).
        The L0 retrieval mode — no ranking, no fuzzy match."""
        where = ["lower(m.name) = lower(?)"]
        params: list = [query.strip()]
        if kind:
            where.append("m.kind = ?")
            params.append(kind)
        if project:
            where.append("m.project = ?")
            params.append(project)
        if not include_superseded:
            where.append("m.superseded_time IS NULL")
        if since:
            where.append("(m.event_time >= ? OR m.event_time_end >= ?)")
            params += [since, since]
        if until:
            where.append("m.event_time < ?")
            params.append(until)
        sql = (f"SELECT m.* FROM memory m WHERE {' AND '.join(where)} "
               "ORDER BY m.event_time DESC LIMIT ?")
        params.append(limit)
        rows = [dict(r) for r in self.conn.execute(sql, params)]
        for r in rows:
            r["score"] = 1.0  # exact hit; keeps the result shape uniform
        return rows

    def _resolve_embedder(self, embedder):
        if embedder is False:
            return None
        if embedder is not None:
            return embedder
        if not hasattr(self, "_auto_embedder"):
            self._auto_embedder = self._auto_resolve_embedder()
            if self._auto_embedder is not None:
                self._maybe_backfill_vectors(self._auto_embedder)
        return self._auto_embedder

    def _maybe_backfill_vectors(self, emb):
        """Self-healing, on first real vector use per store open: any memory
        lacking vectors for this model gets embedded. That covers both the
        store that predates vectors (nothing embedded yet) and the store that
        LOST coverage — vector-dropping edits (set_gist before it re-embedded,
        writes from an environment without the model) used to leave permanent
        holes, because this guard bailed the moment ANY embedding existed.
        Semantic recall is silently blind to an unembedded row, so gaps must
        close themselves rather than wait for a manual `embed`. Cost: one
        indexed lookup when coverage is full; embedding work only for the gap
        rows (backfill is incremental). Never blocks or raises; it triggers on
        store()/recall(), not on bare open or admin commands."""
        try:
            gap = self.conn.execute(
                """SELECT 1 FROM memory m
                   LEFT JOIN embedding e ON e.memory_id = m.id AND e.model = ?
                   WHERE e.memory_id IS NULL LIMIT 1""", (emb.name,)).fetchone()
            if gap is None:
                return  # full coverage — embed-on-write maintains it from here
            from .vectors import backfill
            n = backfill(self, emb)
            if n:
                import sys
                print(f"FornixDB: embedded {n} memories that were missing "
                      f"vectors ({emb.name}) — semantic recall now covers them.",
                      file=sys.stderr)
        except Exception:
            pass  # backfill is best-effort; never break a store/recall over it

    def _auto_resolve_embedder(self):
        """Vectors are on by default: load the bundled model so a fresh store
        embeds from its first write. Three ways it stays off: a machine-wide
        env switch (`FORNIXDB_VECTORS=off`), a per-store opt-out
        (`config vectors off`), and incapable hardware where the model can't
        import/load — get_default_embedder() returns None there, so recall and
        writes fall back to keyword + time and nothing breaks."""
        import os
        _OFF = ("off", "0", "false", "no")
        env = os.environ.get("FORNIXDB_VECTORS")
        if env is not None:
            if env.strip().lower() in _OFF:        # env is the machine-wide override
                return None
        else:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key = 'vectors'").fetchone()
            if row and str(row["value"]).strip().lower() in _OFF:
                return None
        from .vectors import get_default_embedder
        return get_default_embedder()

    def _blend_vectors(self, fts_rows, query, emb, limit, kind, project,
                       include_superseded, since=None, until=None):
        """Merge keyword hits with vector neighbors into one ranked list.
        relevance := -bm25 + VECTOR_WEIGHT * cosine, then the usual
        salience/recency blend in _score."""
        from .vectors import similar
        try:
            neighbors = {mid: cos for mid, cos in
                         similar(self, emb, query, limit=max(limit * 3, 25),
                                 include_superseded=include_superseded)
                         if cos >= VECTOR_MIN_COS}
        except Exception:
            return fts_rows  # vectors must never break recall

        vec_ranked = sorted(neighbors, key=lambda m: neighbors[m], reverse=True)
        by_id = {r["id"]: r for r in fts_rows}
        missing = [mid for mid in vec_ranked if mid not in by_id]
        if missing:
            ph = ",".join("?" * len(missing))
            where = [f"m.id IN ({ph})"]
            params: list = list(missing)
            if kind:
                where.append("m.kind = ?")
                params.append(kind)
            if project:
                where.append("m.project = ?")
                params.append(project)
            if not include_superseded:
                where.append("m.superseded_time IS NULL")
            if since:  # vector neighbors honor the time window too
                where.append("(m.event_time >= ? OR m.event_time_end >= ?)")
                params += [since, since]
            if until:
                where.append("m.event_time < ?")
                params.append(until)
            for r in self.conn.execute(
                    f"SELECT m.*, 0.0 AS relevance FROM memory m WHERE {' AND '.join(where)}",
                    params):
                by_id[r["id"]] = dict(r)

        # Top-up: keyword candidates that didn't make the neighbor shortlist
        # still need their TRUE cosine — reading them as 0.0 stripped their
        # vector relevance, made rankings shift with `limit` (the shortlist
        # scales with it), and false-abstained the gate on keyword-anchored
        # rank-1 hits (seen live: a 0.37-cosine top hit read as 0.0 < gate).
        # The same VECTOR_MIN_COS noise floor applies as for the shortlist.
        unscored = [mid for mid in by_id if mid not in neighbors]
        if unscored:
            try:
                from .vectors import cosines_for
                for mid, cos in cosines_for(self, emb, query, unscored).items():
                    if cos >= VECTOR_MIN_COS:
                        neighbors[mid] = cos
            except Exception:
                pass  # exact-cosine top-up is an upgrade, never a requirement

        now = datetime.now()
        out = []
        for mid, row in by_id.items():
            cos = max(neighbors.get(mid, 0.0), 0.0)
            row["vec_cos"] = cos            # exposed for the abstention gate
            row["relevance"] = (float(row.get("relevance") or 0.0)
                                + VECTOR_WEIGHT * cos)
            row["score"] = self._score(row, now)
            out.append(row)
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:limit]

    def _recall_fts(self, query, mode, limit, kind, project, include_superseded,
                    since=None, until=None, keep=None):
        where = ["memory_fts MATCH ?"]
        params: list = [_fts_query(query, mode)]
        if kind:
            where.append("m.kind = ?")
            params.append(kind)
        if project:
            where.append("m.project = ?")
            params.append(project)
        if not include_superseded:
            where.append("m.superseded_time IS NULL")
        if since:  # a spanned event (event_time_end) overlaps the window too
            where.append("(m.event_time >= ? OR m.event_time_end >= ?)")
            params += [since, since]
        if until:
            where.append("m.event_time < ?")
            params.append(until)
        # column weights (name, gist, detail): a hit in the title outranks the
        # same hit buried in detail — names are searchable as of schema v2
        sql = f"""
            SELECT m.*, -bm25(memory_fts, 3.0, 2.0, 1.0) AS relevance
            FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid
            WHERE {' AND '.join(where)}
            ORDER BY relevance DESC LIMIT ?
        """
        keep = limit if keep is None else keep
        # overfetch generously for the salience/recency re-rank headroom, then
        # return `keep` rows (= limit normally; the wider blend set when vectors
        # follow, so their bm25 relevance survives into _blend_vectors)
        params.append(max(keep, limit * 5, 25))
        try:
            rows = [dict(r) for r in self.conn.execute(sql, params)]
        except sqlite3.OperationalError:
            return []
        now = datetime.now()
        for r in rows:
            r["score"] = self._score(r, now)
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:keep]

    def _decay_cfg(self) -> tuple[dict, dict]:
        if not hasattr(self, "_decay_cache"):
            half, floor = dict(DECAY_HALFLIFE), dict(DECAY_FLOOR)
            for r in self.conn.execute(
                    "SELECT key, value FROM meta WHERE key LIKE 'decay_%'"):
                _, which, kind = r["key"].split("_", 2)
                target = half if which == "halflife" else floor
                if kind in target:
                    target[kind] = float(r["value"])
            self._decay_cache = (half, floor)
        return self._decay_cache

    def effective_salience(self, row: dict, now: datetime | None = None) -> float:
        """Stored salience decayed since last recall (or storage), floored per
        kind so load-bearing kinds (feedback) never sink out of reach."""
        now = now or datetime.now()
        half, floor = self._decay_cfg()
        kind = row["kind"]
        anchor = row.get("last_recalled") or row.get("recorded_time")
        try:
            days = max((now - datetime.fromisoformat(anchor)).days, 0)
        except (ValueError, TypeError):
            days = 365
        decayed = float(row["salience"]) * math.exp(
            -days / half.get(kind, 120.0))
        return max(floor.get(kind, 0.1), decayed)

    def stale_days(self, row: dict, now: datetime | None = None) -> int | None:
        """Days since this memory was last reinforced (or stored), when that
        age exceeds the kind's decay half-life — the "verify before trusting"
        flag: a fact this old and unused may describe a world that has moved
        on. Episodic rows are history, not claims, so they never flag."""
        if row["kind"] == "episodic":
            return None
        now = now or datetime.now()
        half, _ = self._decay_cfg()
        # anchor on REINFORCEMENT (detail engagement), not passive listing —
        # otherwise the flag would vanish the first time anyone saw it
        anchor = row.get("last_reinforced") or row.get("recorded_time")
        try:
            days = (now - datetime.fromisoformat(anchor)).days
        except (ValueError, TypeError):
            return None
        return days if days > half.get(row["kind"], 120.0) else None

    def _attach_neighbors(self, rows: list[dict], per: int = 3) -> None:
        """Spreading activation: each recalled row gains its 1-hop typed
        links (`related` key) — the association a person follows from one
        memory to the next. Tombstoned neighbors are skipped; capped per row
        so recall output stays context-affordable."""
        ids = [r["id"] for r in rows]
        if not ids:
            return
        ph = ",".join("?" * len(ids))
        found: dict[int, list[dict]] = {}
        for ln in self.conn.execute(
                f"""SELECT ml.memory_id AS src, ml.relation, ml.related_id AS nid,
                           m2.gist AS ngist
                    FROM memory_link ml JOIN memory m2 ON m2.id = ml.related_id
                    WHERE ml.memory_id IN ({ph}) AND m2.superseded_time IS NULL
                    UNION ALL
                    SELECT ml.related_id, ml.relation || '-by', ml.memory_id, m1.gist
                    FROM memory_link ml JOIN memory m1 ON m1.id = ml.memory_id
                    WHERE ml.related_id IN ({ph}) AND m1.superseded_time IS NULL""",
                ids + ids):
            bucket = found.setdefault(ln["src"], [])
            if len(bucket) < per and ln["nid"] not in {n["id"] for n in bucket}:
                bucket.append({"id": ln["nid"], "relation": ln["relation"],
                               "gist": ln["ngist"]})
        for r in rows:
            r["related"] = found.get(r["id"], [])

    def _score(self, row: dict, now: datetime) -> float:
        relevance = float(row.get("relevance") or 0.0)
        try:
            age_days = max((now - datetime.fromisoformat(row["event_time"])).days, 0)
        except (ValueError, TypeError):
            age_days = 365
        recency = RECENCY_WEIGHT * math.exp(-age_days / RECENCY_HALFLIFE_DAYS)
        eff = self.effective_salience(row, now)
        return (relevance * (1 + SALIENCE_WEIGHT * eff + self._usefulness(row))
                + recency)

    def _usefulness(self, row: dict) -> float:
        """A saturating bonus for memories that have proven useful — explicit
        "this helped" endorsements (strongest) plus scan-verified referenced use
        (weaker). Folded into the salience multiplier (not added flat) so it
        scales a real relevance match rather than lifting unrelated rows: a used
        memory outranks an equally-relevant unused one, but usefulness alone
        never makes an irrelevant memory surface. recall_count does NOT feed
        rank: pull counts carry listing-era inflation and would entrench old
        rows against new ones at relevance parity (the rich-get-richer
        crowding); referenced_count is the honest engagement signal."""
        bonus = 0.0
        h = float(row.get("helpful_count") or 0)
        if h > 0:
            bonus += USEFULNESS_WEIGHT * (1.0 - math.exp(-h / USEFULNESS_SATURATION))
        r = float(row.get("referenced_count") or 0)
        if r > 0:
            bonus += REFERENCED_WEIGHT * (1.0 - math.exp(-r / REFERENCED_SATURATION))
        return bonus

    def effective_floor(self, row: dict, base_floor: float,
                        active_project: str | None = None,
                        aliases: set[str] | tuple = ()) -> float:
        """The PUSH relevance floor for ONE memory: `base_floor`, adjusted by two
        independent dials (each its own config switch).

        Usefulness (`usefulness_floor_adapt`, default on): used vs ignored from the
        durable PUSH-outcome counts — uses = HELPFUL_USE_WEIGHT*helpful_count +
        REFERENCED_USE_WEIGHT*referenced_count (endorsements and pushes that were
        actually used downstream), impressions = surfaced_count (proactive pushes).
        A used memory gets a discount (easier to surface); one pushed many times
        but never used gets a penalty (quieter). referenced_count closes the loop:
        a pushed memory is used in-context WITHOUT a pull, so without it a
        proven-useful push would look identical to ignored noise. recall_count
        deliberately does NOT count here: pulls are the other channel (a pulled
        memory needs no pushing to be found), and on a lived-in store the
        listing-era inflation saturated the discount for every row and masked the
        never-used population entirely — measured 2026-07-02: 190/324 rows at max
        discount, zero penalties on the 75 pushed-but-never-used rows.

        Project scope (`project_scoped_pulse`, default on; only when `active_project`
        is given): a memory that does NOT belong to the active context clears a
        higher bar, so off-context memories stop leaking into the stream on weak
        matches while a strongly-relevant one still surfaces. "Belongs" unifies
        project and topics — on-context if the memory's project OR any of its
        (non-structural) topics matches the active label or one of `aliases`.
        Memories with no scoping tags (general facts) are never penalized.

        Both dials only ever move the floor within sane bounds: never above
        FLOOR_CAP, never below 0, and explicit recall ignores it entirely."""
        floor = base_floor
        if not self._setting_off("usefulness_floor_adapt"):
            uses = (HELPFUL_USE_WEIGHT * float(row.get("helpful_count") or 0)
                    + REFERENCED_USE_WEIGHT * float(row.get("referenced_count") or 0))
            impressions = float(row.get("surfaced_count") or 0)
            floor -= FLOOR_DISCOUNT_MAX * (1.0 - math.exp(-uses / FLOOR_USE_SATURATION))
            if impressions >= FLOOR_MIN_IMPRESSIONS:
                ignored = max(0.0, impressions - uses)
                floor += FLOOR_PENALTY_MAX * (1.0 - math.exp(-ignored / FLOOR_IGNORE_SATURATION))
        if active_project and not self._setting_off("project_scoped_pulse"):
            ctx = {active_project.strip().lower()}
            ctx |= {str(a).strip().lower() for a in aliases}
            proj = (row.get("project") or "").strip().lower()
            topics = {str(t).strip().lower() for t in (row.get("topics") or [])}
            tags = (({proj} if proj else set()) | topics) - STRUCTURAL_TOPICS
            if tags and not (tags & ctx):     # tagged, but for another context
                floor += PROJECT_MISMATCH_PENALTY
        return max(0.0, min(floor, FLOOR_CAP))

    def timeline(
        self,
        start: str,
        end: str,
        *,
        kind: str | None = None,
        project: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Time-axis recall: what happened in [start, end), oldest first.
        Includes superseded rows (they happened) — flagged by the caller."""
        where = ["((m.event_time >= ? AND m.event_time < ?) "
                 "OR (m.event_time_end IS NOT NULL AND m.event_time < ? AND m.event_time_end >= ?))"]
        params: list = [start, end, end, start]
        if kind:
            where.append("m.kind = ?")
            params.append(kind)
        if project:
            where.append("m.project = ?")
            params.append(project)
        sql = f"""SELECT m.* FROM memory m WHERE {' AND '.join(where)}
                  ORDER BY m.event_time ASC LIMIT ?"""
        params.append(limit)
        rows = [dict(r) for r in self.conn.execute(sql, params)]
        # a timeline sweep LISTS rows, it doesn't engage with them — an
        # impression, not a use (engagement = show / mark_helpful / referenced)
        self.record_surfaced([r["id"] for r in rows])
        return rows

    def show(self, ref: int | str, reinforce: bool = True) -> dict | None:
        """Fetch a single memory (by id or name) with full detail, topics, and
        links. Detail recall reinforces salience — like human memory."""
        if isinstance(ref, str) and not ref.isdigit():
            row = self.conn.execute("SELECT * FROM memory WHERE name = ?", (ref,)).fetchone()
        else:
            row = self.conn.execute("SELECT * FROM memory WHERE id = ?", (int(ref),)).fetchone()
        if row is None:
            return None
        mem = dict(row)
        if mem["detail"] is None and mem["retention_tier"] in ("consolidated", "cold"):
            from .tiers import load_detail  # transparent restore (P3b)
            mem["detail"] = load_detail(self, mem)
        mem["topics"] = [
            r["name"] for r in self.conn.execute(
                "SELECT t.name FROM topic t JOIN memory_topic mt ON mt.topic_id = t.id "
                "WHERE mt.memory_id = ?", (mem["id"],))
        ]
        mem["links"] = [
            dict(r) for r in self.conn.execute(
                """SELECT ml.relation, ml.related_id, m2.gist AS related_gist
                   FROM memory_link ml JOIN memory m2 ON m2.id = ml.related_id
                   WHERE ml.memory_id = ?
                   UNION ALL
                   SELECT ml.relation || '-by', ml.memory_id, m1.gist
                   FROM memory_link ml JOIN memory m1 ON m1.id = ml.memory_id
                   WHERE ml.related_id = ?""",
                (mem["id"], mem["id"]))
        ]
        mem["stale_days"] = self.stale_days(mem)
        if reinforce:
            self._mark_recalled([mem["id"]], reinforce=True)
        return mem

    def _mark_recalled(self, ids: list[int], reinforce: bool) -> None:
        if not ids or self.frozen():  # frozen stores recall without writing
            return
        bump = REINFORCE_BUMP if reinforce else 0.0
        reinforced = ", last_reinforced = ?" if reinforce else ""
        ts = now_iso()
        self.conn.executemany(
            f"""UPDATE memory SET last_recalled = ?, recall_count = recall_count + 1,
                                  salience = min(salience + ?, ?){reinforced}
                WHERE id = ?""",
            [((ts, bump, SALIENCE_CAP, ts, i) if reinforce
              else (ts, bump, SALIENCE_CAP, i)) for i in ids],
        )
        self.conn.commit()
        # Redemption: reinforcement is a deliberate single-target engagement
        # (show, explicit recall detail) — the host demonstrated the memory
        # matters, so it earns its way back into the push channels. Passive
        # listing (reinforce=False, e.g. a push candidate-gather) does NOT redeem.
        if reinforce:
            self.clear_proactive_suppression(ids, "reinforced")

    def record_surfaced(self, ids: list[int]) -> None:
        """Count a proactive PUSH impression: this memory was injected unsolicited
        (L3 once-per-turn / L4 rhythmic) rather than pulled by an explicit recall.
        Deliberately NOT a recall — it never bumps recall_count or salience, so a
        memory the system keeps pushing but no one ever uses accrues impressions
        without ever looking 'used'. That gap (surfaced_count vs recall_count/
        helpful_count) is the implicit noise signal `effective_floor` acts on.
        Frozen/read-only stores skip silently, like _mark_recalled."""
        if not ids or self.frozen():
            return
        ts = now_iso()
        self.conn.executemany(
            """UPDATE memory SET surfaced_count = surfaced_count + 1,
                                 last_surfaced = ? WHERE id = ?""",
            [(ts, i) for i in ids],
        )
        self.conn.commit()

    def record_referenced(self, counts: dict[int, int]) -> int:
        """Materialize the honest push-usefulness signal: for each memory id, how
        many of its proactive PUSHES were actually USED downstream (cited in the
        host's later reasoning — the usefulness-scan result). This is the credit
        `effective_floor` folds into `uses` so a proven-useful push isn't scored as
        ignored noise.

        Set ABSOLUTELY, not incremented: the scan is authoritative over the whole
        transcript window it can see, so re-running is idempotent (never
        double-counts). `last_referenced` is stamped only for ids getting a positive
        credit. Returns the number of memories credited (>0). Frozen/read-only
        stores skip silently, like the other counters."""
        if not counts or self.frozen():
            return 0
        ts = now_iso()
        # last_referenced marks genuine downstream use, so only advance it for a
        # positive count; a reset to 0 clears the count but leaves the timestamp.
        self.conn.executemany(
            """UPDATE memory
                  SET referenced_count = ?,
                      last_referenced = CASE WHEN ? > 0 THEN ? ELSE last_referenced END
                WHERE id = ?""",
            [(int(n), int(n), ts, i) for i, n in counts.items()],
        )
        self.conn.commit()
        return sum(1 for n in counts.values() if int(n) > 0)

    # ---------------------------------------------------- proactive suppression
    # A memory chronically PUSHED but never REFERENCED is push-noise the cosine
    # floor provably can't filter (useful vs noise cosines overlap — measured
    # 2026-07-12). Push OUTCOME history separates them cleanly, so such a memory
    # is proactive-SUPPRESSED: excluded from the L3/L4/L5 push channels only.
    # These methods are mechanical (set / clear / list + a beside-the-store
    # audit log); the RULE that decides which ids qualify lives in suppress.py,
    # exactly as usefulness_scan owns policy and record_referenced owns the write.
    # INVARIANT: suppression never touches recall/show/timeline — a suppressed
    # memory is always still explicitly reachable.

    def _suppress_log_path(self) -> str | None:
        """suppress_log.jsonl beside the store db (like floor_log) — an audit
        trail of every suppress/un-suppress with its justification. None for an
        in-memory store."""
        from .proactive import floor_log_path_for
        from pathlib import Path
        p = floor_log_path_for(self)
        return str(Path(p).with_name("suppress_log.jsonl")) if p else None

    def _log_suppression(self, event: str, records: list[dict]) -> None:
        path = self._suppress_log_path()
        if not path or not records:
            return
        import json
        ts = now_iso()
        try:
            with open(path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps({"ts": ts, "event": event, **r},
                                       ensure_ascii=False) + "\n")
        except Exception:
            pass

    def suppress_proactive(self, stats: dict[int, tuple], at: str | None = None) -> int:
        """Mark memories proactive-suppressed. `stats` is {id: (pushed, referenced)}
        — the justifying counts, stored so `suppress --list` can show WHY without
        re-scanning. Only rows that EXIST and are NOT already suppressed are touched
        (idempotent; re-running the scan never re-stamps or double-logs). Returns the
        number newly suppressed. Frozen/read-only stores skip silently."""
        if not stats or self.frozen():
            return 0
        newly = []
        for i, pr in stats.items():
            row = self.conn.execute(
                "SELECT proactive_suppressed_at FROM memory WHERE id = ?", (int(i),)
            ).fetchone()
            if row is None or row["proactive_suppressed_at"] is not None:
                continue                 # gone, or already suppressed
            newly.append((int(i), int(pr[0]), int(pr[1])))
        if not newly:
            return 0
        ts = at or now_iso()
        self.conn.executemany(
            """UPDATE memory SET proactive_suppressed_at = ?,
                                 suppressed_pushed = ?, suppressed_referenced = ?
               WHERE id = ?""",
            [(ts, p, r, i) for (i, p, r) in newly])
        self.conn.commit()
        self._log_suppression("suppress", [
            {"id": i, "pushed": p, "referenced": r, "reason": "scan"}
            for (i, p, r) in newly])
        return len(newly)

    def clear_proactive_suppression(self, ids, reason: str) -> int:
        """Redeem memories — un-suppress so they can push again. Called on the
        deliberate single-target signals that a suppressed memory actually matters:
        show/explicit-recall reinforcement, mark_helpful, and content change
        (supersede/set-gist). Logs only rows that were genuinely suppressed (so a
        no-op reinforce doesn't spam the log). Returns the number cleared. Frozen
        stores skip silently."""
        ids = [int(i) for i in ids]
        if not ids or self.frozen():
            return 0
        ph = ",".join("?" * len(ids))
        cleared = [r["id"] for r in self.conn.execute(
            f"SELECT id FROM memory WHERE id IN ({ph}) "
            "AND proactive_suppressed_at IS NOT NULL", ids)]
        if not cleared:
            return 0
        ph2 = ",".join("?" * len(cleared))
        self.conn.execute(
            f"""UPDATE memory SET proactive_suppressed_at = NULL,
                                  suppressed_pushed = NULL, suppressed_referenced = NULL
                WHERE id IN ({ph2})""", cleared)
        self.conn.commit()
        self._log_suppression("unsuppress",
                              [{"id": i, "reason": reason} for i in cleared])
        return len(cleared)

    def proactive_suppressed(self) -> list[dict]:
        """Every currently-suppressed memory with its justifying stats — the
        `suppress --list` view. Ordered by push count (loudest noise first)."""
        return [dict(r) for r in self.conn.execute(
            """SELECT id, gist, kind, project, suppressed_pushed,
                      suppressed_referenced, proactive_suppressed_at
               FROM memory WHERE proactive_suppressed_at IS NOT NULL
               ORDER BY suppressed_pushed DESC, id ASC""")]

    def topics_for(self, ids: list[int]) -> dict[int, list[str]]:
        """Batch-fetch topic names for several memories in one query (the proactive
        belongs test needs topics, which plain recall rows don't carry). Returns
        {memory_id: [topic, ...]}; ids with no topics are absent."""
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        out: dict[int, list[str]] = {}
        for mid, name in self.conn.execute(
                f"""SELECT mt.memory_id, t.name FROM memory_topic mt
                    JOIN topic t ON t.id = mt.topic_id
                    WHERE mt.memory_id IN ({ph})""", ids):
            out.setdefault(mid, []).append(name)
        return out

    # ---------------------------------------------------------------- misc

    def record_session(self, session_id: str, *, project=None, started=None,
                       ended=None, source=None, source_ref=None) -> None:
        self._check_writable()
        self.conn.execute(
            """INSERT INTO session(id, project, started, ended, source, source_ref)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 project=excluded.project, started=excluded.started,
                 ended=excluded.ended, source=excluded.source,
                 source_ref=excluded.source_ref""",
            (session_id, project, started, ended, source, source_ref),
        )
        self.conn.commit()

    def brief(self, *, project: str | None = None, days: int = 7,
              recent_limit: int = 8, salient_limit: int = 10,
              useful_limit: int = 5) -> dict:
        """Session-start context brief: recent activity + most salient
        standing knowledge. Gist-only and capped — this is the cheap recall
        that opens every session; detail is always a `show` away."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        pw, pp = ("AND m.project = ?", [project]) if project else ("", [])
        recent = [dict(r) for r in self.conn.execute(
            f"""SELECT m.* FROM memory m
                WHERE m.kind = 'episodic' AND m.event_time >= ? {pw}
                ORDER BY m.event_time DESC LIMIT ?""",
            [since, *pp, recent_limit])]
        # overfetch, then rank by EFFECTIVE salience so stale high-salience
        # rows sink and reinforced ones surface (P3a decay)
        cand = [dict(r) for r in self.conn.execute(
            f"""SELECT m.* FROM memory m
                WHERE m.kind != 'episodic' AND m.superseded_time IS NULL {pw}
                ORDER BY m.salience DESC, m.event_time DESC LIMIT ?""",
            [*pp, salient_limit * 4])]
        now = datetime.now()
        for r in cand:
            r["eff_salience"] = round(self.effective_salience(r, now), 3)
        cand.sort(key=lambda r: r["eff_salience"], reverse=True)
        salient = cand[:salient_limit]
        # listing in the brief is an unsolicited PUSH, not engagement: counting
        # it as a recall let every listed row refresh its decay anchor and pump
        # recall_count each session — the rich-get-richer loop that froze old
        # rows at the top of ranking (measured 2026-07-02). Impressions only.
        self.record_surfaced([r["id"] for r in recent + salient])
        # the usefulness rollup is META about what has proven worth surfacing —
        # it is NOT itself a content recall, so it counts nothing at all (even
        # an impression would let the rollup feed its own noise signal)
        useful = self.top_useful(useful_limit) if useful_limit else []
        return {"since": since[:10], "recent": recent, "salient": salient,
                "useful": useful}

    def stats(self) -> dict:
        q = self.conn.execute
        return {
            "memories": q("SELECT count(*) c FROM memory").fetchone()["c"],
            "by_kind": {r["kind"]: r["c"] for r in q(
                "SELECT kind, count(*) c FROM memory GROUP BY kind")},
            "superseded": q(
                "SELECT count(*) c FROM memory WHERE superseded_by IS NOT NULL").fetchone()["c"],
            "proactive_suppressed": q(
                "SELECT count(*) c FROM memory WHERE proactive_suppressed_at IS NOT NULL"
            ).fetchone()["c"],
            "topics": q("SELECT count(*) c FROM topic").fetchone()["c"],
            "links": q("SELECT count(*) c FROM memory_link").fetchone()["c"],
            "sessions": q("SELECT count(*) c FROM session").fetchone()["c"],
            "span": dict(q(
                "SELECT min(event_time) AS earliest, max(event_time) AS latest FROM memory"
            ).fetchone()),
        }
