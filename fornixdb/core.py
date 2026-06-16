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

from .db import KINDS, RELATIONS, connect

SALIENCE_WEIGHT = 1.0     # how much a salient memory outranks an equally-relevant one
RECENCY_WEIGHT = 2.0      # max score bonus for a memory from "right now"
RECENCY_HALFLIFE_DAYS = 90.0
REINFORCE_BUMP = 0.05     # salience bump each time detail is recalled
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

# Negative feedback (owner decisions 2026-06-12: explicit-only signal,
# query-conditional penalty). When the current query is similar to a query a
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
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self._check_writable()
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
        emb = self._resolve_embedder(embedder)
        if emb is not None:
            try:
                from .vectors import embed_memory
                embed_memory(self, emb, mem_id)
            except Exception:
                pass
        return mem_id

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

    def set_gist(self, memory_id: int, gist: str) -> None:
        """In-place gist rewrite (consolidation, Design §13.5 decision 2): the
        gist is derived presentation, the detail/source is the record, so no
        supersession. A meaning change is a new memory + supersede, not this.
        The FTS index updates via trigger; the stale vector is dropped so the
        next `embed` re-embeds the row."""
        self._check_writable()
        self.conn.execute("UPDATE memory SET gist = ? WHERE id = ?",
                          (gist, memory_id))
        self.conn.execute("DELETE FROM embedding WHERE memory_id = ?", (memory_id,))
        self.conn.commit()

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
        self._mark_recalled([r["id"] for r in rows], reinforce=False)
        return rows

    def _resolve_embedder(self, embedder):
        if embedder is False:
            return None
        if embedder is not None:
            return embedder
        if not hasattr(self, "_auto_embedder"):
            # Only auto-load a model if this store actually uses vectors —
            # a vector-free deployment never pays the import.
            has_vectors = self.conn.execute(
                "SELECT 1 FROM embedding LIMIT 1").fetchone()
            if has_vectors:
                from .vectors import get_default_embedder
                self._auto_embedder = get_default_embedder()
            else:
                self._auto_embedder = None
        return self._auto_embedder

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
        return relevance * (1 + SALIENCE_WEIGHT * eff) + recency

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
        self._mark_recalled([r["id"] for r in rows], reinforce=False)
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
              recent_limit: int = 8, salient_limit: int = 10) -> dict:
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
        self._mark_recalled([r["id"] for r in recent + salient], reinforce=False)
        return {"since": since[:10], "recent": recent, "salient": salient}

    def stats(self) -> dict:
        q = self.conn.execute
        return {
            "memories": q("SELECT count(*) c FROM memory").fetchone()["c"],
            "by_kind": {r["kind"]: r["c"] for r in q(
                "SELECT kind, count(*) c FROM memory GROUP BY kind")},
            "superseded": q(
                "SELECT count(*) c FROM memory WHERE superseded_by IS NOT NULL").fetchone()["c"],
            "topics": q("SELECT count(*) c FROM topic").fetchone()["c"],
            "links": q("SELECT count(*) c FROM memory_link").fetchone()["c"],
            "sessions": q("SELECT count(*) c FROM session").fetchone()["c"],
            "span": dict(q(
                "SELECT min(event_time) AS earliest, max(event_time) AS latest FROM memory"
            ).fetchone()),
        }
