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
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from .db import KIND_ALIASES, KINDS, RELATIONS, connect

SALIENCE_WEIGHT = 1.0     # how much a salient memory outranks an equally-relevant one
RECENCY_WEIGHT = 2.0      # max score bonus for a memory from "right now"
RECENCY_HALFLIFE_DAYS = 90.0
REINFORCE_BUMP = 0.05     # salience bump each time detail is recalled
STATUS_TIP_MAX_EDITIONS = 200  # how far the brief counts back before saying "+"
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
# everywhere). Excluded from the belongs test; domain topics (fornixdb, studio,
# security, …) still count.
STRUCTURAL_TOPICS = frozenset({
    "reference", "feedback", "project", "milestone", "distilled", "pickup",
    "publication", "documentation", "roadmap", "naming",
})
SALIENCE_CAP = 1.0


def _canonical_project(store, project):
    """`context.canonical_project`, but never fatal to a write. A store that
    cannot answer (peer, mid-migration, missing config table) keeps the caller's
    label trimmed — a fragmented label is a bad day, a failed store() is a lost
    memory, and this module's whole contract is that the write lands."""
    try:
        from .context import canonical_project   # lazy: import cycle via multistore
        return canonical_project(store, project)
    except Exception:
        return project.strip() if isinstance(project, str) else project


# Write-path gist ceiling. A gist is what recall returns and what proactive.py
# pushes — and that push truncates at MAX_GIST=200, so an oversized gist reaches
# the consumer as a headline cut mid-sentence. Measured 2026-08-23 over 49
# sessions of live transcripts, downstream reference rate by gist length peaks
# in the 301-400 band (25%) and falls monotonically above it: 401-500 13.6%,
# 501-600 10.0%, 601-800 0%, 1201+ 3.2%. So the write path splits here rather
# than trusting each writer to be disciplined — 74% of gists written in August
# blew the advisory limit, 212 of 235 from an agent calling
# `store --gist "<wall of text>"` with no --detail at all.
#
# Deliberately NOT the same number as consolidate.GIST_MAX_CHARS (200): that one
# is an advisory dream complaint ("a gist this long is a summary that failed"),
# and splitting every row down to it would cut into the best-measured band.
# Kept as a separate constant rather than shared because core cannot import
# consolidate (cycle) and the two answer different questions.
GIST_MAX_CHARS = 400
GIST_MIN_HEAD = 120       # never leave a stub gist: a split this early is worse
                          # than a slightly long one, so fall back instead
_GIST_BOUNDARIES = ("\n\n", ". ", "! ", "? ", ".\n", "!\n", "?\n")


def split_gist(gist: str, detail: str | None = None,
               limit: int = GIST_MAX_CHARS) -> tuple[str, str | None]:
    """Fold an oversized gist's overflow into detail. Content is never lost —
    only moved to where drill-down already reads it (`show`).

    Splits at the last paragraph or sentence boundary at or before `limit`, else
    the last word boundary, so what stays in the gist is a whole thought rather
    than a fragment. Returns its arguments unchanged when the gist already fits,
    which makes it idempotent and safe to call on any write path."""
    if not gist or len(gist) <= limit:
        return gist, detail
    window = gist[:limit + 1]
    cut = max((window.rfind(b) for b in _GIST_BOUNDARIES), default=-1)
    if cut >= GIST_MIN_HEAD:
        cut += 1                                   # keep the terminator
    else:
        cut = window.rfind(" ", GIST_MIN_HEAD)     # no sentence end: word break
        if cut < GIST_MIN_HEAD:
            cut = limit                            # one unbroken token
    head, tail = gist[:cut].rstrip(), gist[cut:].strip()
    if not tail:                                   # split fell in trailing space
        return head, detail
    return head, (tail + "\n\n" + detail) if detail else tail


# Default output budget for the recall/timeline SURFACES. Not a store concern —
# it lives here only so the CLI and the MCP server cannot drift apart again, as
# they had: MCP defaulted to 4000 while the CLI had no default at all, and the
# writeback hint points agents at the CLI, so the guard was absent on the path
# actually used. 0 or None means unlimited (see cli.fit_chars).
DEFAULT_MAX_CHARS = 4000


VECTOR_WEIGHT = 15.0      # scales cosine into the -bm25 range. Tuned 2026-06-11
                          # via the eval fence: at 6.0, OR-mode keyword noise
                          # (bm25 ≈ 7-9 from common tokens) buried the clearly
                          # best vector hit (cos .57 vs .41 → eval miss #17).
                          # Pure rank fusion (RRF) was tried first: fixed the
                          # miss but flattened hit@1 78→56% — margins matter.
VECTOR_MIN_COS = 0.30     # noise floor: weaker similarity is no evidence at all
# How many keyword rows survive into the vector blend (and how far FTS
# overfetches for the salience/recency re-rank). A CONSTANT on purpose: when
# this scaled with `limit`, the same query returned a different ORDER at
# different limits, because a mid-ranked bm25 row kept its keyword relevance
# in a wide fetch but lost it in a narrow one (2026-07-16, eval case #17).
FTS_BLEND_KEEP = 100

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
# The OR leg of that same calibration (implemented 2026-07-16 after a live
# false-abstain: a rank-1 hit anchored by literal tokens — "qwen 72b … consumer",
# keyword relevance 18.4 — sat at TRUE cosine 0.297, just under the 0.30
# shortlist floor, so vec_cos read 0.0 and the cosine-only gate suppressed a
# real answer). Keyword-ONLY recall already trusts every FTS anchor; hybrid
# recall now trusts one too, but with BOTH checks — bm25 alone regressed a
# clean negative ('capital of France' matched common tokens at kw_rel 9.26,
# raw cosine 0.001), so the anchor must also be corroborated by the raw
# (unfloored) cosine at a low bar: the 2026-06-13 calibration put clear
# negatives at cos < 0.12. bm25 magnitudes are store-dependent, so these are
# provisional like the other constants — tune against the eval fence.
RECALL_ANSWER_KW_REL = 7.1   # literal-token anchor: calibrated positive band
# Raised 0.15 -> 0.22 on 2026-08-03: the 0.15 bar rested on "clear negatives sit
# at cos < 0.12", and store growth expired that premise. A negative reached raw
# cosine 0.167 with kw_rel 8.99 ("how do I bake sourdough bread" matched a row
# about BAKING skeletal meshes to static ones — one shared word, two unrelated
# senses, which a static embedder cannot tell apart) and leaked through this leg.
# Re-measured across both live stores: the leg's only golden positive sits at
# raw 0.297 / kw_rel 18.32, every leg-eligible negative at raw <= 0.167. 0.22
# splits that gap, leaning toward the positive because false-abstain is the
# regression this leg was added to fix. Store-dependent like the rest — re-measure
# it, don't inherit it.
RECALL_ANSWER_KW_COS = 0.22  # raw-cosine corroboration for that anchor
# The cosine leg's own corroboration band (2026-07-17, after the reverse leak):
# best-chunk scoring over long DOCUMENT rows (the markdown bridge ingests whole
# files; an 8-chunk 9.5KB file was the first) hands each chunk a lottery ticket
# on the 0.30 floor — one deep chunk brushed 0.335 against a nonsense query
# with almost no literal-token support (kw_rel 1.73). Measured on the golden
# set: every real near-floor positive shares SOME query vocabulary (minimum
# kw_rel 3.54); the noise mode doesn't. So the cosine leg mirrors the keyword
# leg: unconditional only when genuinely strong, and inside the floor band
# [COS, COS_STRONG) it needs a pinch of the other signal.
RECALL_ANSWER_COS_STRONG = 0.40  # cosine alone suffices above this
RECALL_ANSWER_COS_KW = 3.0       # minimal keyword corroboration in the band
# ...and the count that magnitude alone could not supply (2026-08-03). Both weak
# legs below rest on MAGNITUDES (bm25 sum, cosine), and one accidentally-shared
# word in a short row can clear any magnitude floor that still admits real
# answers. A sweep of twenty ordinary out-of-store questions leaked six of them
# through the two weak legs — household repair, geography, sport, cooking — each
# on a SINGLE common word. Measured on both live stores: real answers decided by
# a weak leg share 3-7 distinct content words with the query, noise shares 0-1.
# Nothing sat at 2. So the weak legs also ask what a magnitude cannot answer —
# how many DISTINCT things do the query and the hit agree on? — because a real
# answer agrees about several and noise agrees about one. A count does not scale
# with store size or row length, which is exactly the portability the bm25 floors
# beside it lack. 2 sits in the measured gap and leans toward answering.
RECALL_ANSWER_MIN_TERMS = 2      # distinct shared content words, both weak legs
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

# Function words carry no topic, so sharing one is not agreement ABOUT anything —
# they are excluded when counting how much a query and a hit really have in common.
# English-only and deliberately short: this is a noise filter, not linguistics, and
# a word wrongly left in costs at most one point of a count that needs two. FTS5's
# own tokenizer does no stopping, so the list lives here rather than in the schema.
_STOPWORDS = frozenset("""
a an the is are was were be been being am do does did doing done have has had having
how what when where why who whom which that this these those there here can could
should would will shall may might must i you he she it we they me him her us them my
your his its our their mine yours hers ours theirs of to in on for with from by at as
and or but if then than so not no nor yes about into over under out up down off again
more most some any all each every both few other another new own same too very just
only also even still much many such get gets got make makes made use uses used using
go goes going come comes came know knows knew think thinks want wants need needs
""".split())


def _content_terms(text: str) -> list[str]:
    """Topic-bearing words of `text`, lowercased and de-duplicated in order.
    Two-character and shorter tokens go with the stopwords: they are overwhelmingly
    function words or fragments, and keeping them would let 'do'/'go'-style noise
    count as agreement."""
    seen = {}
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(w) > 2 and w not in _STOPWORDS:
            seen[w] = None
    return list(seen)


# How much text may contribute one agreement. A row's DETAIL counts only within
# a window this wide, because agreement has to be CONCENTRATED to mean anything:
# a real answer says several of the query's things close together, while a long
# memory about something else collects the same words scattered across pages of
# unrelated prose. Set to the gist ceiling — a paragraph's worth — so the gist
# and a comparable slice of detail are weighed alike.
AGREEMENT_WINDOW_CHARS = GIST_MAX_CHARS


def shared_term_count(query: str, row: dict) -> int:
    """How many DISTINCT content words the query and a recalled row agree on,
    counting the gist plus the single most agreeing WINDOW of the detail.

    The question bm25 cannot answer. A relevance score is a magnitude: one
    accidentally-shared word in a short row can outscore several genuinely-shared
    words in a long one, so magnitude alone cannot tell "this answers the question"
    from "this happens to contain that word". A count can, because a real answer
    agrees with its question about several things and noise agrees about one.

    But a count over the WHOLE row is not length-independent, which is what it was
    claimed to be. Measured on a lived-in store against queries it has no answer
    to, the chance of accidentally agreeing about two things rose with row size —
    0% up to 800 characters, 0.47% past 1600. A long memory is a haystack: given
    enough unrelated prose it will contain any two words you like, in senses that
    have nothing to do with the question. Counting the best window instead makes
    the measure honest about length: on the same store it cut those accidental
    agreements from eight to one and cost none of the golden positives, including
    the six whose agreement lives in their detail rather than their gist.

    The gist always counts, wherever the window falls: it is the summary, and it
    is capped, so it cannot become a haystack of its own."""
    if not query:
        return 0
    q_terms = _content_terms(query)
    if not q_terms:
        return 0
    wanted = set(q_terms)
    gist_hits = {w for w in _content_terms(row.get("gist") or "") if w in wanted}
    detail = row.get("detail") or ""
    if not detail:
        return len(gist_hits)
    # Only occurrences of the query's own words can ever contribute, so walk
    # those rather than re-tokenizing overlapping windows of the whole detail.
    hits = [(m.start(), m.group()) for m in re.finditer(r"[a-z0-9]+", detail.lower())
            if m.group() in wanted]
    best = len(gist_hits)
    for i, (start, _) in enumerate(hits):
        window = set(gist_hits)
        for pos, term in hits[i:]:
            if pos - start >= AGREEMENT_WINDOW_CHARS:
                break
            window.add(term)
        best = max(best, len(window))
    return best


def recall_has_answer(rows: list[dict]) -> bool:
    """True if recall's best hit is a real match; False if the store has
    nothing relevant (the consumer should then act / use its own knowledge /
    say it doesn't know — recall must NOT pose as the answer). Reports presence
    only; never prescribes the next action.

    The gate lives primarily in the VECTOR regime: cosine is a normalized,
    store-independent signal, so a weak top cosine means weak-noise-as-answer
    (the failure that strands a small model). In pure keyword-only recall there
    is no cosine on the rows — an FTS hit there is a literal token anchor by
    definition, so it is trusted (this is also the pre-gate behavior; bm25
    magnitudes are store-dependent and make no portable threshold). In HYBRID
    recall the same literal anchor deserves the same trust: a top hit whose
    pre-blend keyword relevance clears the calibrated positive band
    (RECALL_ANSWER_KW_REL) is a real match even when its cosine is weak —
    otherwise turning vectors ON makes a keyword-answerable question abstain.

    Inside the weak-cosine floor band the gate also counts SHARED CONTENT WORDS
    (`shared_terms`, set by recall()). Every other signal here is a magnitude, and
    magnitudes cannot separate "answers the question" from "happens to contain
    that word" — a single common word in a short row clears any floor that still
    admits real answers. Agreement about several distinct things can."""
    if not rows:
        return False
    top = rows[0]
    if top.get("vec_cos") is None:        # keyword-only recall: trust the FTS anchor
        return True
    vc = float(top["vec_cos"])
    if vc >= RECALL_ANSWER_COS_STRONG:
        return True
    if vc >= RECALL_ANSWER_COS:
        # floor band: a real match here shares at least a little literal
        # vocabulary with the query; a deep-chunk cosine brush does not
        if float(top.get("kw_rel") or 0.0) < RECALL_ANSWER_COS_KW:
            return False
        return _agrees_enough(top)
    # hybrid keyword anchor: a strong literal-token match whose raw (unfloored)
    # cosine corroborates it — semantically-unrelated common-token overlap
    # (raw cosine ~0) stays abstained no matter how big its bm25 sum
    raw = float(top.get("raw_cos", top["vec_cos"]) or 0.0)
    if (float(top.get("kw_rel") or 0.0) < RECALL_ANSWER_KW_REL
            or raw < RECALL_ANSWER_KW_COS):
        return False
    return _agrees_enough(top)


def _agrees_enough(top: dict) -> bool:
    """Does the top hit agree with the query about SEVERAL things, not one word?
    The shared check both weak legs end on. Absent means UNMEASURED, not zero —
    a row built by something other than recall() never had a query to measure
    against, and must keep the pre-2026-08-03 behavior rather than abstain."""
    terms = top.get("shared_terms")
    return terms is None or int(terms) >= RECALL_ANSWER_MIN_TERMS

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
    """One handle to one store. Safe to share across threads: each thread
    gets its own SQLite connection to the same file (WAL serializes them,
    exactly as separate processes are serialized). An injected `conn` or an
    in-memory store pins a single fixed connection instead — those stay
    single-thread, which sqlite3's own cross-thread check enforces."""

    def __init__(self, db_path=None, conn: sqlite3.Connection | None = None):
        self._db_path = db_path
        self._fixed_conn: sqlite3.Connection | None = None
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._embedder_lock = threading.Lock()
        if conn is not None:
            self._fixed_conn = conn
        elif db_path is not None and Path(db_path).name == ":memory:":
            # per-thread connections would each open a DIFFERENT empty db
            self._fixed_conn = connect(db_path)
        else:
            _ = self.conn  # eager: creation/migration happen at construction

    @property
    def conn(self) -> sqlite3.Connection:
        if self._fixed_conn is not None:
            return self._fixed_conn
        c = getattr(self._local, "conn", None)
        if c is None:
            # check_same_thread=False ONLY so close() can run from another
            # thread — each connection is still used by its own thread alone
            # (that confinement is exactly what this property provides).
            c = connect(self._db_path, check_same_thread=False)
            self._local.conn = c
            with self._conns_lock:
                self._conns.append(c)
        return c

    @contextmanager
    def write_txn(self):
        """BEGIN IMMEDIATE transaction for every multi-statement or
        read-then-act write. Two guarantees a bare implicit transaction does
        not give: (1) the write lock is taken BEFORE the reads, so what was
        read is still true when the writes land — no second process can slip
        a write between a SELECT and its UPDATE; (2) all statements commit or
        roll back together. Re-entrant: inside an open transaction it joins
        rather than nests, and the outermost owner commits."""
        conn = self.conn
        if conn.in_transaction:
            yield conn
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def close(self) -> None:
        """Release every SQLite connection this store opened (one per thread
        that used it). Required on Windows before the db file can be moved or
        deleted (open files can't be unlinked there). Call it after worker
        threads are done — closing a connection mid-query raises in that
        thread."""
        if self._fixed_conn is not None:
            self._fixed_conn.close()
            return
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        self._local = threading.local()  # a later use reopens, never sees a closed conn

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
        _in_txn=None,   # private seam: callable(conn, mem_id) run INSIDE the
                        # insert transaction, so a companion row (e.g. a
                        # prospective reminder) commits or rolls back WITH the
                        # memory row. Raw statements on `conn` only — the
                        # self-committing store methods would end the
                        # transaction early.
    ) -> int:
        kind = KIND_ALIASES.get(kind, kind)
        if kind not in KINDS:
            raise ValueError(
                f"kind must be one of {KINDS} (got {kind!r}); "
                f"or a known alias {tuple(KIND_ALIASES)}")
        self._check_writable()
        # One project, one spelling. Capture takes its label from the cwd
        # basename, so the same project fragments the moment a session runs from
        # a differently-cased or differently-named directory (2026-08-03: the
        # per-project directory split put 47 rows under `AIMemory` alongside 219
        # `fornixdb` + 19 `FornixDB`, splitting one thread three ways in brief).
        # Folding here rather than at every call site means no writer — CLI, MCP,
        # hooks, importers — can reintroduce a variant.
        project = _canonical_project(self, project)
        # One gist, one size. Folded here for the same reason as the project
        # label above: enforcing at each call site means the next writer — a new
        # adapter, a hook, an importer — reintroduces the wall of text. The
        # overflow moves into detail, so nothing is lost and `show` still has it.
        gist, detail = split_gist(gist, detail)
        # Resolve the embedder BEFORE inserting: first resolution runs the
        # missing-vector backfill, and with the new row already committed the
        # backfill would count it as a gap — embedding it a first time and
        # announcing a heal on every write from a fresh store handle (seen
        # live 2026-07-10: every local-model sense call printed "embedded 1").
        emb = self._resolve_embedder(embedder)
        from .budget import make_room  # lazy: avoids import cycle, free when no budget set
        make_room(self)
        # One transaction for the row, its topics, and any _in_txn companion:
        # a reminder used to be able to lose its prospective row to a crash
        # between two commits (a dud that reads as a memory but never fires),
        # and a row could land with only some of its topics.
        with self.write_txn() as conn:
            cur = conn.execute(
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
                self._apply_topic(conn, mem_id, topic)
            if _in_txn is not None:
                _in_txn(conn, mem_id)
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
        # Lineage heal (enrichment, never blocks the write): a near-identical
        # row tombstoned successor-less moments ago is this row's predecessor.
        try:
            self._adopt_orphan_tombstones(mem_id, kind)
        except Exception:
            pass
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

    @staticmethod
    def _apply_topic(conn: sqlite3.Connection, memory_id: int, topic: str) -> None:
        """The tag statements without a commit — for callers composing them
        into a larger transaction (store's insert txn) as well as tag()."""
        topic = topic.strip().lower()
        conn.execute("INSERT OR IGNORE INTO topic(name) VALUES (?)", (topic,))
        conn.execute(
            """INSERT OR IGNORE INTO memory_topic(memory_id, topic_id)
               SELECT ?, id FROM topic WHERE name = ?""",
            (memory_id, topic),
        )

    def tag(self, memory_id: int, topic: str) -> None:
        self._check_writable()
        self._apply_topic(self.conn, memory_id, topic)
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
        # write_txn: the name probe and the handoff must see one consistent
        # state — two processes superseding the same row concurrently could
        # otherwise both read the old name and both hand it off
        with self.write_txn() as conn:
            # the successor inherits the unique name handle unless it has its
            # own, so `show <name>` keeps resolving to the live version
            names = {r["id"]: r["name"] for r in conn.execute(
                "SELECT id, name FROM memory WHERE id IN (?, ?)", (old_id, new_id))}
            if names.get(old_id) and not names.get(new_id):
                conn.execute("UPDATE memory SET name = NULL WHERE id = ?", (old_id,))
                conn.execute("UPDATE memory SET name = ? WHERE id = ?",
                             (names[old_id], new_id))
            conn.execute(
                "UPDATE memory SET superseded_by = ?, superseded_time = ? WHERE id = ?",
                (new_id, now_iso(), old_id),
            )
            conn.execute(
                "INSERT OR IGNORE INTO memory_link(memory_id, related_id, relation) "
                "VALUES (?,?, 'supersedes')",
                (new_id, old_id),
            )
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

    # A rewrite stored within this window of a successor-less tombstone is
    # treated as its replacement (same near-duplicate bar as consolidation's
    # MERGE_COSINE). The forget-then-rewrite flow orders the tombstone BEFORE
    # its successor exists, so supersede() can never record the lineage — the
    # 2026-07-25 audit found #534 orphaned exactly this way, six seconds ahead
    # of its rewrite. The store repairs the link at the rewrite instead.
    ORPHAN_ADOPT_COSINE = 0.88
    ORPHAN_ADOPT_WINDOW_MIN = 60.0

    def _adopt_orphan_tombstones(self, mem_id: int, kind: str) -> list[int]:
        """Write superseded_by (+ the supersedes link) on any same-kind row
        tombstoned successor-less within the adoption window that the new row
        near-duplicates. Returns the ids adopted. Vector stores only — without
        an embedding for the new row there is no similarity bar to clear."""
        new = self.conn.execute(
            "SELECT model, vector FROM embedding WHERE memory_id = ? AND chunk = 0",
            (mem_id,)).fetchone()
        if new is None:
            return []
        cutoff = (datetime.now() - timedelta(minutes=self.ORPHAN_ADOPT_WINDOW_MIN)
                  ).isoformat(timespec="seconds")
        orphans = self.conn.execute(
            """SELECT m.id, e.vector FROM memory m
               JOIN embedding e ON e.memory_id = m.id AND e.model = ? AND e.chunk = 0
               WHERE m.superseded_by IS NULL AND m.superseded_time >= ?
                 AND m.kind = ? AND m.id != ?""",
            (new["model"], cutoff, kind, mem_id)).fetchall()
        if not orphans:
            return []
        from .vectors import cosine, from_blob
        nv = from_blob(new["vector"])
        adopted: list[int] = []
        for r in orphans:
            if cosine(nv, from_blob(r["vector"])) < self.ORPHAN_ADOPT_COSINE:
                continue
            with self.write_txn() as conn:
                cur = conn.execute(
                    "UPDATE memory SET superseded_by = ? "
                    "WHERE id = ? AND superseded_by IS NULL", (mem_id, r["id"]))
                if cur.rowcount:
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_link(memory_id, related_id, relation) "
                        "VALUES (?,?, 'supersedes')", (mem_id, r["id"]))
                    adopted.append(r["id"])
        return adopted

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
        # The overfetch is a CONSTANT, not limit-scaled: a row's score must not
        # depend on how many rows the caller asked for. When keep was
        # max(limit*5, 25), a keyword row ranked 26th by bm25 kept its keyword
        # relevance at limit=15 but lost it at limit=5 — the same query returned
        # a different ORDER at different limits (seen live 2026-07-16: eval #17
        # rank 4 at k=5 vs rank 2 at limit=15).
        if self._setting_off("associative_recall"):
            # L0/L1 boundary (ROADMAP: L0 = "exact lookups, no ranking"). When
            # associative recall is disabled the store behaves as a plain keyed
            # get/put: exact name lookup only, no FTS/vector ranking. (Keyed
            # access via show_memory by id/name still works regardless.)
            rows = self._recall_exact_name(query, limit, kind, project,
                                           include_superseded, since, until)
        else:
            emb = self._resolve_embedder(embedder)
            keep = FTS_BLEND_KEEP if emb is not None else limit
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
            # Set on EVERY path — keyword-only, hybrid and exact-name alike — so
            # the field means "measured, and this is the answer" wherever a row
            # comes from. A field that exists on only some paths reads as 0 on
            # the others, and 0 is exactly the value that makes the gate abstain.
            r["shared_terms"] = shared_term_count(query, r)
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

    def _project_clause(self, project, col="m.project"):
        """(sql, params) matching EVERY spelling of `project` this store holds —
        `("m.project = ?", ["fornixdb"])` in the normal one-spelling case, an IN
        over the variants otherwise. Both forms use idx_memory_project; folding in
        SQL (LOWER(project) = ?) would not. Returns ("", []) for no filter.

        Filters go through here rather than comparing the raw label because an
        exact match silently under-reports: a store written before labels were
        canonicalized, or a read-only peer that will never be rewritten, still
        holds `FornixDB` and `AIMemory` rows that `--project fornixdb` must find."""
        if not project:
            return "", []
        from .context import project_equivalents  # lazy: import cycle via multistore
        labels = project_equivalents(self, project)
        if len(labels) <= 1:
            return f"{col} = ?", list(labels) or [project]
        return f"{col} IN ({','.join('?' * len(labels))})", labels

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
            pc, pcp = self._project_clause(project)
            where.append(pc)
            params += pcp
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
            # double-checked: first vector use from N threads at once must
            # load the model and run the missing-vector backfill ONCE, not N
            # times (observed live: six threads, six model loads, six
            # identical backfills of the same 56 rows)
            with self._embedder_lock:
                if not hasattr(self, "_auto_embedder"):
                    emb = self._auto_resolve_embedder()
                    self._auto_embedder = emb
                    if emb is not None:
                        self._maybe_backfill_vectors(emb)
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
                pc, pcp = self._project_clause(project)
                where.append(pc)
                params += pcp
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
        raw_cos: dict[int, float] = {}      # sub-floor cosines, kept for the gate
        if unscored:
            try:
                from .vectors import cosines_for
                for mid, cos in cosines_for(self, emb, query, unscored).items():
                    raw_cos[mid] = cos
                    if cos >= VECTOR_MIN_COS:
                        neighbors[mid] = cos
            except Exception:
                pass  # exact-cosine top-up is an upgrade, never a requirement

        now = datetime.now()
        out = []
        for mid, row in by_id.items():
            cos = max(neighbors.get(mid, 0.0), 0.0)
            row["vec_cos"] = cos            # exposed for the abstention gate
            row["raw_cos"] = max(cos, raw_cos.get(mid, 0.0))  # unfloored, for the gate
            row["kw_rel"] = float(row.get("relevance") or 0.0)  # pre-blend FTS
            row["relevance"] = (row["kw_rel"] + VECTOR_WEIGHT * cos)
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
            pc, pcp = self._project_clause(project)
            where.append(pc)
            params += pcp
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
        # follow, so their bm25 relevance survives into _blend_vectors). The
        # overfetch depth is constant so the re-ranked order is limit-stable.
        params.append(max(keep, FTS_BLEND_KEEP))
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
            pc, pcp = self._project_clause(project)
            where.append(pc)
            params += pcp
        # When a window holds more than `limit` rows, keep the MOST RECENT ones
        # (a busy day must never drop what just happened — the freshly-recorded
        # diary entry is exactly what "what happened today" wants), but present
        # them oldest-first for natural reading. Windows within `limit` are
        # unaffected.
        sql = f"""SELECT * FROM (
                      SELECT m.* FROM memory m WHERE {' AND '.join(where)}
                      ORDER BY m.event_time DESC LIMIT ?
                  ) ORDER BY event_time ASC"""
        params.append(limit)
        rows = [dict(r) for r in self.conn.execute(sql, params)]
        # a timeline sweep LISTS rows, it doesn't engage with them — an
        # impression, not a use (engagement = show / mark_helpful / referenced)
        self.record_surfaced([r["id"] for r in rows])
        return rows

    def show(self, ref: int | str, reinforce: bool = True) -> dict | None:
        """Fetch a single memory (by id or name) with full detail, topics, and
        links. Detail recall reinforces salience — like human memory."""
        row = self._resolve_row(ref)
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

    def lineage(self, ref: int | str, depth: int = 25) -> list[dict]:
        """The supersede chain a memory belongs to, newest edition first.

        Hand it ANY edition — the live tip or a tombstoned ancestor — and it
        walks forward to the current row first, so an old id and a new one
        return the same chain. Gist-only and capped by design: a project's
        arc is already recorded as a run of superseded status rows, and
        reading it as a listing costs a fraction of reconstructing it from
        detail. Detail stays one `show` away, and cold-tier detail is
        deliberately not restored here.
        """
        row = self._resolve_row(ref)
        if row is None:
            return []
        # forward to the live tip FIRST, and deliberately not bounded by
        # `depth` — depth caps how many editions are returned, and letting it
        # stop this leg would report a tombstoned ancestor as the current row.
        # The `seen` set is the termination guard (a cycle revisits an id).
        cur, seen = dict(row), set()
        while cur.get("superseded_by") and cur["id"] not in seen:
            seen.add(cur["id"])
            nxt = self.conn.execute(
                "SELECT * FROM memory WHERE id = ?", (cur["superseded_by"],)).fetchone()
            if nxt is None:
                break
            cur = dict(nxt)
        chain = []
        for mid, siblings in self._mainline(cur["id"], depth):
            row = self.conn.execute(
                "SELECT * FROM memory WHERE id = ?", (mid,)).fetchone()
            if row is None:
                break
            m = dict(row)
            m["merged_siblings"] = siblings
            chain.append(m)
        # a lineage walk LISTS editions the way `timeline` does — an
        # impression, not engagement with any one of them
        self.record_surfaced([m["id"] for m in chain])
        return chain

    def _mainline(self, tip_id: int, depth: int) -> list[tuple[int, int]]:
        """Walk one supersede chain back from a tip: [(id, merged_siblings)].

        Shared by `lineage` and `status_tips` on purpose — a chain length
        quoted in the brief has to be the same number the walk will show, or
        the brief is advertising an arc that does not exist. More than one row
        can point at the same successor (a merge); the most recent is the
        mainline and the rest are counted, never silently dropped.
        """
        out, seen = [], {tip_id}
        cur = tip_id
        while len(out) < depth:
            prevs = [r["id"] for r in self.conn.execute(
                "SELECT id FROM memory WHERE superseded_by = ? "
                "ORDER BY event_time DESC", (cur,)) if r["id"] not in seen]
            out.append((cur, max(0, len(prevs) - 1)))
            if not prevs:
                break
            cur = prevs[0]
            seen.add(cur)
        return out

    def status_tips(self, *, project: str | None = None,
                    limit: int = 5) -> list[dict]:
        """Where each live thread currently stands — one row per project.

        A memory that supersedes another IS a status update, so the live tip
        of a supersede chain is that thread's current state by construction.
        This needs no naming convention and no new column; it just reads a
        structure the store already has. Ranked by how recently the thread
        moved, because a pickup cares about what is warm, and carrying the
        chain depth so the reader knows an arc is there to walk.
        """
        _pc, pp = self._project_clause(project)
        pw = f"AND {_pc}" if _pc else ""
        rows = [dict(r) for r in self.conn.execute(
            f"""WITH tip AS (
                    SELECT m.* FROM memory m
                    WHERE m.superseded_time IS NULL AND m.kind != 'episodic'
                      AND EXISTS (SELECT 1 FROM memory p WHERE p.superseded_by = m.id)
                      {pw}
                )
                SELECT * FROM tip t
                WHERE t.event_time = (
                    SELECT max(t2.event_time) FROM tip t2
                    -- Case-folded so one project spelled two ways is ONE thread.
                    -- `IS` (not `=`) keeps the unlabelled rows grouping together.
                    WHERE t2.project IS t.project
                       OR LOWER(TRIM(t2.project)) = LOWER(TRIM(t.project)))
                ORDER BY t.event_time DESC LIMIT ?""",
            [*pp, limit])]
        for r in rows:
            chain = self._mainline(r["id"], STATUS_TIP_MAX_EDITIONS)
            r["editions"] = len(chain)
            r["editions_capped"] = len(chain) == STATUS_TIP_MAX_EDITIONS
        self.record_surfaced([r["id"] for r in rows])
        return rows

    def _resolve_row(self, ref: int | str):
        """A memory row by id or by name — the lookup `show` and `lineage` share."""
        if isinstance(ref, str) and not ref.isdigit():
            return self.conn.execute(
                "SELECT * FROM memory WHERE name = ?", (ref,)).fetchone()
        return self.conn.execute(
            "SELECT * FROM memory WHERE id = ?", (int(ref),)).fetchone()

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
            from .db import append_log_line
            append_log_line(path, "\n".join(
                json.dumps({"ts": ts, "event": event, **r}, ensure_ascii=False)
                for r in records))
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
        # write_txn: the already-suppressed probe and the stamp are one unit,
        # so two concurrent scans can't both classify a row as new (idempotence
        # would survive a double-stamp, but the audit log would double-log)
        with self.write_txn() as conn:
            newly = []
            for i, pr in stats.items():
                row = conn.execute(
                    "SELECT proactive_suppressed_at FROM memory WHERE id = ?", (int(i),)
                ).fetchone()
                if row is None or row["proactive_suppressed_at"] is not None:
                    continue                 # gone, or already suppressed
                newly.append((int(i), int(pr[0]), int(pr[1])))
            if not newly:
                return 0
            ts = at or now_iso()
            conn.executemany(
                """UPDATE memory SET proactive_suppressed_at = ?,
                                     suppressed_pushed = ?, suppressed_referenced = ?
                   WHERE id = ?""",
                [(ts, p, r, i) for (i, p, r) in newly])
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
        stores skip silently.

        Records `redeemed_pushes` so the redemption STICKS: see the column note
        in db.py. A redemption that the next scan reverses is not a redemption."""
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
        # Keep the push count the redemption overruled (v14). A scan re-derives
        # from the same transcripts, so without this the row re-qualifies on the
        # identical evidence and the redemption is undone before it can mean
        # anything. Suppression now has to be re-earned from pushes that happen
        # AFTER this moment.
        self.conn.execute(
            f"""UPDATE memory SET proactive_suppressed_at = NULL,
                                  redeemed_pushes = COALESCE(suppressed_pushed,
                                                             surfaced_count, 0),
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
        project = _canonical_project(self, project)   # same fold as memory rows
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

    def project_labels(self) -> list[dict]:
        """Every project spelling in the store with its row count and the label it
        canonicalizes to — the `projects` view. `canonical` == `label` means the
        row is already the settled spelling for that project."""
        from .context import project_canon_map
        cmap = project_canon_map(self)
        rows = []
        for label, n in self._all_project_labels():
            rows.append({"label": label, "memories": n,
                         "canonical": cmap.get(label.strip().lower(), label.strip())})
        return rows

    def _all_project_labels(self) -> list[tuple[str, int]]:
        """Distinct project spellings across BOTH memory and session, with memory
        row counts (0 for a label only sessions use). Ordered most-used first."""
        counts: dict[str, int] = {}
        for p, n in self.conn.execute(
                "SELECT project, COUNT(*) FROM memory "
                "WHERE project IS NOT NULL AND project <> '' GROUP BY project"):
            counts[p] = n
        for (p,) in self.conn.execute(
                "SELECT DISTINCT project FROM session "
                "WHERE project IS NOT NULL AND project <> ''"):
            counts.setdefault(p, 0)
        return sorted(counts.items(), key=lambda r: (-r[1], r[0]))

    def normalize_projects(self, *, apply: bool = False) -> dict:
        """Rewrite every project label to its canonical spelling, in memory AND
        session. Propose-not-dispose: DRY RUN unless `apply` — the caller sees the
        exact rewrites first, because merging two labels is a judgement about what
        is one project, and only case-folding is safe to assume.

        Idempotent (a second run proposes nothing) and reversible in principle,
        but it overwrites the only copy of the old label — take a backup first."""
        from .context import project_canon_map
        cmap = project_canon_map(self)
        changes = []
        for label, n_mem in self._all_project_labels():
            canon = cmap.get(label.strip().lower(), label.strip())
            if canon == label:
                continue
            n_ses = self.conn.execute(
                "SELECT COUNT(*) FROM session WHERE project = ?", (label,)).fetchone()[0]
            changes.append({"from": label, "to": canon,
                            "memories": n_mem, "sessions": n_ses})
        if apply and changes:
            self._check_writable()
            with self.write_txn() as conn:
                for c in changes:
                    conn.execute("UPDATE memory SET project = ? WHERE project = ?",
                                 (c["to"], c["from"]))
                    conn.execute("UPDATE session SET project = ? WHERE project = ?",
                                 (c["to"], c["from"]))
        return {"changes": changes, "applied": bool(apply and changes),
                "memories": sum(c["memories"] for c in changes),
                "sessions": sum(c["sessions"] for c in changes)}

    def brief(self, *, project: str | None = None, days: int = 7,
              recent_limit: int = 8, salient_limit: int = 10,
              useful_limit: int = 5) -> dict:
        """Session-start context brief: recent activity + most salient
        standing knowledge. Gist-only and capped — this is the cheap recall
        that opens every session; detail is always a `show` away."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        _pc, pp = self._project_clause(project)
        pw = f"AND {_pc}" if _pc else ""
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
        # where live threads stand. This is a RECENCY-of-thread axis, not the
        # importance axis `salient` ranks on, which is why a resume row at
        # default salience never reached that list: measured 2026-08-02, the
        # pointer to the current state of the active project did not make the
        # top-40 salience pool, so a pickup fell back to re-reading a 42k-token
        # narrative file that the chain already summarises for ~800.
        return {"since": since[:10], "recent": recent, "salient": salient,
                "useful": useful,
                "threads": self.status_tips(project=project)}

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
