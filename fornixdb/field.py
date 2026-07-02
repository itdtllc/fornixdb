"""L5 parallel multi-domain activation — the field.

L4 gave memory a beat inside the thought (cadence.py: WHEN to re-activate).
L5 makes each beat WIDE: instead of one recall, a field of domain-scoped
recalls fires on the same evolving thought — standing knowledge, recent
events, the deep past, learned guidance, reference pointers, the active
project's context, and the associative neighborhood of what is already lit —
and a settling integrator resolves the field into one small directed block.
Not one recall, but a field of simultaneous local recalls resolving into a
direction (ROADMAP L5).

Wide input, narrow output: the field's internal width costs milliseconds and
zero context tokens (the thought is embedded ONCE and shared across domains);
only the settled pattern ever reaches the model, at the same block budget as
an L4 pulse.

Settling is CORROBORATION, not summarization — an algorithmic floor, no model
required (no rung is gated on a local model). Rows that several domains
return, or that link/topic-connect across domains, cluster; the cluster with
the highest cross-domain mass is the pattern the field settles into. When the
field finds no corroboration it degrades gracefully to L4 behavior (best
single hits, no direction line), and when nothing clears the floors it stays
silent. It must never fabricate a pattern.

The neighborhood domain is corroboration-only BY CONSTRUCTION: its rows are
query-free link-spread (no cosine, so no floor to clear honestly), so they can
reach the block only by clustering with a row a query domain vouched for.
Activation spread alone is never enough; it has to resonate with the current
thought through another domain.

PORTABLE BY DESIGN (#276/#332): this module knows nothing about any AI host.
`cadence.pulse()` routes through it when `parallel_recall` is on (P2); each
host keeps its existing L4 wiring. Memory stays substrate, not actor: the
direction line DESCRIBES the settled structure; it never instructs.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dcfield
from datetime import datetime, timedelta

from .core import RHYTHMIC_RECALL_COS, STRUCTURAL_TOPICS, MemoryStore
from .multistore import get_config
from .proactive import (HEADER, _is_low_information, _log_floor_decision,
                        row_line)

DEFAULT_K = 3               # per-domain top-k gathered into the field
DEFAULT_RECENT_DAYS = 14    # the recent/deep-past episodic split
DEFAULT_MAX_CHARS = 400     # block budget — an L5 beat costs what an L4 beat costs
DEFAULT_LIMIT = 3           # settled gists emitted (the narrow output)

# settling bonuses — corroboration raises SCORE, floors stay honest (§6 of the
# design: a corroborated row clears more easily because agreement is evidence,
# not because the bar dropped)
DOMAIN_BONUS = 0.15         # per additional domain returning the same row
LINK_BONUS = 0.10           # per link edge to a row from a different domain
TOPIC_BONUS = 0.05          # per shared non-structural topic across domains
LINK_BONUS_CAP = 0.30
TOPIC_BONUS_CAP = 0.15


@dataclass(frozen=True)
class Domain:
    """One scoped view over the store. Every query domain is composition over
    existing `recall()` scopes — the field adds no query machinery."""
    id: str
    label: str                     # the word the direction line uses
    kind: str | None = None
    recent: bool | None = None     # True = since-cut, False = until-cut (episodic)
    project_scoped: bool = False   # scope to the active project (any kind)
    neighborhood: bool = False     # query-free link-spread (corroboration-only)


# Single source of truth, ordered as the design doc §3 table. All seven on by
# default (owner decision 1); a deployment trims via `config parallel_domains`.
DOMAINS: tuple[Domain, ...] = (
    Domain("knowledge", "knowledge", kind="semantic"),
    Domain("recent", "recent", kind="episodic", recent=True),
    Domain("deep", "deep-past", kind="episodic", recent=False),
    Domain("guidance", "guidance", kind="feedback"),
    Domain("reference", "reference", kind="reference"),
    Domain("context", "context", project_scoped=True),
    Domain("neighborhood", "neighborhood", neighborhood=True),
)
_BY_ID = {d.id: d for d in DOMAINS}


def configured_domains(store: MemoryStore) -> list[Domain]:
    raw = (get_config(store, "parallel_domains", "") or "").strip()
    if not raw:
        return list(DOMAINS)
    ids = [t.strip().lower() for t in raw.split(",") if t.strip()]
    return [_BY_ID[i] for i in ids if i in _BY_ID] or list(DOMAINS)


class _SharedEmbedder:
    """One embed, N domain queries: caches per-text vectors so the field embeds
    the thought once no matter how many domains fire. Transparent wrapper —
    satisfies the Embedder protocol, delegates anything uncached."""

    def __init__(self, inner):
        self._inner = inner
        self.name = inner.name
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for t, v in zip(missing, self._inner.embed(missing)):
                self._cache[t] = v
        return [self._cache[t] for t in texts]


@dataclass
class FieldResult:
    """The raw field: what each domain returned, before settling."""
    by_domain: dict[str, list[dict]] = _dcfield(default_factory=dict)
    domains_of: dict[int, set[str]] = _dcfield(default_factory=dict)
    rows: dict[int, dict] = _dcfield(default_factory=dict)

    def _add(self, domain_id: str, row: dict) -> None:
        rid = row["id"]
        self.by_domain.setdefault(domain_id, []).append(row)
        self.domains_of.setdefault(rid, set()).add(domain_id)
        # keep the best-scored copy (domains overfetch the same store)
        best = self.rows.get(rid)
        if best is None or float(row.get("score") or 0) > float(best.get("score") or 0):
            self.rows[rid] = row


def _neighborhood(store: MemoryStore, lit_ids: set[int],
                  exclude: set[int]) -> list[dict]:
    """1-hop link spread from what is already lit — the already-active memories
    of this episode plus this beat's own query returns. Query-free: no score,
    no floor; these rows are corroboration-only candidates."""
    if not lit_ids:
        return []
    marks = ",".join("?" for _ in lit_ids)
    ids = list(lit_ids)
    sql = f"""
        SELECT m2.* FROM memory_link ml JOIN memory m2 ON m2.id = ml.related_id
        WHERE ml.memory_id IN ({marks}) AND m2.superseded_time IS NULL
        UNION
        SELECT m1.* FROM memory_link ml JOIN memory m1 ON m1.id = ml.memory_id
        WHERE ml.related_id IN ({marks}) AND m1.superseded_time IS NULL"""
    rows = [dict(r) for r in store.conn.execute(sql, ids + ids)]
    out = []
    for r in rows:
        if r["id"] in lit_ids or r["id"] in exclude or _is_low_information(r):
            continue
        r["score"] = 0.0    # structural candidate: earns its place via settling
        out.append(r)
    return out


def run_field(store: MemoryStore, thought: str, *,
              episode_ids: set[int] | frozenset = frozenset(),
              exclude_ids=(), active_project: str | None = None,
              k: int | None = None, floor: float | None = None,
              recent_days: int = DEFAULT_RECENT_DAYS) -> FieldResult:
    """Fire every configured domain on `thought` and gather the field.

    Each query domain's rows pass the SAME honest gate as an L4 pulse — the
    per-memory effective floor on the vector cosine (keyword-anchor trust in a
    vectorless store) — so the field inherits the usefulness/scoping dials for
    free. The neighborhood domain is gathered floor-free but corroboration-only
    (see module docstring)."""
    result = FieldResult()
    if not (thought or "").strip():
        return result
    if k is None:
        k = int(get_config(store, "parallel_domain_k", str(DEFAULT_K)))
    if floor is None:
        floor = float(get_config(store, "rhythmic_recall_floor",
                                 str(RHYTHMIC_RECALL_COS)))
    exclude = set(exclude_ids)
    emb = store._resolve_embedder(None)
    shared = _SharedEmbedder(emb) if emb is not None else False
    has_vectors = emb is not None
    cut = (datetime.now() - timedelta(days=recent_days)).isoformat(timespec="seconds")

    domains = configured_domains(store)
    aliases: set[str] = set()
    if active_project:
        from . import context
        aliases = context.aliases_for(store, active_project)

    for d in domains:
        if d.neighborhood:
            continue                       # gathered last, from what lit
        if d.project_scoped and not active_project:
            continue                       # no room to stand in — skip
        kwargs: dict = {"limit": k * 3, "count_recall": False, "embedder": shared}
        if d.kind:
            kwargs["kind"] = d.kind
        if d.recent is True:
            kwargs["since"] = cut
        elif d.recent is False:
            kwargs["until"] = cut
        if d.project_scoped:
            kwargs["project"] = active_project
        kept = 0
        for r in store.recall(thought, **kwargs):
            if kept >= k:
                break
            if r["id"] in exclude or _is_low_information(r):
                continue
            cos = r.get("vec_cos")
            if has_vectors:
                if cos is None:            # couldn't clear the vector noise
                    _log_floor_decision(store, "L5", thought, r, cos, floor,
                                        floor, "weak_vector_skip")
                    continue               # floor — semantically unrelated
                r.setdefault("topics", store.topics_for([r["id"]]).get(r["id"], []))
                eff = store.effective_floor(r, floor, active_project=active_project,
                                            aliases=aliases)
                if float(cos) < eff:
                    _log_floor_decision(store, "L5", thought, r, cos, eff,
                                        floor, "below_floor")
                    continue
                # clearing the floor only ADMITS a row to the field; whether it
                # SURFACES is settling's call — the outcome is logged by the
                # caller that emits the block, keeping "surfaced" = injected.
                r["_eff_floor"] = eff
            result._add(d.id, r)
            kept += 1

    if any(d.neighborhood for d in domains):
        lit = set(episode_ids) | set(result.rows)
        for r in _neighborhood(store, lit, exclude):
            result._add("neighborhood", r)
    return result


# ------------------------------------------------------------------ settling

@dataclass
class Settled:
    """What the field resolved to. `settled` False = graceful L4 fallback
    (best single hits, no direction — the field found no pattern)."""
    rows: list[dict] = _dcfield(default_factory=list)
    direction: str | None = None
    dissent: dict | None = None
    settled: bool = False
    scores: dict[int, float] = _dcfield(default_factory=dict)
    clusters: list[set[int]] = _dcfield(default_factory=list)
    # beat telemetry (field_log): the winning cluster, what glued it, and the
    # dissent SHADOW — the minority-report row computed on every settled beat
    # even while the dissent line is config-off, so the on/off decision can
    # later be made from evidence instead of a guess.
    winner: set[int] = _dcfield(default_factory=set)
    link_glue: int = 0
    topic_glue: int = 0
    dissent_shadow: dict | None = None


class _UnionFind:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _edges(store: MemoryStore, ids: set[int]) -> tuple[set, set]:
    """(link_edges, topic_edges) among the field's rows. Topic edges use only
    non-structural topics — 'distilled' etc. must not glue unrelated rows."""
    if len(ids) < 2:
        return set(), set()
    marks = ",".join("?" for _ in ids)
    lst = list(ids)
    link_edges = {tuple(sorted((r["memory_id"], r["related_id"])))
                  for r in store.conn.execute(
                      f"SELECT memory_id, related_id FROM memory_link "
                      f"WHERE memory_id IN ({marks}) AND related_id IN ({marks})",
                      lst + lst)}
    by_topic: dict[str, list[int]] = {}
    for rid, topics in store.topics_for(lst).items():
        for t in topics:
            if t.strip().lower() not in STRUCTURAL_TOPICS:
                by_topic.setdefault(t.strip().lower(), []).append(rid)
    topic_edges = set()
    for members in by_topic.values():
        members = sorted(set(members))
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                e = (a, b)
                if e not in link_edges:
                    topic_edges.add(e)
    return link_edges, topic_edges


def settle(store: MemoryStore, fr: FieldResult, *,
           limit: int | None = None, dissent: bool | None = None) -> Settled:
    """Corroboration clustering — the attractor the field falls into.

    Per row: best relevance score + a bonus per additional agreeing domain +
    bounded bonuses per link/topic edge to rows other domains returned. Rows
    connected by links or shared topics form clusters; cluster mass =
    (sum of member scores) x (distinct QUERY domains represented), so a
    cluster the field lit from several directions beats a single loud hit,
    and a pure-neighborhood cluster has mass 0 — structurally unable to win."""
    out = Settled()
    if not fr.rows:
        return out
    if limit is None:
        limit = int(get_config(store, "parallel_limit", str(DEFAULT_LIMIT)))
    if dissent is None:
        dissent = (get_config(store, "parallel_dissent", "off")
                   not in ("off", "0", "false"))

    ids = set(fr.rows)
    link_edges, topic_edges = _edges(store, ids)

    def other_domain(a: int, b: int) -> bool:
        return bool(fr.domains_of.get(a, set()) - fr.domains_of.get(b, set())) \
            or bool(fr.domains_of.get(b, set()) - fr.domains_of.get(a, set()))

    for rid, row in fr.rows.items():
        query_domains = fr.domains_of[rid] - {"neighborhood"}
        score = float(row.get("score") or 0.0)
        score += DOMAIN_BONUS * max(0, len(query_domains) - 1)
        lb = sum(LINK_BONUS for e in link_edges if rid in e and other_domain(*e))
        tb = sum(TOPIC_BONUS for e in topic_edges if rid in e and other_domain(*e))
        score += min(lb, LINK_BONUS_CAP) + min(tb, TOPIC_BONUS_CAP)
        out.scores[rid] = score

    uf = _UnionFind(ids)
    for a, b in link_edges | topic_edges:
        uf.union(a, b)
    clusters: dict[int, set[int]] = {}
    for i in ids:
        clusters.setdefault(uf.find(i), set()).add(i)
    out.clusters = sorted(clusters.values(), key=len, reverse=True)

    def cluster_mass(c: set[int]) -> float:
        qd = set()
        for i in c:
            qd |= fr.domains_of[i] - {"neighborhood"}
        return sum(out.scores[i] for i in c) * len(qd)

    def query_domains_in(c: set[int]) -> set[str]:
        qd = set()
        for i in c:
            qd |= fr.domains_of[i] - {"neighborhood"}
        return qd

    winner = max(out.clusters, key=cluster_mass, default=set())
    # corroborated = the winner was lit from at least two directions. The
    # neighborhood counts as a direction here — a link-attached neighbor IS
    # structure resonating with a query domain's hit — but it can never be the
    # ONLY direction, because a pure-neighborhood cluster has mass 0.
    all_domains = set()
    for i in winner:
        all_domains |= fr.domains_of[i]
    corroborated = (len(all_domains) >= 2
                    or any(len(fr.domains_of[i] - {"neighborhood"}) >= 2
                           for i in winner))
    if winner and cluster_mass(winner) > 0 and corroborated:
        out.settled = True
        out.winner = set(winner)
        out.link_glue = sum(1 for a, b in link_edges
                            if a in winner and b in winner)
        out.topic_glue = sum(1 for a, b in topic_edges
                             if a in winner and b in winner)
        ranked = sorted(winner, key=lambda i: out.scores[i], reverse=True)
        out.rows = [fr.rows[i] for i in ranked[:limit]]
        out.direction = _direction(store, fr, out, winner)
        rest = [i for i in ids - winner
                if fr.domains_of[i] - {"neighborhood"}]
        if rest:
            out.dissent_shadow = fr.rows[max(rest, key=lambda i: out.scores[i])]
            if dissent:
                out.dissent = out.dissent_shadow
    else:
        # no pattern — degrade to L4 behavior: best single query-domain hits
        ranked = sorted((i for i in ids if fr.domains_of[i] - {"neighborhood"}),
                        key=lambda i: out.scores[i], reverse=True)
        out.rows = [fr.rows[i] for i in ranked[:limit]]
    return out


def _direction(store: MemoryStore, fr: FieldResult, st: Settled,
               winner: set[int]) -> str:
    """One DESCRIPTIVE line: what the field settled on, over what span, lit
    from which directions. Never imperative — memory surfaces structure;
    judgment stays with the reasoning model."""
    tags: dict[str, int] = {}
    for rid, topics in store.topics_for(list(winner)).items():
        for t in topics:
            t = t.strip().lower()
            if t not in STRUCTURAL_TOPICS:
                tags[t] = tags.get(t, 0) + 1
    for i in winner:
        p = (fr.rows[i].get("project") or "").strip().lower()
        if p:
            tags[p] = tags.get(p, 0) + 1
    subject = max(tags, key=tags.get) if tags else None
    times = sorted((fr.rows[i].get("event_time") or "")[:10]
                   for i in winner if fr.rows[i].get("event_time"))
    span = ""
    if times:
        span = times[0] if times[0] == times[-1] else f"{times[0]}→{times[-1]}"
    lit = sorted({_BY_ID[d].label for i in winner
                  for d in fr.domains_of[i] if d in _BY_ID})
    parts = [p for p in (subject, span, "+".join(lit)) if p]
    return "settled: " + " · ".join(parts)


# ------------------------------------------------------------------ beat log

def field_log_path_for(store: MemoryStore) -> str | None:
    """The field log's location for THIS store (field_log.jsonl beside the db),
    regardless of whether logging is currently on. None for in-memory stores."""
    from .proactive import floor_log_path_for
    p = floor_log_path_for(store)
    if not p:
        return None
    from pathlib import Path as _P
    return str(_P(p).with_name("field_log.jsonl"))


def _log_field_beat(store: MemoryStore, thought: str, fr: FieldResult,
                    st: Settled, emitted: list[int], ms: float) -> None:
    """One JSONL record per field beat — the per-BEAT telemetry the per-ROW
    floor log can't carry: settle-vs-degrade, which domains lit the winner,
    what glued it (links vs topics — the topic-noise evidence), the dissent
    shadow, and wall-time. Rides the SAME `floor_log` switch (one dial, all
    diagnostics) and the same beside-the-store, never-in-the-db rules."""
    if get_config(store, "floor_log", "off") in ("off", "0", "false"):
        return
    path = field_log_path_for(store)
    if not path:
        return
    import json
    from datetime import datetime as _dt
    winner_domains = sorted({d for i in st.winner
                             for d in fr.domains_of.get(i, ())})
    rec = {
        "ts": _dt.now().astimezone().isoformat(timespec="seconds"),
        "query": (thought or "")[:80],
        "settled": st.settled,
        "direction": st.direction,
        "domains_lit": {d: len(rows) for d, rows in fr.by_domain.items() if rows},
        "winner": sorted(st.winner),
        "winner_domains": winner_domains,
        "link_glue": st.link_glue,
        "topic_glue": st.topic_glue,
        "neighborhood_emitted": any(
            fr.domains_of.get(i) == {"neighborhood"} for i in emitted),
        "emitted": emitted,
        "dissent_shadow": (st.dissent_shadow or {}).get("id"),
        "dissent_emitted": st.dissent is not None,
        "ms": round(ms, 1),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ------------------------------------------------------------------ block

def field_block(st: Settled, max_chars: int) -> str | None:
    """The narrow output: standard provenance header, the direction line when
    the field settled, the winning gists, and (config-gated) one tension line —
    the field's minority report. Same whole-line budget trim as L3/L4."""
    if not st.rows:
        return None
    lines = [HEADER]
    if st.direction:
        lines.append(st.direction)
    lines += [row_line(m) for m in st.rows]
    if st.dissent is not None:
        lines.append("tension: " + row_line(st.dissent))
    while len(lines) > 1 and len("\n".join(lines)) > max_chars:
        lines.pop()
    # a header-plus-direction husk with no memory lines is not a block
    if all(l is st.direction or l == HEADER or l.startswith("tension:")
           for l in lines):
        return None
    return "\n".join(lines)


def field_recall(store: MemoryStore, thought: str, *,
                 episode_ids: set[int] | frozenset = frozenset(),
                 exclude_ids=(), active_project: str | None = None,
                 max_chars: int | None = None) -> tuple[str | None, list[int]]:
    """One L5 beat, end to end: field → settle → block. Returns (block, ids
    actually emitted) so the caller can dedup/count impressions exactly as it
    does for an L4 pulse. None block = nothing worth interrupting for."""
    if max_chars is None:
        max_chars = int(get_config(store, "parallel_block_max_chars",
                                   str(DEFAULT_MAX_CHARS)))
    from time import perf_counter
    t0 = perf_counter()
    fr = run_field(store, thought, episode_ids=episode_ids,
                   exclude_ids=exclude_ids, active_project=active_project)
    st = settle(store, fr)
    ms = (perf_counter() - t0) * 1000.0
    block = field_block(st, max_chars)
    ids: list[int] = []
    if block:
        ids = [r["id"] for r in st.rows]
        if st.dissent is not None and "tension:" in block:
            ids.append(st.dissent["id"])
    # floor-log the field's OUTCOMES, preserving the L3/L4 meaning of
    # "surfaced" = actually injected: a floor-clearing row that lost the
    # settling logs the L5-specific "cleared_not_settled" instead.
    base = float(get_config(store, "rhythmic_recall_floor",
                            str(RHYTHMIC_RECALL_COS)))
    emitted = set(ids)
    for rid, r in fr.rows.items():
        if "_eff_floor" not in r and r.get("vec_cos") is None:
            # keyword-anchor / neighborhood candidates faced no floor test —
            # they are floor-log material only when they actually surface
            if rid in emitted:
                _log_floor_decision(store, "L5", thought, r, None, base, base,
                                    "surfaced")
            continue
        eff = r.get("_eff_floor", base)
        _log_floor_decision(store, "L5", thought, r, r.get("vec_cos"), eff, base,
                            "surfaced" if rid in emitted else "cleared_not_settled")
    _log_field_beat(store, thought, fr, st, ids, ms)
    if not block:
        return None, []
    return block, ids


# ------------------------------------------------------------------ debug view

def format_field_debug(store: MemoryStore, thought: str, *,
                       active_project: str | None = None,
                       episode_ids: set[int] | frozenset = frozenset()) -> str:
    """The design's REPL (`fornixdb field "<thought>"`): every domain's returns,
    corroboration scores, clusters, and the final block — the instrument the
    verification doc drives."""
    fr = run_field(store, thought, episode_ids=episode_ids,
                   active_project=active_project)
    st = settle(store, fr)
    lines = [f"field: {thought!r}"
             + (f"  [project: {active_project}]" if active_project else "")]
    for d in configured_domains(store):
        rows = fr.by_domain.get(d.id, [])
        lines.append(f"  {d.id:<13} {len(rows)} row(s)")
        for r in rows:
            cos = r.get("vec_cos")
            lines.append(f"    #{r['id']:<5} score={float(r.get('score') or 0):.3f}"
                         + (f" cos={cos:.3f}" if cos is not None else "")
                         + f"  {(r.get('gist') or '')[:60]}")
    if st.scores:
        lines.append("  corroboration (settled scores):")
        for rid in sorted(st.scores, key=st.scores.get, reverse=True)[:10]:
            doms = ",".join(sorted(fr.domains_of[rid]))
            lines.append(f"    #{rid:<5} {st.scores[rid]:.3f}  [{doms}]")
    lines.append(f"  clusters: {[sorted(c) for c in st.clusters]}")
    lines.append(f"  settled: {st.settled}")
    block = field_block(st, int(get_config(store, "parallel_block_max_chars",
                                           str(DEFAULT_MAX_CHARS))))
    lines.append("  block:")
    lines += ["    " + l for l in (block or "(none)").splitlines()]
    return "\n".join(lines)
