#!/usr/bin/env python3
"""Phase 0 premise probe for Reasoned Dreaming — does this store's content
actually yield enough TRUE, USEFUL reasoned pairs to justify the machinery?

Design: Design/ReasonedDreaming_Feature_Design.md §2. This is a probe, not a
feature: it writes NOTHING to any store and adds no code under fornixdb/. The
premise being tested is NOT "can an LLM classify relations" (it can) — it is
whether a real, lived-in store holds enough reasoned structure to be worth
mining. Two features have been built and reverted on refuted premises; the
pre-registered kill rule below exists so this one cannot become the third.

    PRE-REGISTERED DECISION RULE (fixed before the first run):
      build iff the best model reaches
        >= 50% TRUE-and-useful on causal + principle proposals   (new capability)
        AND
        >= 80% correct triage on the contradiction stratum       (workload cut)
      Below both -> record the negative in Design/FornixDB_Landscape.md
      next to the dissent trial, and STOP.

Three strata are sampled per store (~20 pairs each, §2):
  A  contradiction triage, AGAINST HUMAN GROUND TRUTH (see below)
  B  time-adjacent episodic pairs in one project — where causality lives, and
     invisible to the pair scan, which excludes episodic rows outright
  C  mid-distance cross-topic pairs, cosine 0.35-0.55 — the AnalogyIncubation
     band, where shared principles hide beneath low surface similarity

DEVIATION FROM §2, and why it is a strengthening rather than a shortcut: the
design sampled stratum A from the pair scan's live cosine-near candidates and
had the owner review them afterwards. Measured 2026-08-01, that pool is EMPTY
on both live stores — on the main store all 23 same-kind pairs in the
[0.70,0.88) band have already been adjudicated, and the second store has 17
non-episodic rows total. A well-groomed store does not carry a backlog of
un-triaged contradictions; grooming is what dream passes have been doing all
along. But those passes left 235 supersede links ("a human decided the newer
row replaces the older") and 171 distinct links ("a human decided these are
NOT the same thing") — real labels, accumulated over months. Stratum A is
therefore drawn from those, half from each class, and scored against the
labels with no review step and no reviewer bias.

Triage correctness was defined before the first run, and REPAIRED 2026-08-01
after that (now-archived) run exposed two label defects; the repairs landed
before any post-repair scoring run, so the bar still cannot drift against a
result:
  * ground truth REPLACE (supersede-linked): correct iff the model answers
    `contradicts`, `duplicate`, or `refines` — all three mean "the newer one
    stands in for the older". (`refines` restored per ReasonedDreaming §1's
    four triage classes — its omission made "B updates A", which is what
    every supersede pair IS, inexpressible, and the 80% bar unmeasurable.)
  * ground truth DISTINCT (distinct-linked): correct iff the model answers
    `distinct` — BUT a stored `distinct` link means "reviewed, never
    re-propose", a STORAGE decision, not "semantically unrelated" (measured
    live: at cos 0.932 the model said `duplicate` against the label and was
    right — file-mirror bridge anchors are accepted distinct precisely
    BECAUSE they look like duplicates, and hardest-cosine-first sampling
    concentrates exactly those). So the sampled DISTINCT pairs are RELABELLED
    semantically by hand BEFORE any model run: --sample writes
    _probe/relabel.md, the labels go to _probe/relabel.json blind to model
    output (REPLACE/DISTINCT/DROP; DROP = not judgeable from the gists
    alone). Without relabel.json the triage bar prints as NOT MEASURABLE.
  * `unclear` is scored as an ABSTENTION, not a success: it leaves the human
    exactly the work the feature promises to remove.
  * A `contradicts` answer on a DISTINCT-labelled pair is counted and reported
    separately as a FALSE ALARM. That is the destructive direction — it is the
    shape of the bug fixed on 2026-08-01, where a proposal to close a standing
    owner directive would have tombstoned a live rule in one command.
Triage is scored on every PARSED answer (relation in the enum), not only
quote-passing ones: the labels are objective, and the quote guard exists to
kill invented causal stories — it has no business shrinking this denominator.
It still gates strata B and C.

Each pair goes to each model under a strict, abstain-friendly JSON contract.
`unclear` is a first-class answer, and a rationale that fails to quote BOTH
gists voids the row — the hallucination guard (§6's quote-grounding), because a
fluent model will happily invent a causal story between two unrelated rows.

TWO FURTHER STRATA, added 2026-08-01, SCORED SEPARATELY:
  D  analogy      — AnalogyIncubation §3A/§3B: mid-distance cross-topic pairs
                    where the model must GIVE THE MAPPING, not tick a label
  E  proportion   — AnalogyIncubation §3C: A:B::C:D over FOUR memories, the
                    proportional form a two-memory prompt cannot express
ReasonedDreaming §8 places the analogy stratum in PHASE 3, gated behind Phase
2, so neither of these may move the Phase 0 verdict — `score` prints them under
a separate heading. Run `--no-analogy` to sample Phase 0's strata alone. The
quote-grounding guard is NOT applied to analogy claims: §2 defines the target
as low surface similarity with high relational correspondence, so a good
analogy ("both are a gate that admits only reviewed work") quotes neither
memory. Analogy claims are entity-grounded instead — every mapped role must
name something that really appears on its own side.

Usage:
    .venv/bin/python examples/reasoned_pairs_probe.py --sample      # strata -> JSON + relabel sheet
      (fill examples/_probe/relabel.json from the sheet BEFORE the first run)
    .venv/bin/python examples/reasoned_pairs_probe.py --run MODEL   # ask ONE model
    .venv/bin/python examples/reasoned_pairs_probe.py --review      # review table
    .venv/bin/python examples/reasoned_pairs_probe.py --score       # the gate

Artifacts land in examples/_probe/ (gitignored like other local-only outputs).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fornixdb.consolidate import _distinct_pairs  # noqa: E402
from fornixdb.core import MemoryStore  # noqa: E402
from fornixdb.vectors import cosine, from_blob  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "_probe"

STORES = {
    "fornix": Path.home() / "dev/ITDT/AIMemory/store/fornix.db",
    "engram": Path.home() / "dev/ITDT/AI/memories/engram.db",
}

# The three residents, base models — NOT the persona variants, which
# carry a personality system prompt that would bias a classification task.
MODELS = ["qwen2.5:72b-instruct-q4_K_M", "qwen3.5:122b-a10b-q4_K_M", "gpt-oss:120b"]

OLLAMA = "http://localhost:11434/api/chat"

PER_STRATUM = 20
TRIAGE_N = 30                  # the objectively-scored stratum: 15 labelled
                               # REPLACE + the 15 hardest labelled DISTINCT
CONTRA_BAND = (0.70, 0.88)     # the pair scan's own contradiction-candidate band
MID_BAND = (0.35, 0.55)        # ReasonedDreaming §2's mid-distance stratum
TIME_ADJACENT_HOURS = 48
RELATIONS = ("causes", "contradicts", "same_principle", "refines",
             "duplicate", "distinct", "unclear")

# ---------------------------------------------------------------- analogy
# AnalogyIncubation_Feature_Design.md, added to this probe 2026-08-01 at the
# owner's direction. NOTE ON SCOPE, because it matters for how results are read:
# ReasonedDreaming §8 defers the analogy stratum to PHASE 3, gated behind Phase
# 2. So everything below is an EXTENSION beyond Phase 0 and is scored and
# reported SEPARATELY — it must not move the pre-registered Phase 0 bars.
#
# What it fixes about the first attempt, which asked for a LABEL:
#   * §3A requires the model to "give the MAPPING", not tick a box. With no slot
#     for a structure, the first run produced topic soup ("both discuss
#     improvements to memory systems") and 10 of 18 claims were false.
#   * §2 defines an analogy candidate as LOW surface similarity + HIGH
#     relational correspondence, and says value RISES with domain distance
#     provided the structure still maps. A verbatim-quote guard (right for
#     causal claims per §6) therefore penalises exactly the good analogies,
#     which are stated abstractly: "both are a gate that admits only reviewed
#     work" quotes neither side. Analogy claims are grounded on ENTITIES
#     instead — the mapped roles must name things that really appear in each
#     gist.
#   * §3C is proportional: match the RELATION PATTERN while the ENTITIES DIFFER
#     ("deploy —gated_by→ approval" :: "merge —gated_by→ review"). That is a
#     four-memory question. A two-memory prompt cannot express it, so the
#     proportional form below hands the model TWO pairs.
ANALOGY_BAND = (0.30, 0.50)    # §3B, deliberately excluding near-duplicates
SERENDIPITY_FRACTION = 0.25    # §10: a quota of genuinely distant pairs
SERENDIPITY_MAX_COS = 0.20

ANALOGY_PROMPT = """Two memories from one person's memory store. They are from
different areas of their life or work.

MEMORY A (recorded {date_a}):
{gist_a}

MEMORY B (recorded {date_b}):
{gist_b}

Is there a deeper STRUCTURE shared by these two — the same pattern of roles and
relations, even though the subjects are completely different? If so, give the
mapping: what in A plays the same part as what in B.

A real example of the kind of thing meant: "deploy is gated by approval" and
"a merge is gated by review" share the structure "an action gated by a check",
where deploy maps to merge and approval maps to review.

Answer with ONE JSON object and nothing else:
{{"analogy": true|false,
 "schema": "<the shared structure in the abstract, using role words, or null>",
 "mapping": [{{"a": "<thing named in A>", "b": "<the thing in B playing that part>"}}],
 "surprise": 1-5,
 "rationale": "<one sentence>"}}

Rules:
- Say false unless the STRUCTURE really matches. Being about the same topic is
  NOT an analogy. "Both are about memory systems" is not a structure.
- Every "a" value must be something actually named in MEMORY A, and every "b"
  value something actually named in MEMORY B.
- surprise: 1 = the two are near-identical so the mapping is trivial, 5 = the
  domains are far apart and the mapping still holds exactly.
- false, with schema null and an empty mapping, is a perfectly good answer.
"""

PROPORTION_PROMPT = """Four memories from one person's memory store, as two
pairs.

PAIR ONE
  A: {gist_a}
  B: {gist_b}

PAIR TWO
  C: {gist_c}
  D: {gist_d}

Does the relation between A and B match the relation between C and D? That is:
is A to B as C is to D? The subjects will differ; what must match is the
RELATION.

Answer with ONE JSON object and nothing else:
{{"holds": true|false,
 "relation": "<the relation both pairs share, in the abstract, or null>",
 "rationale": "<one sentence naming something from each pair>"}}

Rules:
- Say false unless the two relations really are the same relation.
- Sharing a topic, a project, or a date is NOT a matching relation.
- false with relation null is a perfectly good answer.
"""

PROMPT = """You are reading two memories from a personal AI memory store.

MEMORY A (recorded {date_a}):
{gist_a}

MEMORY B (recorded {date_b}):
{gist_b}

What, if anything, is the real relation between them?

Answer with ONE JSON object and nothing else:
{{"relation": "causes"|"contradicts"|"same_principle"|"refines"|"duplicate"|"distinct"|"unclear",
 "direction": "a->b"|"b->a"|null,
 "rationale": "<one sentence that quotes a phrase from EACH memory>"}}

Rules:
- "causes" means one brought the other about, not merely that they co-occurred.
- "same_principle" means one underlying lesson generalizes over both, even if
  the subjects differ.
- "refines" means the newer memory updates, extends, or replaces the older one
  — the same subject, in a newer state. A revision, not a contradiction.
- "duplicate" means they state the same fact.
- "distinct" means genuinely unrelated or merely on the same topic.
- "unclear" is a fully acceptable answer. Prefer it to guessing.
- The rationale MUST quote a phrase from each memory verbatim. Do not invent
  facts that are not in the two texts.
- "direction" is null unless the relation is "causes".
"""


# --------------------------------------------------------------- sampling

def _rows(store: MemoryStore, model: str):
    return store.conn.execute(
        """SELECT m.id, m.kind, m.gist, m.project, m.event_time, m.recorded_time,
                  e.vector
           FROM memory m
           JOIN embedding e ON e.memory_id = m.id AND e.model = ? AND e.chunk = 0
           WHERE m.superseded_time IS NULL AND m.gist IS NOT NULL""",
        (model,)).fetchall()


def _dominant_model(store: MemoryStore) -> str | None:
    row = store.conn.execute(
        "SELECT model, count(*) c FROM embedding GROUP BY model "
        "ORDER BY c DESC LIMIT 1").fetchone()
    return row["model"] if row else None


def _topics(store: MemoryStore) -> dict:
    out: dict[int, set] = {}
    for r in store.conn.execute(
            "SELECT memory_id, topic_id FROM memory_topic"):
        out.setdefault(r["memory_id"], set()).add(r["topic_id"])
    return out


def _hours_apart(a, b) -> float:
    ta = (a["event_time"] or a["recorded_time"] or "")[:19]
    tb = (b["event_time"] or b["recorded_time"] or "")[:19]
    if not ta or not tb:
        return 1e9
    try:
        from datetime import datetime
        fmt = "%Y-%m-%dT%H:%M:%S"
        da = datetime.strptime(ta.replace(" ", "T")[:19], fmt)
        db = datetime.strptime(tb.replace(" ", "T")[:19], fmt)
        return abs((da - db).total_seconds()) / 3600.0
    except Exception:
        return 1e9


def sample(store_name: str, rng: random.Random) -> list[dict]:
    path = STORES[store_name]
    if not path.exists():
        print(f"  {store_name}: no store at {path} — skipped")
        return []
    store = MemoryStore(str(path))
    try:
        emb_model = _dominant_model(store)
        if emb_model is None:
            print(f"  {store_name}: no embeddings — skipped")
            return []
        rows = _rows(store, emb_model)
        vecs = {r["id"]: from_blob(r["vector"]) for r in rows}
        by_id = {r["id"]: r for r in rows}
        # Stratum A needs the TOMBSTONED side of every supersede pair, which
        # _rows() deliberately excludes (live recall must never see it). Load
        # the full set separately — strata B and C stay live-rows-only.
        all_rows = store.conn.execute(
            """SELECT m.id, m.kind, m.gist, m.project, m.event_time,
                      m.recorded_time, e.vector
               FROM memory m
               JOIN embedding e ON e.memory_id = m.id AND e.model = ? AND e.chunk = 0
               WHERE m.gist IS NOT NULL""", (emb_model,)).fetchall()
        all_vecs = {r["id"]: from_blob(r["vector"]) for r in all_rows}
        by_id.update({r["id"]: r for r in all_rows if r["id"] not in by_id})
        topics = _topics(store)
        skip = _distinct_pairs(store)     # pairs a human already ruled distinct
        linked = {frozenset((r["memory_id"], r["related_id"])) for r in
                  store.conn.execute("SELECT memory_id, related_id FROM memory_link")}

        non_epi = [r for r in rows if r["kind"] != "episodic"]
        epi = [r for r in rows if r["kind"] == "episodic"]

        stratum_a, stratum_b, stratum_c = [], [], []

        # A: triage against HUMAN GROUND TRUTH — half pairs a human superseded
        #    (newer replaces older), half a human ruled distinct. See the module
        #    docstring for why this replaces §2's live-candidate sampling.
        truth = {}
        for r in store.conn.execute(
                "SELECT id, superseded_by FROM memory WHERE superseded_by IS NOT NULL"):
            if r["id"] in all_vecs and r["superseded_by"] in all_vecs:
                stratum_a.append((r["id"], r["superseded_by"], None))
                truth[(r["id"], r["superseded_by"])] = "REPLACE"
        rng.shuffle(stratum_a)
        stratum_a = stratum_a[:TRIAGE_N // 2]
        # HARD negatives on purpose: the distinct pairs are taken highest-cosine
        # first, so the model is asked about pairs that look like duplicates to
        # an embedder (up to 0.93) and were still ruled distinct by a human. A
        # randomly-drawn negative at cosine 0.3 is trivially separable and would
        # inflate the triage score; these are the pairs the workload is made of.
        distinct_pool = [tuple(sorted(p)) for p in skip
                         if all(i in all_vecs for i in p)]
        distinct_pool.sort(key=lambda p: cosine(all_vecs[p[0]], all_vecs[p[1]]),
                           reverse=True)
        for aid, bid in distinct_pool[:TRIAGE_N - len(stratum_a)]:
            stratum_a.append((aid, bid, None))
            truth[(aid, bid)] = "DISTINCT"
        stratum_a = [(a, b, round(cosine(all_vecs[a], all_vecs[b]), 3))
                     for a, b, _ in stratum_a]

        # B: time-adjacent episodic pairs in the SAME project — where causality
        #    lives. The pair scan cannot see these: it excludes episodic rows.
        for a, b in combinations(epi, 2):
            if (a["project"] or "").lower() != (b["project"] or "").lower():
                continue
            if _hours_apart(a, b) > TIME_ADJACENT_HOURS:
                continue
            key = frozenset((a["id"], b["id"]))
            if key in skip or key in linked:
                continue
            c = cosine(vecs[a["id"]], vecs[b["id"]])
            stratum_b.append((a["id"], b["id"], round(c, 3)))

        # C: mid-distance, CROSS-topic (no shared topic id) — the analogy band
        for a, b in combinations(rows, 2):
            key = frozenset((a["id"], b["id"]))
            if key in skip or key in linked:
                continue
            if topics.get(a["id"], set()) & topics.get(b["id"], set()):
                continue
            c = cosine(vecs[a["id"]], vecs[b["id"]])
            if MID_BAND[0] <= c <= MID_BAND[1]:
                stratum_c.append((a["id"], b["id"], round(c, 3)))

        out = []
        for name, pool in (("triage", stratum_a), ("time_adjacent", stratum_b),
                           ("mid_distance", stratum_c)):
            if name != "triage":
                rng.shuffle(pool)
            cap = TRIAGE_N if name == "triage" else PER_STRATUM
            for aid, bid, c in pool[:cap]:
                ra, rb = by_id[aid], by_id[bid]
                out.append({
                    "store": store_name, "stratum": name, "cosine": c,
                    "a_id": aid, "b_id": bid,
                    "truth": truth.get((aid, bid)),
                    "a_gist": ra["gist"], "b_gist": rb["gist"],
                    "a_date": (ra["event_time"] or ra["recorded_time"] or "")[:10],
                    "b_date": (rb["event_time"] or rb["recorded_time"] or "")[:10],
                })
            print(f"  {store_name}/{name}: {len(pool)} candidates -> "
                  f"{min(len(pool), cap)} sampled")
        return out
    finally:
        store.close()


_ROLE_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been it its this "
    "that at by from as not no thing things one both each their there".split())


def _grounded_in(text: str, gist: str) -> bool:
    """Entity grounding, the analogy-safe counterpart to _quotes_both: does this
    role name something that actually appears in the gist? One shared content
    word is enough — the point is that the model maps REAL things from each
    memory rather than inventing entities, while staying free to state the
    schema abstractly, which §2 requires and verbatim quoting forbids."""
    words = {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
             if w not in _ROLE_STOPWORDS}
    return bool(words & set(re.findall(r"[a-z0-9]{3,}", (gist or "").lower())))


def analogy_grounded(rec: dict) -> bool:
    """A usable analogy answer: a clean 'no analogy' is valid; a claimed analogy
    must carry a schema and mapped roles that are each grounded on their side."""
    if not rec.get("analogy"):
        return True
    if not rec.get("schema") or not rec.get("mapping"):
        return False
    for m in rec["mapping"]:
        if not isinstance(m, dict):
            return False
        if not (_grounded_in(str(m.get("a", "")), rec["a_gist"])
                and _grounded_in(str(m.get("b", "")), rec["b_gist"])):
            return False
    return True


def sample_analogy(store_name: str, rng: random.Random, n: int = 24) -> list[dict]:
    """Mid-distance cross-topic pairs (§3B) + a serendipity quota of genuinely
    distant ones (§10), plus proportional A:B::C:D quads (§3C)."""
    path = STORES[store_name]
    if not path.exists():
        return []
    store = MemoryStore(str(path))
    try:
        emb = _dominant_model(store)
        if emb is None:
            return []
        rows = _rows(store, emb)
        vecs = {r["id"]: from_blob(r["vector"]) for r in rows}
        by_id = {r["id"]: r for r in rows}
        topics = _topics(store)
        linked = {frozenset((r["memory_id"], r["related_id"])) for r in
                  store.conn.execute("SELECT memory_id, related_id FROM memory_link")}
        mid, far = [], []
        for a, b in combinations(rows, 2):
            if frozenset((a["id"], b["id"])) in linked:
                continue
            if topics.get(a["id"], set()) & topics.get(b["id"], set()):
                continue                  # cross-topic only: analogy needs distance
            c = cosine(vecs[a["id"]], vecs[b["id"]])
            if ANALOGY_BAND[0] <= c <= ANALOGY_BAND[1]:
                mid.append((a["id"], b["id"], round(c, 3)))
            elif c < SERENDIPITY_MAX_COS:
                far.append((a["id"], b["id"], round(c, 3)))
        rng.shuffle(mid)
        rng.shuffle(far)
        n_far = int(n * SERENDIPITY_FRACTION)
        picked = mid[:n - n_far] + far[:n_far]

        out = []
        for aid, bid, c in picked:
            ra, rb = by_id[aid], by_id[bid]
            out.append({"store": store_name, "stratum": "analogy", "cosine": c,
                        "a_id": aid, "b_id": bid,
                        "a_gist": ra["gist"], "b_gist": rb["gist"],
                        "a_date": (ra["event_time"] or "")[:10],
                        "b_date": (rb["event_time"] or "")[:10]})

        # Proportional quads (§3C). Two supersede pairs are two instances of ONE
        # real relation ("this row replaced that one"), so a model that grasps
        # relations should say the proportion holds; four rows drawn at random
        # share no relation and it should not. Both are present so the stratum
        # catches a model that simply answers true.
        #
        # HONEST WEAKNESS OF THE 'HOLDS' LABEL, seen in the first smoke test:
        # `supersedes` is a STORAGE relation, and two supersede pairs can encode
        # semantically different relations — "a reminder followed by its
        # outcome" vs "a document revised to a new edition". The 72B called such
        # a quad false and its reasoning was defensible. So HOLDS is a WEAK
        # positive; the FALSE quads (four unrelated rows) are the solid ground
        # truth, and a model claiming those hold is unambiguously wrong. Read
        # the two classes accordingly rather than as one accuracy figure.
        # the OLD side of a supersede pair is tombstoned, so _rows() excludes it
        for r in store.conn.execute(
                "SELECT id, gist FROM memory WHERE gist IS NOT NULL"):
            by_id.setdefault(r["id"], r)
        sup = [(r["id"], r["superseded_by"]) for r in store.conn.execute(
            "SELECT id, superseded_by FROM memory WHERE superseded_by IS NOT NULL")
            if r["id"] in by_id and r["superseded_by"] in by_id]
        rng.shuffle(sup)
        quads = [(sup[i][0], sup[i][1], sup[i + 1][0], sup[i + 1][1], "HOLDS")
                 for i in range(0, min(len(sup) - 1, 12), 2)]
        if len(rows) >= 4:
            pool = [r["id"] for r in rows]
            quads += [(*rng.sample(pool, 4), "FALSE") for _ in range(6)]
        for a, b, c_, d, truth in quads:
            out.append({"store": store_name, "stratum": "proportion",
                        "cosine": None, "truth": truth,
                        "a_id": a, "b_id": b, "c_id": c_, "d_id": d,
                        "a_gist": by_id[a]["gist"], "b_gist": by_id[b]["gist"],
                        "c_gist": by_id[c_]["gist"], "d_gist": by_id[d]["gist"],
                        "a_date": "", "b_date": ""})
        print(f"  {store_name}: {len(mid)} mid-band, {len(far)} distant -> "
              f"{len(picked)} analogy pairs + {len(quads)} proportional quads")
        return out
    finally:
        store.close()


# ------------------------------------------------------------------ model

# Two of the three residents are REASONING models: left to themselves they emit
# a long hidden chain-of-thought before the JSON, into a `thinking` field this
# probe never reads. Measured 2026-08-01, that plus Ollama's default 262144-token
# context put qwen3.5 at ~150s/pair against the 72B's 8.4s/pair — 18x slower, and
# a projected ~10 hours of saturated GPU for the full three-model run. Both are
# pinned here so every model answers under the SAME one-shot contract:
#   think=False  — no hidden reasoning budget; the JSON contract is the whole task
#   num_ctx      — prompts are ~500 tokens, so a 262k KV allocation is pure cost
#                  (a trivial 17-token request measured 18s of prompt-eval alone)
# If a model lands near a bar under this contract, re-run THAT model with
# --think to see whether reasoning carries it over; do not mix the two in one
# comparison.
NUM_CTX = 8192
CALL_TIMEOUT = 120     # a single pair may never take longer than this
MAX_CONSECUTIVE_FAILS = 3


def ask(model: str, prompt: str, timeout: int = CALL_TIMEOUT,
        think: bool | str = False) -> str:
    # gpt-oss (Harmony format) cannot disable thinking: think=false +
    # format=json returns EMPTY content (measured 2026-08-01 — the tokens go
    # to the reasoning channel). "low" is its floor and complies cleanly, so
    # that model runs at minimum reasoning instead of none.
    if model.startswith("gpt-oss") and think is False:
        think = "low"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "think": think,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def _quotes_both(rationale: str, gist_a: str, gist_b: str) -> bool:
    """Hallucination guard: the rationale must lift a real phrase from EACH
    gist. Checked as a >=3-word shingle match, case-folded — quoting is the
    point, exact punctuation is not."""
    def shingles(text, n=3):
        w = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
    rs = shingles(rationale)
    return bool(rs & shingles(gist_a)) and bool(rs & shingles(gist_b))


def _one_pair(model: str, p: dict, think: bool) -> dict:
    """One question to one model. The stratum decides BOTH the prompt and the
    grounding check: causal/triage claims are quote-grounded per §6, analogy
    claims are entity-grounded because §2's whole point is that a good analogy
    is stated abstractly and would fail a verbatim-quote test."""
    stratum = p.get("stratum")
    rec = dict(p, model=model)
    try:
        if stratum == "analogy":
            obj = json.loads(ask(model, ANALOGY_PROMPT.format(
                date_a=p["a_date"], gist_a=p["a_gist"],
                date_b=p["b_date"], gist_b=p["b_gist"]), think=think))
            rec["analogy"] = bool(obj.get("analogy"))
            rec["schema"] = (str(obj.get("schema"))[:300]
                             if obj.get("schema") else None)
            mapping = obj.get("mapping") or []
            rec["mapping"] = mapping if isinstance(mapping, list) else []
            try:
                rec["surprise"] = int(obj.get("surprise") or 0)
            except (TypeError, ValueError):
                rec["surprise"] = 0
            rec["rationale"] = str(obj.get("rationale", ""))[:600]
            rec["grounded"] = analogy_grounded(rec)
            rec["relation"] = "analogy" if rec["analogy"] else "no_analogy"
        elif stratum == "proportion":
            obj = json.loads(ask(model, PROPORTION_PROMPT.format(
                gist_a=p["a_gist"], gist_b=p["b_gist"],
                gist_c=p["c_gist"], gist_d=p["d_gist"]), think=think))
            rec["holds"] = bool(obj.get("holds"))
            rec["shared_relation"] = (str(obj.get("relation"))[:300]
                                      if obj.get("relation") else None)
            rec["rationale"] = str(obj.get("rationale", ""))[:600]
            rec["relation"] = "holds" if rec["holds"] else "no_hold"
            rec["grounded"] = True
        else:
            obj = json.loads(ask(model, PROMPT.format(
                date_a=p["a_date"], gist_a=p["a_gist"],
                date_b=p["b_date"], gist_b=p["b_gist"]), think=think))
            rel = str(obj.get("relation", "")).strip().lower()
            rec["relation"] = rel if rel in RELATIONS else "unparseable"
            rec["direction"] = obj.get("direction")
            rec["rationale"] = str(obj.get("rationale", ""))[:600]
            rec["quotes_both"] = _quotes_both(rec["rationale"],
                                              p["a_gist"], p["b_gist"])
    except Exception as e:      # timeout, transport, or malformed JSON
        rec.update(relation="error", rationale=f"{type(e).__name__}: {e}",
                   quotes_both=False, grounded=False)
    return rec


def run_model(model: str, pairs: list[dict], *, think: bool = False,
              budget_min: float = 45.0) -> list[dict]:
    """One model, one run, with three independent stops so it can never grind
    on unattended: a per-call timeout (CALL_TIMEOUT), a whole-run wall-clock
    budget, and an abort after MAX_CONSECUTIVE_FAILS timeouts in a row (which
    means the model is unhealthy, not slow). Results are written after EVERY
    pair — stopping the run early keeps everything already answered."""
    out_f = OUT_DIR / f"results_{model.split(':')[0].replace('.', '')}.json"
    out, t0, fails = [], time.time(), 0
    for i, p in enumerate(pairs, 1):
        elapsed = (time.time() - t0) / 60
        if elapsed > budget_min:
            print(f"  ABORT: {budget_min:.0f} min budget spent at pair {i}"
                  f"/{len(pairs)} — keeping the {len(out)} answered", flush=True)
            break
        rec = _one_pair(model, p, think)
        out.append(rec)
        out_f.write_text(json.dumps(out, indent=1))     # incremental: never lose work
        fails = fails + 1 if rec["relation"] == "error" else 0
        if fails >= MAX_CONSECUTIVE_FAILS:
            print(f"  ABORT: {fails} consecutive failures — model unhealthy",
                  flush=True)
            break
        if i % 10 == 0:
            rate = (time.time() - t0) / i
            print(f"    {model}: {i}/{len(pairs)}  {rate:.1f}s/pair  "
                  f"eta {rate * (len(pairs) - i) / 60:.0f} min", flush=True)
    ok = sum(1 for r in out if r.get("quotes_both"))
    voided = sum(1 for r in out if r["relation"] in RELATIONS
                 and not r.get("quotes_both"))
    bad = sum(1 for r in out if r["relation"] in ("error", "unparseable"))
    print(f"  {model}: {len(out)} answered — {ok} usable, {voided} voided, "
          f"{bad} error/unparseable, {(time.time() - t0) / 60:.1f} min",
          flush=True)
    print(f"  -> {out_f}", flush=True)
    return out


def timeit(model: str, pairs: list[dict], n: int = 3,
           think: bool = False) -> float:
    """Measure this model on n pairs and project the full run BEFORE committing
    the machine to it. Added after extrapolating one model's speed to all three
    and being wrong by an order of magnitude."""
    t0 = time.time()
    for p in pairs[:n]:
        _one_pair(model, p, think)
    rate = (time.time() - t0) / n
    print(f"  {model}: {rate:.1f}s/pair  ->  {rate * len(pairs) / 60:.0f} min "
          f"for {len(pairs)} pairs", flush=True)
    return rate


# ------------------------------------------------------------------ review

# A causal claim can be TRUE and WORTHLESS ("session started causes session
# ended"), so the review marks usefulness separately from truth. Filled in by
# the reviewing AI and spot-checked by the owner; the scorer reads the verdicts.
VERDICTS = ("TRUE_USEFUL", "TRUE_USELESS", "FALSE", "")


def needs_verdict(r: dict) -> bool:
    """Rows a human/AI must judge, because truth is not mechanically knowable:
    a causal or shared-principle claim on the exploratory strata (the 50% bar),
    and any CLAIMED, grounded analogy (the Phase 3 extension). Triage and
    proportional rows carry ground-truth labels and are scored mechanically;
    `distinct`/`unclear`/`no_analogy` answers assert nothing to assess."""
    if r.get("stratum") == "analogy":
        return bool(r.get("analogy") and r.get("grounded"))
    return (r.get("stratum") not in ("triage", "proportion")
            and r.get("quotes_both")
            and r.get("relation") in ("causes", "same_principle"))


def review_table(results: list[dict]) -> str:
    todo = [r for r in results if needs_verdict(r)]
    lines = [
        "# Reasoned-pairs probe — review sheet", "",
        f"{len(todo)} proposals to judge (of {len(results)} answers). Each is a",
        "CAUSAL or SHARED-PRINCIPLE claim — the capability cosine does not have.",
        "", "Mark each verdict:",
        "  TRUE_USEFUL   — the relation holds AND surfacing it would help later",
        "  TRUE_USELESS  — it holds but is worthless "
        "(\"session started causes session ended\")",
        "  FALSE         — the relation does not hold",
        "",
        "Only TRUE_USEFUL counts toward the 50% bar. Copy the key line into",
        "_probe/verdicts.json as {\"<key>\": \"TRUE_USEFUL\"}.", ""]
    for r in todo:
        lines += [f"## {_key(r)}",
                  f"   {r['store']}/{r['stratum']}  cos {r['cosine']}",
                  f"A: {r['a_gist'][:260]}",
                  f"B: {r['b_gist'][:260]}"]
        if r["stratum"] == "analogy":
            pairs = ", ".join(f"{m.get('a')} :: {m.get('b')}"
                              for m in r.get("mapping", []) if isinstance(m, dict))
            lines += [f"-> SCHEMA: {r.get('schema')}",
                      f"   MAPPING: {pairs}",
                      f"   surprise {r.get('surprise')}"]
        else:
            lines += [f"-> **{r['relation']}** {r.get('direction') or ''}"]
        lines += [f"   {r.get('rationale', '')[:320]}", "   verdict: ", ""]
    return "\n".join(lines)


def relabel_sheet(pairs: list[dict]) -> str:
    """The blind semantic re-adjudication sheet for the DISTINCT triage pairs.
    A stored `distinct` link means "reviewed, never re-propose" — a storage
    decision — so it cannot serve as semantic ground truth; these pairs get a
    fresh human label on the gists alone, written BEFORE any model answer is
    read (that ordering is what keeps the relabel unbiased). REPLACE labels
    (supersede links) are sound as recorded and are not re-adjudicated."""
    todo = [p for p in pairs if p.get("stratum") == "triage"
            and p.get("truth") == "DISTINCT"]
    lines = [
        "# Stratum-A DISTINCT pairs — semantic relabel (BEFORE any model run)",
        "",
        f"{len(todo)} pairs. A stored `distinct` link is a storage decision",
        "(\"reviewed, never re-propose\"), not semantic truth — file-mirror",
        "bridge anchors are accepted distinct precisely BECAUSE they look like",
        "duplicates. Relabel each on the two gists alone:",
        "  REPLACE  — same fact, or one is a newer state of the other",
        "  DISTINCT — genuinely different things",
        "  DROP     — not judgeable from the gists alone",
        "",
        "Write _probe/relabel.json as",
        "  {\"<store>:<a_id>x<b_id>\": \"REPLACE\"|\"DISTINCT\"|\"DROP\", ...}",
        "Do this blind — before reading any model's answers.", ""]
    for p in todo:
        lines += [f"## {p['store']}:{p['a_id']}x{p['b_id']}",
                  f"   cos {p['cosine']}",
                  f"A ({p['a_date']}): {p['a_gist'][:300]}",
                  f"B ({p['b_date']}): {p['b_gist'][:300]}",
                  "   label: ", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------- score

def triage_correct(r: dict) -> str:
    """Score one ground-truth triage answer: CORRECT / ABSTAIN / WRONG.
    Definition fixed before the first run, repaired 2026-08-01 (refines
    restored) — see the module docstring."""
    rel, truth = r.get("relation"), r.get("truth")
    if rel == "unclear":
        return "ABSTAIN"
    if truth == "REPLACE":
        return "CORRECT" if rel in ("contradicts", "duplicate", "refines") \
            else "WRONG"
    if truth == "DISTINCT":
        return "CORRECT" if rel == "distinct" else "WRONG"
    return "WRONG"


def score(results: list[dict], verdicts: dict,
          relabels: dict | None = None) -> None:
    """Apply the pre-registered rule. Prints per-model numbers and the verdict.

    `relabels` is the blind semantic re-adjudication of the DISTINCT triage
    pairs (_probe/relabel.json). Without it the triage bar is NOT MEASURABLE:
    the stored `distinct` labels are storage decisions, not semantic truth."""
    print("\n=== PRE-REGISTERED GATE ===")
    print("build iff best model >=50% TRUE-and-useful on causal+principle "
          "AND >=80% correct triage\n")
    for model in sorted({r["model"] for r in results}):
        rows = [r for r in results if r["model"] == model]
        usable = [r for r in rows if r.get("quotes_both")
                  and r.get("relation") in RELATIONS]

        # new capability — needs human verdicts (a causal claim can be true and
        # worthless), drawn from the two exploratory strata only
        new_cap = [r for r in usable if r["stratum"] != "triage"
                   and r["relation"] in ("causes", "same_principle")]
        nc_true = [r for r in new_cap if verdicts.get(_key(r)) == "TRUE_USEFUL"]
        nc_judged = [r for r in new_cap if verdicts.get(_key(r)) in
                     ("TRUE_USEFUL", "TRUE_USELESS", "FALSE")]
        pct_new = 100 * len(nc_true) / len(new_cap) if new_cap else 0.0

        # workload reduction — objective labels, so scored on every PARSED
        # answer: the quote guard exists to kill invented causal stories and
        # must not shrink this denominator (it still gates strata B and C).
        # DISTINCT truths come from the blind semantic relabel; DROP excludes
        # a pair nobody could judge from the two gists alone.
        tri, dropped = [], 0
        for r in rows:
            if r.get("stratum") != "triage" or r.get("relation") not in RELATIONS:
                continue
            truth = r.get("truth")
            if relabels is not None:
                truth = relabels.get(f"{r['store']}:{r['a_id']}x{r['b_id']}",
                                     truth)
            if truth == "DROP":
                dropped += 1
                continue
            if truth:
                tri.append(dict(r, truth=truth))
        marks = [triage_correct(r) for r in tri]
        ok = marks.count("CORRECT")
        abst = marks.count("ABSTAIN")
        pct_tri = 100 * ok / len(tri) if tri else 0.0
        false_alarm = [r for r in tri if r.get("truth") == "DISTINCT"
                       and r.get("relation") == "contradicts"]
        missed = [r for r in tri if r.get("truth") == "REPLACE"
                  and r.get("relation") == "distinct"]

        print(f"{model}")
        print(f"  usable answers      : {len(usable)}/{len(rows)} "
              f"(rest voided by the quote guard, or unparseable)")
        print(f"  causal+principle    : {len(nc_true)}/{len(new_cap)} "
              f"true-and-useful = {pct_new:.0f}%   (bar: 50%)"
              + (f"  [{len(new_cap) - len(nc_judged)} awaiting verdicts]"
                 if len(nc_judged) < len(new_cap) else ""))
        print(f"  triage vs ground truth: {ok}/{len(tri)} correct "
              f"= {pct_tri:.0f}%   (bar: 80%)  [{abst} abstained"
              + (f", {dropped} dropped by relabel]" if dropped else "]"))
        print(f"     false alarms (said CONTRADICTS on a human-distinct pair): "
              f"{len(false_alarm)}  <- the destructive direction")
        print(f"     missed replacements (said DISTINCT on a superseded pair): "
              f"{len(missed)}")
        blockers = []
        if new_cap and len(nc_judged) < len(new_cap):
            blockers.append("causal+principle claims await verdicts.json")
        if relabels is None:
            blockers.append("triage DISTINCT pairs await relabel.json "
                            "(fill it from _probe/relabel.md)")
        if blockers:
            print(f"  PHASE 0 VERDICT: NOT MEASURABLE — {'; '.join(blockers)}")
        else:
            verdict = "PASS" if (pct_new >= 50 and pct_tri >= 80) else "FAIL"
            print(f"  PHASE 0 VERDICT: {verdict}")

        # Phase 3 extension — reported, never folded into the Phase 0 verdict
        ana = [r for r in rows if r["stratum"] == "analogy"]
        claimed = [r for r in ana if r.get("analogy")]
        grounded = [r for r in claimed if r.get("grounded")]
        ana_true = [r for r in grounded if verdicts.get(_key(r)) == "TRUE_USEFUL"]
        prop = [r for r in rows if r["stratum"] == "proportion"]
        p_ok = sum(1 for r in prop
                   if (r.get("holds") and r.get("truth") == "HOLDS")
                   or (not r.get("holds") and r.get("truth") == "FALSE"))
        if ana or prop:
            print(f"  -- analogy stratum (Phase 3 extension, NOT in the "
                  f"verdict above) --")
            print(f"     claimed an analogy : {len(claimed)}/{len(ana)} pairs")
            print(f"     survived grounding : {len(grounded)}")
            print(f"     judged real+useful : {len(ana_true)}"
                  + (f" = {100 * len(ana_true) / len(grounded):.0f}%"
                     if grounded else ""))
            if prop:
                print(f"     A:B::C:D correct   : {p_ok}/{len(prop)}"
                      f" = {100 * p_ok / len(prop):.0f}%")
        print()


def _key(r: dict) -> str:
    return f"{r['store']}:{r['a_id']}x{r['b_id']}:{r['model']}"


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true", help="build the strata")
    ap.add_argument("--run", metavar="MODEL",
                    help="ask ONE model (run them one at a time, cooling "
                         "between; three resident models at once is what made "
                         "the first attempt a 10-hour GPU burn)")
    ap.add_argument("--timeit", metavar="MODEL",
                    help="measure s/pair on 3 pairs and project the full run")
    ap.add_argument("--review", action="store_true", help="write the review sheet")
    ap.add_argument("--relabel-sheet", action="store_true",
                    help="regenerate the semantic relabel sheet for the "
                         "DISTINCT triage pairs from pairs.json")
    ap.add_argument("--score", action="store_true", help="apply the gate")
    ap.add_argument("--think", action="store_true",
                    help="let a reasoning model think (much slower; only for "
                         "re-running a single model that landed near a bar)")
    ap.add_argument("--budget-min", type=float, default=45.0,
                    help="wall-clock cap for one model's run (default 45)")
    ap.add_argument("--no-analogy", action="store_true",
                    help="Phase 0 strata only (the analogy/proportional strata "
                         "are a Phase 3 extension, scored separately)")
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    pairs_f = OUT_DIR / "pairs.json"
    results_f = OUT_DIR / "results.json"

    if args.sample:
        rng = random.Random(args.seed)
        pairs = []
        for name in STORES:
            print(f"sampling {name}…")
            pairs += sample(name, rng)
            if not args.no_analogy:
                pairs += sample_analogy(name, rng)
        pairs_f.write_text(json.dumps(pairs, indent=1))
        print(f"\n{len(pairs)} questions -> {pairs_f}")
        sheet = OUT_DIR / "relabel.md"
        sheet.write_text(relabel_sheet(pairs))
        print(f"relabel sheet -> {sheet}  "
              f"(fill _probe/relabel.json BEFORE the first run)")

    if args.relabel_sheet and not args.sample:
        pairs = json.loads(pairs_f.read_text())
        sheet = OUT_DIR / "relabel.md"
        sheet.write_text(relabel_sheet(pairs))
        print(f"relabel sheet -> {sheet}")

    if args.timeit:
        pairs = json.loads(pairs_f.read_text())
        timeit(args.timeit, pairs, think=args.think)

    if args.run:
        pairs = json.loads(pairs_f.read_text())
        print(f"running {len(pairs)} pairs on {args.run} "
              f"(think={args.think}, budget {args.budget_min:.0f} min)…")
        run_model(args.run, pairs, think=args.think,
                  budget_min=args.budget_min)

    if args.review or args.score:
        # merge whatever per-model files exist
        results = []
        for f in sorted(OUT_DIR.glob("results_*.json")):
            if pairs_f.exists() and f.stat().st_mtime < pairs_f.stat().st_mtime:
                print(f"WARNING: {f.name} predates pairs.json — answers from a "
                      f"stale question set; archive it or re-run that model")
            results += json.loads(f.read_text())
        if results:
            results_f.write_text(json.dumps(results, indent=1))

    if args.review:
        sheet = OUT_DIR / "review.md"
        sheet.write_text(review_table(results))
        print(f"review sheet -> {sheet}")

    if args.score:
        vf = OUT_DIR / "verdicts.json"
        verdicts = json.loads(vf.read_text()) if vf.exists() else {}
        if not verdicts:
            print(f"note: no verdicts yet — fill {vf} with "
                  '{"<store>:<a>x<b>:<model>": "TRUE_USEFUL"|...}; '
                  "the mechanical strata still score below, but the verdict "
                  "reads NOT MEASURABLE until the claims are judged")
        rl_f = OUT_DIR / "relabel.json"
        relabels = json.loads(rl_f.read_text()) if rl_f.exists() else None
        score(results, verdicts, relabels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
