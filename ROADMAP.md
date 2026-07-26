# FornixDB Roadmap — the operating-levels ladder

FornixDB's direction is not a feature backlog; it is a climb along one axis:
**how tightly, how often, and how much in parallel memory is fused into the
act of thinking.**

This axis is *orthogonal* to the two hardware dials described in the design
(memory capacity/speed, and where reasoning runs). The dials configure a
*single* capability to fit the host — a microcontroller and a workstation can
sit on the **same rung** of this ladder and simply differ in how much they
remember and how coarsely. The ladder, by contrast, measures the *maturity of
the coupling itself*: from "the program must ask" to "memory activates itself,
across many domains at once, and steers the next thought."

The organizing principle of the climb is **autonomy, then rhythm, then
parallelism** in the memory↔cognition loop.

## The levels

| Level | Name | How memory couples to cognition | Example endpoint / system | Status |
|---|---|---|---|---|
| **L0** | **Explicit store / retrieve** | A passive keyed store. The program must *deliberately* put and get; exact lookups, no ranking, no automation. Memory does nothing on its own. | A microcontroller or embedded device; a hand-wired `put(key)` / `get(key)`. | Trivially supported (the floor). |
| **L1** | **Associative recall on demand** | Still *pull-based* — the AI must choose to ask — but a query returns **relevance-ranked, associative** recall (vectors + full-text + a time axis), gist→detail. Human-like *retrieval*, but only when invoked. | An agent or chatbot that calls `recall` when it decides it needs context. | **Shipped** — the core query surface. |
| **L2** | **Automatic capture** *(write-side autonomy)* | The **write** side becomes autonomous: experience is captured and consolidated after each prompt or session with no explicit "store." The read side is still pull-based. This is *storing memories after each prompt* — the foothold for everything above it. | A coding-agent or local-model session whose transcript is auto-captured and consolidated. | **Shipped.** |
| **L3** | **Proactive recall injection** *(read-side autonomy, one pulse per turn)* | Memory **pushes** relevant context into the thinking *without being asked* — once per turn, relevance-gated, additive (never replacing the host's own memory). The first heartbeat of memory "eventing back" to the thinker. | A prompt-submit hook that surfaces a provenance-tagged "possibly-relevant past" block each turn. | **Shipped — lived-in; recall quality tuned via usefulness feedback.** Wired into the daily loop and surfacing each turn; the relevance floor is now per-memory, tightened by the usefulness signal below (impressions vs uses) so the block fires on the right memories, not merely plausible ones. |
| **L4** | **Rhythmic in-thought recall** *(the "metronome")* | Memory is re-activated **many times within a single reasoning episode** — pulsed as the thought evolves, each pulse re-querying on the *current* state and steering the next step. **Event-driven cadence, not a constant beat.** The loop tightens from once-per-turn to many-per-thought. | A debounced local recall loop / inner agent that re-queries at reasoning checkpoints. | **Shipped, default-on — proven by the usefulness gate**: scan-verified downstream reference rate for L4 pushes beat the per-turn L3 channel (20% vs 13%) on lived-in usage. Portable cadence controller (`fornixdb.cadence`) wired into both a local model's inner tool-loop (a local model) and the Claude Code tool seam. |
| **L5** | **Parallel multi-domain activation** *(the analog, human-like target)* | Many lightweight agents fire **simultaneously across different information domains** (episodic, semantic, feedback, by-project, by-person, by-salience…), **all local**, and their returns are **integrated and settle into a single pattern** that directs the next thought. Not one recall, but a *field* of simultaneous local recalls resolving into a direction. | A local orchestrator spawning N domain-scoped recall agents per reasoning step, with a settling/attractor integrator. | **Shipped, default-on** — the field (`fornixdb.field`): seven domain-scoped recalls on one shared query embedding, settled by corroboration clustering (cross-domain agreement + link/topic edges; no corroboration degrades to L4, nothing fabricated), riding the L4 metronome via `config parallel_recall` — **default-on since 0.5.0**. Measured honestly against the gate L4 passed: the first true scan readout (2026-07-18, after fixing a marker bug that had credited every settled push to L4) put L5 at 4–6% referenced vs L4's 12–17% while it was the largest injected-token channel, so the field was **trimmed to its proven domains** (`parallel_limit 2`; knowledge, guidance, reference, context, neighborhood; deep + recent dropped). The 2026-07-26 re-measure: **17% of L5 pushes referenced downstream** (4× the pre-trim rate; L4 ran 32% in the same window) — owner decision: keep the trim. L5 is an ambient cross-domain association channel that earns its keep at that measured rate; it is not human-grade association (see "The association gap" below). `config parallel_recall off` steps back to L4 at any time. **◀ we are here — the default operating rung since 0.5.0, trimmed to proven domains in the 0.8.14 honing round.** |
| **L6** | **Federated / distributed memory** *(an extension beyond human-like)* | The parallel model extended **across endpoints** — machines, agents, eventually a household — federating many FornixDB stores behind one recall. Reached only **after L5**: a single mind is not federated, so this sits *above* the human-likeness climb as a super-human extension, not a step toward it. | A cross-machine aggregator tier; a fleet or household memory. | Far out; encryption-gated. |

## Why this is the shape of the climb

L0–L3 are about **autonomy** — memory learning to write itself and to recall
without being told. L4–L5 are about **rhythm and parallelism**, and that is
where the human-likeness actually lives.

Today's AI *thinks a lot without memory injected into the thinking in analog
fashion*: it reasons in a long serial stream, and memory — at best — is
consulted once, at the edges of a turn (L1–L3). A human mind, mid-thought,
lights up **in parallel across many domains at once**, and the action-state
*settles* on a pattern. L5 is the computational mirror of that: not a single
lookup but a field of simultaneous local recalls that resolve into a
direction for the next thought.

So the rungs are cumulative heartbeats of the same idea:

- **L2** is the foothold — memory writes itself.
- **L3** is the first heartbeat — memory speaks back, once per turn.
- **L4** makes it beat repeatedly *within* a thought.
- **L5** *(where we are)* makes it beat *in parallel across domains*.
- **L6** then extends that beating mind across machines — beyond what a single
  human mind is, and deliberately last.

Endpoint-local memory storage is the correct foundation, and these are the
stages that grow it from a store the AI queries into something that
approximates the analog, parallel character of remembering.

## The association gap — what L5 is, and what it is not

*(added at 1.0.0, after the honing rounds put numbers on the field)*

Human cross-domain association — the "common sense" by which a mind knows the
real world through its interconnections — is the capability L5 aims at, and
it is where current AI is weakest relative to people. Honesty about the gap:

**What L5 actually is.** Retrieval-side spreading activation over stored
gists: static embeddings, links, and topics, settled by corroboration.
Trimmed to its proven domains it earns a measured **17% referenced-downstream
rate** — a genuinely useful ambient channel that surfaces cross-domain
candidates a serial lookup would miss. That is a prosthetic, not cognition.

**Why its ceiling is structural, not a tuning problem.** Two binds, neither
of which more FLOPs at recall time fixes:

1. *Where association happens.* In a mind, spreading activation runs inside
   the model; the LLM analog is attention over context — the forward pass IS
   the association engine. The field imitates association *outside* the model
   precisely because context is scarce; injecting winners is the affordable
   substitute for letting attention see the whole store. Bigger contexts and
   better resident models raise this ceiling; a cleverer field does not.
2. *The data.* A human associates over a lifetime of continuous, multimodal,
   causally-structured experience, re-woven nightly. A store holds hundreds
   of text gists. The interconnection density is not there to be found.

**The next achievable rung — model-mediated consolidation ("reasoned
dreaming").** In people the interconnections are precomputed during sleep;
recall is fast because consolidation already wove the graph. FornixDB has
exactly this architecture — dream — but dream today weaves by cosine, the
same shallow signal the field searches by. The one abundant local resource is
an idle resident LLM at consolidation time (batch, latency-irrelevant, zero
marginal cost): pointed at the store during dream, it can propose *reasoned*
links — "X causes Y", "these contradict", "same principle underneath" —
instead of similarity pairs. The runtime field then inherits richer glue for
free, at the only point in the loop where heavyweight intelligence is
affordable. This stays propose-only (§6.5): the reviewing AI applies.

**The long view — grounded association.** Human-grade common sense requires
embodied, causally-structured experience streams (robotics is the data path)
plus model-side world-model research (the training path). Neither is a memory
library's to build, and this roadmap does not pretend the field will grow
into it. FornixDB's claim in that future is the **episodic substrate**: the
senses (SENSES.md) already capture the embryonic version of exactly such a
stream, and a local, time-indexed, consolidating store is what an embodied
system would sleep on. The memory-side seat is ours to hold; the model-side
advances are awaited, not promised.

## Cross-cutting work that strengthens every rung

The ladder measures the *maturity of the coupling*. Orthogonal to it — and to
the two hardware dials — are signals that make every recall, at every rung,
land better. These are not rungs (climbing them does not tighten the
memory↔cognition loop), but the rungs lean on them: a sharper relevance signal
makes L1's ranking, L3's once-per-turn gate, and L4/L5's repeated/parallel
pulses all fire on the *right* memories instead of merely plausible ones.

- **Per-memory usefulness feedback** *(built)*. Each memory carries a usefulness
  signal — an explicit "this helped" mark (`helpful_count`) and scan-verified
  downstream use of its pushes (`referenced_count`), rolled up at session start —
  and it feeds back into both ranking *and* the relevance floor. The closing
  move, **directly downstream of L3**: every *unsolicited* surfacing — a
  proactive push, a `brief`/`timeline` listing — is counted as an *impression*
  (`surfaced_count`), kept strictly apart from evidence of use, so the system
  can tell "this memory keeps getting surfaced but no one ever uses it" from
  "this memory is used." That gap nudges a **per-memory** push floor —
  proven-useful memories surface a touch more easily, chronically-ignored ones
  go quiet — which is the implicit, additive attack on the cross-project noise
  observed in live dogfooding. Genuine PULLs (`recall_count`) are tracked but
  deliberately count toward neither ranking nor the floor: a pulled memory
  needs no pushing to be found, and on a lived-in store pull counts inflate
  with listing traffic until they mask the never-used population. Bounded,
  never hides a memory, reversible via `config usefulness_floor_adapt off`.

- **Project-scoped pulse recall** *(built)*. The complementary half of the
  noise fix: a pulse that knows its active context raises the push floor for
  memories that don't *belong* to it (belonging unifies both axes — project OR
  any topic, with aliases), so off-context memories stop leaking into the ambient
  stream on weak matches — while a strongly-relevant one still surfaces and
  untagged/structural-only facts are never scoped out. The active context is a
  pinned `active_project`, else the project DECLARED in a prompt ("continue the X
  project", sticky per session), else the host's working directory — so it works
  even when every session shares one cwd. Push-only (explicit recall is
  unscoped); reversible via `config project_scoped_pulse off`.

## Relationship to the rest of the design

- **Orthogonal to the two dials.** A given rung runs at whatever memory and
  cognition-locus settings the host hardware dictates. Climbing the ladder
  never requires more powerful hardware on its own — L2's automatic capture
  and L5's parallel activation are both *local* by construction.
- **The "memory, not a mind" line holds at every rung.** Even at L5, the
  parallel recalls *surface and settle structure*; judgment and action stay in
  the reasoning model. A higher rung means memory participates in thinking more
  rhythmically — never that it decides or acts.
- **No rung is gated on a local model.** As everywhere in FornixDB, the
  algorithmic reflex layer is the floor; a local model is an optional
  enhancement to the work each rung does, never a requirement to be on it.
- **Performance tuning lives in the deployment, not the core.** The embedding
  model is **cold-loaded per turn** by design — the only behavior portable
  across the hardware FornixDB targets (microcontroller → server). A deployment
  on *known* hardware that runs inside one long-lived process (a humanoid robot,
  a resident agent) can keep the model **warm** for a large per-turn speedup, but
  that needs host-specific lifecycle/IPC glue the core can't write blind. So
  FornixDB ships the **seam** for it — inject a warm `Embedder` via
  `FORNIXDB_EMBEDDER` or `set_default_embedder()` — and keeps warm as a
  per-deployment adapter. See INTEGRATION.md, "Warm embedding on dedicated
  hardware."
