<p align="center">
  <img src="assets/fornixdb_logo.png" alt="FornixDB" width="300">
</p>

# FornixDB

**Persistent local memory for any AI — private, model-agnostic, and inspectable.**

## The problem

An AI assistant forgets everything between sessions. You pay for that every day: re-explaining context, re-stating preferences, watching the model re-read files and re-derive decisions it already made. Some questions have no answer at all in a stateless chat — *"what day did the pool guy come by?"* — no matter how capable the model is.

FornixDB is a memory any AI can read and write. It runs entirely on your machine, in a single SQLite file you own, and works with anything that can call a tool or run a shell command — Claude, a local Llama/Qwen, a robot's onboard model.

It is a **memory, not a mind**: it stores, indexes, ranks, and retrieves. It never decides, never acts, never calls a tool. All judgment and guardrails stay in the AI you connect it to.

**Connect it in one line.** `claude mcp add fornixdb -- fornixdb-mcp` wires it into Claude Code or any [MCP](https://modelcontextprotocol.io) client; a local Llama/Qwen reaches the same tools through the shim or a shell call. Details in [Connecting an AI](#connecting-an-ai).

## What it does

- **Recall by time.** Natural phrases — *"what did we do last Thursday?"*, *"this morning"* — return everything from that window. Sessions are captured automatically (owner-toggleable), so the answer exists without anyone deciding to save it.
- **Remembering to remember** *(prospective memory)*. *"Remind me tomorrow morning to talk to Joe about his vacation plans"* stores an intention with a due time; when the clock arrives it is delivered through the host's own heartbeat — the next chat turn, a voice assistant's idle tick, or session start — exactly once. Mark it **urgent** and it nags instead: repeated every few minutes until the owner responds with anything at all, and resurfaced at the next session start rather than ever dying silently. Future phrases work on the timeline too: *"what's coming up?"* reads the same rows forward.
- **Recall by subject.** Keyword + ranked relevance returns a one-line **gist** first; full **detail** is fetched only when the conversation drills in, so recall stays cheap in the context window.
- **Recall by meaning** *(on by default, never required)*. A small local embedding model adds similarity by meaning — *"the glitch where her eyes sparkled"* finds the right memory with zero shared words. It ships by default but is never required: switch it off (`FORNIXDB_VECTORS=off`) or run on hardware that can't load it, and pure keyword + time still works.
- **A memory that adapts.** New knowledge supersedes old *without erasing it* (the trail of corrections stays queryable) — just **re-`remember` under the same title** and the prior version is kept as history, so a finding that evolves mid-investigation (hypothesis → corrected diagnosis → fix) stays *one* current memory instead of three contradictory ones. Unused memories fade in ranking while frequently-used ones stay sharp; an explicit "not that one" downweights a wrong recall hit for similar queries only — retractable, never deleted. A memory that keeps getting *pushed* proactively but is never acted on eventually stops being pushed — muted from the proactive channels while staying fully available to a direct recall, and un-muted the moment it earns a reference again.
- **Honesty flags.** Stale, unverified facts come back marked as such; duplicates across stores answer once; a downweighted or auto-captured result says so. Provenance travels with every answer.
- **You stay in charge.** Capture policy is yours (only-when-asked / offer / auto). Never-delete is the default; true deletion happens only at your explicit consent, and forgets least-important-first.

**The measured case** — what your AI gains, the token economics, and who stays in charge: [BENEFITS.md](BENEFITS.md).

## Design principles

- **Local and private.** One store per endpoint (computer, robot, device). Your memories never leave the machine. Only the *reasoning* may be a remote service; the memory never is.
- **Model-agnostic and vendor-neutral.** The core has zero dependencies on any AI vendor. Integrations with specific ecosystems (e.g. importing Claude Code's memory files or session transcripts) are optional adapters.
- **Two recall axes.** Episodic (time-stamped events: "on June 5 we designed the schema") and semantic (distilled facts: "the owner prefers specs over UI driving"). Semantic memories are consolidated *from* episodic ones over time.
- **Progressive disclosure.** Recall returns gists cheaply; detail is fetched only when the conversation drills in. This is what keeps recall affordable in a context window.
- **Supersede, don't overwrite.** Conflicting newer knowledge tombstones the old with a timestamp and a typed link. History is the record of how understanding changed.
- **Scales by forgetting.** Salience, reinforcement-on-recall, decay, and tiered storage (hot → consolidated → cold archive) — tuned by two dials derived from the host's hardware (storage capacity/speed, and where reasoning runs).
- **No model required.** The baseline reflex layer (ranking, eviction, clustering) is purely algorithmic, so it runs on hardware that can't host a local model. A local model is an optional enhancement, never a requirement.

## Status

Published (MIT) and in daily use. Shipped: the hot spine (SQLite + FTS5, time + subject recall, supersede-with-history), associative recall on by default (model2vec vectors, switchable off), decay + retention tiers + the consolidation pass, explicit negative feedback (mark a wrong recall hit irrelevant to a query — downweighted for similar queries only, retractable, never deleted), multi-AI topology (per-agent stores + machine shared tier + capture modes, cross-store recall deduped), disk-budget cap with prune/freeze boundary policies and frozen read-only stores, import adapters (Claude Code transcripts + a SessionEnd hook for live passive session capture), a bidirectional Markdown bridge (heading-chunked import of arbitrary documents, plus Markdown export of the store), a dream-time markdown staleness cross-check (a hand-maintained file's open PICKUP/NEXT block that a later captured session appears to have overtaken gets flagged — the flat file has no machinery for noticing it is stale; the store does), prospective memory (reminders with natural due phrases, delivered exactly once on the host's own heartbeat), and a configurable MCP tool surface (all tools on by default; optional ones trimmable per store via `fornixdb tools` to cut prefill on token-tight devices — no universal ceiling). Three consumers proven: Claude Code, a local Qwen-72B agent, and a 14B via the MCP/shim surface (cold-installed and verified on Windows). Extensively tested.

**Accountable in numbers.** The store can tell you what it is worth. `fornixdb value` leads with an estimated net token verdict — *"Estimated tokens SAVED: ~N/session"* (or EXTRA, when that's the truth) — built from a **measured** cost side (tool schemas, startup context, and the actual size of every proactively injected memory, summed from your own session transcripts) against an **explicitly assumed** savings side (a printed low/mid/high band for what each downstream-used memory replaced; no session-without-memory exists to compare against, so it is a range, never one confident number). Behind the verdict sits the honest usage signal: the referenced-push rate, scanned from transcripts — how many proactive recalls your AI actually used, not how many it was shown. Proactive recall itself is **self-maintaining**: opt-in logging (`config floor_log on` → `floor-stats` / `field-stats`) records every push decision, and the periodic dream pass turns accrued evidence into reviewable worklists (credit memories proven useful, flag chronic noise, propose dial changes) — propose-not-dispose, the owner reviews. The verdict is context-space (each token counted once); host usage panels count token-turns — the same content re-read on every API request, 30–150× larger for both sides of the ledger equally — and `fornixdb tokens --billed` measures that billed view directly from your transcripts' per-request usage records so the two numbers finally match in kind. See [BENEFITS.md](BENEFITS.md) for the full token-economics case.

**Where this is heading** — FornixDB's direction is a climb along one axis: how tightly, how often, and how much in parallel memory is fused into the act of thinking, from "the program must ask" up toward memory that activates itself across many domains at once and steers the next thought. Today the ladder is lived-in **through its full built height: L5 parallel multi-domain activation — the field — is default-on since 0.5.0**, after L4 passed its scan-verified usefulness gate and live L5 dogfooding showed the field surfaces useful context without measurable harm (~96% of surfaced pushes scored useful in the floor-log join, ~0.3 s per beat); the L5-vs-L4 reference-rate readout keeps accruing as the revert signal, and `config parallel_recall off` steps a store back to L4 at any time. With the ladder's built height default-on, the active workstream is **multimodal senses memory** — sight, sound, and sensor streams entering the same store ([SENSES.md](SENSES.md)). The full ladder — and why parallel, in-thought recall is the human-like target — is in [ROADMAP.md](ROADMAP.md).

## Requirements

- Python 3.10+ with SQLite compiled with FTS5 (standard on macOS, most Linux distributions, and the python.org Windows builds). The keyword + time core runs on **only the standard library**; the default install also pulls a small embedding model (model2vec) so semantic recall works out of the box.
- **No SQLite is installed or bundled.** FornixDB uses Python's built-in `sqlite3` module, so it relies on whatever SQLite your Python already links — it can't touch or conflict with a system SQLite.
- **model2vec is unpinned**, so pip accepts any version already on the machine and never overrides or downgrades a model2vec another project is using; if it's missing or too old to load, FornixDB falls back to keyword-only.
- **Windows:** the interpreter is `python` (or `py`), not `python3` — on stock Windows, `python3` is a Microsoft Store stub. Substitute accordingly in every command below.

```bash
# optional but recommended: install the package so `fornixdb` and
# `python -m fornixdb` work from any directory
pip install -e .          # includes model2vec; semantic recall on by default
```

**Pin a released version** so a clone never lands mid-change. `main` is the
active development branch and can change through the day; for a stable checkout,
install a tagged release instead:

```bash
pip install "git+https://github.com/itdtllc/fornixdb@v0.8.10"
```

Releases are listed at <https://github.com/itdtllc/fornixdb/releases>; see
[CHANGELOG.md](CHANGELOG.md) for what each version contains.

## Semantic recall (on by default)

Recall ranks by keyword + time, blended with *similarity by meaning* from a small local embedding model — "the glitch where her eyes sparkled" finds the eye-twinkle memory with zero shared words. The model ships by default and is tiny (~30MB static embeddings, CPU-only, no torch); a fresh store embeds memories automatically from the first write, and an existing store from before vectors backfills itself the first time it's used (no manual step). `python3 -m fornixdb embed` still forces a backfill on demand.

The embedder is pluggable (any object with `.name` and `.embed()`), the model is configurable via `FORNIXDB_EMBED_MODEL` (a HuggingFace repo id or a local directory), and air-gapped machines can drop model files into `~/.cache/fornixdb-models/<model>/` so no network is ever attempted.

**Capability, reconciled — many machines can't (or shouldn't) run a model, and that's fine.** Vectors are a layer *on top of* a fully-working keyword + time core, capability is detected at runtime, and the same store works everywhere — degrading gracefully in both directions:

- **No model** (hardware too small to load it — a microcontroller, a small robot): the model import fails, recall falls back to keyword + time, `store()` embeds nothing. Everything else (time recall, supersede, decay, budget, links) is pure SQLite and unaffected.
- **A vector-built store opened on a model-less machine:** the embeddings sit unused on disk; recall falls back to keyword. Nothing breaks.
- **Model present (the default):** recall blends similarity and new writes embed automatically.

Two things to know: (1) vectors are **on by default but easy to switch off** for a deliberately lean deployment — set `FORNIXDB_VECTORS=off` (machine-wide) or `fornixdb config vectors off` (one store), or pass `--keyword-only` on a single recall; (2) the tradeoff is *quality, not function* — keyword-only ranks the single best hit less often than hybrid, but still finds memories and answers time questions nothing else can.

## Quick start

```bash
# initialize a store
python3 -m fornixdb init

# store a memory
python3 -m fornixdb store --gist "Decided to use SQLite FTS5 for subject recall" \
    --detail "Considered a dedicated vector DB but ..." --topic architecture

# recall by subject (gist-first, ranked)
python3 -m fornixdb recall "subject recall ranking"

# recall by time (natural phrases go in the positional argument;
# --since/--until take explicit dates)
python3 -m fornixdb timeline "last thursday"

# drill into detail
python3 -m fornixdb show <id>

# a recalled hit was wrong for that query? downweight it for similar queries
# (query-conditional: it stays fully ranked for everything else; retractable)
python3 -m fornixdb irrelevant <id> "the query it was wrong for"
```

## Anatomy of a memory

Transparency is the point: you (and your AI) can always see what is
remembered, what it costs, and how to go deeper. Here is what one memory
actually is.

**Three levels of content.** Every memory is a disclosure ladder:

1. **gist** — one line, what every recall and listing returns first
   (~20–80 tokens; recall stays cheap because this is all it ships).
2. **detail** — the full body, fetched only when someone drills down
   (`show <id>`). Drilling down *reinforces* the memory, so frequently-used ones stay ranked.
3. **source_ref** — a pointer to the raw original (a session's full
   transcript file, an imported document). Total reconstruction is possible
   without the database carrying a second copy of the heavy source.

**Unbounded structure on top.** Memories **link** to related memories
(recall can attach 1-hop neighbors with `--related`; the graph walks as far
as it goes), and **supersession chains** keep every correction: the live
version answers, the history of what was believed before stays queryable
(`recall --all`). Topics, time spans, and a session id tie each memory to
when and where it happened.

**Not every word gets stored — selectivity is the design.** A whole chat
session becomes *one* episodic row (a one-sentence gist + a compact digest
of the user's turns); the full transcript stays on disk as the source_ref.
Facts are stored deliberately, gated by the owner's capture mode — memory
is curated, not vacuumed up. After capture, three more stages keep the
store lean:

- **Decay** — unused memories sink in ranking (per-kind half-lives, owner
  feedback floors highest); recalled ones stay sharp. Nothing is deleted.
- **Consolidation** — a periodic background pass proposes distilling
  verbose gists, merging near-duplicates, and flagging contradictions.
  Propose-not-dispose: the owner reviews, the pass never deletes on its own.
- **Retention tiers** — old, unused detail is compressed in place, then
  archived to cold storage; `show` restores it transparently if ever asked.

True deletion exists only at the owner's explicit consent (the disk-budget
`prune` policy and `budget shrink`), in humane order: least-salient first,
owner feedback last.

**Multimodal senses memory is the active workstream — the design is
published, the APIs are declared.** [SENSES.md](SENSES.md) lays out the
architecture now being worked toward: dense sensory sampling through a
salience gate so only events become memories, captions as gists (recall
across modalities through the existing text space from day one), modality
embeddings in the same pluggable slot the text model uses, artifacts on
disk as `source_ref`, streams as episodic time spans, and decay as a
fidelity ladder from vivid to verbal.
[`fornixdb/senses.py`](fornixdb/senses.py) holds the entry points: `see`
(images), `hear` (audio), and `feel` (sensor readings — machine
proprioception, no robot required) are implemented for single artifacts
today — caption/state gist, optional modality vector, pointer-not-blob —
with local models plugging in as callables; the salience gate ships
hardware-free as `fornixdb.salience`. The live loops are here: `watch`
([`fornixdb.watchloop`](fornixdb/watchloop.py)) drives a camera, the screen, or
a video file through the gate into `see` memories with event-time spans — the
Mac adapters ([`fornixdb.adapters.mac_camera`](fornixdb/adapters/mac_camera.py)
frame sources + an MLX image embedder) and a `fornixdb watch` command are live
(`pip install 'fornixdb[mac]'`) — and `feel`'s change-gated loop
([`fornixdb.feelloop`](fornixdb/feelloop.py)) runs with a reference Mac power
adapter and a `fornixdb feel [--live]` command. Capture is always
owner-started: no always-on sampling, no watch/feel MCP tool. Sound is treated as
meaning, not only words: a sound-scene caption is the required lane, a
transcript is the additional one.

## Layout

```
fornixdb/            core package (vendor-neutral, stdlib-only)
  adapters/          optional importers for specific ecosystems
tests/               test suite
examples/            reference shim + runnable smoke test
```

## Multiple AIs on one machine

Memory topology is configurable, not fixed. Each AI gets its **own store** (its working memory), and every AI also reads a **machine-level shared tier** (`~/.fornixdb/shared.db`, or `$FORNIXDB_SHARED_DB`) holding owner facts and preferences all of them should know. Recall, timeline, and brief merge both automatically; write owner-level knowledge with `store --shared`. An aggregator across agent stores is the planned next level.

Each store also carries an owner-settable **capture mode** (`config capture_mode explicit|suggest|auto`) that connected AIs read at startup: remember only when asked, offer to remember at checkpoints (default), or store autonomously.

All of this is safe to run **at the same time**: a store file can be hit concurrently by several agents, several processes (MCP server, hooks, CLI), and several threads sharing one `MemoryStore` handle — writers serialize through WAL + per-store busy timeout (`config busy_timeout_ms`), schema migrations are single-winner, and reminders fire exactly once no matter how many hosts poll. Details in INTEGRATION.md §Concurrency.

## Long-running agents and loops

A looping agent — a session re-woken on a schedule (Claude Code's `/loop`, a cron-driven agent, any host that re-prompts the same conversation on an interval) — is where a memory earns its keep, because loops are exactly where context windows fail. What FornixDB adds to a loop:

- **Proactive recall fires every iteration.** Injection is seam-driven — each wakeup's prompt and each tool call gets its beat — so the loop receives "possibly-relevant past" continuously for as long as it runs, with no extra wiring beyond the host's normal hooks.
- **Recall survives context compaction.** Long-running sessions get their conversation summarized or truncated by the host; the store doesn't. Checkpoint decisions and durable facts mid-loop (`store`, or `jot` for cheap raw capture) and iteration 40 can retrieve associatively what iteration 2 decided, after the conversation text is gone.
- **Reminders make loops punctual.** A loop that polls `due` each wakeup is a delivery host for prospective memory — and delivery is exactly-once across hosts, so a reminder set for 9am fires in one session even when a loop *and* an interactive session are open on the same store.
- **Checkpoint explicitly; don't rely on session capture.** Passive episodic capture writes at session *end*, so a loop that never ends never auto-captures. The mid-loop checkpoint habit above is the fix.
- **Sub-agents don't inherit pulses.** Helper agents a loop spawns run outside the host's injection seam; give them the recall tools and they use memory explicitly, but ambient injection belongs to the main conversation.
- **Parallel loops stay coherent, not entangled.** Several loops on one store — even in different projects — each pulse under their own active project: off-context memories must clear a higher relevance floor (`project_scoped_pulse`), unscoped general facts flow to all, and nothing is ever hidden from explicit recall. Concurrent access is safe per the section above.

One loop is one remembering agent; parallelism comes from running several sessions, and the store is the shared substrate that keeps them coherent rather than a coordinator between them.

## Configuration at a glance

`fornixdb config` with no arguments prints **every** store setting at once — capture mode, ingest mode, vectors, disk budget, frozen state, proactive recall, and the MCP tool surface — alongside the **suggested defaults** for each (and which aren't applied yet). `fornixdb doctor` adds a health pass: schema currency, the host-side hooks that make capture and proactive recall actually fire (the most common silent gap, since those live in the host's `settings.json`, not in FornixDB), config smells, and a **config-integrity** check that flags any setting you've stored that no code actually reads (a typo, or a key with no effect). The one recommended setting **not** applied out of the box is a disk cap (never-delete is the default) — `fornixdb doctor --apply-suggested` sets it to a figure scaled to the device (20% of free disk, capped at 2 GB).

```bash
python3 -m fornixdb config                 # all settings + suggested defaults
python3 -m fornixdb doctor                 # health check + host-hook detection
python3 -m fornixdb doctor --apply-suggested   # apply the unmet defaults (e.g. a disk cap)
```

### The configuration wizard

`config` prints and `config <key> <value>` sets one thing at a time. If you'd
rather be walked through every setting — with the suggested default, a short
explanation, and the current value for each, confirming as you go — run the
**interactive wizard**. It's the same engine (`fornixdb configure`), wrapped in
a launcher that needs **no arguments**: it finds your store from the machine
registry, and if you keep more than one store it asks which to configure. The
wizard is non-destructive — it reports "No changes" when you keep everything.

```bash
./fornix-config                 # macOS / Linux / Git Bash
fornix-config.cmd               # Windows (also double-clickable)

./fornix-config --db /path/to/other.db   # advanced: target a specific store
python3 -m fornixdb configure            # same wizard, no launcher
```

## Disk budget

Never-delete is the default: with no cap set, nothing is ever removed — forgetting is only a ranking and tier effect. On devices where that's not affordable, cap the store's **total on-disk footprint** (db + WAL + cold archives) and choose what happens at the boundary:

```bash
python3 -m fornixdb config disk_budget_mb 500     # cap: MBs on a microcontroller … 1 TB on a workstation
python3 -m fornixdb config budget_policy freeze   # at the cap: refuse new memories (default)
python3 -m fornixdb config budget_policy prune    # at the cap: truly forget the least-salient memories
python3 -m fornixdb budget                        # footprint / headroom / policy
python3 -m fornixdb budget enforce --dry-run      # see what a pass would do
python3 -m fornixdb budget shrink 200             # ONE-SHOT: reduce the store to 200 MB right now
python3 -m fornixdb budget shrink 200 --dry-run   # preview what shrinking would forget
```

At the cap, mechanical tier escalation (compress, archive) always runs first; only if that can't fit the budget does the policy apply. **freeze** stops accepting new memories while keeping everything recallable. **prune** is the one true delete in FornixDB — choosing it is your explicit consent to forgetting, and it forgets least-important-first: tombstoned rows first, then the least-salient episodic detail, owner feedback last. Enforcement is purely algorithmic (no model needed).

There is also a **machine-wide cap** across every store on the box. A fresh install defaults it to **20% of free disk space, at most 2 GB** — never silently: the moment it is set, the CLI says so, and every AI surface keeps flagging it as the unreviewed install default until the owner sets it themselves (`config machine_budget_mb <MB> --shared`, or `off` to run uncapped; policy likewise). Each AI's store holds the line by fixing its own side only: it compresses and (policy prune) forgets its own least-salient memories; it never deletes another AI's. If that isn't enough, the write is refused with the per-store breakdown so the owner can shrink the right store. `fornixdb usage` shows every store, the total, and the cap.

**shrink** is the one-shot sibling of the cap — "reduce this space to X MB" — for when you want the store smaller *now* without setting a standing limit. It runs the same chain (compress, then truly forget least-salient-first, then vacuum) straight to the named target and leaves `disk_budget_mb`/`budget_policy` untouched. Asking for it is the explicit consent to forgetting, so it does not consult the boundary policy; if even forgetting everything cannot reach the target (the db file has a size floor), the result says so honestly.

A store can also be frozen outright, independent of any cap — `config frozen on` — for vendor-shipped, read-only memory DBs: recall works (without reinforcement writes), every mutation is refused. This is policy, not security; ship the file without write permission when you need a hard guarantee.

## Connecting an AI

The fastest path is MCP: `fornixdb-mcp` is a zero-dependency [Model Context Protocol](https://modelcontextprotocol.io) server over stdio, so any MCP-capable client connects with one config line (e.g. `claude mcp add fornixdb -- fornixdb-mcp`). For everything else, tooling is the standard way to add capabilities to a model, and FornixDB is designed to be reached through tools. **See [INTEGRATION.md](INTEGRATION.md)** for the recommended nine-tool surface (subject recall, time-axis recall, remember/update, list, forget, negative feedback, usage, shrink, startup context), the system-prompt guidance that makes small models use it well, and the two capture layers — **passive** episodic session capture (`session_capture on|off`, the shell remembers each session like a person remembers their day) and **interactive** semantic capture governed by the capture mode.

## Markdown in, Markdown out

FornixDB reads and writes Markdown — the format the AI ecosystem already lives in (Obsidian vaults, design docs, READMEs, Claude Code's own memory files).

**Import — a document becomes recallable by section.** `import-markdown` splits a Markdown file along its headings into one memory per section (gist = the heading, detail = that section's text), preserving the heading hierarchy as `refines` links and `[[wikilinks]]` as `relates` links. Point it at one file or a folder:

```bash
python3 -m fornixdb import-markdown notes/homelab.md        # an arbitrary doc, chunked by heading
python3 -m fornixdb import-markdown ./memory --frontmatter  # a folder of frontmatter memory files (one file = one memory)
```

Why chunk instead of storing the whole document? Because an AI re-reads whatever recall returns *on every turn*. A question whose answer lives in one section ("when does the backup run?") then comes back as that one small section instead of the entire document — measurably cheaper and more precise. The bundled walkthrough makes this concrete (no network, nothing written outside a temp folder):

```bash
python3 -m examples.markdown_bridge_demo   # ingest → recall → measured benefit → export
```

On the sample note it returns the right section ranked first every time and costs **~7.6× fewer tokens to answer** than the whole-document baseline; the ratio grows with document size. The benefit is guarded by `tests/test_markdown_benefit.py`, so it cannot silently regress.

**Export — your memory as readable, git-diffable files.** `export-markdown` writes one `.md` per memory (frontmatter + detail + a `## Related` links footer) plus a `FornixDB.md` index:

```bash
python3 -m fornixdb export-markdown ./memory-export
```

These are ordinary Markdown files: read or edit them in any editor, commit the folder to track how memory changed over time, or edit one and re-import with `import-markdown --frontmatter` to feed the changes back. Export round-trips with `import-markdown --frontmatter`. Both directions are also exposed as MCP tools (`import_markdown`, `export_markdown`) for clients that drive FornixDB through MCP rather than the shell.

## Security posture

A memory store is personal data, so the posture is explicit:

- **Local-first, no network.** SQLite files on disk, stdio transports, no listeners, no telemetry. The embedding model resolves from a local cache before any network source, so air-gapped installs are first-class.
- **OS boundaries, not application crypto.** Stores FornixDB creates are owner-only (`0600` files, `0700` for a created `~/.fornixdb`), and full-disk encryption (FileVault, BitLocker, LUKS) is the right layer against device theft. There is deliberately **no built-in database encryption**: an always-on agent needs the key resident on the same machine, readable by the same processes as the database — encryption at rest against a same-user attacker is theater, and we don't ship theater. What that means for you: keep disk encryption on, and **encrypt any backup or copy of a store that leaves the machine**.
- **SQL is parameterized throughout**, and free-text recall input is reduced to quoted tokens before it reaches FTS5 (`_fts_query`), so neither values nor FTS query operators can be injected. Contributors: every new query uses `?` parameters and FTS `MATCH` input goes through `_fts_query`, no exceptions.
- **Provenance over trust.** The store never verifies truth — it preserves where every memory came from (source, writer, supersede history) and surfaces it at recall (`[auto-captured]`, `[by X]` — see [INTEGRATION.md](INTEGRATION.md)) so the consuming model can judge. Recalled content is data about the past, never instructions.
- **The MCP server authenticates nothing itself.** It is a local stdio process: whoever can start it against a store file has that store's access. File permissions are the boundary; the MCP client's tool-approval prompt is the write gate.

## The name

The *fornix* is the brain's memory tract — the fiber bundle that carries what the hippocampus has stored out to the rest of the brain. FornixDB plays that role for an AI: the store that holds memories and the path the model reaches them through — never the thinker.

## License

MIT — see LICENSE.
