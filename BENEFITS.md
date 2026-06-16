# Why give your AI a local memory

The source document for FornixDB's benefit story — GitHub copy and website
pages draw from here. Every claim below is implemented and measured, not
aspirational.

## The problem it solves

An AI assistant forgets everything between sessions. The user pays for that
every day: re-explaining context, re-stating preferences, watching the AI
re-read files and re-derive decisions it already made. And some questions
can't be answered at all — "what day did the pool guy come by?" has no answer
in a stateless chat, no matter how good the model is.

FornixDB is a persistent local memory any AI can use. It runs entirely
on your machine, in a single SQLite file you own.

## What your AI gains

- **Recall by time.** "What did we do last Thursday?" / "what happened this
  morning?" — natural time phrases return everything from that window.
  Sessions are remembered automatically (a diary, kept by the shell, owner
  toggleable), so the answer exists without anyone deciding to save it.
- **Recall by meaning.** Paraphrases work — "the glitch where her eyes
  sparkled" finds the eye-twinkle memory with zero keyword overlap. No exact
  wording required.
- **A memory that learns.** New knowledge supersedes old without erasing it
  (the trail of corrections is kept), unused memories fade in ranking instead
  of cluttering it, frequently-used ones stay sharp, and explicit "not that
  one" feedback teaches recall what to stop surfacing.
- **Honesty flags.** Old, unverified facts come back marked stale; duplicate
  facts across stores answer once; a downweighted result says so.

## Private by construction

Nothing leaves your machine. There is no account, no cloud, no telemetry —
memory is a local file, readable and deletable by you. The core is
vendor-neutral: it works with any AI that can call a tool or run a shell
command (Claude, local Llama/Qwen models, anything MCP-capable), and the
store outlives any one vendor's product decisions.

## The token economics

Memory must earn its context space. Measured on a live store
(`fornixdb tokens` prints this for yours):

- Fixed cost: ~1,700 tokens once per session (tool schemas + startup
  context).
- Per recall: ~300 tokens at the default settings, hard-capped by a
  configurable character budget.
- What one recall replaces: the user re-explaining history by hand or the AI
  re-reading files and re-deriving past decisions — typically hundreds to
  thousands of tokens, every session — plus the time-axis answers that are
  otherwise impossible at any price.

For local models, prompt size is also response latency. FornixDB ships the
measuring tool, output budgets (`max_chars`), and lean-by-design tool
descriptions so the integration stays affordable on a laptop-class model.

## The owner stays in charge

- **Capture policy is yours:** explicit (only when you ask), suggest (the AI
  offers, you decide), or auto — set per store.
- **Never-delete by default:** "forgetting" is a tombstone, recoverable; the
  history of corrections is preserved.
- **True deletion only with your consent:** a standing disk cap with a
  prune-or-freeze boundary policy, and a one-shot "reduce this space to X MB"
  command. Both forget the way a person does — least-important first, your
  stated preferences and lessons last.
- **Scales to the device:** the cap works from megabytes on a microcontroller
  to a terabyte on a workstation; a vendor can even ship a frozen, read-only
  curated store.

## Built for more than one AI

Each agent gets its own store, plus a machine-level shared tier for the facts
every assistant should know ("prefers short answers in the morning" — tell
one AI, they all know). Stores are portable files; the architecture was
cold-installed and verified on Windows from the README alone.

## Transparent by design

The store can answer questions about itself, in plain English, through any
connected AI: how much disk space it uses (per AI and machine-wide), what
its token footprint is, what is remembered (titles → gist → full detail →
raw source, four levels of drill-down), and why a result ranked where it
did (staleness, downweighting, and duplicate flags travel with every
answer). Most memory systems are black boxes; this one is inspectable all
the way down — see "Anatomy of a memory" in the README.

## Engineering posture

- Keyword + time recall is dependency-free Python (SQLite + FTS5); associative
  recall ships by default as a ~30 MB CPU embedding model (no GPU, no torch, no
  network) and switches off (`FORNIXDB_VECTORS=off`) for lean or incapable
  hardware, falling back to keyword + time.
- One config line connects any MCP client; a six-line shim connects anything
  else.
- Recall quality is eval-fenced: a golden-query suite scores every ranking
  change before it ships (current: 89% top-hit, 100% top-5).
- The memory is substrate, not actor: it never decides or acts — your AI
  does, and every write path is gated by the owner's policy.
