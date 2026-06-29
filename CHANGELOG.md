# Changelog

All notable changes to FornixDB are recorded here. Versions follow semantic
versioning. While the project is pre-1.0 the public API may still evolve between
minor versions; pin a tag (`@vX.Y.Z`) for a stable checkout — `main` is the
active development branch and can change through the day.

## [Unreleased]

### Added
- **`fornixdb reproject` — re-derive project labels from CONTENT.** For a store
  whose auto-captured history was mislabeled (the pre-0.3.1 launch-dir bug, or any
  single-home setup where the working directory carries no project signal), cwd is
  the wrong thing to scope by — the portable signal is what a memory is ABOUT. The
  tool classifies suspect memories against per-project profiles built from the
  reliably-labeled (non-episodic) anchors: VECTOR mode (mean-centered centroids,
  so a broad catch-all project stops attracting everything) when the store has
  embeddings, KEYWORD mode (inverse-class-frequency weighted overlap) otherwise.
  Safe by construction: dry-run by default; only UNSCOPED (NULL-project) memories
  are reconsidered unless a known-bad label is named with `--suspect LABEL`, so a
  correctly-labeled session is never overridden because its transcript mentions
  another project; alias families are never churned; `--apply` records an undo set
  and `--undo` restores it. Pure classifiers (`classify_vec`/`classify_words`) are
  testable with no store.

### Fixed
- **Proactive recall now suppresses navigational session-openers, not just empty
  greetings.** The low-information filter that keeps bland auto-captured episodics
  out of proactive surfacing only caught near-empty greetings ("Chat …: Hello").
  Content-bearing-but-navigational openers ("come up to speed on X", "let's resume
  where we left off", "what's next") cleared the bar and leaked as noise — and,
  being near-identical, recalled against each other. The filter now strips the
  auto-capture scaffold ("Session <date> (<n> turns, branch <b>): …") and a
  navigation/greeting stoplist before counting distinct content words, so a pure
  pickup turn reads as low-information while an opener that ALSO states real work
  ("…finished the video pipeline between Mac and PC…") keeps its words and still
  surfaces. Curated (non-episodic) memories remain exempt. On the live store this
  drops 28 of 133 episodics from proactive push (was 0); all stay explicitly
  recallable.
- **L4 no longer pulses on FornixDB's own memory operations (self-recall guard).**
  The Claude Code rhythmic-recall adapter built its query from each tool call,
  including FornixDB's own MCP tools — so a `remember` self-matched the text it
  had just stored and a `show_memory {ref: N}` self-matched `#N`, surfacing the
  memory right back at cosine ~0.9. These degenerate hits were never useful and
  skewed the floor log's noise picture. `build_thought` now returns no thought
  for any FornixDB memory operation (matched by the `mcp__fornixdb__*` prefix, or
  by tool basename so a renamed MCP server is still caught), so the pulse is
  skipped. On a 3-day live floor log this removes 19 spurious surfacings (cos
  0.68–0.92) and stops 37 pulses from firing at all.

### Added
- **Floor logging — opt-in pulse-cosine instrumentation.** With `config floor_log on`
  (also offered in the `configure` wizard; default off), every candidate evaluated at
  the proactive/cadence relevance floor appends one JSONL record — channel (L3/L4),
  cosine, effective vs base floor, margin, and decision — to `floor_log.jsonl` beside
  the store. It captures below-floor near-misses too, not just what surfaced. No effect
  on recall behavior and a true no-op when off. Driven by store config, never an
  environment variable.
- **`fornixdb floor-stats`** — turns the floor log into cosine distributions, floor-dial
  activity (how often the effective floor is raised/lowered from the base), and an
  evidence-based floor recommendation. It joins each surfaced memory to its use outcome
  (ever used vs pushed-but-never-used) and suggests a data-driven floor — or honestly
  reports insufficient evidence / no lossless cutoff when the data doesn't support one.

## [0.3.1] — 2026-06-25

### Fixed
- **Auto-captured sessions are scoped to the project actually worked in.** The
  two transcript ingesters (batch back-fill and the live session-end hook) used
  to derive a session's `project` from the `~/.claude/projects/<dir>` name,
  which encodes the session's *launch* directory. In a setup where every session
  launches from one home directory, that mislabeled all work with that
  directory's name. The project is now read from the session's own recorded
  working directory, falling back to the directory-name parse only when no cwd
  is recorded.

### Added
- **`remember` / `remember_many` take an optional `project`.** A manually stored
  memory can now declare the project it belongs to (the batch form scopes the
  whole batch, with an optional per-item override). When omitted it inherits a
  pinned `config active_project`; otherwise it is left unscoped, as before.

## [0.3.0] — 2026-06-25

Recall-quality and noise-reduction work, plus a configurable operating surface —
fixes from dogfooding FornixDB as the live memory during real (unrelated) work
sessions. All new behavior defaults on and is reversible.

### Added
- **`doctor` config-integrity check.** `doctor` now flags any config key SET in a
  store that **no code reads** — a typo (`proactive_reall`), a stale/removed
  setting, or a key set expecting an effect it can't have (the "declarative-only"
  class, e.g. a dial that was never wired). The reader set is scanned from source
  so it stays current; dynamic per-session keys and indirectly-read ones are
  recognized, not flagged. A dev-time test additionally guards that every
  operating-level dial and behavioral toggle actually has a runtime reader, so a
  declared-but-unwired dial fails CI rather than silently doing nothing.
- **Cross-pulse dedup (L3 + L4 share one anti-repeat set).** A memory pushed
  once this session by *either* rung — the per-turn L3 hook or an L4 tool-call
  tick — is no longer pushed again that session. Previously L3 deduped per
  session and L4 only per turn, and the two didn't share, so the same memory
  could be re-injected. They now read/write one shared per-session set, so a
  pulse never repeats what you've already been shown (it's still in context from
  the first push, and explicit `recall_memory` ignores the set). Reversible via
  `config cross_pulse_dedup off`.
- **Project-scoped pulse recall.** The other half of the cross-project noise
  fix: when a pulse knows its active context, a memory that doesn't *belong* to
  it clears a higher push floor — so off-context memories stop leaking into the
  ambient stream on weak matches, while a strongly-relevant one (high cosine)
  still surfaces. "Belongs" unifies both axes — a memory is on-context if its
  **project OR any of its topics** matches the active label or an alias — so the
  two signals reinforce instead of competing. Memories with no scoping tags
  (general/curated facts) and ones tagged only with structural topics
  (`reference`, `feedback`, …) are never penalized. PUSH-only — explicit
  `recall_memory` still searches everything. Reversible via
  `config project_scoped_pulse off`.
  - **The active context is learned three ways**, by precedence: a pinned
    `config active_project` > **what you declare in a prompt** ("continue the
    fornixdb project" / "switch to videos" — detected from a cue phrase + a known
    project name, then sticky for the rest of the session) > the host's working
    directory. Declaration is what makes scoping work when every session runs
    from one directory.
  - **Aliases** (`config project_aliases "fornixdb=engramdb,aimemory; videos=elira"`)
    bridge a project's messy historical names so one declaration catches them all.
- **Usefulness-feedback loop (closes the PUSH side).** A proactively pushed
  memory now accrues a `surfaced_count` (an *impression*) kept strictly separate
  from `recall_count` (a genuine *pull*) — proactive candidate-gathers no longer
  inflate the recall count (`recall(count_recall=False)`). Ranking now also gives
  a small saturating bonus for passive recalls (well below an explicit "this
  helped"), and the proactive/rhythmic relevance floor is now **per-memory**: a
  proven-useful memory clears a slightly lower bar, while one pushed many times
  but never used clears a higher one — the implicit signal that quiets
  cross-project noise in the ambient stream. Bounded and additive; never hides a
  memory (explicit recall ignores the push floor). Reversible via
  `config usefulness_floor_adapt off`. Schema → v7 (`surfaced_count`,
  `last_surfaced`; auto-migrates).
- **`recent_writes` (MCP).** Lists the memories saved this session (this
  connection), in write order, marking any since superseded — a checkpoint view
  for end-of-session dedup/supersede review. Optional tool (trimmable via
  `fornixdb tools`).
- **L4 rhythmic in-thought recall (first build).** `fornixdb.cadence` — a
  portable "metronome" controller that fires many recall pulses within one
  reasoning episode (event-driven debounce, per-episode dedup, a relevance floor
  a notch above the L3 push). Host-neutral; wired into a local model's inner
  tool-loop (Elira) as the reference caller. Shared the L3 relevance gate +
  formatter by extracting them into a vendor-neutral `fornixdb.proactive`
  module. Switches: `ingest_mode=explicit` / `config rhythmic_recall off`. Also
  wired as a Claude Code `PostToolUse` hook
  (`fornixdb.adapters.claude_code_cadence`) so the metronome rides the tool-call
  seam on any OS Claude Code runs on.
- **Operating-levels ladder as a configurable surface (`fornixdb level`).** The
  ROADMAP's L0–L6 ladder is now something you set: `level` shows the rungs and
  the current one, `level L3` selects a rung, `level L4 off` toggles one. The
  ladder is cumulative (a rung is on only if every rung below it is) and each
  rung is a view over the dial that drives it. L1 (associative recall) is
  toggleable, so a constrained device can sit at a pure keyed-store L0.
- **Interactive configuration wizard (`fornixdb configure`).** Walks every
  setting — operating level, capture style, session capture, vectors, ingest
  mode, disk budget/policy, and MCP tools — showing each current value and its
  default; collects your choices, shows a diff, and writes only after you
  confirm (Ctrl-C aborts clean). A `./fornix-config` launcher runs it with one
  command (no venv activation or `$FORNIXDB_DB` needed; defaults to the repo's
  store).
- **Read-only `config` overview.** Bare `fornixdb config` now prints every
  option with its `[default: …]` plus the full ladder — a no-change way to see
  the whole configuration at a glance.
- **`fornixdb tools --full`.** Prints each MCP tool's complete explanation (not
  truncated); the wizard's MCP-tools step (keep / minimal / custom) shows the
  same explanations inline while you choose which tools to advertise.

### Changed
- **L1 associative recall is a real runtime toggle.** With
  `associative_recall=off` (rung L0) `recall()` collapses to exact keyed-name
  lookup only — no FTS/vector ranking — matching the ROADMAP's "L0 = exact
  lookups, no ranking." `show_memory` by id/name still works.
- **Default filenames are FornixDB-branded.** The default store is now
  `~/.fornixdb/fornix.db` (was `memory.db`), the shared tier `fornix-shared.db`,
  and the Markdown export `FornixDB.md` — so a FornixDB store is never mistaken
  for a host AI's own memory file. Existing stores (referenced by explicit path
  or `$FORNIXDB_DB`) are unaffected.
- **L4 relevance floor lowered 0.60 → 0.50** after measuring genuine
  signal/noise separation on a live store (noise ≈ 0 cosine vs signal 0.42–0.92);
  still a notch above the L3 once-per-turn floor.
- **`kind` accepts native-taxonomy aliases.** `remember(kind="project")` /
  `kind="user"` no longer bounce — both map to `semantic` (standing knowledge).
  A model reaching for a host's native memory kinds just works; an unknown kind
  still errors, now with the offending value and the known aliases in the
  message. Documented in the `remember` tool description.
- **README** now shows *how* to trigger supersede-by-same-title (re-`remember`
  under the same title), the quiet feature that keeps an evolving finding as one
  current memory.

## [0.2.0] — 2026-06-17

The first tagged release after the initial publish. Adds semantic recall by
default, ambient (proactive) recall, a usefulness-feedback loop, a Markdown
bridge, and a batch of capture conveniences — plus Windows-correctness and
tooling fixes from real-world dogfooding.

### Added
- **Semantic recall on by default.** `model2vec` ships as a default (unpinned)
  dependency; new stores embed on write, and a pre-vectors store backfills itself
  on first use. Disable with `FORNIXDB_VECTORS=off` / `config vectors off`, or it
  falls back automatically on hardware that can't load a model.
- **Proactive recall injection (L3).** A UserPromptSubmit hook
  (`fornixdb.adapters.claude_code_recall`) surfaces a provenance-tagged
  "possibly-relevant past" block each turn, additive to the host's own memory.
- **Per-memory usefulness feedback.** `mark_helpful` endorsements plus recall
  hit-count / last-recalled, rolled up at session start and fed into ranking.
- **Bidirectional Markdown bridge.** Import arbitrary heading-chunked documents
  and export memories to Markdown — `import-markdown` / `export-markdown` (CLI)
  and `import_markdown` / `export_markdown` (MCP).
- **Capture conveniences.** `remember_many` (batch), `jot` / `review_candidates`
  (cheap staged capture), and auto-linking of `[[wikilinks]]` with a near-dup
  nudge at store time.
- **Additive native-memory following** via `ingest_mode` (never a takeover).
- **Configurable MCP tool surface** — all tools on by default, optional ones
  trimmable.
- **Warm-embedder seam** for dedicated hardware — inject a long-lived embedder
  via `FORNIXDB_EMBEDDER="pkg.module:factory"` or `set_default_embedder()`
  instead of the default cold per-turn load (see INTEGRATION.md).
- **`config` + `doctor`.** `config` shows every setting at once with its
  suggested default; `doctor` health-checks the install (schema currency, wired
  hooks, config smells) and can apply suggested defaults.

### Changed
- **Install-default machine cap raised 500 MB → 2 GB** (a ceiling; the
  `min(cap, 20% of free disk)` rule is unchanged).
- **Proactive recall gates stricter than explicit recall** (`PROACTIVE_RECALL_COS`
  = 0.45, vs the 0.30 include floor for `recall_memory`) and, when vectors are
  on, no longer admits keyword-only anchors — cutting wrong-topic noise from the
  ambient block. Override per store with `config proactive_recall_floor`.

### Fixed
- **Windows data loss in `export-markdown`:** a colon (or other Windows-illegal
  character) in a memory name routed content into an NTFS Alternate Data Stream,
  leaving a 0-byte file. The filename sanitizer now strips the full illegal set.
- **Windows mojibake:** the MCP server and both Claude Code hook adapters now
  force UTF-8 on stdio, so non-ASCII (`—`, `→`, accents…) is no longer corrupted
  at rest or in the recall block on cp1252 hosts.
- **`eval` / `answer-eval` empty-store guard:** running against a 0-memory store
  now fails loudly (exit 2) instead of reporting a false 0% that looked like a
  total regression.
- **Quieter recall:** suppressed the Hugging Face "Fetching N files" progress
  bar that printed to stderr on every recall.

## [0.1.0] — 2026-06-15

Initial public release under the MIT license: the SQLite spine (bi-temporal,
FTS5, topics, typed links, sessions), ranked subject recall and natural-time
timeline recall, supersede-with-history, salience/decay, reversible tiers and
disk budgets, the consolidation ("dream") pass, the shared machine tier, the MCP
server, and Claude Code passive session capture.

[0.2.0]: https://github.com/itdtllc/fornixdb/releases/tag/v0.2.0
[0.1.0]: https://github.com/itdtllc/fornixdb/releases/tag/v0.1.0
