# Changelog

All notable changes to FornixDB are recorded here. Versions follow semantic
versioning. While the project is pre-1.0 the public API may still evolve between
minor versions; pin a tag (`@vX.Y.Z`) for a stable checkout — `main` is the
active development branch and can change through the day.

## [Unreleased]

Fixes from dogfooding FornixDB as the live memory during a real (unrelated)
work session — see the in-use session report's recommendations.

### Added
- **`recent_writes` (MCP).** Lists the memories saved this session (this
  connection), in write order, marking any since superseded — a checkpoint view
  for end-of-session dedup/supersede review. Optional tool (trimmable via
  `fornixdb tools`).

### Changed
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
