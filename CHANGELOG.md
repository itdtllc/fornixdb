# Changelog

All notable changes to FornixDB are recorded here. Versions follow semantic
versioning. While the project is pre-1.0 the public API may still evolve between
minor versions; pin a tag (`@vX.Y.Z`) for a stable checkout — `main` is the
active development branch and can change through the day.

## [Unreleased]

## [0.8.3] - 2026-07-09

### Fixed
- **Every connection now sets `PRAGMA busy_timeout=5000`.** WAL lets readers and
  a writer coexist, but a checkpoint or a recall that writes could still collide
  with a concurrent writer — most sharply when the live watch thread commits
  keyframes while the main connection recalls — and with no busy-timeout that
  collision was an immediate `SQLITE_BUSY`. SQLite now waits up to 5s and
  retries instead of erroring.
- **Camera-index probe no longer prints backend chatter.** Probing an index past
  the last real camera made AVFoundation shout `out device of bound` /
  `camera failed to properly initialize!` straight to fd 2 (bypassing Python and
  OpenCV logging). The probe — which already handles the failure via
  `isOpened()`/frame checks — now runs under an fd-2 silencer, so `look` / watch
  start cleanly on a single-camera Mac.

## [0.8.2] - 2026-07-09

### Added
- **The live senses are now MCP tools — `look`, `feel`, `see`, `recaption` —
  default-OFF.** An MCP client (Claude, Elira, any) can now perceive through the
  standard tool interface, not just the CLI: `look` (one-shot "what do you see
  right now?"), `feel` (capture the machine's battery/body state as a tactile
  memory), `see` (caption + remember an image on disk), `recaption` (dream-pass
  fill of watch placeholders). Because they reach hardware (a camera, the
  battery) and a local model, they ship **off** — a new default-off tier
  (`DEFAULT_OFF_TOOLS`, tracked via a separate `mcp_tools_enabled` key, tier
  label `sense`). Turn them on per store with `fornixdb tools enable <name>` or
  in `fornixdb configure`, where they now appear as toggles. A disabled sense
  stays callable if a client already knows it; default-off only keeps it out of
  `tools/list` (the prefill), so the memory-tool footprint is unchanged for
  stores that don't opt in. Each fails loudly (a clean tool error, never a
  crash) when the camera or model is unavailable.

## [0.8.1] - 2026-07-09

### Added
- **Look once, right now — `senses.glance` + `fornixdb look`.** A synchronous,
  gate-free counterpart to `watch()`: grab the CURRENT frame from a source
  ("camera" / "camera:N" / "screen" / a video path), caption it immediately with
  a local VLM, and return the sentence — the answer to "what do you see right
  now?". Unlike the watch loop it never waits for a salient scene change, so an
  assistant asked on demand describes the moment of asking, not the last thing
  that crossed the gate. Ephemeral by default (caption the frame, delete the
  still, write nothing); `remember=True` records it as a `see` memory and, like
  a live watch commit, drops the still to the `vector + gist` rung unless
  `keep_keyframe`. CLI: `fornixdb look [--source S] [--model ID] [--remember]
  [--keep-keyframe]`.

### Fixed
- **`watch` rejects `drop_keyframe_after_commit` without a captioner.** The
  combination stranded a placeholder the dream pass could never fulfil (the
  still is gone, but only a templated gist was written). `run_watch` now raises
  `ValueError` instead of silently stranding it — the two are only coherent
  together (caption inline, then drop).

## [0.8.0] - 2026-07-09

### Added
- **Live watch controls — `should_continue` and `drop_keyframe_after_commit`.**
  `watchloop.run_watch` / `senses.watch` gained two options for running the
  watch loop as a background "live eyes" thread rather than a fixed-duration
  capture. `should_continue()` is polled once per frame and stops the loop
  cleanly when it returns False (the stop signal for a thread with no
  `max_seconds`); on stop the frame generator is now closed immediately so an
  adapter's release (e.g. the camera's `cv2.release`) runs at once instead of at
  GC. `drop_keyframe_after_commit` deletes each keyframe and nulls its
  `source_ref` once the caption gist and modal vector are stored — the fidelity
  ladder (`keyframe+vector+gist → vector+gist`) applied at capture time, so live
  capture with an inline `captioner` leaves no stills on disk while
  same-modality recall (the surviving vector) still works.
- **Dream-pass captioning for the watch loop — `fornixdb.recaption`.** The
  watch loop keeps its 2 Hz path model-free: every committed keyframe lands
  under a *templated* placeholder gist (`watch[screen]: scene change`). This new
  pass fills the real ones. `recaption.pending_captions(store)` is the worklist
  (live sight memories still holding a placeholder, oldest scene first, keyframe
  verified on disk); `recaption.recaption(store, captioner)` runs a local VLM
  `(keyframe_path) -> str` over each and rewrites the gist in place via
  `set_gist` (re-embedding the text lane) — so a *text* consumer can finally
  recall what was seen. Pure and injectable like `watchloop`: the captioner is a
  callable, nothing here imports a model, and the pass is idempotent (keyed on
  the placeholder text), non-destructive (a decayed keyframe or an empty caption
  is skipped, never clobbers), and crash-proof (a per-frame captioner error is a
  skip, not an abort).
- **`fornixdb recaption` CLI** — `fornixdb recaption [--dry-run] [--limit N]
  [--model ID]`. `--dry-run` lists the keyframes awaiting a caption with no VLM
  loaded (model-free); a real run captions them and prints each rewrite. The
  `dream` report now surfaces the same worklist (`👁 N watch keyframes still
  hold a templated placeholder`) so the perceptual backlog shows up alongside
  the text consolidation work — dream itself stays stdlib-only; captioning is
  the separate model-bearing command.
- **Ollama-backed captioner — `mac_vision.vlm_captioner(model="qwen2.5vl:7b")`.**
  A reference captioner that is *pure stdlib* (urllib POST to the local Ollama
  daemon), so the `[mac]` extras don't grow. `ollama pull <model>` is the only
  setup; any permissive local vision model serves (qwen2.5vl / llama3.2-vision /
  minicpm-v / moondream). Defaults to `qwen2.5vl:7b` (Apache-2.0) — on a
  same-frame A/B it read scene detail llava missed. Local-first (defaults to
  localhost) and fails loudly with an actionable message when the daemon is
  unreachable or the model isn't pulled.

## [0.7.1] - 2026-07-08

### Added
- **The watch loop has Mac stream adapters — `senses.watch()` now runs.** P2
  of the watch() design. `fornixdb.adapters.mac_camera` yields
  `(timestamp, frame)` pairs from the webcam (`cv2`), the screen
  (`screencapture`, zero-dependency), or a video file; `fornixdb.adapters.mac_vision`
  embeds each frame with a CLIP image tower on MLX (Apple-silicon-native — the
  same vectors feed both the salience gate's hot path and the latent lane).
  `senses.watch(store, "camera"|"screen"|<file>)` resolves the source, builds
  the embedder, and drives `watchloop.run_watch`; committed keyframes only land
  under `<store_dir>/senses/watch`. Everything Mac-specific imports lazily, so
  the core and the whole test suite stay stdlib-only.
- **`fornixdb watch` CLI** — `fornixdb watch --source screen|camera|<file>
  [--seconds N] [--rate HZ] [--window S] [--threshold D] [--max-commits N]
  [--project P]` prints each committed frame as it lands
  (`#id  reason  span  gist`) and stops cleanly on Ctrl-C. Owner-started only —
  no watch MCP tool, no always-on capture, and the default source is `screen`
  so a bare invocation never opens the camera. A frame commits when its CLIP
  distance from the recent scene exceeds the salience threshold (default
  `0.20`, field-tuned for camera+CLIP; `--threshold` overrides). Captions are
  templated at commit (a dream pass fills real ones later — the hot path stays
  model-free).
- **Optional `[mac]` extras** — `pip install 'fornixdb[mac]'` adds the
  camera/embedder dependencies (`opencv-python`, `pillow`, `mlx`,
  `huggingface-hub`, `numpy`), all imported lazily inside the adapters only.
  The watch embedder is a standard CLIP image tower on MLX: the CLIP code is
  vendored from Apple's mlx-examples (MIT), and weights are pulled once from
  the permissively licensed `mlx-community` repos (Apache-2.0). No PyPI package
  in the extra pulls in a heavy tree, and nothing is GPL or unlicensed.

## [0.7.0] - 2026-07-07

### Added
- **The watch-loop core is real** (`fornixdb.watchloop.run_watch`) — P1 of
  the watch() design: pure and injectable (no camera, no clock, no sleep),
  it drives any adapter's `(timestamp, frame)` iterator through the salience
  gate and turns each commit into a `see` memory with a real event-time
  span. Frames that never commit never touch disk (bytes frames are written
  under `keyframe_dir` only at commit time; path frames pass through);
  windows cut at boundaries with `window_seconds` as the maximum; captions
  come from an optional local captioner, else a recallable templated gist a
  later dream pass can upgrade; gate lane (`embed(frame)`, fast, pre-file)
  and latent lane (`ModalEmbedder`, committed keyframes only) are separate
  on purpose. `see()` gained an optional `event_time_end` so a sight memory
  can carry a span. `senses.watch()` still raises honestly — what remains is
  the stream-source adapter layer (camera / screen / file).
- **`senses.feel()` — machine proprioception, no robot required.** Remember
  one sensor reading (power source, charge, thermal, lid) as an ordinary
  episodic memory: a reading is a dict of named values (or a plain string),
  the gist is a templated state line, and NO embedder is needed — the gist
  lane alone makes it recallable ("when did the laptop go on battery?"). The
  full dict lands in `detail` as JSON; the source is `sensor:<name>`. Robot
  endpoints (force, contact, IMU) are the same pattern, later adding a
  sensor-domain `ModalEmbedder` for the latent lane.
- **The feel-loop core is real** (`fornixdb.feelloop.run_feel`) — the
  proprioception twin of the watch loop: pure and injectable (no sensor, no
  clock, no sleep), it drives any adapter's `(timestamp, reading)` iterator
  through a plain field-diff gate and turns each commit into a `feel` memory.
  It commits the first reading, any reading whose watched fields change, and
  a heartbeat after a quiet stretch; `ignore_fields` drops noisy drift (a
  minute-to-minute `percent`) from change detection while still recording it.
  Ships with a reference Mac adapter (`fornixdb.adapters.mac_proprioception`)
  that reads power/battery state via `pmset` — pure `parse_batt` core plus a
  thin subprocess wrapper and a coarsened frame generator.
- **`fornixdb feel` — proprioception from the command line.** `fornixdb feel`
  captures the Mac's power/battery state now; `fornixdb feel "lid closed"
  --sensor lid` captures any literal reading (platform-neutral); `fornixdb
  feel --live [--seconds N]` runs the change-gated loop, printing each commit
  and stopping cleanly on Ctrl-C. `examples/feel_demo.py` shows the same
  machinery a level down against a throwaway store.

## [0.6.0] - 2026-07-06

### Added
- **`fornixdb tokens --billed` — the billed (token-turn) view, measured from
  your host's own transcripts.** `tokens` counts each token once
  (context-space), but a stateless host re-sends the whole conversation on
  every API request — one request per tool call, 30–150+ per session — so
  host usage panels attribute *token-turns*, 30–150× the once-counted
  figure. The new report reads the per-request `usage` records in the
  host's transcript JSONL (the same numbers its usage panel aggregates) and
  weighs every proactive push (per channel L3/L4/L5), `mcp__fornixdb` tool
  call, and id-matched tool result by the requests that re-read it; the
  resident schema/instruction surface is shown as a deferred…eager band,
  since transcripts don't record how the host loaded schemas. Live 7-day
  readout on the reference store: all FornixDB content = **1.9%** of 625M
  billed tokens, nearly all ~0.1×-price prompt-cache reads. `tokens`,
  `value`, README, and BENEFITS now state both units explicitly — the
  multiplier applies to the cost *and* savings sides equally, so the net
  verdict's direction survives the unit change.
- **The senses capture core: `see` and `hear` are real** (single artifacts;
  the live `watch`/`feel` loops remain honest TBD). `see(image)` stores a
  percept the SENSES.md way — caption as gist (a passed string or a local
  captioner callable; the gist embeds through the ordinary text path, so
  cross-modal recall works immediately), the file on disk as `source_ref`
  (never the blob; a missing file is refused, no dangling pointers), and an
  optional `ModalEmbedder` vector in the new latent lane. `hear(audio)` runs
  **two lanes with the non-speech one required**: sound carries meaning with
  zero words (a crosswalk beep, whistling, a kettle), so every sound memory
  needs a sound-scene caption, and speech — when present — rides *with* it
  (gist leads with the scene and quotes the words; the full transcript goes
  to detail). A transcript alone is rejected as an incomplete memory of a
  sound.
- **`fornixdb.salience` — the gate that turns dense sampling into sparse
  memories.** Pure and hardware-free: EMA-referenced embedding distance,
  one commit per scene change (hysteresis re-arm), heartbeat commits to
  anchor quiet stretches, caller-supplied clock. The capture loops will feed
  it; anything producing vectors can use it today.
- **Schema v10: `modal_embedding` (new table only).** The latent lane — a
  perceptual memory's modality vector, one row per model, beside its
  ordinary caption embedding; similarity (`senses.modal_neighbors`) is
  scored strictly within one model's space, so embedding spaces never mix.
  The hot text path is untouched.
- **SENSES.md — the multimodal design, published.** The architecture behind
  the declared-TBD `see`/`watch`/`hear`/`feel` entry points: a RAM-only
  sensory buffer sampled densely, a salience gate that commits only on
  divergence from the recent past (plus heartbeat anchors), and ordinary
  store rows (caption gist + `ModalEmbedder` vector + `source_ref` artifact
  + real event-time spans). Two independent recall lanes (captions through
  the existing text space = cross-modal recall from day one; per-modality
  latents for similarity and gating), a fidelity-ladder decay
  (clip → keyframe → vector → gist-only), and `feel()` provable with no
  extra hardware as machine proprioception. Design only — the stubs still
  raise `NotImplementedError`; README and the stub message now link the doc.

## [0.5.0] - 2026-07-04

### Changed
- **L5 parallel multi-domain activation is now DEFAULT-ON.** `parallel_recall`
  reads **on** when unset, so a fresh store starts at rung **L5**: every L4 beat
  gathers through the field (seven domain-scoped recalls settling by
  corroboration) unless the owner steps back down (`level L4`, or
  `config parallel_recall off`). 0.5.0 was reserved for exactly this flip.
  The evidence behind it, from live dogfooding on a 476-memory store:
  - **No harm**: in the floor-log join against scan outcomes, 783 of 818
    surfaced pushes scored *useful* vs 35 *noise* (~96%) with L5 contributing
    the large majority of floor evaluations; field beats cost ~307 ms median
    (343 ms max) and honesty properties held (no corroboration → clean L4
    degrade, nothing fabricated).
  - **The gate readout keeps accruing**: the usefulness scan has not yet
    banked enough L5-settled impressions for the L5-vs-L4 reference-rate
    comparison (scan attribution credits the most-recent injecting channel,
    which leans the split toward L3/L4). That readout — the same gate L4
    passed — now serves as the **revert signal**, surfaced by the dream dial
    report; the flip is reversible with one config.
- **Ladder surfaces reflect the new status**: `level` shows L5 as built (no
  more "under evaluation" tag), and the dial report labels the gate
  `on (default)` and phrases its AGAINST branch as a revert suggestion
  (`config parallel_recall off`) instead of "keep dogfooding".

## [0.4.3] - 2026-07-03

### Added
- **`fornixdb value` leads with a net token verdict.** The report's first line
  is now *"Estimated tokens SAVED/EXTRA: ~N/session"*, followed by the
  supporting data: a **measured** cost side (fixed integration surfaces plus
  the actual size of every proactively injected memory block, summed from the
  host's own session transcripts — `usefulness-scan` now records per-push
  block sizes, totals and per-channel) against an **explicitly assumed**
  savings side (a printed low/mid/high band per downstream-referenced push;
  there is no session-without-memory to compare against, so the verdict is a
  range, never one confident number). The report also states what it does not
  count (explicit pull results; session-end capture costs zero prompt tokens;
  time-axis answers have no re-derivation path), and reminds the owner that
  `config floor_log on` collects per-push / per-beat detail
  (`floor-stats` / `field-stats`) beyond what transcripts alone show.
- **L5 parallel multi-domain activation — the field (first build, default OFF).**
  With `config parallel_recall on`, each L4 beat gathers through `fornixdb.field`
  instead of a single recall: seven domain-scoped recalls (semantic knowledge /
  recent episodes / deep past / feedback guidance / reference / active-project
  context / 1-hop link-spread neighborhood) fire on the same evolving thought
  sharing ONE query embedding, then settle by corroboration clustering — rows
  several domains return, or that link/topic-connect across domains, form the
  pattern, emitted under a descriptive `settled:` direction line. Honesty
  properties: every row clears the same per-memory effective floor as an L4
  pulse; the neighborhood is corroboration-only (query-free rows can never
  surface alone); no corroboration degrades to plain L4 behavior; nothing
  clearing the floors stays silent. L4 owns WHEN (debounce, episode budget,
  dedup — unchanged); L5 owns HOW WIDE. The dial ships off while L5 walks the
  same scan-verified usefulness gate L4 passed before its default flipped.
- **`fornixdb field "<thought>"`** — debug view of one field beat: every
  domain's returns, corroboration scores, clusters, and the block it would
  inject (read-only).
- **Per-beat field telemetry + `fornixdb field-stats`.** With `floor_log on`,
  each field beat appends one record to `field_log.jsonl` beside the store:
  settle/degrade/abstain, which domains lit the winning cluster, link-vs-topic
  glue, the dissent *shadow* (what the minority-report line would have shown
  while `parallel_dissent` is off — so that decision can be made from data),
  emitted ids, and wall time. `field-stats` renders settle rate, domain
  contribution, glue split, shadow counts, and cost.
- **L5 instrumentation is channel-aware end to end**: field floor decisions log
  `channel="L5"` (with the L5-specific `cleared_not_settled` outcome so
  "surfaced" keeps meaning *actually injected*), and the usefulness scan
  tallies SETTLED field blocks as channel L5 — a degraded field block is L4
  behavior and is credited to L4.
- **Ladder and config surfaces know the new rung**: `level` shows L5 (dogfood,
  dial `parallel_recall`, reads off when unset), the `configure` wizard derives
  its rung menu from the ladder (every built rung offered; planned stay off the
  menu) and asks the one L5 behavior choice (`parallel_dissent`) only at L5,
  and the read-only `config` view gained the `parallel_dissent` row.
- **Dream housekeeping — push-noise gets both of its halves.** The mechanical
  half: opening a dream pass now refreshes the push use-credit automatically
  (`usefulness-scan --apply` folded into the dream) so the floor's credit side
  never goes stale while its penalty side accrues at push time. The pairing is
  EXPLICIT: it runs only on a store whose `transcripts_path` config names the
  host's transcript dir (`fornixdb config transcripts_path ~/.claude/projects`
  on the one store the host's hooks inject from; env `FORNIXDB_TRANSCRIPTS`
  overrides, `off` skips, `dream_use_credit off` hard-disables) — transcript
  `#id`s belong to that store and ids collide across stores, so an implicit
  default would write phantom counts onto every other store on the machine
  (caught live on the second store's rows during first cross-store testing).
  The judgment half: the dream worklist gained a **chronic push-noise**
  section — live rows pushed ≥6 times with zero downstream use (endorsements
  and scan-verified references both count; lifetime pulls are reported but
  never exempt). The per-memory floor already quiets these mechanically; the
  dream asks whether they should keep living: forget/supersede if obsolete,
  `reproject` if mis-scoped, or accept with `tag <id> noise-ok` (the
  `reality-ok` analogue). Propose-not-dispose as ever — the reviewing AI/owner
  decides.
- **Dream worklist: mis-scoped memories.** `reproject`'s confident
  content-based proposals (unscoped/suspect rows whose content points at a
  project) now surface as a dream section, best margin first — the other root
  of cross-project push noise, caught in the same pass as its symptoms. Apply
  via `reproject --apply` (undo-able) or relabel the accepted rows.
- **`distinct` links — accept a reviewed pair (the pair-level
  `reality-ok`/`noise-ok`).** A contradiction/merge/resolution pair the
  reviewer judges legitimately distinct kept re-appearing in every dream pass
  and wake nudge. `link <a> <b> --relation distinct` (CLI) or the `link` MCP
  tool with `relation="distinct"` marks the pair reviewed — never re-proposed,
  either direction, both rows stay live and recallable. Schema v9 rebuilds
  `memory_link` in place on older stores (its CHECK bakes the relation list);
  rows are preserved.
- **Dream dial report — sleep as self-review of the dials.** Every dream reads
  the accrued telemetry back as evidence-attached config PROPOSALS (never
  applied): `parallel_dissent` when the field log shows a minority report
  existed on many settled beats but was never shown; the `parallel_recall`
  gate readout (scan-verified L5 settled-push reference rate vs L4, FOR /
  AGAINST / still-accruing) once both channels clear an evidence minimum; and
  a push-floor suggestion when scan-labeled useful/noise cosines separate
  cleanly. Each rule has an evidence floor so a thin log cannot produce a
  confident-sounding lie; the scan-derived rules run only at pass open, from
  the same scan the use-credit refresh already paid for.

### Changed
- **L4 rhythmic recall is marked BUILT** (was "under evaluation"): its
  usefulness gate resolved — scan-verified downstream reference rate for L4
  pushes beats the per-turn L3 channel (20% vs 13%) on lived-in usage.

## [0.4.2] - 2026-07-02

### Fixed
- **Push floor runs on push-outcome evidence only.** `effective_floor`'s
  usefulness dial counted `recall_count` toward `uses`, so on a lived-in store
  the listing-era inflation saturated the discount for every row and the
  ignored-noise penalty never fired (measured: 190/324 rows at max discount,
  zero penalties across the 75 rows pushed ≥3× and never used). Floor `uses`
  is now `helpful_count` + `referenced_count` — endorsements and scan-verified
  downstream use. Pulls are the other channel: a pulled memory needs no
  pushing to be found, and explicit recall ignores the floor entirely.
- The `set-gist` CLI message claimed the vector was dropped and needed a
  manual `embed`; since 0.4.1 the gist is re-embedded in place (the message
  now says so).

## [0.4.1] - 2026-07-02

### Fixed
- **Rich-get-richer crowding: listing is no longer engagement.** Every `brief`
  and `timeline` sweep counted each *listed* row as a recall, which both pumped
  `recall_count` into the hundreds and refreshed `last_recalled` — the decay
  anchor — so chronically-listed rows never decayed and new memories lost at
  relevance parity (measured on the live store: winners at eff 0.7–0.85 with
  recall counts no honest use could produce, e.g. 269, vs new rows at 0.5).
  Listings now record impressions (`surfaced_count`), the same currency as the
  proactive push path; genuine pulls (`recall`) still count. The ranking
  usefulness bonus switches its second term from `recall_count` to
  `referenced_count` (scan-verified downstream use) — pull ranking and the push
  floor now run on one honest use signal (`helpful_count` + `referenced_count`).
  Eval fence: hit@5 81%→86% (two crowded-out rows reclaimed), hit@1 61%→58%
  (the one lost @1 was itself a contamination-funded win), MRR flat, all
  abstains held.

### Added
- **Reality check: review verbs + fewer false flags.** The first full review
  of a live store's flags shaped the workflow: rows tagged `reality-ok` are
  reviewed-and-accepted (a historical mention, a documented default, a
  described absence) and stay accepted on every future dream; the CLI/MCP
  sections say so. Extraction now clears a path with a SPACE in a segment
  (`Test Cases/…` truncated at the space and read as missing) by testing
  space-extended candidates, and skips matches truncated by a placeholder
  (`AppStore/v<X.Y.Z>`). On the live store: 11 flags → 4 false positives
  cleared by extraction, 6 accepted, 1 genuine rot superseded → 0.
- **Reality check in the dream pass — memory grows its first sense organ.**
  A memory that points at the filesystem can silently rot: the file moves or is
  deleted and the pointer stays live and recallable (motivating case: a design
  doc lost in a disk reorg whose pointer memories sat stale for two weeks).
  `consolidate propose` / `dream` now verify file-path claims in live
  non-episodic memories against the world and list pointers to missing files
  (CLI + MCP sections, dream narrative line). Propose-not-dispose as always: a
  missing path may be an unmounted volume or a moved file — the reviewing
  AI/owner judges. Episodic rows are exempt (history, not claims — same
  principle as the staleness flag); only paths under this machine's home are
  judged, so pointers to other machines never false-positive; elided/template/
  ephemeral-container patterns are skipped (tuned on the live store's first
  run, which surfaced 12 genuinely rotted pointers).

### Fixed
- **Every recall candidate now carries its TRUE cosine.** The hybrid blend only
  had similarity scores for rows on the nearest-neighbor shortlist (25–3×limit
  rows); a keyword-anchored candidate outside it read as cosine 0.0 — it lost
  its whole vector relevance term, results shifted with `limit` (the shortlist
  scales with it), and the abstention gate mistook "not shortlisted" for
  "nothing similar", false-abstaining on correct rank-1 hits whose real cosine
  cleared the gate. `cosines_for()` now tops up exact best-chunk cosines for
  the rest of the candidate set (same noise floor); on the live golden set this
  eliminated every remaining false-abstain with ranking aggregates unchanged.
- **Vector coverage can no longer silently rot.** `set_gist` dropped the row's
  stale embedding and waited for a manual `embed` run to re-embed it — so a bulk
  consolidation pass left most of the store semantically invisible until someone
  remembered (live store 2026-07-01: 250 of 317 live rows had NO vector after a
  distill pass; the recall eval fell from hit@1 79%/MRR 0.882 to 54%/0.616 and
  the abstention gate false-abstained on 9 of 28 golden positives). Two-sided
  fix: `set_gist` now re-embeds in place when a model is available (embed-on-write
  parity with `store()`; still drop-only without one), and the first-use auto-
  backfill now heals any coverage GAP instead of bailing the moment one embedding
  exists — one indexed lookup when coverage is full, incremental embedding of
  just the gap rows when it is not. Post-repair eval on the same store:
  hit@1 64% / hit@k 89% / MRR 0.741, false-abstains 9 → 1 (the residual gap vs
  the 188-memory baseline is store growth, tracked in `eval --record` history).
- **The recall-quality eval no longer perturbs what it measures.** Each fence
  run counted its sweep as genuine recalls, inflating `recall_count` (a use
  signal feeding the `_usefulness` ranking bonus and the push floor) on every
  row it returned. Eval recall now passes `count_recall=False`.

## [0.4.0] — 2026-07-01

### Added
- **Close the loop: a pushed memory that gets USED now counts as used.** The
  per-memory usefulness floor scored a proactive push as ignored noise whenever
  `surfaced_count` outran `recall_count` — but a pushed memory sits in context and
  is used in reasoning WITHOUT ever being pulled, so a proven-useful push looked
  identical to genuine noise. Schema **v8** adds `referenced_count`/`last_referenced`;
  `effective_floor` folds a per-memory reference use-credit into `uses`, so a push
  that was actually referenced downstream sheds the ignored-noise penalty while a
  never-referenced one stays quiet (the credit only ever LOWERS a floor — noise,
  referenced=0, is untouched). `usefulness-scan --apply` materializes the honest
  transcript reference signal into the store (idempotent absolute set). The credit
  reads per-memory reference totals, so it is immune to the L3/L4 channel-attribution
  bias. Idempotent auto-migration; the recall/PULL path never touches the floor, so
  the recall-quality eval fence is unaffected.
- **`fornixdb usefulness-scan` — an HONEST push-usefulness signal from session
  transcripts.** The usefulness loop credits a memory as "used" only on an
  explicit pull or endorsement, but a proactively PUSHED memory is already in
  context — the model references it in its reasoning without ever pulling it — so
  a useful push and an ignored one look identical to the counters, and any outcome
  join keyed on recall_count measures "is this a frequently-pulled memory", not
  "was THIS push used". (Live: floor-stats called 265/271 surfaced rows "useful"
  on lifetime pulls; the real number is far lower.) The scan recovers the true
  signal from the host's own transcripts — FornixDB's injected block and the
  assistant's later `#id` citations are both visible there — attributing each
  citation to the injection that preceded it. Live result: only **18%** of pushes
  were referenced downstream. `floor-stats --transcripts PATH` now joins outcomes
  to this real reference signal instead of the lifetime-recall proxy. Read-only
  analysis; no schema or ranking change. The scan also breaks the reference rate
  down **by channel** (L3 per-turn vs L4 rhythmic in-thought), reading the L4
  block from the PostToolUse `stdout` field as well as the L3 `content` field —
  the data point for the operating-level decision (live: L3 13%, L4 20%).
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
