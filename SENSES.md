# Senses — the multimodal design

**Status: the capture core and two live loops are implemented.**
[`see`](fornixdb/senses.py) and `hear` work today for single artifacts (an
image file, an audio clip): caption gist + optional modality vector +
`source_ref` pointer, exactly the row shape below, with local models plugging
in as callables (nothing is bundled). The salience gate ships as
`fornixdb.salience` (pure, hardware-free, tested), and schema v10 adds the
latent-lane `modal_embedding` table. Two of the live loops are running:
`watch` ([`fornixdb.watchloop`](fornixdb/watchloop.py) + the Mac camera/screen/
file adapters and the `fornixdb watch` command) and `feel`
([`fornixdb.feelloop`](fornixdb/feelloop.py) + a Mac power adapter and
`fornixdb feel --live`). The audio loop that feeds `hear` is still declared
intent. Capture is always owner-started — no always-on sampling. Signatures
may still move.

FornixDB is a human-like memory, and humans don't remember only words. The
claim this design makes is narrow and testable: **adding senses to the store
requires new encoders, not new architecture.** Everything below is the
existing machinery — gists, vectors, source refs, event-time spans, tiers,
consolidation, the disk budget — fed by new capture paths.

## The one idea everything hangs on

Text memory works here because words are stored as *distributed activations*:
the embedding model turns a sentence into a weight vector, and recall is
nearest-neighbor in that space. The senses are the same trick with different
encoders. An image, an audio window, or a sensor frame each map to a vector in
some local model's space — modality is an encoder detail. The field has
already converged on this (see prior art), and the `ModalEmbedder` protocol in
`senses.py` is the seam: one interface, N local encoders, vectors scored
within each model's own space (the model name already partitions every
embedding row in the store).

**Cross-modal recall needs no new machinery.** Every perceptual memory
carries a textual gist — a caption written by a small local model — and gists
embed through the ordinary text path. "When did I hear the doorbell?" is a
text query that lands on an audio memory *via its caption*. The modality
vector is a second, independent lane:

| lane        | vector                      | answers                                        |
|-------------|-----------------------------|------------------------------------------------|
| gist lane   | text-embed(caption)         | any-modality recall by meaning, from day one    |
| latent lane | ModalEmbedder(artifact)     | same-modality similarity, dedup, salience gating |

## Buffer → gate → store

Humans sample their senses densely and keep almost nothing: iconic (visual)
sensory memory lasts roughly a quarter-second to a second, echoic (auditory)
a few seconds, and episodic commits cluster at *event boundaries* — where
prediction error spikes. The design copies that shape, and the storage math
independently forces it (embeddings sampled at 10 Hz are gigabytes per day of
"same room, nothing happened").

**Stage 1 — sensory buffer (RAM only).** A ring buffer of samples: frames at
~10 Hz a few seconds deep, a rolling audio window, current sensor readings.
It evaporates on session end and never touches disk — deliberately not a
store concept at all.

**Stage 2 — salience gate.** The only genuinely new algorithm. Commit a
snapshot only when the present diverges from the recent past: compare each
sample's embedding against a moving average of recent ones and commit when
the distance clears a per-modality threshold, with hysteresis so one event
commits once rather than ten times a second. A heartbeat commit (e.g. every
ten minutes of quiet) anchors the timeline cheaply — "nothing happened,
here's proof." Discrete sensors gate on change and need no threshold.

**Sound is meaning, not only words.** A crosswalk beep, a kettle, whistling,
a dog at the door all carry meaning with zero speech in them — so the
non-speech lane is the *required* one, never a fallback: every committed
audio window gets a sound-scene caption ("crosswalk signal beeping, light
traffic"), and when voice-activity detection finds speech, transcription runs
*additionally* — the gist leads with the scene and quotes the words, the full
transcript lands in detail. `hear` enforces this shape today: a transcript
alone is rejected as an incomplete memory of a sound.

**Stage 3 — ordinary store rows.** A committed snapshot is a normal episodic
memory: the caption as **gist** (recalled first, like everything), the
modality embedding as **vector**, the artifact on disk as **source_ref**
(keyframe, clip, reading log — the store keeps the pointer, never the blob),
and a real **event_time / event_time_end** span so stream windows answer
"what happened at the front door yesterday afternoon" as plain timeline
recall. One integration requirement follows: `forget`, `shrink`, and the
disk budget treat the artifact as part of the memory — forgetting a sight
deletes its keyframe, and the budget counts artifact bytes.

**Windows cut at boundaries, not the clock.** `watch()`/`hear()` window
parameters are *maximum* lengths; the gate ends a window early when something
happens. A boring hour is one heartbeat row; a busy minute may be three
distinct events. A fixed-window integrator is just the degenerate case of a
zero threshold, so the declared signatures hold.

## Consolidation and the fidelity ladder

The dream pass extends naturally: temporally adjacent snapshots with similar
latents merge into one episodic summary, members linked, redundant members
pruned — the same shape as session distillation. Decay becomes a **fidelity
ladder** — losing vividness, not existence:

```
clip + keyframe + vector + gist
     → keyframe + vector + gist
     → vector + gist
     → gist only
```

Each rung drops the largest remaining artifact, governed by the existing
per-kind decay halflives and retention tiers. Old perceptual memories fade
from vivid to verbal — which is how human episodic memory actually degrades.

## `feel()` without a robot: machine proprioception

The cheapest proof that the abstraction generalizes needs no extra hardware:
bind `feel()` to the machine sensing itself — power and battery state, lid
state, thermal pressure, network identity, display and audio-route changes.
Commits are change-driven, gists are templated text ("switched to home
network, on power, external display connected"), and notably **no
ModalEmbedder is required at all**: the gist lane alone makes machine state
recallable, demonstrating that the two lanes are genuinely independent.
Shipped (0.8.4): `adapters/mac_proprioception.read_temperature()` reads the
Apple Silicon die sensors (IOHID, no sudo) and the battery thermistor, so
"how do you feel?" can honestly include how warm the thinking has made the
machine. The
same pattern with real sensor-domain embedders is the robot-endpoint story
(force, contact, IMU streams) that FornixDB's store-per-endpoint design
anticipates.

## Reference binding: a laptop's camera and microphone

Local-first holds — captioners and modality embedders run on device, like
everything else here. The first bindings have shipped: `senses.watch()`
samples camera frames with a CLIP-family embedder driving the gate and a
compact vision-language model writing captions *only on commit* (never at
sample rate); `senses.glance()` is the look-once, right-now variant; and as
of 0.8.4 `senses.listen()` is the ear — a short microphone clip answered
with a sound-scene gist from a CLAP model (`adapters/mac_audio.py`), with a
speech lane feeding transcripts into the existing text path untouched. The
cost profile is the point: the sampling loop runs only the cheap embedder,
the expensive models run only on committed events, and an idle scene costs
approximately nothing.

**Privacy stance.** Capture is explicit and session-scoped — the owner starts
`watch`/`hear` and sessions end by command or timeout; the OS camera/mic
indicator is the ground truth; there is no always-on default. Frames, clips,
embeddings, and captions all stay on device, under the store's own directory
tree where the disk budget and shrink govern them like everything else.

**Storage envelope.** An active day at conservative settings — a couple
hundred committed events of keyframe + embedding + gist — lands in the tens
of megabytes before the ladder decays it toward kilobytes of gists. Clips
dominate if enabled and are opt-in per session. No new budget concept is
needed.

## Prior art

This is the well-trodden road, not an invention:

- **CLIP / SigLIP** (image↔text), **CLAP** (audio↔text), **Whisper**
  (speech→text) — the per-modality workhorses.
- **ImageBind** — six modalities in one joint embedding space; the existence
  proof for cross-modal recall, and a candidate single `ModalEmbedder` later.
- **Event segmentation theory** (Zacks et al.) — human episodic boundaries
  sit at prediction-error spikes; the salience gate is that, mechanized.
- **Iconic/echoic memory** (Sperling; Darwin et al.) — the sensory buffer's
  durations are the measured human constants, not invented ones.
- **World models** (e.g. DreamerV3) — store latents, predict the next latent.
  The far end of this road is memory that notices when the present diverges
  from what it expected; the salience gate is the first rung of that ladder.
