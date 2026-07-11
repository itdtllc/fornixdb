"""macOS audio adapters — the ear for `senses.hear()`/`senses.listen()`.

Three pieces, mirroring what `mac_camera` + `mac_vision` are to sight:

  capture_clip()   record N seconds of microphone audio to a WAV via ffmpeg's
                   avfoundation input — the cochlea. macOS shows a mic
                   permission prompt the first time; concurrent captures are
                   fine (CoreAudio serves multiple clients), so a one-shot
                   listen can run beside a voice loop's continuous capture.
  clap_tagger()    the sound-scene captioner — the REQUIRED lane of hear()
                   ("sound is meaning, not only words", owner principle
                   2026-07-04). Zero-shot: a CLAP model scores the clip
                   against a label vocabulary and the top labels become the
                   one-line gist ("acoustic guitar, a person singing").
  clap_embedder()  the ModalEmbedder (senses.ModalEmbedder) over the SAME
                   CLAP audio tower — the latent lane, audio↔audio
                   similarity. One model serves both lanes so their spaces
                   agree, exactly like the CLIP tower on the vision side.

The default model is `laion/larger_clap_music_and_speech` (Apache-2.0),
loaded lazily through `transformers` + `torch` — heavier than the core's
stdlib-only ethos, so everything imports lazily and a missing dependency
raises a pointed install message, the same pattern as the [mac] vision
extras. The model/processor pair is cached per model_id so a tagger and an
embedder in one process share a single loaded tower. `name` keys every
modal_embedding row, so it must be STABLE across runs.

The speech lane (transcript) is deliberately NOT here: hear()/listen() take a
`transcriber` callable, and any local STT (whisper.cpp is the proven one)
plugs in at the call site. Everything here is generic to any Mac — only the
shape of capture is encoded; `run` and the encode seams are injectable so the
logic is unit-testable with no microphone and no model.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

__all__ = ["capture_clip", "clap_tagger", "clap_embedder", "AudioEmbedder",
           "DEFAULT_SOUND_LABELS"]


# ------------------------------------------------------------------ capture

def capture_clip(seconds: float = 8.0, *, device: str = "0",
                 sample_rate: int = 48000, out_path: str | None = None,
                 run: Callable[..., "subprocess.CompletedProcess"] | None = None,
                 ) -> str:
    """Record `seconds` of microphone audio to a mono WAV and return its path.

    `device` is the avfoundation audio index (the voice loop's mic is "0").
    48 kHz is CLAP's native rate, so a captured clip feeds the tagger with no
    resampling. `out_path=None` records to a fresh temp file — the caller owns
    deleting it (senses.listen() applies the fidelity ladder: caption, then
    drop the clip). Inject `run` to test without a microphone."""
    if out_path is None:
        fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fd.close()
        out_path = fd.name
    runner = run or subprocess.run
    proc = runner(
        # -nostdin: with the terminal as its stdin ffmpeg flips it to raw mode
        # for interactive keys and eats keystrokes meant for the caller (froze
        # Elira's voice loop live 2026-07-10) — recording needs no keyboard.
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "avfoundation", "-i", f":{device}",
         "-ac", "1", "-ar", str(sample_rate), "-t", str(seconds), out_path],
        capture_output=True, text=True, timeout=seconds + 30)
    if proc.returncode != 0 or not Path(out_path).is_file():
        detail = (proc.stderr or "").strip()[:300]
        raise RuntimeError(
            f"capture_clip: ffmpeg could not record from mic :{device} — has "
            f"macOS granted microphone permission to this terminal? {detail}")
    return out_path


# ------------------------------------------------- the shared CLAP tower

def _load_audio(path: str, target_sr: int = 48000):
    """Read a WAV/AIFF to a mono float array at `target_sr`. Stereo is averaged;
    an off-rate file is linearly resampled (numpy — good enough for tagging,
    and it keeps librosa out of the dependency tree)."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n = int(len(audio) * target_sr / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    return audio


_towers: dict = {}          # model_id -> (model, processor); one load per process


def _clap(model_id: str):
    """Load (and cache) the CLAP model+processor for `model_id`. Lazy so
    importing this module costs nothing; a missing dependency points at the
    fix instead of a bare ImportError."""
    if model_id in _towers:
        return _towers[model_id]
    try:
        import torch  # noqa: F401 - presence check before the heavy load
        from transformers import ClapModel, ClapProcessor
    except ImportError as e:                       # pragma: no cover - env-dep
        raise ImportError(
            "the audio adapters need transformers + torch + soundfile — "
            "install them into this environment: "
            "pip install torch transformers soundfile") from e
    model = ClapModel.from_pretrained(model_id)
    model.eval()
    processor = ClapProcessor.from_pretrained(model_id)
    _towers[model_id] = (model, processor)
    return model, processor


DEFAULT_CLAP_MODEL = "laion/larger_clap_music_and_speech"

# The zero-shot vocabulary the sound-scene gist is written from. Deliberately
# everyday: instruments, voices, and the household/street sounds the design
# doc calls out (a crosswalk beep, a kettle, a dog at the door). Pass your own
# `labels=` for a specialized space; more labels cost one text-encoder pass.
DEFAULT_SOUND_LABELS = (
    "acoustic guitar", "electric guitar", "piano", "violin", "drums",
    "a person singing", "a person speaking", "people having a conversation",
    "whistling", "humming", "laughter", "applause", "footsteps",
    "a dog barking", "a cat meowing", "birds chirping",
    "a phone ringing", "an alarm beeping", "a kettle whistling",
    "a doorbell", "a door closing", "keyboard typing", "a vacuum cleaner",
    "running water", "rain", "wind", "thunder", "traffic and car engines",
    "a car horn", "a siren", "glass breaking", "a baby crying",
    "music playing from speakers", "television in the background",
    "silence", "static noise",
)


def clap_tagger(model_id: str = DEFAULT_CLAP_MODEL, *,
                labels: tuple = DEFAULT_SOUND_LABELS,
                min_prob: float = 0.15,
                score: Callable[[str], list[float]] | None = None,
                ) -> Callable[[str], str]:
    """A sound-scene captioner for hear()/listen(): `(clip_path) -> gist`.

    Zero-shot CLAP: the clip and every label are embedded in the shared
    audio↔text space; labels scoring ≥ `min_prob` (softmax) join the caption,
    strongest first — "acoustic guitar, a person singing". When nothing
    clears the bar the top label is reported hedged ("possibly footsteps"),
    so the gist never comes back empty. Inject `score` (path -> per-label
    probabilities) to test without a model."""
    if score is None:
        def score(clip_path: str) -> list[float]:
            import torch
            model, processor = _clap(model_id)
            inputs = processor(text=list(labels), audio=[_load_audio(clip_path)],
                               sampling_rate=48000, return_tensors="pt",
                               padding=True)
            with torch.no_grad():
                out = model(**inputs)
            return out.logits_per_audio.softmax(dim=-1)[0].tolist()

    def caption(clip_path: str) -> str:
        probs = score(clip_path)
        ranked = sorted(zip(labels, probs), key=lambda p: -p[1])
        strong = [l for l, p in ranked if p >= min_prob]
        if strong:
            return ", ".join(strong[:3])
        return f"possibly {ranked[0][0]}"

    return caption


# ------------------------------------------------------------ latent lane

class AudioEmbedder:
    """A senses.ModalEmbedder over any local audio tower. `encode(path) ->
    vector` is the whole model contract; this class owns L2 normalization and
    the protocol surface. `encode` is injectable so it is unit-testable with
    no model, the way the other adapters take injectable readers."""

    def __init__(self, encode: Callable[[str], list[float]], *, name: str):
        self._encode = encode
        self.name = name

    def embed_artifact(self, paths: list[str]) -> list[list[float]]:
        out = []
        for p in paths:
            v = self._encode(p)
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v] if n else list(v))
        return out


def clap_embedder(model_id: str = DEFAULT_CLAP_MODEL, *,
                  encode: Callable[[str], list[float]] | None = None,
                  ) -> AudioEmbedder:
    """The default hear() ModalEmbedder: the CLAP audio tower, keyed by
    `model_id` (stable — it names the latent-lane space, and it is the same
    tower the tagger scores with). Built lazily on the first embed."""
    if encode is None:
        def encode(clip_path: str) -> list[float]:  # noqa: F811 - lazy default
            import torch
            model, processor = _clap(model_id)
            inputs = processor(audio=[_load_audio(clip_path)],
                               sampling_rate=48000, return_tensors="pt")
            with torch.no_grad():
                feats = model.get_audio_features(**inputs)
            # transformers ≤4 returns the [1, dim] tensor directly; 5.x wraps
            # it in an output object whose pooler_output is the embedding
            feats = getattr(feats, "pooler_output", feats)
            return [float(x) for x in feats.reshape(-1).tolist()]

    return AudioEmbedder(encode, name=model_id)
