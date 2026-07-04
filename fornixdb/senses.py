"""Multimodal capture — the human senses. ALL TBD: this module is the
declared intent, not a working feature.

FornixDB is a human-like memory, and humans don't remember only words.
These APIs stake out how sight, sound, and touch will enter the store, so
integrators can see where the architecture is headed before it ships. Every
function below raises NotImplementedError today; signatures may still move.

The design pattern is already proven by the text path and will not change:

  gist        a one-line caption (written by a small local model),
              recalled first like any other memory
  vector      a modality embedding (image/audio/sensor model) in the same
              pluggable Embedder slot model2vec occupies for text — recall
              by meaning works across modalities
  source_ref  the artifact stays on disk (frame, clip, waveform, reading
              log); the store keeps the pointer, never the blob
  time        streams are the episodic axis taken literally — event_time /
              event_time_end spans, so "what happened at the front door
              yesterday afternoon" is ordinary timeline recall

Local-first still holds: captioners and modality embedders must run on
device, like everything else here.
"""

from __future__ import annotations

from typing import Protocol

_TBD = ("TBD — multimodal capture is declared intent, not yet implemented. "
        "See SENSES.md for the full design (buffer -> salience gate -> store, "
        "two recall lanes, fidelity-ladder decay) and the README's "
        "'Anatomy of a memory' for how it fits the store.")


class ModalEmbedder(Protocol):
    """TBD. The modality twin of vectors.Embedder: any local model that maps
    an artifact (image, audio window, sensor frame) to a vector. One store
    may hold vectors from several modalities; recall scores within each
    model's space (model name is already part of every embedding row)."""

    name: str

    def embed_artifact(self, paths: list[str]) -> list[list[float]]: ...


# ------------------------------------------------------------------ sight

def see(store, image_path: str, *, caption: str | None = None,
        event_time: str | None = None, topics: list[str] | None = None) -> int:
    """TBD — remember one image. Planned: caption becomes the gist (a local
    VLM writes one if not given), an image embedding lands in the vector
    slot, the file stays on disk as source_ref. Returns the memory id."""
    raise NotImplementedError(_TBD)


def watch(store, stream_source: str, *, window_seconds: float = 30.0):
    """TBD — remember a video stream (camera, screen, file). Planned: the
    stream is chunked into time windows; each window stores one episodic
    memory (keyframe caption + embedding + clip source_ref) with a real
    event_time span, so timeline recall answers "what happened while…".
    Retention tiers and the disk budget govern how much footage survives."""
    raise NotImplementedError(_TBD)


# ------------------------------------------------------------------ sound

def hear(store, audio_source: str, *, transcribe: bool = True,
         window_seconds: float = 60.0):
    """TBD — remember audio (microphone stream or file). Planned: local
    transcription feeds the existing TEXT path (a transcript is words —
    today's machinery already handles it); non-speech audio stores a sound
    embedding + caption ("glass breaking, two dogs barking") per window."""
    raise NotImplementedError(_TBD)


# ------------------------------------------------------------------ touch

def feel(store, reading, *, sensor: str, event_time: str | None = None):
    """TBD — remember tactile/proprioceptive sensor data (robotics: force,
    temperature, contact, IMU). Planned: readings aggregate into episodic
    windows ("gripper slipped twice on the glass jar") with the raw reading
    log as source_ref; embeddings come from a sensor-domain model. This is
    the robot-endpoint sense — FornixDB's per-endpoint design (a store on
    every robot) is why it exists."""
    raise NotImplementedError(_TBD)
