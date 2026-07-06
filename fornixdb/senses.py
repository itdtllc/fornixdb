"""Multimodal capture — the human senses. Design: SENSES.md.

FornixDB is a human-like memory, and humans don't remember only words.
`see` and `hear` are IMPLEMENTED for single artifacts (an image file, an
audio clip), and `feel` is IMPLEMENTED for single sensor readings (machine
proprioception first — no robot required). The live loops live beside this
surface: vision's core is `fornixdb.watchloop` (stream-source adapters
pending, so `watch` here still raises honestly); proprioception's
change-gated loop is the next build. The pattern is the one the text path proved and
SENSES.md publishes:

  gist        a one-line caption; recalled first like any other memory, and
              embedded through the ordinary text path — so recall by meaning
              works ACROSS modalities from day one (the gist lane)
  vector      a modality embedding (image/audio/sensor model) in the v10
              modal_embedding table — same-modality similarity, scored only
              within its own model's space (the latent lane)
  source_ref  the artifact stays on disk (frame, clip, reading log); the
              store keeps the pointer, never the blob
  time        streams are the episodic axis taken literally — event_time /
              event_time_end spans, so "what happened at the front door
              yesterday afternoon" is ordinary timeline recall

SOUND IS MEANING, NOT ONLY WORDS (owner principle, 2026-07-04): a crosswalk
beep, a kettle, whistling, a dog at the door all carry meaning with zero
speech in them. `hear` therefore treats the sound-scene caption as the
REQUIRED lane and the transcript as the additional one — never the reverse.

Local-first still holds: captioners, transcribers, and modality embedders
must run on device, like everything else here. They plug in as callables /
protocols so any local model can serve; nothing here imports one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

from .salience import cosine
from .vectors import from_blob, to_blob

__all__ = ["ModalEmbedder", "see", "hear", "watch", "feel",
           "modal_vector", "modal_neighbors"]

_TBD = ("TBD — this sense's live capture loop is declared intent, not yet "
        "implemented. See SENSES.md for the design (buffer -> salience gate "
        "-> store); `see` and `hear` already work for single artifacts.")


class ModalEmbedder(Protocol):
    """The modality twin of vectors.Embedder: any local model that maps an
    artifact (image, audio window, sensor frame) to a vector. One store may
    hold vectors from several modalities; similarity is only ever scored
    within one model's space (the model name keys every modal_embedding row).
    """

    name: str

    def embed_artifact(self, paths: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------- the latent lane

def _save_modal_vector(store, memory_id: int, embedder: ModalEmbedder,
                       path: str) -> None:
    vec = embedder.embed_artifact([path])[0]
    store.conn.execute(
        "INSERT OR REPLACE INTO modal_embedding(memory_id, model, dim, vector) "
        "VALUES (?, ?, ?, ?)",
        (memory_id, embedder.name, len(vec), to_blob(vec)))
    store.conn.commit()


def modal_vector(store, memory_id: int, model: str | None = None):
    """The stored modality vector for a memory: (model, vector) or None.
    With several models on one memory, pass `model` to pick one."""
    q = "SELECT model, vector FROM modal_embedding WHERE memory_id = ?"
    args: list = [memory_id]
    if model is not None:
        q += " AND model = ?"
        args.append(model)
    row = store.conn.execute(q, args).fetchone()
    return (row[0], from_blob(row[1])) if row else None


def modal_neighbors(store, memory_id: int, *, model: str | None = None,
                    k: int = 5) -> list[tuple[int, float]]:
    """Same-modality similarity: live memories whose modal vector (SAME model
    only — spaces never mix) is nearest this memory's. [(memory_id, cos)]."""
    anchor = modal_vector(store, memory_id, model)
    if anchor is None:
        return []
    model_name, vec = anchor
    rows = store.conn.execute(
        "SELECT e.memory_id, e.vector FROM modal_embedding e "
        "JOIN memory m ON m.id = e.memory_id "
        "WHERE e.model = ? AND e.memory_id != ? AND m.superseded_time IS NULL",
        (model_name, memory_id)).fetchall()
    scored = [(mid, cosine(vec, from_blob(blob))) for mid, blob in rows]
    scored.sort(key=lambda p: p[1], reverse=True)
    return scored[:k]


# ------------------------------------------------------------------ helpers

def _percept(store, gist: str, *, sense: str, artifact: str,
             detail: str | None, event_time: str | None,
             event_time_end: str | None, topics: list[str] | None,
             project: str | None, session_id: str | None,
             embedder: ModalEmbedder | None) -> int:
    mid = store.store(
        gist, detail, kind="episodic", topics=topics, project=project,
        event_time=event_time, event_time_end=event_time_end,
        session_id=session_id, source=f"senses:{sense}", source_ref=artifact)
    if embedder is not None:
        _save_modal_vector(store, mid, embedder, artifact)
    return mid


def _resolve(path: str, sense: str) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"{sense}: no artifact at {path!r} — the store keeps a pointer to "
            "the file on disk, so the file must exist when the memory is made")
    return str(p.resolve())


# -------------------------------------------------------------------- sight

def see(store, image_path: str, *, caption: str | None = None,
        captioner: Callable[[str], str] | None = None,
        embedder: ModalEmbedder | None = None,
        event_time: str | None = None, event_time_end: str | None = None,
        topics: list[str] | None = None,
        project: str | None = None, session_id: str | None = None) -> int:
    """Remember one image. The caption becomes the gist (pass one, or pass a
    local captioner callable that writes one); the image stays on disk as
    source_ref; an optional ModalEmbedder adds the latent-lane vector.
    Returns the memory id."""
    artifact = _resolve(image_path, "see")
    if caption is None:
        if captioner is None:
            raise ValueError(
                "see: a caption is the gist every recall path leans on — "
                "pass caption=..., or captioner=<local VLM callable> to "
                "write one from the image")
        caption = captioner(artifact)
    return _percept(store, caption.strip(), sense="sight", artifact=artifact,
                    detail=None, event_time=event_time,
                    event_time_end=event_time_end,
                    topics=topics, project=project, session_id=session_id,
                    embedder=embedder)


# -------------------------------------------------------------------- sound

def hear(store, audio_path: str, *, sound_caption: str | None = None,
         sound_tagger: Callable[[str], str] | None = None,
         transcript: str | None = None,
         transcriber: Callable[[str], str | None] | None = None,
         embedder: ModalEmbedder | None = None,
         event_time: str | None = None, event_time_end: str | None = None,
         topics: list[str] | None = None, project: str | None = None,
         session_id: str | None = None) -> int:
    """Remember audio. TWO lanes, and the non-speech one is the required one:

    sound scene (REQUIRED) — what the audio *was*: "crosswalk signal beeping",
        "someone whistling a tune", "glass breaking, two dogs barking". Pass
        sound_caption=..., or sound_tagger=<local audio-caption callable>.
        Sound carries meaning with zero words in it; this lane always runs.
    speech (ADDITIONAL) — when words were spoken, pass transcript=..., or
        transcriber=<local STT callable> (returning None when there is no
        speech). The gist leads with the sound scene and quotes the speech;
        the full transcript lands in detail for drill-down.

    The clip stays on disk as source_ref; an optional ModalEmbedder (e.g. a
    CLAP-family model) adds the latent-lane vector. Returns the memory id."""
    artifact = _resolve(audio_path, "hear")
    if sound_caption is None:
        if sound_tagger is None:
            raise ValueError(
                "hear: the sound-scene caption is the required lane (sound "
                "means things without words — a crosswalk beep, whistling) — "
                "pass sound_caption=..., or sound_tagger=<local audio-caption "
                "callable>. A transcript alone is not enough.")
        sound_caption = sound_tagger(artifact)
    if transcript is None and transcriber is not None:
        transcript = transcriber(artifact)

    gist = sound_caption.strip()
    detail = None
    if transcript:
        transcript = transcript.strip()
        quoted = transcript if len(transcript) <= 120 else transcript[:117] + "…"
        gist = f'{gist} — said: "{quoted}"'
        detail = transcript
    return _percept(store, gist, sense="sound", artifact=artifact,
                    detail=detail, event_time=event_time,
                    event_time_end=event_time_end, topics=topics,
                    project=project, session_id=session_id, embedder=embedder)


# ---------------------------------------------------- streams (still TBD)

def watch(store, stream_source: str, *, window_seconds: float = 30.0):
    """TBD at this surface — remember a video stream (camera, screen, file).
    The LOOP CORE is real: `fornixdb.watchloop.run_watch` takes any
    (timestamp, frame) iterator plus an embed callable and does the rest
    (salience gate, keyframes on commit only, `see` rows with event-time
    spans; window_seconds is the MAXIMUM window — boundaries cut early when
    something happens). What remains TBD here is the stream-source adapter
    layer that turns a source string into that frame iterator — camera /
    screen / file adapters per Design/Watch_Loop_Implementation_Spec.md."""
    raise NotImplementedError(_TBD)


def _feel_gist(sensor: str, reading) -> str:
    if isinstance(reading, dict):
        state = ", ".join(f"{k}={v}" for k, v in reading.items())
    else:
        state = str(reading).strip()
    return f"feel[{sensor}]: {state}"


def feel(store, reading, *, sensor: str, gist: str | None = None,
         event_time: str | None = None, event_time_end: str | None = None,
         topics: list[str] | None = None, project: str | None = None,
         session_id: str | None = None) -> int:
    """Remember one tactile/proprioceptive sensor reading. The first binding
    needs no robot: machine proprioception (power, thermal, network, lid) —
    a reading is a dict of named values (or a plain string), the gist is a
    templated state line unless you pass one, and NO embedder is required:
    the gist lane alone makes it recallable ("when did the laptop go on
    battery?"). Robot endpoints (force, contact, IMU) are the same pattern,
    later adding a sensor-domain ModalEmbedder for the latent lane. A
    change-gated live loop is the next build (see the watch spec)."""
    detail = (json.dumps(reading, sort_keys=True, default=str)
              if isinstance(reading, dict) else None)
    return _percept(store, (gist or _feel_gist(sensor, reading)).strip(),
                    sense="feel", artifact=f"sensor:{sensor}", detail=detail,
                    event_time=event_time, event_time_end=event_time_end,
                    topics=topics, project=project, session_id=session_id,
                    embedder=None)
