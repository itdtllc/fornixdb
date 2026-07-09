"""The watch-loop core — dense frames in, sparse `see` memories out.

P1 of the watch() design (SENSES.md; Design/Watch_Loop_Implementation_Spec.md):
the pure, injectable half. No camera, no clock, no sleep lives here — an
adapter owns sampling and yields (timestamp, frame) pairs; this module embeds
each frame, asks the salience gate whether the moment deserves to become a
memory, and turns each commit into an ordinary `see` percept with a real
event-time span. Frames that never commit never touch disk.

A frame may be a str path (already on disk — screencapture-style adapters) or
raw encoded image bytes (camera adapters); bytes are written under
keyframe_dir only at commit time. The gate lane and the latent lane take
different embedders on purpose: the gate needs a fast vector per frame BEFORE
any file exists (`embed(frame) -> vector`, caller's frame format), while the
ModalEmbedder protocol embeds artifact paths and runs only on committed
keyframes.

Windows cut at boundaries, not the clock: a committed memory's span runs from
the first frame after the previous commit to the committing frame, capped at
window_seconds — a busy minute can be three memories, a boring hour is one
heartbeat row. Timestamps from the adapter are monotonic seconds; start_wall
anchors them to calendar time for event_time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from . import senses
from .salience import SalienceGate

__all__ = ["WatchEvent", "run_watch"]


@dataclass(frozen=True)
class WatchEvent:
    memory_id: int
    reason: str        # "first" | "event" | "heartbeat"
    t_start: float     # adapter (monotonic) seconds — span start
    t_end: float       # the committing frame's timestamp
    keyframe: str      # path the memory's source_ref points at
    gist: str


def _template_gist(source_label: str, reason: str) -> str:
    return {"first": f"watch[{source_label}]: session start",
            "event": f"watch[{source_label}]: scene change",
            "heartbeat": f"watch[{source_label}]: quiet interval"}[reason]


def _drop_keyframe(store, memory_id: int, keyframe: str) -> None:
    """The fidelity ladder (Senses_Design §4) applied at capture time. Once the
    caption gist and the modal (image) vector are stored — both happen inside
    the senses.see() call before this — the raw frame is no longer needed to
    recall the moment. Delete the file and null the now-dangling source_ref so
    the memory settles to the `vector + gist` rung: same-modality recall still
    works (the vector survives), only the displayable still is gone. Meant for
    live capture where lingering stills on disk are undesirable; pair it with a
    `captioner` so the gist is a real caption, not a placeholder. Never raises —
    a failed unlink must not abort the loop."""
    try:
        Path(keyframe).unlink()
    except OSError:
        pass
    store.conn.execute("UPDATE memory SET source_ref = NULL WHERE id = ?",
                       (memory_id,))
    store.conn.commit()


def _persist(frame, keyframe_dir: str | None, session_id: str | None,
             t: float) -> str:
    """A committed frame becomes a file; a path frame already is one."""
    if isinstance(frame, str):
        return frame
    if not isinstance(frame, (bytes, bytearray)):
        raise TypeError(f"frame must be a path or encoded bytes, "
                        f"got {type(frame).__name__}")
    if keyframe_dir is None:
        raise ValueError("bytes frames need keyframe_dir=... — committed "
                         "keyframes live on disk so the memory's source_ref "
                         "has something to point at")
    d = Path(keyframe_dir).expanduser()
    if session_id:
        d = d / session_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{t:012.3f}.jpg"
    p.write_bytes(bytes(frame))
    return str(p)


def run_watch(store, frames, *,
              embed: Callable[[object], list[float]],
              gate: SalienceGate | None = None,
              captioner: Callable[[str], str] | None = None,
              modal_embedder=None,
              window_seconds: float = 30.0,
              keyframe_dir: str | None = None,
              source_label: str = "stream",
              start_wall: datetime | None = None,
              max_seconds: float | None = None,
              max_commits: int | None = None,
              should_continue: Callable[[], bool] | None = None,
              drop_keyframe_after_commit: bool = False,
              topics: list[str] | None = None,
              project: str | None = None,
              session_id: str | None = None,
              on_commit: Callable[[WatchEvent], None] | None = None,
              ) -> list[WatchEvent]:
    """Drive one watch session over an adapter's frame iterator.

    frames yields (t, frame) with t in monotonic seconds. Every frame is
    embedded and shown to the gate; every gate commit becomes a `see` memory
    (captioner writes the gist from the committed keyframe when provided,
    else a templated gist recall can find — a later dream pass may caption
    properly). Returns the committed events in order. Stops on iterator
    exhaustion, max_seconds of stream time, or max_commits.

    `should_continue`, when given, is checked once per frame BEFORE any work;
    returning False stops the loop cleanly. It is the stop signal for a
    background/'live' watch thread (e.g. an assistant opening and closing its
    eyes) that has no fixed max_seconds. `drop_keyframe_after_commit` deletes
    each committed keyframe and nulls its source_ref once the gist and modal
    vector are stored (see _drop_keyframe) — pair it with `captioner` for live
    capture that leaves no stills on disk.
    """
    if drop_keyframe_after_commit and captioner is None:
        # Dropping the still while writing only a placeholder gist would strand a
        # row the dream pass can never fulfill: recaption needs the keyframe on
        # disk, but there is nothing left to caption. The two are only coherent
        # together — caption inline, THEN drop. Fail loudly instead.
        raise ValueError(
            "drop_keyframe_after_commit=True requires a captioner: dropping the "
            "keyframe leaves only a placeholder gist, and the dream pass "
            "(recaption) cannot caption a still that is already gone. Pass a "
            "captioner to caption inline, or keep the keyframe for the dream pass.")
    gate = gate or SalienceGate()
    wall0 = start_wall or datetime.now()
    events: list[WatchEvent] = []
    t_first: float | None = None
    window_start: float | None = None

    def _iso(t: float) -> str:
        return (wall0 + timedelta(seconds=t - t_first)).isoformat(
            timespec="seconds")

    try:
        for t, frame in frames:
            if should_continue is not None and not should_continue():
                break
            if t_first is None:
                t_first = t
            if max_seconds is not None and t - t_first > max_seconds:
                break
            if window_start is None:
                window_start = t

            decision = gate.observe(embed(frame), t)
            if not decision.commit:
                continue

            span_start = max(window_start, t - window_seconds)
            keyframe = _persist(frame, keyframe_dir, session_id, t)
            gist = (captioner(keyframe) if captioner
                    else _template_gist(source_label, decision.reason))
            et, ete = _iso(span_start), _iso(t)
            mid = senses.see(
                store, keyframe, caption=gist, embedder=modal_embedder,
                event_time=et, event_time_end=(ete if ete != et else None),
                topics=topics, project=project, session_id=session_id)

            ev = WatchEvent(mid, decision.reason, span_start, t, keyframe, gist)
            events.append(ev)
            if on_commit is not None:
                on_commit(ev)          # callback still sees the keyframe on disk
            if drop_keyframe_after_commit:
                _drop_keyframe(store, mid, keyframe)
            window_start = None        # the next frame opens the next window
            if max_commits is not None and len(events) >= max_commits:
                break
    finally:
        # Close the frame generator NOW so an adapter's release (e.g. the camera
        # capture, cv2.release) runs the moment we stop — not whenever GC gets to
        # it. Critical for a stop-hook'd 'live eyes' loop: the camera light must
        # go off on close_eyes(). A plain iterator has no close(); that's fine.
        closer = getattr(frames, "close", None)
        if callable(closer):
            closer()
    return events
