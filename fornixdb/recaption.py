"""Dream-pass captioning for the watch loop — the model-bearing half, run in
batch off the hot path.

The watch loop keeps the 2 Hz path model-free: every committed keyframe lands
under a *templated* placeholder gist (`watch[screen]: scene change`), never a
real caption (Senses_Design §4; Watch_Loop_Implementation_Spec Open Decision 3
— "template-now, batch-caption in dream"). This module is that later pass. It
finds the placeholders, runs a local VLM captioner over each committed keyframe,
and rewrites the gist in place so a *text* consumer can recall what was seen
("what did you see at the front door yesterday?") — the whole point of the gist
lane crossing modalities.

Pure and injectable, exactly like `watchloop`: the captioner is a callable
`(keyframe_path) -> str`, so nothing here imports a model and the core + full
test suite stay stdlib-only. The Mac VLM adapter that supplies a real captioner
lives beside the CLIP embedder in `fornixdb.adapters` and is imported lazily by
the CLI, never here.

Propose-not-dispose, and non-destructive: a caption is derived presentation, so
the rewrite goes through `store.set_gist` (in-place, re-embeds the text lane) —
NOT a supersession (Design §13.5 decision 2). A keyframe that decayed off disk
is skipped, and an empty caption never clobbers a placeholder, so a half-finished
or model-less pass leaves the store exactly recallable as it was. Keyed on the
placeholder text, the pass is naturally idempotent: a row it already captioned no
longer matches, so re-running only touches what still awaits a caption.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

__all__ = ["is_templated_watch_gist", "pending_captions", "pending_count",
           "recaption"]

# The exact placeholders watchloop._template_gist emits, and nothing else. A
# single see() with a real caption, or a row this pass already rewrote, will not
# match — that is what makes the worklist self-limiting and the pass idempotent.
_TEMPLATE_RE = re.compile(
    r"^watch\[[^\]]+\]: (session start|scene change|quiet interval)$")


def is_templated_watch_gist(gist: str) -> bool:
    """True if `gist` is a watch placeholder still awaiting a real caption."""
    return bool(_TEMPLATE_RE.match((gist or "").strip()))


def pending_captions(store, *, limit: int | None = None
                     ) -> list[tuple[int, str, str]]:
    """The dream worklist: live sight memories still holding a templated
    placeholder, oldest scene first. Each row is (memory_id, gist, keyframe).
    Rows whose keyframe has decayed off disk are dropped — there is nothing left
    to caption."""
    rows = store.conn.execute(
        "SELECT id, gist, source_ref FROM memory "
        "WHERE source = 'senses:sight' AND source_ref IS NOT NULL "
        "AND superseded_time IS NULL "
        "ORDER BY event_time, id").fetchall()
    out = [(mid, g, ref) for mid, g, ref in rows
           if is_templated_watch_gist(g) and Path(ref).is_file()]
    return out[:limit] if limit is not None else out


def pending_count(store) -> int:
    """How many committed watch keyframes still await a real caption."""
    return len(pending_captions(store))


def recaption(store, captioner: Callable[[str], str], *,
              limit: int | None = None, embedder=None,
              on_caption: Callable[[int, str, str], None] | None = None
              ) -> list[tuple[int, str, str]]:
    """Rewrite templated watch gists with real captions from a local VLM.

    `captioner(keyframe_path) -> str` runs once per committed keyframe (never on
    the hot path). For each rewritten row the gist becomes the caption and the
    text lane is re-embedded via `store.set_gist` (embedder=None uses the store
    default). A caption that comes back empty/whitespace is skipped — a
    placeholder is more recallable than nothing. Returns the applied rewrites as
    (memory_id, old_gist, caption), and calls `on_caption` per rewrite for live
    output. Never raises for a single bad frame: a captioner error skips that
    row and the pass continues (a dream must not die on one frame)."""
    applied: list[tuple[int, str, str]] = []
    for mid, old, keyframe in pending_captions(store, limit=limit):
        try:
            caption = (captioner(keyframe) or "").strip()
        except Exception:
            continue
        if not caption:
            continue
        store.set_gist(mid, caption, embedder=embedder)
        applied.append((mid, old, caption))
        if on_caption is not None:
            on_caption(mid, old, caption)
    return applied
