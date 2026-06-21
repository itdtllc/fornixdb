"""L4 rhythmic in-thought recall — the portable "metronome" controller.

L3 fires one recall pulse per turn. L4 fires MANY pulses within a single
reasoning episode: as the thought evolves (each tool call, each reasoning step),
memory re-queries on the *current* state and, when something genuinely relevant
turns up that hasn't already surfaced this episode, pushes it back in to steer
the next step.

The defining property is **event-driven cadence, not a constant beat**: a pulse
fires only when the thought has meaningfully MOVED since the last one (a debounce
on token overlap), and only when a hit clears a floor set a notch above the
per-turn L3 push — a mid-thought interruption is more intrusive than a
once-per-turn block, so it gates harder.

PORTABLE BY DESIGN (#276/#332): this module knows nothing about any AI host. It
takes the evolving-thought text plus an in-memory `Episode` and returns a block
to inject, or None. Each host wires it to its own inner loop:
  - a local model you own (Elira): the tool-call loop ticks `pulse()` between
    reasoning steps — see `Videos/Elira/elira_engram.rhythmic_pulse`;
  - any other agent: call `pulse()` at its reasoning checkpoints.
Nothing here is OS-specific; the only host-specific code is the thin caller.

ADDITIVE, never a takeover: it only ADDS a block; it never replaces or owns the
host's native context. Respects the same switches as L3 (`ingest_mode=explicit`
off entirely; `config rhythmic_recall off` off on its own).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .adapters.native_memory import auto_background_enabled
from .core import RHYTHMIC_RECALL_COS, MemoryStore
from .multistore import get_config
from .proactive import (cross_pulse_dedup_on, format_block, injected_this_session,
                        mark_injected, relevant_memories, resolve_active_project)

DEFAULT_LIMIT = 2         # tinier than L3's per-turn block — a nudge mid-thought
DEFAULT_MAX_CHARS = 400
MIN_THOUGHT_CHARS = 24    # a reasoning step carries more than a bare "yes"
DEFAULT_MAX_PULSES = 4    # bound pulses per episode so it can't fire every step
DEBOUNCE_OVERLAP = 0.6    # skip if the thought barely moved since the last pulse

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _overlap(a: str, b: str) -> float:
    """Jaccard token overlap — a cheap, embedder-free "has the thought moved?"
    measure for the cadence debounce. 1.0 = identical token sets, 0.0 = disjoint."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class Episode:
    """In-memory state for one reasoning episode (one host turn / tool-loop).
    Created fresh per episode; never persisted. Tracks what has already pulsed so
    each pulse adds NEW context, and the last query so the debounce can tell
    whether the thought has moved."""
    pulsed_ids: set[int] = field(default_factory=set)
    last_query: str = ""
    pulse_count: int = 0


def pulse(store: MemoryStore, thought: str, episode: Episode, *,
          limit: int | None = None, max_chars: int | None = None,
          floor: float | None = None, max_pulses: int | None = None,
          active_project: str | None = None,
          session_id: str | None = None) -> str | None:
    """One metronome beat: a "possibly-relevant past" block for the CURRENT
    evolving `thought`, or None when disabled / the thought is trivial or hasn't
    moved / the episode's pulse budget is spent / nothing clears the floor.
    Mutates `episode` only when it actually returns a block."""
    if not auto_background_enabled(store):              # ingest_mode=explicit
        return None
    if get_config(store, "rhythmic_recall", "on") in ("off", "0", "false"):
        return None
    if not thought or len(thought.strip()) < MIN_THOUGHT_CHARS:
        return None

    if max_pulses is None:
        max_pulses = int(get_config(store, "rhythmic_recall_max_pulses",
                                    str(DEFAULT_MAX_PULSES)))
    if episode.pulse_count >= max_pulses:
        return None
    # event-driven cadence: only fire when the thought has meaningfully moved
    if episode.last_query and _overlap(thought, episode.last_query) >= DEBOUNCE_OVERLAP:
        return None

    if limit is None:
        limit = int(get_config(store, "rhythmic_recall_limit", str(DEFAULT_LIMIT)))
    if max_chars is None:
        max_chars = int(get_config(store, "rhythmic_recall_max_chars",
                                   str(DEFAULT_MAX_CHARS)))
    if floor is None:
        floor = float(get_config(store, "rhythmic_recall_floor",
                                  str(RHYTHMIC_RECALL_COS)))

    # Cross-pulse dedup: also skip anything ALREADY pushed this session by L3 or a
    # prior L4 tick (not just this episode's pulses), so the two rungs don't repeat
    # each other. The session set is shared with the L3 hook.
    dedup = cross_pulse_dedup_on(store)
    exclude = set(episode.pulsed_ids)
    if dedup:
        exclude |= injected_this_session(store, session_id)
    rows = relevant_memories(
        store, thought, limit=limit, floor=floor, exclude_ids=exclude,
        active_project=resolve_active_project(store, active_project,
                                              session_id=session_id))
    block = format_block(rows, max_chars)
    if not block:
        # the thought still counts as the latest query, so an unchanged next
        # step debounces against it rather than re-querying the same miss
        episode.last_query = thought
        return None
    ids = [r["id"] for r in rows]
    episode.pulsed_ids.update(ids)
    episode.last_query = thought
    episode.pulse_count += 1
    if dedup:
        # add to the session-shared set so L3 and later L4 ticks won't re-push these
        mark_injected(store, session_id, ids)
    # count the PUSH impression — per-episode dedup (pulsed_ids) keeps a memory
    # from being counted twice in one reasoning episode (best-effort; read-only
    # stores skip).
    try:
        store.record_surfaced(ids)
    except Exception:
        pass
    return block
