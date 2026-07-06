"""The salience gate — dense sampling in, sparse memories out (SENSES.md §gate).

A sense samples continuously (frames at ~10 Hz, rolling audio windows, sensor
readings). Almost none of that deserves to become a memory: the gate commits a
sample only when the present diverges from the recent past, the way human
episodic boundaries sit at prediction-error spikes.

Mechanism, per sample:

  distance   d = 1 − cos(sample, EMA of recent samples)
  event      d > threshold while the gate is armed → commit, then DISARM so a
             single scene change commits once, not ten times a second
  re-arm     the gate re-arms when d falls back under rearm_below (hysteresis)
  heartbeat  a commit fires anyway after heartbeat_seconds of quiet —
             "nothing happened, here's proof" anchors the timeline cheaply
  first      the first sample always commits (there is no past yet)

The gate is pure and hardware-free: callers feed it embedding vectors and a
monotonic timestamp, it answers commit/hold. Capture loops own the sampling;
`senses` owns what a committed sample becomes. Stdlib only, like the core.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

__all__ = ["Decision", "SalienceGate", "cosine"]


def cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity; 0.0 when either vector has no magnitude."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class Decision:
    commit: bool
    reason: str | None   # "first" | "event" | "heartbeat" | None
    distance: float      # 1 − cos(sample, reference) at decision time


class SalienceGate:
    """One gate per stream (one camera, one microphone, one sensor bundle).

    threshold          embedding distance that counts as "something happened"
    rearm_below        distance the scene must settle back under before the
                       next event can fire (defaults to 60% of threshold)
    ema_alpha          how fast the reference absorbs the present; higher =
                       the new scene becomes "normal" sooner
    heartbeat_seconds  max quiet time between commits (0 disables heartbeats)
    """

    def __init__(self, *, threshold: float = 0.35,
                 rearm_below: float | None = None,
                 ema_alpha: float = 0.15,
                 heartbeat_seconds: float = 600.0) -> None:
        if not 0.0 < threshold <= 2.0:
            raise ValueError(f"threshold must be in (0, 2] (got {threshold})")
        self.threshold = threshold
        self.rearm_below = (threshold * 0.6 if rearm_below is None
                            else rearm_below)
        if self.rearm_below >= threshold:
            raise ValueError("rearm_below must sit under threshold "
                             f"({self.rearm_below} >= {threshold})")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1] (got {ema_alpha})")
        self.ema_alpha = ema_alpha
        self.heartbeat_seconds = heartbeat_seconds
        self._ema: list[float] | None = None
        self._armed = True
        self._last_commit: float | None = None

    def observe(self, vector: list[float], t: float | None = None) -> Decision:
        """Feed one sample; returns whether it should become a memory."""
        if t is None:
            t = time.monotonic()

        if self._ema is None:
            self._ema = list(vector)
            self._last_commit = t
            return Decision(True, "first", 1.0)

        d = 1.0 - cosine(vector, self._ema)
        # reference tracks the present regardless of the decision: a committed
        # novelty must become the new "normal" so the gate can re-arm on it
        a = self.ema_alpha
        self._ema = [(1.0 - a) * r + a * v for r, v in zip(self._ema, vector)]

        if self._armed and d > self.threshold:
            self._armed = False
            self._last_commit = t
            return Decision(True, "event", d)
        if not self._armed and d < self.rearm_below:
            self._armed = True

        if (self.heartbeat_seconds > 0 and self._last_commit is not None
                and t - self._last_commit >= self.heartbeat_seconds):
            self._last_commit = t
            return Decision(True, "heartbeat", d)

        return Decision(False, None, d)
