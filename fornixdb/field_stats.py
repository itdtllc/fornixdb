"""`fornixdb field-stats` — the per-beat readout of the L5 field log.

The floor log answers per-ROW questions (where should the floor sit); this
answers the per-BEAT questions the L5 gate will ask after dogfood accrues:

  - settle rate: how often the field found a pattern vs degraded to L4
  - domain contribution: which domains actually light winning clusters
    (the evidence for trimming `parallel_domains`)
  - glue split: are winners held together by links or by shared topics —
    topic-only glue is the false-positive channel to watch (IDF-damping case)
  - dissent shadow: how often a minority report EXISTED while the tension
    line was off — the data for the `parallel_dissent` on/off decision
  - cost: per-beat wall time (the one-embed promise, on real hardware)

Reads field_log.jsonl (written beside the store whenever `floor_log` is on).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_beats(path: str | Path | None) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    beats = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(d, dict):
            beats.append(d)
    return beats


def summarize(beats: list[dict]) -> dict:
    """Four DISJOINT beat buckets (settled × emitted). A settled beat whose
    gists were all session-deduped or trimmed away emits nothing — it is
    `settled_quiet` (cost ≈ 0), NOT an abstention; conflating them once made
    `degraded` go negative on the live log."""
    n = len(beats)
    settled = [b for b in beats if b.get("settled")]
    settled_emitted = [b for b in settled if b.get("emitted")]
    degraded_emitted = [b for b in beats
                        if not b.get("settled") and b.get("emitted")]
    abstained = [b for b in beats
                 if not b.get("settled") and not b.get("emitted")]
    domain_wins: dict[str, int] = {}
    for b in settled:
        for d in b.get("winner_domains") or ():
            domain_wins[d] = domain_wins.get(d, 0) + 1
    glue = {"links_only": 0, "topics_only": 0, "mixed": 0}
    for b in settled:
        lg, tg = b.get("link_glue") or 0, b.get("topic_glue") or 0
        if lg and tg:
            glue["mixed"] += 1
        elif lg:
            glue["links_only"] += 1
        elif tg:
            glue["topics_only"] += 1
    times = sorted(float(b["ms"]) for b in beats if b.get("ms") is not None)
    return {
        "beats": n,
        "settled": len(settled),
        "settled_emitted": len(settled_emitted),
        "settled_quiet": len(settled) - len(settled_emitted),
        "degraded": len(degraded_emitted),
        "abstained": len(abstained),
        "domain_wins": dict(sorted(domain_wins.items(),
                                   key=lambda kv: -kv[1])),
        "glue": glue,
        "neighborhood_emitted": sum(1 for b in beats
                                    if b.get("neighborhood_emitted")),
        "dissent_shadow": sum(1 for b in settled if b.get("dissent_shadow")),
        "dissent_emitted": sum(1 for b in settled if b.get("dissent_emitted")),
        "ms_median": times[len(times) // 2] if times else None,
        "ms_max": times[-1] if times else None,
    }


def format_report(s: dict, path: str | None) -> str:
    if not s["beats"]:
        return (f"field log: {path or '(none)'}\nno beats recorded yet — the "
                "log fills as L5 pulses fire with `floor_log on`.")
    lines = [f"field log: {path}",
             f"beats: {s['beats']}  settled={s['settled']} "
             f"(emitted={s['settled_emitted']}, quiet={s['settled_quiet']} — "
             "all gists already injected this session, cost ~0)  "
             f"degraded-to-L4={s['degraded']}  abstained={s['abstained']}"]
    if s["domain_wins"]:
        lines.append("domains in winning clusters:")
        for d, c in s["domain_wins"].items():
            lines.append(f"  {d:<13} {c}")
    g = s["glue"]
    lines.append(f"winner glue:  links-only={g['links_only']}  "
                 f"topics-only={g['topics_only']}  mixed={g['mixed']}"
                 + ("   <- watch topics-only (the noise channel)"
                    if g["topics_only"] > g["links_only"] + g["mixed"] else ""))
    lines.append(f"neighborhood rows emitted: {s['neighborhood_emitted']} beat(s)")
    lines.append(f"dissent: shadow existed on {s['dissent_shadow']} settled "
                 f"beat(s), tension line reached the block on {s['dissent_emitted']} "
                 "(emitted = the id survived render+trim into the injected block, "
                 "NOT just computed; shadow>0 with `parallel_dissent` off keeps "
                 "emitted=0 — turning it on is what lets emitted rise)")
    if s["ms_median"] is not None:
        lines.append(f"cost: median {s['ms_median']:.0f} ms/beat, "
                     f"max {s['ms_max']:.0f} ms")
    return "\n".join(lines)
