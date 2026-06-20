"""Claude Code L4 rhythmic-recall adapter — the metronome on the tool-call seam.

L3 (claude_code_recall.py, the UserPromptSubmit hook) fires ONE recall pulse per
user turn, on the prompt. L4 fires MANY pulses WITHIN a turn: as Claude works
through a task, each tool call is a checkpoint where the thought has moved, so
memory re-queries on the *current* state and pushes anything newly relevant back
in to steer the next step.

WHY THIS IS A DELIBERATELY-COARSER L4 (#332/#343): on a host whose reasoning loop
you own (Elira) the metronome can tick between every reasoning step. Claude Code
does NOT expose per-step ticks — the only mid-reasoning seam is the tool-call
hooks (PreToolUse / PostToolUse). So Claude Code's L4 granularity is per-tool-call,
not per-thought. It is the honest realization the host's seams allow, sharing the
SAME portable cadence core as Elira; only this thin edge differs.

SEAM CHOICE — PostToolUse is primary. The cue that makes a memory relevant usually
lives in a tool's RESULT (a grep hit, a test failure, a returned value), so we
pulse on (tool call + result) AFTER it runs, and `additionalContext` is placed
"next to the tool result" — exactly where the next reasoning step will read it.
PreToolUse is also supported (pulse on the intended call, before it runs) for the
"recall before you do X" case; wire whichever (or both) you want.

EPISODE STATE ACROSS PROCESSES. Each hook firing is a SEPARATE short-lived
process, so the in-memory cadence.Episode that Elira keeps for a whole turn does
not survive here. We persist it in the store config keyed by session_id and
reconstruct it each firing. It is reset at the start of each user turn — the L3
UserPromptSubmit hook bumps a per-session turn counter (see claude_code_recall),
and a turn change (or, as a standalone fallback when L3 isn't wired, a long idle
gap) starts a fresh episode so pulse_count/dedup are per-turn, matching Elira.

ADDITIVE and gated exactly like L3 (#2/#276): ingest_mode=explicit turns it off;
`config rhythmic_recall off` turns it off on its own; it only ADDS context.
Silence is the default — unsolicited mid-task interruption gates HARDER than the
per-turn push (RHYTHMIC_RECALL_COS), so most tool calls inject nothing.

Wire it in Claude Code settings.json (matcher "*" = every tool):

    {"hooks": {"PostToolUse": [{"matcher": "*", "hooks": [{"type": "command",
        "command": "/path/.venv/bin/python -m fornixdb.adapters.claude_code_cadence --db /path/store/fornix.db"}]}]}}

Always exits 0 — a hook must never make a tool call look like it failed, and a
silent tick (nothing relevant) is success, not error.
"""

from __future__ import annotations

import argparse
import json
import sys

from .. import cadence
from ..core import MemoryStore
from ..multistore import get_config, set_config

# Standalone fallback when the L3 turn counter isn't wired: a gap longer than
# this between tool calls is treated as a new reasoning episode. Within an active
# task tool calls are seconds apart, so this only fires across genuine idle.
IDLE_RESET_SECONDS = 120
# Bound the recall query: a tool input/result can be a whole file. Only a lead
# slice carries the "what is happening now" signal the pulse queries on.
THOUGHT_CHARS = 600

_EPISODE_KEY = "cadence_episode_"   # + session_id
_TURN_KEY = "cadence_turn_"         # + session_id  (bumped by the L3 hook)


def _episode_state_key(session_id: str | None) -> str:
    return _EPISODE_KEY + (session_id or "_nosess")


def _current_turn(store: MemoryStore, session_id: str | None) -> str:
    return get_config(store, _TURN_KEY + (session_id or "_nosess"), "0") or "0"


def bump_turn(store: MemoryStore, session_id: str | None) -> None:
    """Advance this session's turn counter — called by the L3 UserPromptSubmit
    hook at the start of each user turn so the L4 episode (pulse budget + dedup)
    resets per turn. Best-effort; a read-only store just skips it."""
    try:
        cur = int(_current_turn(store, session_id))
        set_config(store, _TURN_KEY + (session_id or "_nosess"), str(cur + 1))
    except Exception:
        pass


def load_episode(store: MemoryStore, session_id: str | None,
                 now: float) -> cadence.Episode:
    """Reconstruct this session's Episode, or a fresh one when the user turn has
    advanced (L3 bumped the turn counter) or — standalone — after a long idle.
    Carries `turn` and `ts` in the persisted blob so the next firing can decide."""
    raw = get_config(store, _episode_state_key(session_id), "") or ""
    turn = _current_turn(store, session_id)
    if raw:
        try:
            d = json.loads(raw)
            same_turn = str(d.get("turn", "")) == turn
            fresh_enough = (now - float(d.get("ts", 0))) < IDLE_RESET_SECONDS
            if same_turn and fresh_enough:
                ep = cadence.Episode(
                    pulsed_ids=set(d.get("pulsed_ids", [])),
                    last_query=d.get("last_query", ""),
                    pulse_count=int(d.get("pulse_count", 0)))
                ep._turn = turn  # type: ignore[attr-defined]
                return ep
        except Exception:
            pass
    ep = cadence.Episode()
    ep._turn = turn  # type: ignore[attr-defined]
    return ep


def save_episode(store: MemoryStore, session_id: str | None,
                 episode: cadence.Episode, now: float) -> None:
    """Persist episode state for the next tool-call firing (best-effort: a
    read-only store just skips it, like L3's dedup)."""
    try:
        set_config(store, _episode_state_key(session_id), json.dumps({
            "turn": getattr(episode, "_turn", _current_turn(store, session_id)),
            "ts": now,
            "pulsed_ids": sorted(episode.pulsed_ids),
            "last_query": episode.last_query,
            "pulse_count": episode.pulse_count,
        }))
    except Exception:
        pass


def build_thought(event: str, tool_name: str, tool_input: dict,
                  tool_response) -> str:
    """The evolving-thought query for this tool-call checkpoint: the tool and its
    arguments, plus — on PostToolUse — what it returned (the cue usually lives in
    the result). Bounded so a whole-file input/result can't blow up the query."""
    parts = [tool_name or ""]
    if tool_input:
        parts.append(json.dumps(tool_input, default=str)[:THOUGHT_CHARS])
    if event in ("PostToolUse", "PostToolUseFailure") and tool_response is not None:
        resp = (tool_response if isinstance(tool_response, str)
                else json.dumps(tool_response, default=str))
        parts.append(resp[:THOUGHT_CHARS])
    return " ".join(p for p in parts if p)


def main(argv=None) -> int:
    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", help="store path (default: $FORNIXDB_DB or default)")
    args = ap.parse_args(argv)

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    event = payload.get("hook_event_name") or "PostToolUse"
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response")
    session_id = payload.get("session_id")

    thought = build_thought(event, tool_name, tool_input, tool_response)
    if not thought:
        return 0

    import time
    now = time.time()
    try:
        with MemoryStore(db_path=args.db) as store:
            episode = load_episode(store, session_id, now)
            block = cadence.pulse(store, thought, episode)
            save_episode(store, session_id, episode, now)
            if block:
                # additionalContext for tool events lands next to the tool result,
                # where the next reasoning step reads it (plain stdout is NOT shown
                # for PreToolUse/PostToolUse — only via this field).
                print(json.dumps({"hookSpecificOutput": {
                    "hookEventName": event,
                    "additionalContext": block,
                }}))
    except Exception as e:  # never make a tool call look like it failed
        print(f"fornixdb rhythmic-recall: error: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
