"""Claude Code UserPromptSubmit hook — proactive relevance-triggered recall (§15.2 #3).

The PULL gap: today recall is opt-in — the AI must think to call `recall_memory`,
and `startup_context` fires only once, at session start. The most common memory
failure is therefore not bad retrieval but NEVER-TRIGGERED retrieval: as a
conversation drifts to a new topic mid-session, nothing re-surfaces unless the
model remembers to ask. This hook makes recall ambient: on each user turn it
runs a relevance-gated recall against the prompt and, when something genuinely
relevant turns up, emits a small provenance-tagged "possibly-relevant past"
block into the model's context — no tool call needed.

ADDITIVE, never a takeover (the #2 / #276 principle): this only ADDS a block; it
never replaces, intercepts, or owns the host's native memory injection. Delete
FornixDB and Claude Code's own memory is untouched.

Silence is the default. Nothing is injected unless a hit clears the relevance
floor — and unsolicited PUSH gates HIGHER than an explicit pull (PROACTIVE_RECALL_COS,
not the looser recall_memory include floor), with keyword-only anchors trusted
only in a vectors-off store. Unsolicited noise erodes trust faster than a missed
recall, so an empty turn is the common case.
Cost stays lean (top-K + a char budget) because the measured price of memory is
the PREFILL of what it adds to the prompt, not the recall itself.

Respects the same switches as the other background automation:
  - ingest_mode=explicit  → off entirely (the "leave my background alone" switch)
  - `config proactive_recall off` → off while other passive automation stays on
  - cross-turn dedup: a memory already injected this session is not re-injected

Wire it in Claude Code settings.json (stdout of a UserPromptSubmit hook is added
to the model's context):

    {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command":
        "/path/.venv/bin/python -m fornixdb.adapters.claude_code_recall --db /path/store/fornix.db"}]}]}}

Always exits 0 — proactive recall must never make submitting a prompt look like
an error, and a silent turn (no relevant memory) is success, not failure.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..core import MemoryStore
# The L4 cadence adapter owns the per-session turn counter; the L3 hook only
# advances it (one-directional, constant-light import — no heavy deps run).
from .claude_code_cadence import bump_turn as _bump_turn
# The relevance gate, block formatter, and per-turn orchestration are
# host-neutral and live in the core (`fornixdb.proactive`) so the L4 cadence
# controller can reuse them without depending on this Claude-Code adapter
# (#276/#332). Re-exported here for back-compat with existing imports.
from ..proactive import (  # noqa: F401
    HEADER,
    _format_block,
    format_block,
    proactive_recall,
    relevant_memories,
)


def main(argv=None) -> int:
    # Claude Code reads this hook's stdout as UTF-8 and writes the hook JSON as
    # UTF-8, but Python's piped stdio defaults to the OS code page on Windows
    # (cp1252) — so the header's `·`/`—` and any non-ASCII gist would round-trip
    # as `�`. Force UTF-8 on stdin/stdout — and stderr, whose diagnostics carry
    # `—` that would otherwise mojibake — so nothing the hook emits is corrupted.
    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", help="store path (default: $FORNIXDB_DB or default)")
    ap.add_argument("--prompt", help="prompt text (otherwise read from the hook "
                                     "JSON on stdin)")
    args = ap.parse_args(argv)

    prompt, session_id = args.prompt, None
    if prompt is None:
        try:
            payload = json.loads(sys.stdin.read() or "{}")
            prompt = payload.get("prompt") or ""
            session_id = payload.get("session_id")
        except json.JSONDecodeError:
            prompt = ""
    if not prompt:
        return 0  # nothing to recall on — silent success

    try:
        with MemoryStore(db_path=args.db) as store:
            # Bump the per-session turn counter the L4 cadence adapter reads to
            # scope its episode (pulse budget + dedup) to one user turn. Cheap and
            # best-effort: if L4 isn't wired this is just an unused config row, and
            # a read-only store simply skips it.
            if session_id:
                _bump_turn(store, session_id)
            block = proactive_recall(store, prompt, session_id=session_id)
            if block:
                # stdout of a UserPromptSubmit hook is added to the model's
                # context (the additive injection seam)
                print(block)
    except Exception as e:  # never make submitting a prompt look like an error
        print(f"fornixdb proactive-recall: error: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
