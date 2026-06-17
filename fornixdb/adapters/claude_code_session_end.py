"""Claude Code SessionEnd hook — passive episodic capture, live.

OPTIONAL adapter, the hook-shaped twin of claude_code_transcripts: that one
back-fills history in bulk; this one fires when a session ENDS and stores the
session's episodic row immediately, so "what did we do this morning" works
without waiting for a back-fill pass. Parity with the in-process shim's auto session capture.

Wire it in Claude Code settings.json:

    {"hooks": {"SessionEnd": [{"hooks": [{"type": "command", "command":
        "/path/.venv/bin/python -m fornixdb.adapters.claude_code_session_end --db /path/store/memory.db"}]}]}}

The hook reads Claude Code's JSON payload on stdin (session_id,
transcript_path), summarizes that one transcript (read-only, algorithmic — no
model), and writes ONE episodic row. A resumed session that ends again
refreshes its row in place (same session, more complete record — not a
meaning change). The passive layer stays owner-toggleable: `config
session_capture off` disables capture, matching the reference shim's shell. Always exits 0
— memory capture must never make ending a session look like an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core import MemoryStore, now_iso
from ..multistore import get_config
from .claude_code_transcripts import (MAX_DETAIL_PROMPT, MAX_GIST_PROMPT,
                                      summarize_session)

SOURCE = "claude-code-transcript"  # same tag as the back-fill, so each skips
                                   # sessions the other has already captured


def _compose(s: dict, path: Path, project: str) -> tuple[str, str]:
    date = s["started"][:10]
    gist = (f"Session {date} ({s['user_turns']} user turns"
            + (f", branch {s['branch']}" if s["branch"] else "")
            + f"): {s['first_prompt'][:MAX_GIST_PROMPT]}")
    detail = (
        f"Claude Code session {s['session_id']}\n"
        f"Project dir: {s['cwd'] or path.parent}\n"
        f"Span: {s['started']} → {s['ended']}\n"
        f"Turns: {s['user_turns']} user / {s['assistant_turns']} assistant\n\n"
        f"Opening request:\n{s['first_prompt'][:MAX_DETAIL_PROMPT]}\n\n"
        f"Closing request:\n{s['last_prompt'][:MAX_DETAIL_PROMPT]}\n\n"
        f"Full transcript: {path}"
    )
    return gist, detail


def capture_session(store: MemoryStore, transcript_path: str | Path,
                    session_id: str | None = None) -> str:
    """Store (or refresh) the episodic row for one session transcript.
    Returns a short status string for the hook log."""
    from .native_memory import ingest_mode
    if ingest_mode(store) == "explicit":  # the user's "no background" switch
        return "ingest_mode=explicit — background capture off"
    if get_config(store, "session_capture", "on") in ("off", "0", "false"):
        return "session_capture off — skipped"
    path = Path(transcript_path).expanduser()
    if not path.is_file():
        return f"no transcript at {path} — skipped"
    s = summarize_session(path)
    if s is None:
        return "empty session — skipped"
    session_id = session_id or s["session_id"]
    project = path.parent.name.split("-")[-1] or path.parent.name
    gist, detail = _compose(s, path, project)

    existing = store.conn.execute(
        "SELECT id FROM memory WHERE session_id = ? AND source = ?",
        (session_id, SOURCE)).fetchone()
    if existing:
        # the same session ended again (resume): refresh the derived row in
        # place and drop its stale vector, like set_gist does
        store._check_writable()
        store.conn.execute(
            """UPDATE memory SET gist = ?, detail = ?, event_time = ?,
                                 event_time_end = ?, recorded_time = ?
               WHERE id = ?""",
            (gist, detail, s["started"], s["ended"], now_iso(), existing["id"]))
        store.conn.execute("DELETE FROM embedding WHERE memory_id = ?",
                           (existing["id"],))
        store.conn.commit()
        mem_id, verb = existing["id"], "refreshed"
    else:
        mem_id = store.store(
            gist, detail, kind="episodic", project=project,
            event_time=s["started"], event_time_end=s["ended"],
            session_id=session_id, salience=0.4, source=SOURCE,
            source_ref=str(path))
        verb = "captured"
    store.record_session(session_id, project=project, started=s["started"],
                         ended=s["ended"], source="claude-code",
                         source_ref=str(path))
    try:  # vectors are an upgrade, never a requirement — and never a failure
        from ..vectors import embed_memory, get_default_embedder
        emb = get_default_embedder()
        if emb is not None:
            embed_memory(store, emb, mem_id)
    except Exception:
        pass
    return f"{verb} #{mem_id} ({s['user_turns']} user turns)"


def main(argv=None) -> int:
    # Claude Code writes the hook JSON (and the transcript) as UTF-8; Python's
    # piped stdio defaults to the OS code page on Windows (cp1252), which would
    # mangle a non-ASCII transcript_path or prompt; and the status lines this
    # prints to stderr carry `—`, which mojibakes to `�` on cp1252. Force UTF-8
    # on all three streams.
    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", help="store path (default: $FORNIXDB_DB or default)")
    ap.add_argument("--transcript", help="transcript path (otherwise read from "
                                         "the hook JSON on stdin)")
    args = ap.parse_args(argv)

    transcript, session_id = args.transcript, None
    if not transcript:
        try:
            payload = json.loads(sys.stdin.read() or "{}")
            transcript = payload.get("transcript_path")
            session_id = payload.get("session_id")
        except json.JSONDecodeError:
            transcript = None
    if not transcript:
        print("fornixdb session-end: no transcript_path — skipped", file=sys.stderr)
        return 0

    try:
        with MemoryStore(db_path=args.db) as store:
            print(f"fornixdb session-end: "
                  f"{capture_session(store, transcript, session_id)}",
                  file=sys.stderr)
            # passive/both + a configured native dir: follow native memory
            # downstream (additive, never a takeover). explicit mode skips this.
            from .native_memory import auto_background_enabled, ingest, native_dir
            if auto_background_enabled(store) and native_dir(store):
                r = ingest(store)
                if r.get("ok"):
                    print(f"fornixdb session-end: native ingest — imported "
                          f"{r['imported']}, skipped {r['skipped']}", file=sys.stderr)
    except Exception as e:  # never make ending a session look like an error
        print(f"fornixdb session-end: error: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
