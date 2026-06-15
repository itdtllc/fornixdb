"""Back-fill episodic memories from Claude Code session transcripts (JSONL).

OPTIONAL adapter — the fornixdb core has no dependency on Claude Code. This
exists because Claude Code happens to persist full session transcripts at
~/.claude/projects/<project>/<session-id>.jsonl, which is a ready-made raw
episodic archive. We read it (never modify it) and emit one episodic row per
session: time span, project, and an algorithmic gist (Design §12 decision 8:
no model required for the baseline import).

Ingestion is gated to the owner's own words (B2, security assessment #176): a
session row is built ONLY from real user prompts. Tool RESULTS — which Claude
Code records as `tool_result` blocks inside `user`-typed entries, and which
carry fetched web pages, file contents, and command output — are dropped at
this boundary and never enter a gist or detail. Assistant turns are counted,
not quoted.

Usage:
    python3 -m fornixdb.adapters.claude_code_transcripts <projects-dir-or-project> [--db PATH]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..core import MemoryStore

MAX_GIST_PROMPT = 140
MAX_DETAIL_PROMPT = 600


def _to_local(ts: str) -> str:
    """UTC transcript timestamp → naive local ISO, so 'yesterday' means the
    owner's yesterday, not UTC's."""
    try:
        return (datetime.fromisoformat(ts.replace("Z", "+00:00"))
                .astimezone().replace(tzinfo=None, microsecond=0).isoformat())
    except ValueError:
        return ts


def _text_of(content) -> str:
    """Flatten a user-message content field (string or block list) to text.
    Only genuine `text` blocks are kept — tool_result / image / other block
    types are never flattened in (B2: tool payloads stay out of memory)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _is_tool_result(content) -> bool:
    """True if a `user`-typed transcript entry is actually a tool RESULT fed
    back to the model (web pages, file contents, command output) rather than
    the owner speaking. These are dropped wholesale at ingestion (B2) so their
    payload can never reach a memory — we key rows off real prompts only."""
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _is_real_prompt(text: str) -> bool:
    t = text.strip()
    return bool(t) and not t.startswith("<") and not t.startswith("Caveat:")


def summarize_session(path: Path) -> dict | None:
    """One pass over a transcript: time span, branch, counts, first/last prompts."""
    session_id = path.stem
    first_ts = last_ts = None
    branch = cwd = None
    user_prompts: list[str] = []
    assistant_turns = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = d.get("timestamp")
            if ts:
                first_ts = first_ts or ts
                last_ts = ts
            if d.get("type") == "user" and not d.get("isSidechain"):
                content = (d.get("message") or {}).get("content")
                if _is_tool_result(content):
                    continue  # tool payload, not a prompt — never ingested (B2)
                branch = d.get("gitBranch") or branch
                cwd = d.get("cwd") or cwd
                text = _text_of(content)
                if _is_real_prompt(text):
                    user_prompts.append(" ".join(text.split()))
            elif d.get("type") == "assistant" and not d.get("isSidechain"):
                assistant_turns += 1

    if not first_ts or not user_prompts:
        return None  # empty / non-conversation file
    return {
        "session_id": session_id,
        "started": _to_local(first_ts),
        "ended": _to_local(last_ts or first_ts),
        "branch": branch,
        "cwd": cwd,
        "user_turns": len(user_prompts),
        "assistant_turns": assistant_turns,
        "first_prompt": user_prompts[0],
        "last_prompt": user_prompts[-1],
    }


def import_project_dir(store: MemoryStore, project_dir: str | Path) -> dict:
    """Import every session transcript in one ~/.claude/projects/<project> dir."""
    project_dir = Path(project_dir).expanduser()
    project = project_dir.name.split("-")[-1] or project_dir.name
    imported = skipped = 0

    for path in sorted(project_dir.glob("*.jsonl")):
        session_id = path.stem
        if store.conn.execute(
            "SELECT 1 FROM memory WHERE session_id = ? AND source = 'claude-code-transcript'",
            (session_id,),
        ).fetchone():
            skipped += 1
            continue
        s = summarize_session(path)
        if s is None:
            skipped += 1
            continue
        date = s["started"][:10]
        gist = (f"Session {date} ({s['user_turns']} user turns"
                + (f", branch {s['branch']}" if s["branch"] else "")
                + f"): {s['first_prompt'][:MAX_GIST_PROMPT]}")
        detail = (
            f"Claude Code session {session_id}\n"
            f"Project dir: {s['cwd'] or project_dir}\n"
            f"Span: {s['started']} → {s['ended']}\n"
            f"Turns: {s['user_turns']} user / {s['assistant_turns']} assistant\n\n"
            f"Opening request:\n{s['first_prompt'][:MAX_DETAIL_PROMPT]}\n\n"
            f"Closing request:\n{s['last_prompt'][:MAX_DETAIL_PROMPT]}\n\n"
            f"Full transcript: {path}"
        )
        store.store(
            gist, detail,
            kind="episodic",
            project=project,
            event_time=s["started"],
            event_time_end=s["ended"],
            session_id=session_id,
            salience=0.4,  # raw sessions start below hand-stored memories
            source="claude-code-transcript",
            source_ref=str(path),
        )
        store.record_session(session_id, project=project, started=s["started"],
                             ended=s["ended"], source="claude-code",
                             source_ref=str(path))
        imported += 1

    return {"project": project, "imported": imported, "skipped": skipped}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="a ~/.claude/projects/<project> dir, or the "
                                 "projects root to import all of them")
    ap.add_argument("--db")
    args = ap.parse_args(argv)
    store = MemoryStore(db_path=args.db)
    root = Path(args.path).expanduser()

    dirs = [root]
    if not list(root.glob("*.jsonl")):  # projects root → recurse one level
        dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and list(d.glob("*.jsonl"))]

    total = 0
    for d in dirs:
        r = import_project_dir(store, d)
        total += r["imported"]
        print(f"{r['project']}: imported {r['imported']}, skipped {r['skipped']}")
    print(f"total imported: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
