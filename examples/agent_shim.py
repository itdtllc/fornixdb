"""Reference integration shim: the INTEGRATION.md six-tool surface, in-process.

Adapt this to your agent framework: expose the six functions below as tools
(JSON-schema'd for an LLM, or however your shell does tools), inject
`startup_context()` as a system message, and call `record_session(...)` at
shell exit if you want passive episodic capture.

Two hard-won rules are baked in (PC portability test, 2026-06-11):

1. **Write tools are shell-gated.** A small model (14B-q4) will call
   `remember`/`forget` unprompted no matter what the system prompt says about
   capture mode. The capture policy is enforced HERE — the shell asks the
   owner — not by trusting the model. Pass `confirm=` a callable that asks
   your owner (the default auto-denies, which is the safe behavior for an
   unattended agent).
2. **Recall returns rows, not judgment.** Hand the rows to the model and
   instruct it to enumerate them; don't let it summarize "nothing happened"
   over a non-empty result.

Run `python examples/agent_shim.py` for a scripted smoke test on a temp store.
"""

from __future__ import annotations

import os
from datetime import datetime

from fornixdb.core import MemoryStore, now_iso
from fornixdb.multistore import (capture_mode, multi_recall, multi_timeline,
                                 open_stores)
from fornixdb.timeparse import parse_when


def _deny(question: str) -> bool:
    """Default write-gate: refuse. Replace with a real owner prompt."""
    return False


class AgentMemory:
    def __init__(self, db_path=None, agent: str = "agent", shared=True,
                 confirm=_deny):
        self.agent = agent
        self.store = MemoryStore(db_path=db_path)
        self.stores = open_stores(self.store, shared=shared)
        self.confirm = confirm

    def close(self):
        for _, s in self.stores:
            s.close()

    # ------------------------------------------------------------- read tools

    def recall_memory(self, query: str, limit: int = 5) -> list[dict]:
        """Subject-axis recall: ranked gists. For WHEN questions use
        recall_timeline — subject search can't grip time words."""
        rows = multi_recall(self.stores, query, limit=limit)
        return [self._row(r) for r in rows]

    def recall_timeline(self, when: str) -> list[dict]:
        """Time-axis recall: 'yesterday', 'last thursday', 'june 5', …"""
        start, end = parse_when(when)
        rows = multi_timeline(self.stores, start.isoformat(), end.isoformat())
        return [self._row(r) for r in rows]

    def list_memories(self) -> list[dict]:
        """Semantic memories only — sessions live on the time axis, and
        including them here grows the listing with every session."""
        rows = self.store.conn.execute(
            "SELECT * FROM memory WHERE kind != 'episodic' "
            "AND superseded_time IS NULL ORDER BY salience DESC LIMIT 50")
        return [self._row(dict(r)) for r in rows]

    def startup_context(self) -> str:
        """Inject as a system message at session start: the capture policy
        (read from the store, so the owner changes it with one CLI command)
        and the passive-capture disclosure, truthfully stated."""
        mode = capture_mode(self.store)
        lines = [
            f"You have a persistent memory (FornixDB). Capture mode: {mode}.",
            "Use recall_memory for subject questions and recall_timeline for"
            " 'when' questions; enumerate the rows a tool returns rather than"
            " summarizing judgment about them.",
            "Memory writes (remember/forget) require the owner's confirmation"
            " at the shell — request them, don't expect them to be silent.",
            "The shell stores one episodic memory per session at exit"
            " (passive capture); answer honestly if asked whether you record.",
        ]
        return "\n".join(lines)

    # ---------------------------------------------- write tools (shell-gated)

    def remember(self, title: str, content: str, kind: str = "semantic") -> str:
        """Same-title remember = update: stores the new version and
        supersedes the old (kept, tombstoned) — never a duplicate."""
        if not self.confirm(f"store memory '{title}'?"):
            return "not stored: owner declined (capture policy)"
        old = self.store.show(title, reinforce=False) if title else None
        new_id = self.store.store(content[:120], content, kind=kind,
                                  name=None if old else title or None,
                                  source=self.agent)
        if old:
            self.store.supersede(old["id"], new_id)
            return f"stored #{new_id} (supersedes #{old['id']})"
        from fornixdb.consolidate import supersede_suggestion
        sug = supersede_suggestion(self.store, new_id, content, kind)
        if sug and sug.get("reason") == "resolves":
            return (f"stored #{new_id} — looks like it CLOSES open task #{sug['id']} "
                    f"\"{sug['gist'][:60]}\"; supersede that one to close it instead "
                    f"of leaving it open.")
        if sug:
            return (f"stored #{new_id} — looks like an update of #{sug['id']} "
                    f"\"{sug['gist'][:60]}\" (cos {sug['cosine']}); re-remember under "
                    f"that title to supersede it instead of keeping both.")
        return f"stored #{new_id}"

    def forget_memory(self, title: str) -> str:
        """Forget = tombstone. Recoverable — FornixDB never deletes."""
        mem = self.store.show(title, reinforce=False)
        if mem is None:
            return f"no memory named '{title}'"
        if not self.confirm(f"forget memory '{title}' (#{mem['id']})?"):
            return "not forgotten: owner declined"
        self.store.tombstone(mem["id"])
        return f"#{mem['id']} retired (tombstoned, recoverable)"

    # ------------------------------------------------------- passive capture

    def record_session(self, gist: str, detail: str | None = None) -> None:
        """Call from the shell at exit (every exit path, never blocking)."""
        self.store.store(gist, detail, kind="episodic", source=self.agent,
                         session_id=f"{self.agent}-{now_iso()}")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _row(r: dict) -> dict:
        store_tag = r.get("_store") or ""  # '' = the agent's own store
        return {"id": f"{store_tag}:{r['id']}" if store_tag else str(r["id"]),
                "date": (r.get("event_time") or "")[:10],
                "kind": r["kind"], "gist": r["gist"],
                "shared": store_tag == "shared"}


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        mem = AgentMemory(db_path=Path(tmp) / "demo.db", shared=False,
                          confirm=lambda q: print(f"[owner gate] {q} -> yes") or True)
        print(mem.startup_context(), "\n")
        print(mem.remember("gpu-rule", "The LLM and a rendering GPU never coexist."))
        print(mem.recall_memory("can the model stay loaded during a render?"))
        print(mem.recall_timeline("today"))
        print(mem.forget_memory("gpu-rule"))
        mem.record_session("demo session: stored, recalled, forgot")
        mem.close()
    print("\nsmoke test done (temp store removed)")
