import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fornixdb.adapters.claude_code_transcripts import import_project_dir, summarize_session
from fornixdb.adapters.markdown_import import import_directory, parse_frontmatter
from fornixdb.core import MemoryStore
from fornixdb.db import connect

MD_FILE = """---
name: test-memory-one
description: A short description line
metadata:
  type: feedback
---

Body of the memory with a link to [[test-memory-two]].
"""

MD_FILE_2 = """---
name: test-memory-two
description: Second memory
metadata:
  type: reference
---

Plain body.
"""


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


class TestMarkdownImport(unittest.TestCase):
    def test_frontmatter(self):
        meta, body = parse_frontmatter(MD_FILE)
        self.assertEqual(meta["name"], "test-memory-one")
        self.assertEqual(meta["metadata"]["type"], "feedback")
        self.assertTrue(body.startswith("Body of the memory"))

    def test_import_and_links(self):
        s = mem_store()
        with tempfile.TemporaryDirectory() as d:
            Path(d, "one.md").write_text(MD_FILE)
            Path(d, "two.md").write_text(MD_FILE_2)
            Path(d, "MEMORY.md").write_text("# index — must be skipped")
            r = import_directory(s, d)
        self.assertEqual(r["imported"], 2)
        self.assertEqual(r["links"], 1)
        m = s.show("test-memory-one", reinforce=False)
        self.assertEqual(m["kind"], "feedback")
        self.assertEqual(m["links"][0]["related_gist"], "Second memory")

    def test_idempotent(self):
        s = mem_store()
        with tempfile.TemporaryDirectory() as d:
            Path(d, "one.md").write_text(MD_FILE)
            import_directory(s, d)
            r2 = import_directory(s, d)
        self.assertEqual(r2["imported"], 0)
        self.assertEqual(s.stats()["memories"], 1)


def _transcript_line(type_, ts, content=None, **kw):
    d = {"type": type_, "timestamp": ts, "sessionId": "abc", **kw}
    if content is not None:
        d["message"] = {"role": type_, "content": content}
    return json.dumps(d)


def write_session(d, name="abc"):
    lines = [
        json.dumps({"type": "mode", "mode": "normal"}),
        _transcript_line("user", "2026-06-03T12:57:16.875Z",
                         "Fix the login bug please", gitBranch="master",
                         cwd="/tmp/proj"),
        _transcript_line("assistant", "2026-06-03T12:58:00.000Z", "On it"),
        _transcript_line("user", "2026-06-03T14:00:00.000Z",
                         [{"type": "text", "text": "Now add a test"}]),
        _transcript_line("user", "2026-06-03T14:01:00.000Z",
                         "<command-name>/usage</command-name>"),  # filtered
    ]
    path = Path(d, f"{name}.jsonl")
    path.write_text("\n".join(lines))
    return path


class TestTranscriptImport(unittest.TestCase):
    def _write_session(self, d, name="abc"):
        write_session(d, name)

    def test_summarize(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_session(d)
            s = summarize_session(Path(d, "abc.jsonl"))
        self.assertEqual(s["user_turns"], 2)
        self.assertEqual(s["first_prompt"], "Fix the login bug please")
        self.assertEqual(s["last_prompt"], "Now add a test")
        self.assertEqual(s["branch"], "master")
        # timestamps are converted from UTC to local time on import
        def local(ts):
            return (datetime.fromisoformat(ts).astimezone()
                    .replace(tzinfo=None, microsecond=0).isoformat())
        self.assertEqual(s["started"], local("2026-06-03T12:57:16.875+00:00"))
        self.assertEqual(s["ended"], local("2026-06-03T14:01:00+00:00"))

    def test_tool_result_payload_never_ingested(self):
        # B2: a tool RESULT (web page / file / command output) arrives as a
        # `user`-typed entry; its payload must never reach a gist or detail.
        sentinel = "SENSITIVE_WEB_TOKEN_ABC123"
        with tempfile.TemporaryDirectory() as d:
            lines = [
                _transcript_line("user", "2026-06-03T12:00:00.000Z",
                                 "Look up the weather", gitBranch="main",
                                 cwd="/tmp/p"),
                _transcript_line("assistant", "2026-06-03T12:00:05.000Z",
                                 "Checking"),
                _transcript_line("user", "2026-06-03T12:00:10.000Z",
                                 [{"type": "tool_result", "tool_use_id": "t1",
                                   "content": [{"type": "text",
                                                "text": f"{sentinel} https://evil.example"}]}]),
                _transcript_line("user", "2026-06-03T12:01:00.000Z",
                                 [{"type": "text", "text": "Thanks, save that"}]),
            ]
            path = Path(d, "tr.jsonl")
            path.write_text("\n".join(lines))
            s = summarize_session(path)
            self.assertEqual(s["user_turns"], 2)  # tool result not a prompt
            self.assertEqual(s["first_prompt"], "Look up the weather")
            self.assertEqual(s["last_prompt"], "Thanks, save that")
            store = mem_store()
            import_project_dir(store, d)
            row = store.timeline("2026-06-03T00:00:00", "2026-06-04T00:00:00")[0]
            detail = store.show(row["id"], reinforce=False)["detail"] or ""
            blob = row["gist"] + detail
            self.assertNotIn(sentinel, blob)
            self.assertNotIn("evil.example", blob)

    def test_import_idempotent_and_episodic(self):
        s = mem_store()
        with tempfile.TemporaryDirectory() as d:
            self._write_session(d)
            r1 = import_project_dir(s, d)
            r2 = import_project_dir(s, d)
        self.assertEqual(r1["imported"], 1)
        self.assertEqual(r2["imported"], 0)
        st = s.stats()
        self.assertEqual(st["by_kind"]["episodic"], 1)
        self.assertEqual(st["sessions"], 1)
        rows = s.timeline("2026-06-03T00:00:00", "2026-06-04T00:00:00")
        self.assertEqual(len(rows), 1)
        self.assertIn("Fix the login bug", rows[0]["gist"])


class TestSessionEndCapture(unittest.TestCase):
    """The SessionEnd hook adapter: live passive capture at session end."""

    def setUp(self):
        from fornixdb.adapters.claude_code_session_end import capture_session
        self.capture = capture_session
        self.s = mem_store()

    def test_captures_one_episodic_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_session(d)
            out = self.capture(self.s, path)
        self.assertIn("captured #", out)
        st = self.s.stats()
        self.assertEqual(st["by_kind"]["episodic"], 1)
        self.assertEqual(st["sessions"], 1)
        rows = self.s.timeline("2026-06-03T00:00:00", "2026-06-04T00:00:00")
        self.assertIn("Fix the login bug", rows[0]["gist"])

    def test_resumed_session_refreshes_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_session(d)
            self.capture(self.s, path)
            with path.open("a") as fh:  # the session resumed and grew
                fh.write("\n" + _transcript_line(
                    "user", "2026-06-03T15:00:00.000Z", "One more fix"))
            out = self.capture(self.s, path)
        self.assertIn("refreshed #", out)
        self.assertEqual(self.s.stats()["by_kind"]["episodic"], 1)  # same row
        row = self.s.show("1", reinforce=False)
        self.assertIn("One more fix", row["detail"])
        self.assertIn("3 user turns", row["gist"])

    def test_session_capture_off_skips(self):
        from fornixdb.multistore import set_config
        set_config(self.s, "session_capture", "off")
        with tempfile.TemporaryDirectory() as d:
            out = self.capture(self.s, write_session(d))
        self.assertIn("skipped", out)
        self.assertEqual(self.s.stats()["memories"], 0)

    def test_backfill_skips_hook_captured_session(self):
        with tempfile.TemporaryDirectory() as d:
            self.capture(self.s, write_session(d))
            r = import_project_dir(self.s, d)
        self.assertEqual(r["imported"], 0)
        self.assertEqual(r["skipped"], 1)

    def test_missing_transcript_skips(self):
        out = self.capture(self.s, "/nonexistent/nope.jsonl")
        self.assertIn("skipped", out)
        self.assertEqual(self.s.stats()["memories"], 0)

    def test_main_reads_hook_json_on_stdin(self):
        import subprocess
        import sys as _sys
        with tempfile.TemporaryDirectory() as d:
            path = write_session(d)
            db = Path(d, "hook.db")
            payload = json.dumps({"session_id": "abc",
                                  "transcript_path": str(path),
                                  "hook_event_name": "SessionEnd"})
            r = subprocess.run(
                [_sys.executable, "-m", "fornixdb.adapters.claude_code_session_end",
                 "--db", str(db)],
                input=payload, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0)
            self.assertIn("captured #", r.stderr)
            with MemoryStore(db_path=db) as s2:
                self.assertEqual(s2.stats()["by_kind"]["episodic"], 1)


if __name__ == "__main__":
    unittest.main()
