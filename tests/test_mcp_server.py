"""MCP adapter: protocol handling + a real stdio round-trip."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["FORNIXDB_TRANSCRIPTS"] = "off"  # dream must not scan this machine's
                                            # real ~/.claude/projects from tests

from fornixdb.adapters.mcp_server import TOOLS, FornixMCP
from fornixdb.multistore import set_config


def req(mid, method, **params):
    m = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params:
        m["params"] = params
    return m


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.srv = FornixMCP(db_path=Path(self.tmp.name) / "m.db", shared=False)

    def tearDown(self):
        self.srv.store.close()
        self.tmp.cleanup()

    def _call(self, name, **args):
        resp = self.srv.handle(req(9, "tools/call", name=name, arguments=args))
        return resp["result"]

    def test_initialize_and_list(self):
        r = self.srv.handle(req(1, "initialize",
                                protocolVersion="2025-06-18"))["result"]
        self.assertEqual(r["protocolVersion"], "2025-06-18")  # echoes the client
        self.assertEqual(r["serverInfo"]["name"], "fornixdb")
        self.assertIn("recall_timeline", r["instructions"])
        tools = self.srv.handle(req(2, "tools/list"))["result"]["tools"]
        names = {t["name"] for t in tools}
        # tools/list advertises the default-on set: the live senses ship off, so
        # they are defined in TOOLS but not advertised until a store opts in.
        self.assertEqual({t["name"] for t in TOOLS} - names,
                         {"look", "feel", "see", "recaption"})
        self.assertEqual(len(tools), 21)

    def test_remember_recall_show_forget_cycle(self):
        out = self._call("remember", title="gpu-rule",
                         content="The LLM and a rendering GPU never coexist.")
        self.assertIn("stored #1", out["content"][0]["text"])
        out = self._call("recall_memory", query="rendering GPU coexist")
        self.assertIn("never coexist", out["content"][0]["text"])
        out = self._call("show_memory", ref="gpu-rule")
        self.assertIn("name: gpu-rule", out["content"][0]["text"])
        # same-title remember = supersede, history kept
        out = self._call("remember", title="gpu-rule",
                         content="GPU exclusion holds on adopt paths too.")
        self.assertIn("supersedes #1", out["content"][0]["text"])
        out = self._call("forget_memory", ref="gpu-rule")
        self.assertIn("tombstoned, recoverable", out["content"][0]["text"])

    def _project_of(self, name):
        return self.srv.store.conn.execute(
            "SELECT project FROM memory WHERE name = ?", (name,)).fetchone()[0]

    def test_remember_explicit_project_arg(self):
        # The explicit arg is the reliable per-capture scope (the MCP server
        # can't see the host's per-session declared project).
        self._call("remember", title="vid-note", content="A videos finding.",
                   project="videos")
        self.assertEqual(self._project_of("vid-note"), "videos")

    def test_remember_falls_back_to_pinned_active_project(self):
        from fornixdb.multistore import set_config
        set_config(self.srv.store, "active_project", "fornixdb")
        self._call("remember", title="pinned-note", content="No project arg.")
        self.assertEqual(self._project_of("pinned-note"), "fornixdb")

    def test_remember_no_project_stays_null(self):
        self._call("remember", title="bare-note", content="Nothing pinned.")
        self.assertIsNone(self._project_of("bare-note"))

    def test_remember_many_batch_project_with_per_item_override(self):
        self._call("remember_many", project="videos", items=[
            {"title": "b1", "content": "batch one"},
            {"title": "b2", "content": "batch two", "project": "fornixdb"}])
        self.assertEqual(self._project_of("b1"), "videos")       # batch default
        self.assertEqual(self._project_of("b2"), "fornixdb")     # per-item wins

    def test_recent_writes_tracks_session_and_supersede(self):
        self.assertIn("no memories written",
                      self._call("recent_writes")["content"][0]["text"])
        self._call("remember", title="alpha", content="first fact")
        self._call("remember", title="beta", content="second fact")
        # re-remember beta -> supersedes, both ids are session writes
        self._call("remember", title="beta", content="second fact, revised")
        out = self._call("recent_writes")["content"][0]["text"]
        self.assertIn("#1", out)
        self.assertIn("#2", out)
        self.assertIn("#3", out)
        self.assertIn("[superseded]", out)  # #2 was superseded by #3

    def test_recall_max_chars(self):
        for i in range(6):
            self._call("remember", title=f"fact-{i}",
                       content=f"context budget fact number {i} with some padding text")
        full = self._call("recall_memory", query="context budget fact",
                          limit=6)["content"][0]["text"]
        capped = self._call("recall_memory", query="context budget fact",
                            limit=6, max_chars=150)["content"][0]["text"]
        self.assertGreater(full.count("budget fact"), capped.count("budget fact"))
        self.assertIn("more — raise max_chars", capped)
        self.assertLessEqual(len(capped), 150 + 80)  # body capped + one note line

    def test_memory_usage_and_shrink(self):
        for i in range(150):
            self._call("remember", title=f"bulk-{i}", content="z" * 4000,
                       kind="episodic")
        self.srv.store.conn.execute("VACUUM")
        out = self._call("memory_usage")["content"][0]["text"]
        self.assertIn("MB on disk", out)
        self.assertIn("no standing cap", out)
        # owner: "reduce this space to 0.3 MB" — true deletion, cap untouched
        out = self._call("shrink_memory", target_mb=0.3)["content"][0]["text"]
        self.assertIn("permanently forgotten", out)
        self.assertIn("no standing cap",
                      self._call("memory_usage")["content"][0]["text"])
        # already-under target is a stated no-op
        out = self._call("shrink_memory", target_mb=10_000)["content"][0]["text"]
        self.assertIn("nothing was deleted", out)

    def test_markdown_import_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            doc = Path(d) / "doc.md"
            doc.write_text("# Title\n\nIntro.\n\n## Section A\n\nBody A.\n")
            out = self._call("import_markdown", path=str(doc))["content"][0]["text"]
            self.assertIn("imported 2", out)            # Title + Section A
            out_dir = str(Path(d) / "out")
            exp = self._call("export_markdown", out_dir=out_dir)["content"][0]["text"]
            self.assertIn("exported 2", exp)
            # index is FornixDB.md by default, NEVER MEMORY.md (no collision with
            # Claude Code's own memory index)
            self.assertTrue((Path(out_dir) / "FornixDB.md").exists())
            self.assertFalse((Path(out_dir) / "MEMORY.md").exists())

    def test_export_markdown_options(self):
        self._call("remember", title="Title", content="Intro.")
        self._call("remember", title="Section A", content="Body A.")
        with tempfile.TemporaryDirectory() as d:
            # custom index name
            out_dir = str(Path(d) / "named")
            self._call("export_markdown", out_dir=out_dir, index_name="Index.md")
            self.assertTrue((Path(out_dir) / "Index.md").exists())
            self.assertFalse((Path(out_dir) / "FornixDB.md").exists())
            # single consolidated human-readable document
            doc_dir = str(Path(d) / "doc")
            r = self._call("export_markdown", out_dir=doc_dir,
                           single_file=True)["content"][0]["text"]
            self.assertIn("FornixDB-export.md", r)
            text = (Path(doc_dir) / "FornixDB-export.md").read_text()
            self.assertIn("# FornixDB Export", text)
            self.assertIn("## Title", text)
            self.assertNotIn("\nsalience:", text)   # no machine frontmatter
            # subject filter selects a subset
            sub = str(Path(d) / "sub")
            r = self._call("export_markdown", out_dir=sub,
                           query="Body A")["content"][0]["text"]
            self.assertIn("exported 1", r)
            # an unreadable time phrase is a clean tool result, not a crash
            r = self._call("export_markdown", out_dir=str(Path(d) / "x"),
                           when="blarghh")["content"][0]["text"]
            self.assertIn("couldn't export", r)

    def test_mark_irrelevant(self):
        self._call("remember", title="pie", content="apple pie recipe steps")
        out = self._call("mark_irrelevant", ref="1", query="apple pie recipe")
        self.assertIn("downweighted", out["content"][0]["text"])
        out = self._call("mark_irrelevant", ref="99", query="x")
        self.assertIn("no memory", out["content"][0]["text"])

    def test_timeline_and_startup(self):
        self._call("remember", title="t", content="today's fact")
        out = self._call("recall_timeline", when="today")
        self.assertIn("today's fact", out["content"][0]["text"])
        out = self._call("startup_context")
        self.assertIn("capture mode: suggest", out["content"][0]["text"])

    def test_startup_context_flags_consolidation_due(self):
        for i in range(5):
            self._call("remember", title=f"fact-{i}", content=f"durable fact number {i}")
        text = self._call("startup_context")["content"][0]["text"]
        self.assertIn("consolidation DUE", text)
        self.assertIn("sleep/dream", text)

    def test_startup_context_no_due_nag_for_tiny_store(self):
        self._call("remember", title="one", content="a single durable fact")
        text = self._call("startup_context")["content"][0]["text"]
        self.assertNotIn("consolidation DUE", text)

    def test_dream_tool_runs_and_marks_done(self):
        for i in range(3):
            self._call("remember", title=f"f{i}", content=f"durable fact {i}")
        out = self._call("dream", done=True)
        text = out["content"][0]["text"]
        self.assertFalse(out["isError"])
        self.assertIn("woke", text)             # the wake read-back
        # the pass reset the DUE clock
        startup = self._call("startup_context")["content"][0]["text"]
        self.assertNotIn("consolidation DUE", startup)

    def test_dream_tool_refused_on_read_only_store(self):
        set_config(self.srv.store, "frozen", "on")
        out = self._call("dream")
        self.assertTrue(out["isError"])
        self.assertIn("frozen", out["content"][0]["text"])

    def test_link_tool_accepts_a_reviewed_pair_as_distinct(self):
        self._call("remember", title="fact-a", content="durable fact a")
        self._call("remember", title="fact-b", content="durable fact b")
        text = self._call("link", a="fact-a", b="fact-b",
                          relation="distinct")["content"][0]["text"]
        self.assertIn("legitimately distinct", text)
        row = self.srv.store.conn.execute(
            "SELECT count(*) c FROM memory_link WHERE relation='distinct'"
        ).fetchone()
        self.assertEqual(row["c"], 1)

    def test_link_tool_rejects_unsupported_relation(self):
        self._call("remember", title="fact-a", content="durable fact a")
        self._call("remember", title="fact-b", content="durable fact b")
        text = self._call("link", a="fact-a", b="fact-b",
                          relation="supersedes")["content"][0]["text"]
        self.assertIn("unsupported relation", text)

    def test_supersede_tool_reconciles(self):
        self._call("remember", title="rate", content="the api key rotates monthly")
        self._call("remember", title="rate-new", content="the api key rotates weekly")
        out = self._call("supersede", old="rate", new="rate-new")
        self.assertFalse(out["isError"])
        self.assertIn("superseded", out["content"][0]["text"])
        listed = self._call("list_memories")["content"][0]["text"]
        self.assertIn("rotates weekly", listed)         # the new one stands
        self.assertNotIn("rotates monthly", listed)     # the stale one is gone

    def test_supersede_tool_unknown_ref(self):
        out = self._call("supersede", old="nope", new="alsonope")
        self.assertIn("no memory", out["content"][0]["text"])

    def test_list_excludes_episodic(self):
        self.srv.store.store("session row", "x", kind="episodic")
        self.srv.store.store("standing fact", "x")
        text = self._call("list_memories")["content"][0]["text"]
        self.assertIn("standing fact", text)
        self.assertNotIn("session row", text)

    def test_frozen_store_refuses_as_tool_error(self):
        set_config(self.srv.store, "frozen", "on")
        out = self._call("remember", title="x", content="y")
        self.assertTrue(out["isError"])
        self.assertIn("frozen", out["content"][0]["text"])

    def test_protocol_errors_and_notifications(self):
        r = self.srv.handle(req(5, "tools/call", name="no_such_tool", arguments={}))
        self.assertEqual(r["error"]["code"], -32602)
        r = self.srv.handle(req(6, "bogus/method"))
        self.assertEqual(r["error"]["code"], -32601)
        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self.assertIsNone(self.srv.handle(note))
        # an unreadable time phrase degrades gracefully (helpful, not an error)
        r = self._call("recall_timeline", when="gibberish phrase")
        self.assertFalse(r["isError"])
        self.assertIn("couldn't read the time phrase", r["content"][0]["text"])
        # a genuine tool exception is still a result with isError, not a dead session
        r = self._call("recall_memory")   # missing required 'query' -> isError
        self.assertTrue(r["isError"])


class TestSenses(unittest.TestCase):
    """The default-off live-sense tools, dispatched via tools/call. The camera
    and VLM are faked; feel needs neither (a literal reading works anywhere)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.srv = FornixMCP(db_path=Path(self.tmp.name) / "m.db", shared=False)

    def tearDown(self):
        self.srv.store.close()
        self.tmp.cleanup()

    def _call(self, name, **args):
        return self.srv.handle(req(9, "tools/call", name=name, arguments=args))["result"]

    def _fake_vlm(self, caption):
        return mock.patch("fornixdb.adapters.mac_vision.vlm_captioner",
                          return_value=lambda p: caption)

    def test_senses_are_callable_though_default_off(self):
        # tools/call permits a known tool even when unadvertised (default-off)
        r = self._call("feel", reading="charge=80%, charging=yes")
        self.assertFalse(r["isError"])
        self.assertIn("stored #", r["content"][0]["text"])
        # and it is recallable as a real tactile memory
        r = self._call("recall_memory", query="charge charging")
        self.assertIn("charge=80%", r["content"][0]["text"])

    def test_look_captions_the_current_frame(self):
        cam = mock.patch("fornixdb.adapters.mac_camera.open_stream",
                         return_value=(iter([(0.0, b"\xff\xd8x")]), "camera"))
        with cam, self._fake_vlm("a person at a desk"):
            r = self._call("look", source="camera")
        self.assertFalse(r["isError"])
        self.assertEqual(r["content"][0]["text"], "a person at a desk")

    def test_see_remembers_an_image_with_a_caption(self):
        img = Path(self.tmp.name) / "shot.jpg"
        img.write_bytes(b"\xff\xd8fake")
        with self._fake_vlm("a red mug on a windowsill"):
            r = self._call("see", image_path=str(img))
        self.assertFalse(r["isError"])
        self.assertIn("a red mug on a windowsill", r["content"][0]["text"])

    def test_recaption_dry_run_reports_empty_backlog(self):
        r = self._call("recaption", dry_run=True)
        self.assertFalse(r["isError"])
        self.assertIn("no watch keyframes await a caption", r["content"][0]["text"])

    def test_missing_model_surfaces_as_a_clean_tool_error(self):
        # no camera / Ollama in the test env: the handler must fail as an
        # isError result, not crash the session
        r = self._call("look", source="/no/such/clip.mov")
        self.assertTrue(r["isError"])
        self.assertIn("error", r["content"][0]["text"].lower())


class TestStdioRoundTrip(unittest.TestCase):
    def test_subprocess_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "rt.db")
            lines = "\n".join(json.dumps(m) for m in [
                req(1, "initialize", protocolVersion="2024-11-05"),
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                req(2, "tools/call", name="remember",
                    arguments={"title": "wire-test", "content": "over the wire"}),
                req(3, "tools/call", name="recall_memory",
                    arguments={"query": "over the wire"}),
            ]) + "\n"
            proc = subprocess.run(
                [sys.executable, "-m", "fornixdb.adapters.mcp_server",
                 "--db", db, "--no-shared"],
                input=lines, capture_output=True, text=True, timeout=60)
            resps = {r["id"]: r for r in map(json.loads,
                     proc.stdout.strip().splitlines())}
            self.assertEqual(len(resps), 3)         # notification got no reply
            self.assertEqual(resps[1]["result"]["serverInfo"]["name"], "fornixdb")
            self.assertIn("stored #1", resps[2]["result"]["content"][0]["text"])
            self.assertIn("over the wire", resps[3]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
