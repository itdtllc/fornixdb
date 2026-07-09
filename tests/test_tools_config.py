"""Configurable MCP tool surface: memory tools on by default, optional ones can
be disabled per-store to shrink prefill, core tools cannot, and the live senses
ship OFF until opted in. No hardcoded token ceiling — curation is per-deployment
(owner direction 2026-06-16)."""

import tempfile
import unittest
from pathlib import Path

from fornixdb.adapters.mcp_server import (CORE_TOOLS, DEFAULT_OFF_TOOLS, TOOLS,
                                          FornixMCP, active_tools,
                                          set_tool_enabled, tool_tier,
                                          tools_disabled)


class TestToolConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.srv = FornixMCP(db_path=Path(self.tmp.name) / "m.db", shared=False)
        self.store = self.srv.store

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_default_memory_tools_on_senses_off(self):
        # memory tools all on; the live senses are the only default-off set
        self.assertEqual(tools_disabled(self.store), set(DEFAULT_OFF_TOOLS))
        active = {t["name"] for t in active_tools(self.store)}
        self.assertEqual(len(active), len(TOOLS) - len(DEFAULT_OFF_TOOLS))
        self.assertTrue(DEFAULT_OFF_TOOLS.isdisjoint(active))

    def test_core_set_is_a_subset_of_defined_tools(self):
        names = {t["name"] for t in TOOLS}
        self.assertTrue(CORE_TOOLS.issubset(names))
        # the irreducible recall + capture loop
        self.assertIn("recall_memory", CORE_TOOLS)
        self.assertIn("remember", CORE_TOOLS)

    def test_disable_optional_tool_drops_it_from_active(self):
        msg = set_tool_enabled(self.store, "export_markdown", False)
        self.assertIn("disabled", msg)
        active = {t["name"] for t in active_tools(self.store)}
        self.assertNotIn("export_markdown", active)
        self.assertEqual(tools_disabled(self.store),
                         {"export_markdown"} | set(DEFAULT_OFF_TOOLS))

    def test_core_tool_cannot_be_disabled(self):
        msg = set_tool_enabled(self.store, "recall_memory", False)
        self.assertIn("core", msg)
        self.assertIn("recall_memory", {t["name"] for t in active_tools(self.store)})

    def test_unknown_tool_is_rejected(self):
        self.assertIn("unknown", set_tool_enabled(self.store, "nope", False))

    def test_re_enable(self):
        set_tool_enabled(self.store, "dream", False)
        self.assertNotIn("dream", {t["name"] for t in active_tools(self.store)})
        set_tool_enabled(self.store, "dream", True)
        self.assertIn("dream", {t["name"] for t in active_tools(self.store)})

    def test_tools_list_advertises_only_active(self):
        set_tool_enabled(self.store, "import_markdown", False)
        resp = self.srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertNotIn("import_markdown", names)
        self.assertIn("recall_memory", names)

    def test_disabled_tool_still_callable_if_invoked(self):
        # disabling only removes it from the advertised prompt, never breaks a
        # call a client already knows about
        set_tool_enabled(self.store, "memory_usage", False)
        resp = self.srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                "params": {"name": "memory_usage", "arguments": {}}})
        self.assertFalse(resp["result"].get("isError"))

    def test_tier_labels(self):
        self.assertEqual(tool_tier("remember"), "core")
        self.assertEqual(tool_tier("export_markdown"), "optional")
        self.assertEqual(tool_tier("look"), "sense")

    def test_sense_tool_opt_in_round_trip(self):
        # default off, not advertised
        self.assertNotIn("look", {t["name"] for t in active_tools(self.store)})
        # owner opts in -> advertised; other senses stay off (independent)
        self.assertIn("enabled", set_tool_enabled(self.store, "look", True))
        active = {t["name"] for t in active_tools(self.store)}
        self.assertIn("look", active)
        self.assertNotIn("feel", active)
        self.assertNotIn("look", tools_disabled(self.store))
        # and back off
        set_tool_enabled(self.store, "look", False)
        self.assertNotIn("look", {t["name"] for t in active_tools(self.store)})

    def test_enabling_a_sense_does_not_disturb_optional_disables(self):
        set_tool_enabled(self.store, "dream", False)      # a default-on opt-out
        set_tool_enabled(self.store, "feel", True)        # a default-off opt-in
        active = {t["name"] for t in active_tools(self.store)}
        self.assertIn("feel", active)                     # opt-in honored
        self.assertNotIn("dream", active)                 # opt-out still honored
        # other senses remain off
        self.assertNotIn("look", active)


if __name__ == "__main__":
    unittest.main()
