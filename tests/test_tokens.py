"""Token-footprint report (FornixDB #165)."""

import unittest

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.tokens import estimate_tokens, format_report, report


class TestTokens(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore(conn=connect(":memory:"))
        for i in range(8):
            self.s.store(f"fact number {i}", "detail " * 40)

    def test_estimate(self):
        self.assertEqual(estimate_tokens("x" * 400), 100)
        self.assertEqual(estimate_tokens(""), 1)  # never zero, never crashes

    def test_report_shape_and_sanity(self):
        r = report(self.s)
        fixed = r["fixed_per_session"]
        self.assertGreater(fixed["mcp_tool_schemas"]["tokens"], 100)
        self.assertEqual(fixed["mcp_tool_schemas"]["tools"], 14)
        self.assertEqual(fixed["total_tokens"],
                         fixed["mcp_tool_schemas"]["tokens"]
                         + fixed["mcp_instructions"]["tokens"]
                         + fixed["startup_context"]["tokens"])
        self.assertGreater(r["per_call"]["recall_default_limit_5"]["tokens"], 0)
        self.assertIn("re-explaining", r["savings_side"])

    def test_format(self):
        out = format_report(report(self.s))
        self.assertIn("TOTAL fixed", out)
        self.assertIn("Per call:", out)

    def test_fixed_footprint_within_budget(self):
        # Regression fence on the STATIC fixed per-session cost (FornixDB #185):
        # the tool schemas + instructions ride in every prompt and a local
        # model re-prefills them each turn. These budgets are deliberate
        # ceilings — raising one is a conscious act (a new tool must earn its
        # tokens), not something a verbose description should do silently.
        # Trimmed 2026-06-13: schemas 982 -> 807, instructions 231.
        # Raised 2026-06-14: 850 -> 1050 — deliberate, for the Sleep/Dream MCP
        # tools `dream` (11th) and `supersede` (12th, so a shell-less consumer
        # can apply the pass's healing); 12 tools, still far under the 4096
        # ceiling and the ~16-lean-tools guideline.
        # Raised 2026-06-16: 1050 -> 1280 — deliberate, for the Markdown-bridge
        # tools `import_markdown` (13th) and `export_markdown` (14th); 14 tools,
        # still under the 4096 ceiling and the ~16-lean-tools guideline.
        import json

        from fornixdb.adapters.mcp_server import INSTRUCTIONS, TOOLS
        SCHEMA_TOKEN_BUDGET = 1280
        INSTRUCTIONS_TOKEN_BUDGET = 260

        schema_tokens = estimate_tokens(json.dumps(TOOLS))
        instr_tokens = estimate_tokens(INSTRUCTIONS)
        self.assertLessEqual(
            schema_tokens, SCHEMA_TOKEN_BUDGET,
            f"MCP tool schemas {schema_tokens} tok > budget "
            f"{SCHEMA_TOKEN_BUDGET}; trim a description or raise the budget "
            "deliberately")
        self.assertLessEqual(
            instr_tokens, INSTRUCTIONS_TOKEN_BUDGET,
            f"MCP instructions {instr_tokens} tok > budget "
            f"{INSTRUCTIONS_TOKEN_BUDGET}")


if __name__ == "__main__":
    unittest.main()
