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
        self.assertEqual(fixed["mcp_tool_schemas"]["tools"], 22)
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
        # model re-prefills them each turn. This budget is NOT a hard device
        # limit — it is just a speed bump so tool schemas do not grow SILENTLY;
        # raising it is a conscious act (a new tool must earn its tokens). There
        # is no universal token ceiling: Claude Code has a ~200K context and is
        # unaffected; local models (Elira 72B, a 14B) care about prefill *cost*,
        # a soft gradient, not a wall; the only true ~4096 cap belongs to a
        # DIFFERENT deployment (Apple on-device Foundation Models in the iOS AI
        # Assistant), and per-deployment caps are handled by `fornixdb tools`
        # curation, never hardcoded here. `tests/test_tools_config.py` covers it.
        # History: 982->807 (trim) -> 1050 (dream+supersede) -> 1280 (markdown
        # bridge) -> 1340 (link) -> 1480 (remember_many) -> 1650 (jot +
        # review_candidates, §15.2 #1) -> 1750 (mark_helpful, §15.2 #6) -> 1810
        # (export_markdown filters: query/when/since/until + single_file +
        # index_name) -> 1890 (recent_writes — session write-log for
        # end-of-session dedup, dogfooding report §4.4) -> 1960
        # (remember/remember_many `project` arg — explicit capture scope, since
        # the MCP server can't see the host's per-session declared project) ->
        # 2110 (link `relation` param + distinct-ok accept semantics, and the
        # dream description naming its grown worklist: chronic-noise,
        # mis-scoped, dial report; owner principle 2026-06-25: never trim
        # tools/descriptions to fit — raise the budget) -> 2260 (remind_me —
        # prospective memory, 0.8.5) -> 2300 (urgent nag param, 0.8.6) —
        # each a deliberate raise for named tools. This measures
        # ALL defined tools; the
        # live footprint is the
        # smaller ADVERTISED set after `fornixdb tools` curation (jot/review,
        # like every optional tool, ship ON but can be disabled per deployment).
        import json

        from fornixdb.adapters.mcp_server import (DEFAULT_OFF_TOOLS,
                                                  INSTRUCTIONS, TOOLS)
        # Two ceilings now that the live senses (look/feel/see/recaption) ship
        # OFF by default: what a fresh store actually prefills is the memory tools
        # only, and that DEFAULT set is held to the historical budget (the senses
        # add nothing until opted in). The FULL set — every defined tool, senses
        # included — has its own, deliberately higher ceiling for stores that
        # enable the senses (owner principle 2026-06-25: raise the budget, never
        # trim a description to fit).
        DEFAULT_SCHEMA_TOKEN_BUDGET = 2300  # memory tools; link distinct + dream
                                            # + remind_me (prospective, 0.8.5;
                                            # urgent nag param, 0.8.6)
        FULL_SCHEMA_TOKEN_BUDGET = 2900     # + look/feel/see/recaption (opt-in)
        INSTRUCTIONS_TOKEN_BUDGET = 260

        default_schema = [t for t in TOOLS if t["name"] not in DEFAULT_OFF_TOOLS]
        schema_tokens = estimate_tokens(json.dumps(default_schema))
        full_tokens = estimate_tokens(json.dumps(TOOLS))
        instr_tokens = estimate_tokens(INSTRUCTIONS)
        self.assertLessEqual(
            schema_tokens, DEFAULT_SCHEMA_TOKEN_BUDGET,
            f"DEFAULT MCP tool schemas {schema_tokens} tok > budget "
            f"{DEFAULT_SCHEMA_TOKEN_BUDGET}; trim a description or raise the "
            "budget deliberately")
        self.assertLessEqual(
            full_tokens, FULL_SCHEMA_TOKEN_BUDGET,
            f"FULL MCP tool schemas {full_tokens} tok > budget "
            f"{FULL_SCHEMA_TOKEN_BUDGET}; raise the budget deliberately")
        self.assertLessEqual(
            instr_tokens, INSTRUCTIONS_TOKEN_BUDGET,
            f"MCP instructions {instr_tokens} tok > budget "
            f"{INSTRUCTIONS_TOKEN_BUDGET}")


if __name__ == "__main__":
    unittest.main()
