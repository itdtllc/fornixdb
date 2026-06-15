"""Token-footprint report (FornixDB #165): what connecting an AI to this
store costs in prompt tokens — and what that replaces.

A local model re-processes its whole prompt every turn, so FornixDB's
contribution to the prompt (tool schemas, the startup context, recall
results) is paid in prefill time as well as context space. The owner asked
for an approximate measure: is memory costing tokens or saving them?

Estimates use the ~4-chars-per-token rule of thumb — right to within ~20%
for English prose across common tokenizers, which is all a budgeting
decision needs. The report is mechanical about COSTS; the SAVINGS side
(what a recall replaces: the user re-explaining history, the AI re-reading
files or re-deriving decisions) can't be measured from inside the store, so
it is reported as the comparison the owner should make.
"""

from __future__ import annotations

import json

EST_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text or "") / EST_CHARS_PER_TOKEN))


def report(store) -> dict:
    """Estimated prompt-token footprint of this store's integration surfaces."""
    from .adapters.mcp_server import INSTRUCTIONS, TOOLS, FornixMCP

    schemas = json.dumps(TOOLS)
    out = {
        "rule_of_thumb": f"1 token ≈ {EST_CHARS_PER_TOKEN} chars (±20%)",
        "fixed_per_session": {
            "mcp_tool_schemas": {"tools": len(TOOLS),
                                 "tokens": estimate_tokens(schemas)},
            "mcp_instructions": {"tokens": estimate_tokens(INSTRUCTIONS)},
        },
        "per_call": {},
    }

    # startup context against THIS store (capture mode + salient gists)
    srv = FornixMCP.__new__(FornixMCP)  # reuse formatting without re-opening dbs
    srv.store, srv.stores = store, [("", store)]
    startup = srv.startup_context()
    out["fixed_per_session"]["startup_context"] = {
        "tokens": estimate_tokens(startup)}
    out["fixed_per_session"]["total_tokens"] = sum(
        v["tokens"] for v in out["fixed_per_session"].values()
        if isinstance(v, dict))

    # per-call: a brief, and a typical recall result (recent gist lines at the
    # default limit; the MCP max_chars default of 4000 is the hard ceiling)
    b = store.brief()
    brief_chars = sum(len(m.get("gist") or "") + 30
                      for m in b["recent"] + b["salient"])
    recent = [r["gist"] or "" for r in store.conn.execute(
        "SELECT gist FROM memory ORDER BY id DESC LIMIT 25")]
    avg_line = (sum(len(g) + 30 for g in recent) / len(recent)) if recent else 60
    out["per_call"] = {
        "brief": {"tokens": estimate_tokens("x" * brief_chars)},
        "recall_default_limit_5": {
            "tokens": round(5 * avg_line / EST_CHARS_PER_TOKEN),
            "ceiling_tokens": round(4000 / EST_CHARS_PER_TOKEN)},
        "show_one_memory_avg": {"tokens": estimate_tokens("x" * int(
            (store.conn.execute(
                "SELECT avg(length(coalesce(detail, gist))) a FROM memory"
            ).fetchone()["a"] or 200)))},
    }
    out["savings_side"] = (
        "Not measurable from inside the store. The comparison: one recall "
        f"(≈{out['per_call']['recall_default_limit_5']['tokens']} tokens) "
        "replaces the user re-explaining that history by hand, or the AI "
        "re-reading files / re-deriving past decisions — typically hundreds "
        "to thousands of tokens, every session, plus the answers it makes "
        "possible at all ('what day did X happen'). The fixed cost is paid "
        "once per session; trim it with leaner tool descriptions and a "
        "shorter startup listing if prefill speed matters (local models).")
    return out


def format_report(r: dict) -> str:
    f = r["fixed_per_session"]
    p = r["per_call"]
    lines = [
        f"Estimated token footprint ({r['rule_of_thumb']})",
        "",
        "Fixed, once per session (in every prompt for most clients):",
        f"  MCP tool schemas ({f['mcp_tool_schemas']['tools']} tools)"
        f"   ~{f['mcp_tool_schemas']['tokens']:>5} tokens",
        f"  MCP instructions          ~{f['mcp_instructions']['tokens']:>5}",
        f"  startup_context           ~{f['startup_context']['tokens']:>5}",
        f"  TOTAL fixed               ~{f['total_tokens']:>5}",
        "",
        "Per call:",
        f"  brief                     ~{p['brief']['tokens']:>5}",
        f"  recall (limit 5)          ~{p['recall_default_limit_5']['tokens']:>5}"
        f"  (hard ceiling {p['recall_default_limit_5']['ceiling_tokens']} via max_chars)",
        f"  show one memory (avg)     ~{p['show_one_memory_avg']['tokens']:>5}",
        "",
        r["savings_side"],
    ]
    return "\n".join(lines)
