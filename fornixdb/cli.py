"""Command-line interface — the universal integration surface.

Any thinking AI that can run a shell command can use this memory. Output is
gist-first and terse by design: recall costs context, detail is on request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .consolidate import mark_done
from .consolidate import status as consolidate_status
from .core import AUTO_CAPTURE_SOURCES, FrozenStoreError, MemoryStore
from .db import DEFAULT_DB_ENV, KINDS, RELATIONS, default_db_path
from .multistore import (CAPTURE_MODE_HELP, get_config, multi_brief, multi_recall,
                         multi_timeline, open_stores, resolve_ref, set_config,
                         shared_db_path)
from .timeparse import parse_when


def fit_chars(lines: list[str], max_chars: int | None) -> tuple[list[str], int]:
    """Trim a list of output blocks to a character budget — recall costs the
    consuming AI context, so the consumer can say how much it can afford.
    Blocks are kept whole, best-first, until the budget; returns (kept,
    omitted_count). A first block longer than the entire budget is truncated
    rather than dropped — something beats nothing."""
    if not max_chars or max_chars <= 0:
        return lines, 0
    kept: list[str] = []
    used = 0
    for i, block in enumerate(lines):
        cost = len(block) + (1 if kept else 0)  # +1 for the joining newline
        if used + cost > max_chars:
            if not kept:
                return [block[:max(max_chars - 1, 1)] + "…"], len(lines) - 1
            return kept, len(lines) - i
        kept.append(block)
        used += cost
    return kept, 0


def _fmt_gist_line(m: dict) -> str:
    flag = " [superseded]" if m.get("superseded_time") else ""
    if m.get("stale_days"):  # old, never reinforced — verify before trusting
        flag += f" [stale {m['stale_days']}d]"
    if m.get("neg_feedback"):  # marked irrelevant for a query like this one
        flag += " [downweighted]"
    if m.get("source") in AUTO_CAPTURE_SOURCES:  # machine-ingested, unreviewed
        flag += " [auto-captured]"
    if m.get("writer"):  # which agent put this in the shared tier
        flag += f" [by {m['writer']}]"
    date = (m.get("event_time") or "")[:10]
    proj = f" ({m['project']})" if m.get("project") else ""
    sid = f"{m['_store']}:{m['id']}" if m.get("_store") else f"{m['id']}"
    if m.get("also_in"):  # the same fact deduped from another store
        flag += " (=" + ", ".join(f"#{x}" for x in m["also_in"]) + ")"
    return f"#{sid:<7} {date}  {m['kind'][:3]}{proj}{flag}  {m['gist']}"


def _agent_label(store) -> str:
    """The writing agent's identity for shared-tier provenance (B3): the
    store's own label (`config store_label`), else its filename stem — the
    same fallback `usage` shows."""
    label = get_config(store, "store_label")
    if label:
        return label
    try:
        return Path(store.conn.execute(
            "PRAGMA database_list").fetchone()[2]).stem or "unknown"
    except Exception:
        return "unknown"


def _time_bound(p, text: str, which: str) -> str:
    """ISO date, or any parse_when phrase (span start for since, span end for
    until). Unparseable input is an error, never a silent empty result."""
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        pass
    try:
        s, e = parse_when(text)
        return (s if which == "since" else e).isoformat()
    except ValueError:
        p.error(f"--{which} {text!r} is not an ISO date or a time phrase")


def _row_block(m: dict) -> str:
    out = _fmt_gist_line(m)
    for ln in m.get("related") or []:
        out += f"\n        ↳ {ln['relation']} #{ln['id']}: {ln['gist'][:70]}"
    return out


def _print_rows(rows: list[dict], as_json: bool,
                max_chars: int | None = None) -> None:
    if as_json:  # a budget is a text-output concern; JSON stays complete
        print(json.dumps(rows, indent=2, default=str))
        return
    if not rows:
        print("(no memories found)")
        return
    blocks, omitted = fit_chars([_row_block(m) for m in rows], max_chars)
    for b in blocks:
        print(b)
    if omitted:
        print(f"(+{omitted} more — raise --max-chars or narrow the query)")


def registered_consumer_stores() -> list[Path]:
    """Existing consumer stores recorded in the machine registry, shared tier
    excluded. Used by the `configure` wizard so it can run with NO arguments:
    a machine usually has one store, but may have several (one per host AI).
    Paths only; non-existent and the shared tier are filtered out."""
    from .db import registry_path
    reg = registry_path()
    if reg is None or not reg.exists():
        return []
    try:
        paths = json.loads(reg.read_text() or "[]")
    except (OSError, ValueError):
        return []
    try:
        shared = str(shared_db_path().resolve())
    except OSError:
        shared = ""
    out: list[Path] = []
    for raw in paths:
        try:
            rp = Path(raw).expanduser()
            if not rp.exists() or str(rp.resolve()) == shared:
                continue
        except OSError:
            continue
        out.append(rp)
    return out


def resolve_configure_store(args, ask=input, out=print) -> None:
    """For `configure` with no explicit store, pick which one to configure so
    the launcher needs no arguments. Sets ``args.db`` in place.

    Respects an explicit --db / $FORNIXDB_DB (no-op). With exactly one
    registered store, selects it silently. With several, prints a numbered menu
    and asks (the wizard is interactive anyway). With none, leaves args.db None
    so the normal default / no-store guard handles it.
    """
    if args.db or os.environ.get(DEFAULT_DB_ENV):
        return
    stores = registered_consumer_stores()
    if not stores:
        return
    if len(stores) == 1:
        args.db = str(stores[0])
        return
    out("FornixDB found more than one store on this machine. Which to configure?")
    for i, p in enumerate(stores, 1):
        out(f"  {i}) {p.name}    ({p})")
    while True:
        try:
            raw = ask(f"Choose 1-{len(stores)}: ").strip()
        except (EOFError, KeyboardInterrupt):
            out("\nNo store chosen — aborted.")
            raise SystemExit(0)
        if raw.isdigit() and 1 <= int(raw) <= len(stores):
            args.db = str(stores[int(raw) - 1])
            return
        out(f"  '{raw}' is not 1-{len(stores)} — try again")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to legacy code pages (cp1252) that can't print
    # the CLI's own output (e.g. the → in link lines), let alone stored CJK or
    # emoji. The data layer is UTF-8 throughout; make the console match.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(
        prog="fornixdb",
        description="A persistent, human-like memory for any AI. Gist-first recall "
                    "by subject or time; detail on request.",
    )
    p.add_argument("--db", help=f"store path (default: $FORNIXDB_DB or {default_db_path()})")
    p.add_argument("--no-shared", action="store_true",
                   help="this store only; skip the machine-level shared tier "
                        f"($FORNIXDB_SHARED_DB or {shared_db_path()})")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--create", action="store_true",
                   help="allow creating a new store at the default path "
                        "(otherwise only 'init' creates the default store, so a "
                        "read-only command never litters ~/.fornixdb)")
    p.add_argument("--version", action="version", version=f"fornixdb {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create/verify the store")

    sp = sub.add_parser("store", help="store a memory")
    sp.add_argument("--shared", action="store_true",
                    help="write to the machine-level shared tier (owner facts/preferences "
                         "every agent should know) instead of this agent's store")
    sp.add_argument("--gist", required=True, help="one-line summary (always recalled first)")
    sp.add_argument("--detail", help="full detail (recalled on drill-down)")
    sp.add_argument("--kind", choices=KINDS, default="semantic")
    sp.add_argument("--name", help="optional unique slug handle")
    sp.add_argument("--topic", action="append", default=[], help="repeatable")
    sp.add_argument("--project")
    sp.add_argument("--event-time", help="when it happened (ISO; default now)")
    sp.add_argument("--event-time-end", help="end of span (ISO), e.g. for sessions")
    sp.add_argument("--session-id")
    sp.add_argument("--salience", type=float, default=0.5)
    sp.add_argument("--source", default="cli")
    sp.add_argument("--source-ref")

    rp = sub.add_parser("recall", help="recall by subject (ranked gists)")
    rp.add_argument("query")
    rp.add_argument("--limit", type=int, default=10)
    rp.add_argument("--kind", choices=KINDS)
    rp.add_argument("--project")
    rp.add_argument("--when", help='combine with a time window: "last month", "may", …')
    rp.add_argument("--since", help="window start (ISO date or time phrase)")
    rp.add_argument("--until", help="window end (ISO date or time phrase)")
    rp.add_argument("--related", action="store_true",
                    help="show each hit's 1-hop linked memories")
    rp.add_argument("--all", action="store_true", help="include superseded memories")
    rp.add_argument("--max-chars", type=int,
                    help="character budget for the output (recall costs the "
                         "consuming AI context); whole hits kept best-first")

    tp = sub.add_parser("timeline", help='recall by time: fornixdb timeline "last thursday"')
    tp.add_argument("when", nargs="?", help='"yesterday", "last thursday", "2026-06-05", ...')
    tp.add_argument("--since", help="explicit ISO start")
    tp.add_argument("--until", help="explicit ISO end")
    tp.add_argument("--kind", choices=KINDS)
    tp.add_argument("--project")
    tp.add_argument("--limit", type=int, default=50)
    tp.add_argument("--max-chars", type=int,
                    help="character budget for the output")

    rm = sub.add_parser("remind", help='prospective memory: fornixdb remind '
                                       '"call the attorney" --when "tomorrow 9am"')
    rm.add_argument("what", help="the intention to surface when it's time")
    rm.add_argument("--when", required=True,
                    help='"in 20 minutes", "tomorrow morning", "friday at 3pm", ISO')
    rm.add_argument("--urgent", action="store_true",
                    help="nag every few minutes until acknowledged (fornixdb ack); "
                         "dials: config nag_interval_minutes / nag_max_attempts")
    rm.add_argument("--project")
    rm.add_argument("--topic", action="append", default=[], help="repeatable")

    dp = sub.add_parser("due", help="reminders that have come due (marks them "
                                    "delivered; --peek to look without consuming)")
    dp.add_argument("--peek", action="store_true",
                    help="don't mark delivered — just look")
    dp.add_argument("--upcoming", type=float, metavar="HOURS", default=None,
                    help="also list undelivered reminders due within HOURS")

    sub.add_parser("ack", help="acknowledge nagging urgent reminders (stops the "
                               "repeats; hosts call this on any owner turn)")

    hp = sub.add_parser("show", help="full detail of one memory (reinforces it)")
    hp.add_argument("ref", help="memory id or name")
    hp.add_argument("--no-reinforce", action="store_true")

    up = sub.add_parser("supersede", help="newer memory replaces older (older kept, tombstoned)")
    up.add_argument("old_id", type=int)
    up.add_argument("new_id", type=int)

    lp = sub.add_parser("link", help="link two memories")
    lp.add_argument("memory_id", type=int)
    lp.add_argument("related_id", type=int)
    lp.add_argument("--relation", choices=RELATIONS, default="relates")

    jp = sub.add_parser("jot", help="stage a raw thought for later review (cheap "
                                    "mid-work capture, not yet a memory)")
    jp.add_argument("note")
    kp = sub.add_parser("candidates", help="list/discard jotted candidates "
                                           "(promote keepers with `store`)")
    kp.add_argument("--discard", type=int, nargs="+", help="candidate ids to drop")
    kp.add_argument("--clear", action="store_true", help="drop all pending")

    bp = sub.add_parser("brief", help="session-start context brief (recent + salient)")
    bp.add_argument("--project")
    bp.add_argument("--days", type=int, default=7)

    ep = sub.add_parser("embed", help="backfill vector embeddings (needs optional deps)")
    ep.add_argument("--batch", type=int, default=64)

    np = sub.add_parser("consolidate",
                        help="consolidation status / AI worklist / mark a pass done")
    np.add_argument("action", nargs="?", default="status",
                    choices=["status", "propose", "done"])

    dp = sub.add_parser("dream",
                        help="sleep/dream mode: one narrated consolidation pass "
                             "(status + worklist; headlines outdated memories, "
                             "chronic push-noise + new connections to weave; "
                             "opening a pass refreshes push use-credit from "
                             "transcripts)")
    dp.add_argument("--weave", action="store_true",
                    help="also CREATE the proposed new associative links "
                         "(non-destructive — adds 'relates' links, changes nothing else)")
    dp.add_argument("--done", action="store_true",
                    help="close the pass: report what was reconciled (wake summary) "
                         "and reset the DUE clock")

    ip = sub.add_parser("irrelevant",
                        help="negative feedback: a recalled memory was wrong for a "
                             "query — downweight it for similar queries only")
    ip.add_argument("ref", nargs="?", help="memory id, 'shared:id', or name")
    ip.add_argument("query", nargs="?", help="the query it was irrelevant to")
    ip.add_argument("--list", action="store_true", dest="list_feedback",
                    help="show feedback rows (optionally for one memory)")
    ip.add_argument("--retract", type=int, metavar="FEEDBACK_ID",
                    help="tombstone one feedback row (memory ranks normally again)")

    fp = sub.add_parser("helpful",
                        help="positive feedback: a recalled memory actually helped "
                             "— endorse it (ranks higher everywhere, resists staleness)")
    fp.add_argument("ref", help="memory id, 'shared:id', or name")

    gp = sub.add_parser("tag", help="add a topic to a memory")
    gp.add_argument("memory_id", type=int)
    gp.add_argument("topic")

    sg = sub.add_parser("set-gist", help="rewrite a gist in place (consolidation; "
                                         "meaning changes use store+supersede instead)")
    sg.add_argument("ref", help="memory id or name")
    sg.add_argument("gist")

    yp = sub.add_parser("tier", help="retention tiers: status / mechanical tier-down")
    yp.add_argument("action", nargs="?", default="status", choices=["status", "down"])
    yp.add_argument("--dry-run", action="store_true")

    dg = sub.add_parser("budget", help="disk budget (config disk_budget_mb / "
                                       "budget_policy): status / enforce the cap / "
                                       "one-shot shrink to a target size")
    dg.add_argument("action", nargs="?", default="status",
                    choices=["status", "enforce", "shrink"])
    dg.add_argument("mb", nargs="?", type=float,
                    help="shrink target in MB (shrink only) — TRUE deletion to "
                         "reach it; the standing cap/policy are not changed")
    dg.add_argument("--dry-run", action="store_true")

    vp = sub.add_parser("eval", help="recall-quality eval: score a golden-query "
                                     "file against this store (primary only)")
    vp.add_argument("golden", help="JSONL of {query, expect:[id|name], k?, kind?, note?}")
    vp.add_argument("--verbose", action="store_true", help="show passing cases too")
    vp.add_argument("--keyword-only", action="store_true",
                    help="score without vectors (the no-model baseline)")
    vp.add_argument("--min-hitk", type=float,
                    help="exit 1 if hit@k falls below this fraction (CI fence)")
    vp.add_argument("--max-drift", type=int,
                    help="exit 1 if more than N rank1-asserted cases have "
                         "slipped below rank 1 (re-ranking drift fence)")
    vp.add_argument("--record", metavar="PATH",
                    help="append this run to a JSONL history (track precision "
                         "as the live store grows; personal data, keep by the store)")
    vp.add_argument("--max-leaks", type=int,
                    help="exit 1 if more than N abstain cases leaked (recall "
                         "returned a hit when it should have reported nothing)")

    ap = sub.add_parser("answer-eval", help="end-to-end A/B: does recall improve "
                                            "the AI's ANSWER (not just ranking)?")
    ap.add_argument("golden", help="JSONL of {query, answer_contains:[...], k?, "
                                   "kind?, when?, match?, note?}")
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="Claude model for the default answerer")
    ap.add_argument("--keyword-only", action="store_true",
                    help="recall without vectors (the no-model retrieval baseline)")
    ap.add_argument("--verbose", action="store_true", help="show every case")
    ap.add_argument("--record", metavar="PATH",
                    help="append this run to a JSONL history (track lift as the "
                         "live store grows; personal data, keep by the store)")
    ap.add_argument("--min-lift", type=float,
                    help="exit 1 if lift falls below this fraction (CI fence)")
    ap.add_argument("--max-regress", type=int,
                    help="exit 1 if more than N cases regressed (recall made a "
                         "right answer wrong)")

    bp = sub.add_parser("benefit", help="marginal value vs the flat markdown "
                                        "memory: what FornixDB holds that MEMORY.md lost")
    bp.add_argument("--memory-md", required=True,
                    help="path to the flat MEMORY.md index")
    bp.add_argument("--memory-dir", required=True,
                    help="dir of topic .md files (the flat system on disk)")
    bp.add_argument("--golden", help="also count prevented surprises on these "
                                     "golden questions")
    bp.add_argument("--cap-chars", type=int,
                    help="MEMORY.md session-start load cap (default 24400)")

    cp = sub.add_parser("config", help="show ALL settings (no args), or get/set "
                                       "one (e.g. config capture_mode suggest)")
    cp.add_argument("key", nargs="?")
    cp.add_argument("value", nargs="?")
    cp.add_argument("--shared", action="store_true", help="apply to the shared tier")

    lvp = sub.add_parser("level", help="operating-levels ladder (L0–L6): show "
                                       "it, set a rung (level L3), or toggle one "
                                       "(level L4 off)")
    lvp.add_argument("level", nargs="?",
                     help="L0–L6; omit to show the ladder and current rung")
    lvp.add_argument("state", nargs="?", choices=["on", "off"],
                     help="toggle just this one level on/off (cumulative); "
                          "omit to set the rung to this level")
    lvp.add_argument("--shared", action="store_true",
                     help="apply to the shared tier")

    sub.add_parser("configure", help="interactive wizard: walk through every "
                                     "setting and apply changes after a confirm")

    drp = sub.add_parser("doctor", help="health check: schema, host hooks, and "
                                        "suggested default settings")
    drp.add_argument("--apply-suggested", action="store_true",
                     help="set the recommended defaults that aren't satisfied yet "
                          "(e.g. a disk-space cap scaled to this device)")

    ng = sub.add_parser("ingest", help="follow a host AI's native memory dir "
                                       "(additive); set ingest_mode / native_dir")
    ng.add_argument("--mode", choices=["explicit", "passive", "both"],
                    help="explicit = no background automation; passive/both = "
                         "auto-ingest + session capture run on the hook")
    ng.add_argument("--dir", help="native memory directory to follow")
    ng.add_argument("--run", action="store_true", help="ingest now (any mode)")

    flp = sub.add_parser("feel", help="proprioception: capture a sensor reading "
                                      "as memory. No arg = the Mac's power/battery "
                                      "state now; a string = that reading. --live "
                                      "runs the change-gated loop.")
    flp.add_argument("reading", nargs="?",
                     help="a reading to feel (e.g. \"lid closed\"); omit to read "
                          "the Mac battery via pmset")
    flp.add_argument("--sensor", default="power",
                     help="sensor name the memory is filed under (default: power)")
    flp.add_argument("--live", action="store_true",
                     help="watch the battery and commit on every state change "
                          "(+ heartbeat); Ctrl-C stops cleanly")
    flp.add_argument("--seconds", type=float, default=60.0,
                     help="--live: how long to watch (default 60)")
    flp.add_argument("--interval", type=float, default=5.0,
                     help="--live: seconds between samples (default 5)")
    flp.add_argument("--heartbeat", type=float, default=300.0,
                     help="--live: commit a proof row after this much quiet "
                          "(default 300; 0 disables)")
    flp.add_argument("--percent-step", type=int, default=5,
                     help="--live: bucket charge %% to this step so drift doesn't "
                          "commit (default 5; 1 keeps every point)")
    flp.add_argument("--project", help="file the memories under a project")

    wtp = sub.add_parser("watch", help="vision: watch a stream (screen / camera "
                                       "/ video file) and commit only salient "
                                       "frames as memories. Owner-started; "
                                       "Ctrl-C stops cleanly.")
    wtp.add_argument("--source", default="screen",
                     help="screen (default, zero-permission) | camera (prompts "
                          "for camera access) | a video-file path")
    wtp.add_argument("--seconds", type=float, default=60.0,
                     help="how long to watch (default 60)")
    wtp.add_argument("--rate", type=float,
                     help="sampling rate in Hz (default 2 camera/file, 1 screen)")
    wtp.add_argument("--window", type=float, default=30.0,
                     help="MAX event-time window per memory in seconds "
                          "(default 30; a scene change cuts the window earlier)")
    wtp.add_argument("--threshold", type=float,
                     help="salience-gate commit distance (default 0.20, "
                          "field-tuned for camera+CLIP; lower = more sensitive)")
    wtp.add_argument("--max-commits", type=int,
                     help="stop after this many committed memories")
    wtp.add_argument("--project", help="file the memories under a project")

    rcp = sub.add_parser("recaption", help="dream-pass captioning: fill real "
                                           "captions on watch keyframes that "
                                           "were committed under a templated "
                                           "placeholder (needs a local VLM)")
    rcp.add_argument("--dry-run", action="store_true",
                     help="list the keyframes awaiting a caption and stop "
                          "(model-free — no VLM loaded)")
    rcp.add_argument("--limit", type=int,
                     help="caption at most this many keyframes this pass")
    rcp.add_argument("--model",
                     help="local VLM model id for the captioner "
                          "(default: the Mac adapter's default)")

    lkp = sub.add_parser("look", help="vision: look ONCE right now and describe "
                                      "the current frame — a synchronous, "
                                      "gate-free snapshot (needs a local VLM). "
                                      "Ephemeral unless --remember.")
    lkp.add_argument("--source", default="camera",
                     help="camera (default) | camera:N | screen | a video-file path")
    lkp.add_argument("--model",
                     help="local VLM model id for the captioner "
                          "(default: the Mac adapter's default)")
    lkp.add_argument("--remember", action="store_true",
                     help="also store the look as a see memory (default: answer "
                          "only, nothing saved)")
    lkp.add_argument("--keep-keyframe", action="store_true",
                     help="keep the still on disk (default: caption then delete)")
    lkp.add_argument("--project", help="file the memory under a project "
                                       "(with --remember)")

    sub.add_parser("usage", help="disk usage of EVERY FornixDB store on this "
                                 "machine (per AI + total)")
    tkp = sub.add_parser("tokens", help="estimated prompt-token footprint of this "
                                        "store's AI integration (cost vs savings)")
    tkp.add_argument("--billed", action="store_true",
                     help="measure the BILLED share instead: token-turns from the "
                          "host's session transcripts (what usage panels attribute), "
                          "not the once-counted context-space footprint")
    tkp.add_argument("--transcripts",
                     help="transcript .jsonl file or directory for --billed "
                          "(default ~/.claude/projects)")
    tkp.add_argument("--since-days", type=float,
                     help="only transcripts modified in the last N days (--billed)")
    tlp = sub.add_parser("tools", help="list/enable/disable the MCP tools this "
                                       "store advertises (all on by default)")
    tlp.add_argument("action", nargs="?", default="list",
                     choices=["list", "enable", "disable"])
    tlp.add_argument("name", nargs="?", help="tool name for enable/disable")
    tlp.add_argument("--profile", choices=["full", "minimal"],
                     help="full = all tools on; minimal = core only")
    tlp.add_argument("--full", action="store_true",
                     help="show each tool's COMPLETE explanation (not truncated)")
    sub.add_parser("topics", help="list topics with counts")
    sub.add_parser("stats", help="store statistics")

    fdp = sub.add_parser("field",
                         help="L5 debug view: fire the parallel multi-domain field "
                              "on a thought and show every domain's returns, the "
                              "corroboration scores, the clusters, and the settled "
                              "block")
    fdp.add_argument("thought", help="the evolving-thought text to fire the field on")
    fdp.add_argument("--project", help="active project context (default: none)")
    fdp.add_argument("--lit", metavar="IDS",
                     help="comma-separated memory ids already active this episode "
                          "(seeds the neighborhood domain)")

    fbp = sub.add_parser("field-stats",
                         help="analyze the L5 field log (settle rate, which domains "
                              "light winners, link-vs-topic glue, dissent shadow, "
                              "per-beat cost)")
    fbp.add_argument("--log", help="field log path (default: field_log.jsonl "
                                   "beside the store)")

    fsp = sub.add_parser("floor-stats",
                         help="analyze the floor log (cosine distributions, dial "
                              "activity, and an evidence-based floor recommendation)")
    fsp.add_argument("--log", help="floor log path (default: floor_log.jsonl beside "
                                   "the store)")
    fsp.add_argument("--transcripts", metavar="PATH",
                     help="session-transcript file or dir; when given, the useful/"
                          "noise outcome join uses whether each PUSH was actually "
                          "referenced downstream (usefulness-scan) instead of the "
                          "lifetime-recall_count proxy")

    usp = sub.add_parser("usefulness-scan",
                         help="honest push-usefulness from session transcripts: how "
                              "often a proactively-pushed memory was actually "
                              "referenced downstream (cited by #id)")
    usp.add_argument("--transcripts", metavar="PATH",
                     default="~/.claude/projects",
                     help="a .jsonl transcript or a directory of them "
                          "(default: ~/.claude/projects)")
    usp.add_argument("--apply", action="store_true",
                     help="close the loop: write each memory's downstream-reference "
                          "count into the store as a use-credit, so effective_floor "
                          "stops scoring proven-useful pushes as ignored noise "
                          "(idempotent absolute set)")

    spp = sub.add_parser("suppress",
                         help="per-memory push suppression: mute memories chronically "
                              "PUSHED but never REFERENCED (push-noise the cosine floor "
                              "can't filter). Never hides from recall/show/timeline.")
    spg = spp.add_mutually_exclusive_group()
    spg.add_argument("--scan", action="store_true",
                     help="scan transcripts and report chronic push-noise (default "
                          "action). Dry-run unless --apply.")
    spg.add_argument("--list", dest="list_", action="store_true",
                     help="list currently-suppressed memories and why")
    spg.add_argument("--undo", metavar="ID", type=int,
                     help="redeem one memory: clear its suppression so it pushes again")
    spp.add_argument("--apply", action="store_true",
                     help="with --scan: write the suppressions (and un-suppress any "
                          "row that has since earned a downstream reference)")
    spp.add_argument("--transcripts", metavar="PATH", default="~/.claude/projects",
                     help="a .jsonl transcript or a directory of them "
                          "(default: ~/.claude/projects)")

    vp = sub.add_parser("value",
                        help="one-shot 'how useful has FornixDB been?': cost (token "
                             "footprint) + reach (vs flat memory) + used (referenced-"
                             "push rate from transcripts)")
    vp.add_argument("--transcripts", metavar="PATH", default="~/.claude/projects",
                    help="transcript file/dir for the used-signal "
                         "(default: ~/.claude/projects; empty string to skip)")
    vp.add_argument("--memory-md", metavar="PATH",
                    help="flat MEMORY.md, to also measure REACH vs the flat memory")
    vp.add_argument("--memory-dir", metavar="PATH",
                    help="flat memory dir (with --memory-md) for the REACH measure")

    rpj = sub.add_parser("reproject",
                         help="re-derive project labels from CONTENT (fixes "
                              "auto-captured memories mislabeled by launch dir / "
                              "single-home cwd). Dry-run by default.")
    rpj.add_argument("--apply", action="store_true",
                     help="write the confident re-projections (records an undo set)")
    rpj.add_argument("--undo", action="store_true",
                     help="restore the labels changed by the last --apply")
    rpj.add_argument("--min-margin", type=float, default=None,
                     help="confidence threshold (vector: cosine gap, default 0.06; "
                          "keyword: overlap gap, default 1)")
    rpj.add_argument("--suspect", action="append", default=[], metavar="LABEL",
                     help="also reconsider memories carrying this (unreliable) "
                          "label — the launch-dir default the bug stamps, e.g. "
                          "--suspect RetirementEstimator. Repeatable. By default "
                          "only unscoped (NULL-project) memories are reconsidered.")

    mp = sub.add_parser("import-markdown",
                        help="import Markdown: a doc chunked by heading into "
                             "section memories (default), or --frontmatter for a "
                             "directory of memory files")
    mp.add_argument("path", help="a .md file or a directory of .md files")
    mp.add_argument("--frontmatter", action="store_true",
                    help="treat PATH as a directory of frontmatter memory files "
                         "(name/description/[[wikilinks]]), one file -> one row, "
                         "instead of heading-chunking an arbitrary document")
    mp.add_argument("--project")
    mp.add_argument("--shared", action="store_true",
                    help="write into the machine-level shared tier")

    xp = sub.add_parser("export-markdown",
                        help="export memories to human-readable Markdown (a "
                             "directory of files + a FornixDB.md index, or one "
                             "doc with --document); round-trips with "
                             "`import-markdown --frontmatter`")
    xp.add_argument("out_dir", help="output directory (created if missing)")
    xp.add_argument("--project", help="only memories in this project")
    xp.add_argument("--kind", choices=KINDS, help="only this kind")
    xp.add_argument("--include-superseded", action="store_true",
                    help="also export tombstoned (superseded) memories")
    xp.add_argument("--shared", action="store_true",
                    help="export the machine-level shared tier instead")
    xp.add_argument("--index-name", default="FornixDB.md",
                    help="directory-mode index filename (default FornixDB.md, "
                         "never MEMORY.md)")
    xp.add_argument("--query", help="only memories matching this subject")
    xp.add_argument("--when", help="time phrase, e.g. 'yesterday', 'last week'")
    xp.add_argument("--since", help="ISO lower bound on event_time")
    xp.add_argument("--until", help="ISO upper bound on event_time")
    xp.add_argument("--document", nargs="?", const="", metavar="FILE",
                    help="write ONE human-readable doc instead of a directory "
                         "(optional FILE path, else <out_dir>/FornixDB-export.md)")

    args = p.parse_args(argv)

    # `configure` is meant to be launched argument-free (the fornix-config
    # wrapper). With no explicit --db, resolve which store to configure from the
    # registry (asking only if there is more than one) BEFORE the no-store guard
    # below — otherwise the guard would short-circuit on the bare default path.
    if args.cmd == "configure":
        resolve_configure_store(args)

    # Never materialize a store as a SIDE EFFECT of falling back to the default
    # path. Only `init` (or an explicit --create) may create the default store;
    # any other command run with no --db / $FORNIXDB_DB and no existing store
    # reports and exits instead of littering ~/.fornixdb with an empty store +
    # registry (decision 2026-06-21 — `doctor`/`config` used to leave strays).
    # An explicit --db / $FORNIXDB_DB path is the caller's choice and still
    # creates on demand, as before.
    if (args.db is None and not os.environ.get(DEFAULT_DB_ENV)
            and args.cmd != "init" and not args.create):
        dpath = default_db_path()
        if not dpath.exists():
            print(f"[warn] no FornixDB store at {dpath}\n"
                  f"       create one with 'fornixdb init', or pass "
                  f"--db <path> / set ${DEFAULT_DB_ENV} / use --create.",
                  file=sys.stderr)
            return 0

    store = MemoryStore(db_path=args.db)
    stores = open_stores(store, shared=not args.no_shared)
    try:
        return _dispatch(p, args, store, stores)
    finally:
        # Windows can't unlink an open file (W1, PC install 2026-06-12), so an
        # in-process caller's TemporaryDirectory teardown needs these closed.
        for _, st in stores:
            st.close()


def _dispatch(p, args, store, stores) -> int:
    if args.cmd == "init":
        print(f"store ready: {args.db or default_db_path()}")
        if not args.no_shared:
            print(f"shared tier:  {shared_db_path()}")

    elif args.cmd == "store":
        target = stores[-1][1] if (args.shared and len(stores) > 1) else store
        try:
            mem_id = target.store(
                args.gist, args.detail, kind=args.kind, name=args.name,
                topics=args.topic, project=args.project,
                event_time=args.event_time, event_time_end=args.event_time_end,
                session_id=args.session_id, salience=args.salience,
                source=args.source, source_ref=args.source_ref,
                # every agent reads the shared tier with full trust, so shared
                # rows carry who wrote them (B3); own-store rows are implicit
                writer=_agent_label(store) if args.shared else None,
            )
        except FrozenStoreError as e:
            print(f"not stored: {e}", file=sys.stderr)
            return 1
        where = "shared:" if (args.shared and len(stores) > 1) else ""
        print(f"stored #{where}{mem_id}")
        linked = target.conn.execute(
            "SELECT related_id FROM memory_link WHERE memory_id = ? "
            "AND relation = 'relates'", (mem_id,)).fetchall()
        if linked:
            print("  linked " + ", ".join(f"#{r['related_id']}" for r in linked)
                  + " (from [[wikilinks]])")
        from .consolidate import supersede_suggestion
        sug = supersede_suggestion(target, mem_id, args.gist + " " + (args.detail or ""),
                                   args.kind)
        if sug and sug.get("reason") == "resolves":
            print(f"  note: looks like it CLOSES open task memory #{sug['id']} "
                  f"\"{sug['gist'][:60]}\" — if so, `supersede {sug['id']} {mem_id}` "
                  f"to close it", file=sys.stderr)
        elif sug:
            print(f"  note: closely matches #{sug['id']} \"{sug['gist'][:60]}\" "
                  f"(cos {sug['cosine']}) — if this updates it, "
                  f"`supersede {sug['id']} {mem_id}`; if related, "
                  f"`link {mem_id} {sug['id']}`", file=sys.stderr)

    elif args.cmd == "remind":
        from .prospective import remind
        try:
            r = remind(store, args.what, args.when, urgent=args.urgent,
                       project=args.project, topics=args.topic, source="cli")
        except (ValueError, FrozenStoreError) as e:
            print(f"not stored: {e}", file=sys.stderr)
            return 1
        tag = " URGENT (nags until `fornixdb ack`)" if args.urgent else ""
        print(f"reminder #{r['id']} due {r['due']}{tag}: {r['gist']}")

    elif args.cmd == "due":
        from .prospective import due as due_now, upcoming
        rows = due_now(store, deliver=not args.peek)
        if not rows:
            print("nothing due.")
        for r in rows:
            mark = "(peek)" if args.peek else "(delivered)"
            if r["urgent"]:
                mark = ("(peek)" if args.peek else
                        f"(URGENT — delivery {r['deliveries']}, repeats until ack)")
            print(f"#{r['id']} due {r['due']} {mark}  {r['gist']}")
        if args.upcoming is not None:
            ahead = upcoming(store, within_hours=args.upcoming)
            if ahead:
                print(f"upcoming (next {args.upcoming:g}h):")
                for r in ahead:
                    print(f"  #{r['id']} due {r['due']}  {r['gist']}")

    elif args.cmd == "ack":
        from .prospective import ack
        n = ack(store)
        print(f"acknowledged {n} urgent reminder{'s' if n != 1 else ''}."
              if n else "nothing was nagging.")

    elif args.cmd == "recall":
        since = until = None
        if args.when:
            try:
                s, e = parse_when(args.when)
            except ValueError as err:
                p.error(str(err))
            since, until = s.isoformat(), e.isoformat()
        if args.since:
            since = _time_bound(p, args.since, "since")
        if args.until:
            until = _time_bound(p, args.until, "until")
        rows = multi_recall(stores, args.query, limit=args.limit, kind=args.kind,
                            project=args.project, since=since, until=until,
                            related=args.related, include_superseded=args.all)
        _print_rows(rows, args.json, args.max_chars)

    elif args.cmd == "brief":
        b = multi_brief(stores, project=args.project, days=args.days)
        if args.json:
            b["consolidation"] = consolidate_status(store)
            print(json.dumps(b, indent=2, default=str))
        else:
            print(f"--- recent sessions (since {b['since']}) ---")
            _print_rows(b["recent"], False)
            print("--- most salient standing knowledge ---")
            _print_rows(b["salient"], False)
            if b.get("useful"):
                print("--- most useful so far (endorsed / recalled) ---")
                for m in b["useful"]:
                    sid = f"{m['_store']}:{m['id']}" if m.get("_store") else m["id"]
                    print(f"#{sid} [helpful x{m['helpful_count']}, "
                          f"recalled x{m['recall_count']}]  {(m.get('gist') or '')[:80]}")
            st = consolidate_status(store)
            if st["due"]:
                print(f"--- consolidation DUE: {st['reason']} "
                      f"(run a pass, then `consolidate done`) ---")

    elif args.cmd == "consolidate":
        if args.action == "done":
            ts = mark_done(store)
            print(f"consolidation pass recorded at {ts}")
        elif args.action == "propose":
            from .consolidate import propose
            work = propose(store)
            if args.json:
                print(json.dumps(work, indent=2, default=str))
            else:
                print(f"--- distill sessions ({len(work['distill'])}) ---")
                for d in work["distill"]:
                    print(f"#{d['id']:<5} eff {d['eff_salience']:.2f}  {d['gist'][:90]}")
                print(f"--- rewrite poor gists ({len(work['gists'])}) ---")
                for g in work["gists"]:
                    print(f"#{g['id']:<5} {g['problem']}: {g['gist'][:90]}")
                print(f"--- close completed tasks ({len(work.get('resolutions', []))}) ---")
                for m in work.get("resolutions", []):
                    print(f"supersede old=#{m['ids'][0]} new=#{m['ids'][1]} "
                          f"cos {m['cosine']:.2f} ({m['kinds'][0]}/{m['kinds'][1]})")
                    for g in m["gists"]:
                        print(f"        {(g or '')[:90]}")
                print(f"--- merge near-duplicates ({len(work['merges'])}) ---")
                for m in work["merges"]:
                    print(f"#{m['ids'][0]} + #{m['ids'][1]} cos {m['cosine']:.2f} ({m['kind']})")
                    for g in m["gists"]:
                        print(f"        {g[:90]}")
                print(f"--- check for contradictions ({len(work['contradictions'])}) ---")
                for m in work["contradictions"]:
                    print(f"#{m['ids'][0]} ~ #{m['ids'][1]} cos {m['cosine']:.2f} ({m['kind']})")
                    for g in m["gists"]:
                        print(f"        {g[:90]}")
                print(f"--- reality check: pointers to missing files "
                      f"({len(work.get('reality', []))}) ---")
                for m in work.get("reality", []):
                    print(f"#{m['id']:<5} MISSING {m['path']}")
                    print(f"        {(m['gist'] or '')[:90]}")
                print(f"--- chronic push-noise: judge disposition "
                      f"({len(work.get('chronic', []))}) ---")
                for m in work.get("chronic", []):
                    print(f"#{m['id']:<5} pushed {m['pushed']}x, used 0, "
                          f"pulled {m['pulls']}x ({m['kind']})")
                    print(f"        {(m['gist'] or '')[:90]}")
                print(f"--- re-project mis-scoped memories "
                      f"({len(work.get('reproject', []))}) ---")
                for m in work.get("reproject", []):
                    print(f"#{m['id']:<5} {m['current'] or '(none)'} -> "
                          f"{m['proposed']}  margin {m['margin']:.3f}")
                    print(f"        {(m['gist'] or '')[:90]}")
                print(f"--- weave new associations ({len(work['associations'])}) ---")
                for m in work["associations"]:
                    print(f"#{m['ids'][0]} <-> #{m['ids'][1]} cos {m['cosine']:.2f} "
                          f"({m['kinds'][0]}/{m['kinds'][1]})")
                    for g in m["gists"]:
                        print(f"        {g[:90]}")
        else:
            print(json.dumps(consolidate_status(store), indent=2, default=str))

    elif args.cmd == "dream":
        from .consolidate import dream
        try:
            rep = dream(store, weave=args.weave, done=args.done)
        except FrozenStoreError as e:
            print(f"not dreamed: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(rep, indent=2, default=str))
        else:
            print(rep["narrative"])
            work = rep["work"]
            # on done the wake narrative already names the remaining heal
            # candidates; the full worklist is the entering (not-done) view
            if work.get("resolutions") and not args.done:
                print(f"\n--- completed tasks to close "
                      f"({len(work['resolutions'])}) — supersede the open one "
                      "(accept a reviewed pair with: link <a> <b> "
                      "--relation distinct) ---")
                for m in work["resolutions"]:
                    print(f"supersede old=#{m['ids'][0]} new=#{m['ids'][1]} "
                          f"cos {m['cosine']:.2f} ({m['kinds'][0]}/{m['kinds'][1]})")
                    print(f"        task:  {(m['gists'][0] or '')[:80]}")
                    print(f"        close: {(m['gists'][1] or '')[:80]}")
            if work["contradictions"] and not args.done:
                print(f"\n--- possible outdated memories to reconcile "
                      f"({len(work['contradictions'])}) — supersede the stale one "
                      "(accept a reviewed pair with: link <a> <b> "
                      "--relation distinct) ---")
                for m in work["contradictions"]:
                    print(f"#{m['ids'][0]} ~ #{m['ids'][1]} cos {m['cosine']:.2f} ({m['kind']})")
                    for g in m["gists"]:
                        print(f"        {g[:90]}")
            if work.get("reality") and not args.done:
                print(f"\n--- reality check ({len(work['reality'])}) — a live memory "
                      "points at a file that isn't there (fix or supersede it; "
                      "accept a reviewed one with: tag <id> reality-ok) ---")
                for m in work["reality"]:
                    print(f"#{m['id']:<5} MISSING {m['path']}")
                    print(f"        {(m['gist'] or '')[:90]}")
            if work.get("chronic") and not args.done:
                print(f"\n--- chronic push-noise ({len(work['chronic'])}) — pushed "
                      "over and over, never used downstream (the floor already "
                      "quiets these; decide their fate: supersede if obsolete, "
                      "reproject if mis-scoped, or accept with: "
                      "tag <id> noise-ok) ---")
                for m in work["chronic"]:
                    print(f"#{m['id']:<5} pushed {m['pushed']}x, used 0, "
                          f"pulled {m['pulls']}x ({m['kind']}"
                          f"{', ' + m['project'] if m['project'] else ''})")
                    print(f"        {(m['gist'] or '')[:90]}")
            if work.get("reproject") and not args.done:
                print(f"\n--- mis-scoped memories to re-project "
                      f"({len(work['reproject'])}) — content points at a "
                      "different project (apply via: reproject --apply, "
                      "undo-able) ---")
                for m in work["reproject"]:
                    print(f"#{m['id']:<5} {m['current'] or '(none)'} -> "
                          f"{m['proposed']}  margin {m['margin']:.3f}")
                    print(f"        {(m['gist'] or '')[:90]}")
            if rep.get("dials") and not args.done:
                print(f"\n--- 🎛 dial report ({len(rep['dials'])}) — evidence-"
                      "backed config suggestions; nothing flipped ---")
                for d in rep["dials"]:
                    print(f"{d['dial']} (current: {d['current']})")
                    print(f"        evidence:   {d['evidence']}")
                    print(f"        suggestion: {d['suggestion']}")
            if work["associations"] and not args.done:
                verb = "wove" if args.weave else "to weave"
                print(f"\n--- new connections {verb} ({len(work['associations'])}) ---")
                for m in work["associations"]:
                    arrow = "<->" if args.weave else "<- ?->"
                    print(f"#{m['ids'][0]} {arrow} #{m['ids'][1]} cos {m['cosine']:.2f} "
                          f"({m['kinds'][0]}/{m['kinds'][1]})")
                    for g in m["gists"]:
                        print(f"        {g[:90]}")
            if work["merges"] and not args.done:
                print(f"\n--- near-duplicates to merge ({len(work['merges'])}) ---")
                for m in work["merges"]:
                    print(f"#{m['ids'][0]} + #{m['ids'][1]} cos {m['cosine']:.2f} ({m['kind']})")
                    for g in m["gists"]:
                        print(f"        {g[:90]}")
            if work["distill"] and not args.done:
                print(f"\n--- sessions to distill ({len(work['distill'])}) ---")
                for d in work["distill"]:
                    print(f"#{d['id']:<5} eff {d['eff_salience']:.2f}  {d['gist'][:90]}")
            if work["gists"] and not args.done:
                print(f"\n--- gists to tidy ({len(work['gists'])}) ---")
                for g in work["gists"]:
                    print(f"#{g['id']:<5} {g['problem']}: {g['gist'][:90]}")

    elif args.cmd == "irrelevant":
        if args.retract is not None:
            target = stores[-1][1] if (args.ref == "shared" and len(stores) > 1) else store
            try:
                target.retract_feedback(args.retract)
            except FrozenStoreError as e:
                print(f"refused: {e}", file=sys.stderr)
                return 1
            print(f"feedback {args.retract} retracted (kept; re-mark to reactivate)")
        elif args.list_feedback:
            target, mem_id = store, None
            if args.ref:
                target, inner = resolve_ref(stores, args.ref)
                mem = target.show(inner, reinforce=False)
                if mem is None:
                    print(f"no memory: {args.ref}", file=sys.stderr)
                    return 1
                mem_id = mem["id"]
            rows = target.list_feedback(mem_id)
            if args.json:
                print(json.dumps(rows, indent=2, default=str))
            elif not rows:
                print("(no feedback)")
            for f in rows if not args.json else []:
                state = " [retracted]" if f["retracted"] else ""
                print(f"{f['id']:>4}  #{f['memory_id']}{state}  "
                      f"irrelevant to: {f['query']!r}  ({f['gist'][:50]})")
        else:
            if not args.ref or not args.query:
                p.error("irrelevant needs <ref> <query> (or --list / --retract)")
            target, inner = resolve_ref(stores, args.ref)
            mem = target.show(inner, reinforce=False)
            if mem is None:
                print(f"no memory: {args.ref}", file=sys.stderr)
                return 1
            try:
                fid = target.mark_irrelevant(mem["id"], args.query)
            except FrozenStoreError as e:
                print(f"refused: {e}", file=sys.stderr)
                return 1
            print(f"#{mem['id']} downweighted for queries like {args.query!r} "
                  f"(feedback {fid}; `irrelevant --retract {fid}` to undo)")

    elif args.cmd == "helpful":
        target, inner = resolve_ref(stores, args.ref)
        try:
            m = target.mark_helpful(inner)
        except ValueError:
            print(f"no memory: {args.ref}", file=sys.stderr)
            return 1
        except FrozenStoreError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"#{m['id']} endorsed (helpful x{m['helpful_count']}) — "
              f"ranks higher everywhere, reinforced against staleness")

    elif args.cmd == "tag":
        try:
            store.tag(args.memory_id, args.topic)
        except FrozenStoreError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"#{args.memory_id} tagged '{args.topic}'")

    elif args.cmd == "set-gist":
        target, ref = resolve_ref(stores, args.ref)
        mem = target.show(ref, reinforce=False)
        if mem is None:
            print(f"no memory: {args.ref}", file=sys.stderr)
            return 1
        try:
            target.set_gist(mem["id"], args.gist)
        except FrozenStoreError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"#{mem['id']} gist rewritten (re-embedded in place; without an "
              f"embedder the stale vector drops and backfill heals it later)")

    elif args.cmd == "tier":
        from .tiers import tier_down, tier_status
        if args.action == "down":
            print(json.dumps(tier_down(store, dry_run=args.dry_run), indent=2))
        else:
            print(json.dumps(tier_status(store), indent=2))

    elif args.cmd == "eval":
        from .evals import format_report, load_history, record_run, run
        if store.conn.execute("SELECT count(*) c FROM memory").fetchone()["c"] == 0:
            print(f"eval: store {args.db or default_db_path()} has 0 memories — "
                  f"every case would falsely score 0%, which looks like a total "
                  f"regression but is just an empty store. Point at a populated "
                  f"one with a PRE-subcommand flag: "
                  f"fornixdb --db <path> eval {args.golden}  (or set $FORNIXDB_DB).",
                  file=sys.stderr)
            return 2
        report = run(store, args.golden,
                     embedder=False if args.keyword_only else None)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(format_report(report, verbose=args.verbose))
        if args.record:
            prev = load_history(args.record)
            rec = record_run(report, args.record, store=store)
            msg = (f"recorded to {args.record}: hit@1 {rec['hit@1']:.0%} "
                   f"hit@k {rec['hit@k']:.0%} MRR {rec['mrr']:.3f} "
                   f"drift {rec['drift']}")
            if prev:
                p = prev[-1]
                msg += f"  (prev {p['when'][:10]}: hit@1 {p['hit@1']:.0%} MRR {p['mrr']:.3f})"
            print(msg, file=sys.stderr)
        if args.min_hitk is not None and report["hit@k"] < args.min_hitk:
            print(f"FAIL: hit@k {report['hit@k']} < required {args.min_hitk}",
                  file=sys.stderr)
            return 1
        if args.max_drift is not None and len(report["drift"]) > args.max_drift:
            print(f"FAIL: {len(report['drift'])} rank1 case(s) drifted "
                  f"> allowed {args.max_drift}", file=sys.stderr)
            return 1
        if args.max_leaks is not None and len(report["abstain_leaks"]) > args.max_leaks:
            print(f"FAIL: {len(report['abstain_leaks'])} abstain case(s) leaked "
                  f"> allowed {args.max_leaks}", file=sys.stderr)
            return 1

    elif args.cmd == "answer-eval":
        from .answer_eval import (default_answerer, format_report as fmt_ans,
                                  load_history, record_run, run as run_ans)
        if store.conn.execute("SELECT count(*) c FROM memory").fetchone()["c"] == 0:
            print(f"answer-eval: store {args.db or default_db_path()} has 0 "
                  f"memories — every case would falsely score 0%. Point at a "
                  f"populated store with a PRE-subcommand flag: "
                  f"fornixdb --db <path> answer-eval {args.golden}  "
                  f"(or set $FORNIXDB_DB).", file=sys.stderr)
            return 2
        answerer = default_answerer(args.model)
        report = run_ans(store, args.golden, answerer,
                         embedder=False if args.keyword_only else None)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(fmt_ans(report, verbose=args.verbose))
        if args.record:
            prev = load_history(args.record)
            rec = record_run(report, args.record, store=store)
            msg = (f"recorded to {args.record}: lift {rec['lift']:.0%} "
                   f"A-correct {rec['a_correct']} B-correct {rec['b_correct']} "
                   f"regressions {rec['regressions']}")
            if prev:
                pr = prev[-1]
                msg += f"  (prev {pr['when'][:10]}: lift {pr['lift']:.0%})"
            print(msg, file=sys.stderr)
        if args.min_lift is not None and report["lift"] < args.min_lift:
            print(f"FAIL: lift {report['lift']} < required {args.min_lift}",
                  file=sys.stderr)
            return 1
        if args.max_regress is not None and len(report["regressions"]) > args.max_regress:
            print(f"FAIL: {len(report['regressions'])} regression(s) "
                  f"> allowed {args.max_regress}", file=sys.stderr)
            return 1

    elif args.cmd == "benefit":
        from .benefit import (coverage, format_report as fmt_benefit,
                              golden_marginal, scan_flat_baseline)
        cap = args.cap_chars or 24_400
        baseline = scan_flat_baseline(args.memory_md, args.memory_dir, cap)
        cov = coverage(store, baseline)
        gold = golden_marginal(store, baseline, args.golden) if args.golden else None
        if args.json:
            print(json.dumps({"coverage": cov, "golden": gold}, indent=2, default=str))
        else:
            print(fmt_benefit(cov, gold))

    elif args.cmd == "budget":
        from .budget import enforce, shrink, status as budget_status
        if args.action == "shrink":
            if args.mb is None:
                p.error("budget shrink needs a target: budget shrink <MB>")
            try:
                print(json.dumps(shrink(store, args.mb, dry_run=args.dry_run),
                                 indent=2))
            except (FrozenStoreError, ValueError) as e:
                print(f"refused: {e}", file=sys.stderr)
                return 1
        elif args.action == "enforce":
            print(json.dumps(enforce(store, dry_run=args.dry_run), indent=2))
        else:
            print(json.dumps(budget_status(store), indent=2))

    elif args.cmd == "configure":
        from .wizard import run_configure
        label = args.db or str(default_db_path())
        try:
            run_configure(store, db_label=label)
        except (FrozenStoreError, ValueError) as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1

    elif args.cmd == "level":
        from . import levels
        target = stores[-1][1] if (args.shared and len(stores) > 1) else store
        if args.level is None:
            rung, _ = levels.current_rung(target)
            print(f"--- operating levels (current rung: {rung}) ---")
            print(levels.format_ladder(target))
            print("\nset a rung: `level L3`   toggle one: `level L4 off`")
        else:
            try:
                if args.state is None:
                    print(levels.set_rung(target, args.level))
                else:
                    print(levels.toggle(target, args.level, args.state == "on"))
            except ValueError as e:
                print(f"refused: {e}", file=sys.stderr)
                return 1
            print(levels.format_ladder(target))

    elif args.cmd == "config":
        target = stores[-1][1] if (args.shared and len(stores) > 1) else store
        if args.key is None:
            from . import levels
            from .doctor import (CONFIG_DEFAULTS, config_overview, format_config,
                                 format_suggested, suggested_settings)
            print("--- current settings (read-only view; nothing changed) ---")
            print(format_config(config_overview(target), CONFIG_DEFAULTS))
            print("\n--- operating-levels ladder (`level` to change) ---")
            print(levels.format_ladder(target))
            print("\n--- suggested defaults ('SET' = not yet applied; "
                  "`doctor --apply-suggested` to apply) ---")
            print(format_suggested(suggested_settings(target)))
        elif args.value is None:
            print(get_config(target, args.key) or "(unset)")
        else:
            set_config(target, args.key, args.value)
            hint = CAPTURE_MODE_HELP.get(args.value, "")
            print(f"{args.key} = {args.value}" + (f"  ({hint})" if hint else ""))

    elif args.cmd == "doctor":
        from .doctor import (apply_suggested, diagnose, format_diagnose,
                             format_suggested, suggested_settings)
        print("--- health ---")
        print(format_diagnose(diagnose(store)))
        print("\n--- suggested defaults ---")
        print(format_suggested(suggested_settings(store)))
        if args.apply_suggested:
            try:
                applied = apply_suggested(store)
            except FrozenStoreError as e:
                print(f"refused: {e}", file=sys.stderr)
                return 1
            print("\n--- applied ---")
            print("\n".join(applied) if applied
                  else "(nothing to apply — all suggestions already satisfied)")

    elif args.cmd == "ingest":
        from .adapters.native_memory import (auto_background_enabled, ingest,
                                             ingest_mode, native_dir,
                                             set_ingest_mode, set_native_dir)
        if args.mode:
            print(set_ingest_mode(store, args.mode))
        if args.dir:
            print(set_native_dir(store, args.dir))
        if args.run:
            r = ingest(store)
            print(f"ingested from {r['dir']} — imported {r['imported']}, "
                  f"skipped {r['skipped']}, links {r['links']}" if r.get("ok")
                  else r["reason"])
        mode = ingest_mode(store)
        d = native_dir(store) or "(unset)"
        bg = "ON" if (auto_background_enabled(store) and native_dir(store)) else "off"
        print(f"\ningest_mode = {mode}  |  native_dir = {d}  |  "
              f"background native ingest: {bg}")
        if mode == "explicit":
            print("explicit: NO background automation (no auto-ingest, no passive "
                  "session capture). FornixDB runs only on deliberate "
                  "remember/recall/`ingest --run`.")
        else:
            print(f"{mode}: native ingest + passive session capture run on the "
                  "session-end hook. `ingest --mode explicit` turns ALL background "
                  "off; set a directory with `ingest --dir <path>`.")

    elif args.cmd == "embed":
        from .vectors import backfill, get_default_embedder
        emb = get_default_embedder()
        if emb is None:
            print("no embedder available — pip install model2vec (see README)",
                  file=sys.stderr)
            return 1
        n = backfill(store, emb, batch=args.batch)
        print(f"embedded {n} memories ({emb.name})")

    elif args.cmd == "timeline":
        if args.since or args.until:
            start = _time_bound(p, args.since, "since") if args.since else "0000"
            end = _time_bound(p, args.until, "until") if args.until else datetime.now().isoformat()
        elif args.when:
            try:
                s, e = parse_when(args.when)
            except ValueError as err:
                p.error(str(err))
            start, end = s.isoformat(), e.isoformat()
        else:
            p.error("timeline needs a phrase ('last thursday') or --since/--until")
        rows = multi_timeline(stores, start, end, kind=args.kind,
                              project=args.project, limit=args.limit)
        _print_rows(rows, args.json, args.max_chars)

    elif args.cmd == "show":
        target, ref = resolve_ref(stores, args.ref)
        mem = target.show(ref, reinforce=not args.no_reinforce)
        if mem is None:
            print(f"no memory: {args.ref}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(mem, indent=2, default=str))
        else:
            print(_fmt_gist_line(mem))
            if mem.get("name"):
                print(f"name:     {mem['name']}")
            print(f"event:    {mem['event_time']}"
                  + (f" → {mem['event_time_end']}" if mem.get("event_time_end") else ""))
            print(f"recorded: {mem['recorded_time']}   salience: {mem['salience']:.2f}"
                  f"   recalls: {mem['recall_count']}")
            if mem.get("topics"):
                print(f"topics:   {', '.join(mem['topics'])}")
            if mem.get("superseded_by"):
                print(f"SUPERSEDED by #{mem['superseded_by']} at {mem['superseded_time']}")
            for ln in mem.get("links", []):
                print(f"  {ln['relation']} → #{ln['related_id']}: {ln['related_gist'][:70]}")
            if mem.get("detail"):
                print("-" * 60)
                print(mem["detail"])

    elif args.cmd == "supersede":
        try:
            store.supersede(args.old_id, args.new_id)
        except FrozenStoreError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"#{args.old_id} superseded by #{args.new_id} (kept, tombstoned)")

    elif args.cmd == "link":
        try:
            store.link(args.memory_id, args.related_id, args.relation)
        except FrozenStoreError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"#{args.memory_id} {args.relation} #{args.related_id}")

    elif args.cmd == "jot":
        try:
            cid = store.jot(args.note)
        except FrozenStoreError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"jotted [{cid}] — {len(store.candidates())} pending "
              "(`fornixdb candidates` to review)")

    elif args.cmd == "candidates":
        if args.clear:
            print(f"discarded all {store.discard_candidates()} pending")
        elif args.discard:
            print(f"discarded {store.discard_candidates(ids=args.discard)}")
        rows = store.candidates()
        if not rows:
            print("no pending candidates")
        else:
            print(f"{len(rows)} pending (promote a keeper with `store`, then "
                  "`candidates --discard <id>` or `--clear`):")
            for r in rows:
                print(f"  [{r['id']}] {r['note'][:100]}")

    elif args.cmd == "feel":
        import subprocess

        from . import senses

        def _commit_line(mid, gist):
            if args.json:
                print(json.dumps({"id": mid, "gist": gist}))
            else:
                print(f"#{mid}  {gist}")

        # A literal reading works on any platform; no arg falls back to the
        # Mac battery adapter (pmset), which only exists on a Mac.
        if not args.live and args.reading is not None:
            gist = senses.feel_gist(args.sensor, args.reading)
            mid = senses.feel(store, args.reading, sensor=args.sensor,
                              gist=gist, project=args.project)
            _commit_line(mid, gist)
            return 0

        from .adapters import mac_proprioception as mp
        try:
            if not args.live:
                # drop unknown (None) fields so the gist stays clean
                reading = {k: v for k, v in mp.read_battery().items()
                           if v is not None}
                gist = senses.feel_gist(args.sensor, reading)
                mid = senses.feel(store, reading, sensor=args.sensor,
                                  gist=gist, project=args.project)
                _commit_line(mid, gist)
                return 0
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as e:
            p.error(f"feel: could not read the battery ({e}) — on a non-Mac, "
                    f"pass a reading string: fornixdb feel \"...\" --sensor NAME")

        from . import feelloop
        seen: list = []

        def _on(ev):
            seen.append(ev)
            print(f"  #{ev.memory_id:<5} {ev.reason:<9} {ev.gist}")

        dbrow = store.conn.execute("PRAGMA database_list").fetchone()
        dbfile = (dbrow[2] if dbrow and dbrow[2] else "in-memory store")
        print(f"feel: watching '{args.sensor}' for {args.seconds:.0f}s "
              f"(sampling every {args.interval:.0f}s; Ctrl-C to stop). "
              f"Committing to {dbfile}")
        frames = mp.battery_frames(interval_seconds=args.interval,
                                   percent_step=args.percent_step)
        try:
            feelloop.run_feel(store, frames, sensor=args.sensor,
                              heartbeat_seconds=args.heartbeat,
                              max_seconds=args.seconds, project=args.project,
                              on_commit=_on)
        except KeyboardInterrupt:
            print()
        print(f"{len(seen)} memories committed.")
        return 0

    elif args.cmd == "watch":
        from . import senses

        seen = []

        def _on(ev):
            seen.append(ev)
            if args.json:
                print(json.dumps({"id": ev.memory_id, "reason": ev.reason,
                                  "gist": ev.gist}))
            else:
                span = f"{ev.t_start:.1f}->{ev.t_end:.1f}s"
                print(f"  #{ev.memory_id:<5} {ev.reason:<9} {span:<16} {ev.gist}")

        if not args.json:
            dbrow = store.conn.execute("PRAGMA database_list").fetchone()
            dbfile = dbrow[2] if dbrow and dbrow[2] else "in-memory store"
            print(f"watch: source '{args.source}' for {args.seconds:.0f}s "
                  f"(Ctrl-C to stop). Committing to {dbfile}")
            if args.source == "camera":
                print("  camera: macOS prompts for permission on first open; "
                      "the green light is on while capturing.")
        try:
            senses.watch(store, args.source, rate_hz=args.rate,
                         window_seconds=args.window, threshold=args.threshold,
                         max_seconds=args.seconds, max_commits=args.max_commits,
                         project=args.project, on_commit=_on)
        except KeyboardInterrupt:
            print()
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            p.error(f"watch: {e}")
        print(f"{len(seen)} memories committed.")
        return 0

    elif args.cmd == "recaption":
        from . import recaption

        pend = recaption.pending_captions(store, limit=args.limit)
        if args.dry_run:
            if args.json:
                print(json.dumps([{"id": m, "gist": g, "keyframe": k}
                                  for m, g, k in pend]))
            else:
                print(f"{len(pend)} keyframe(s) await a caption:")
                for mid, _, k in pend:
                    print(f"  #{mid:<5} {k}")
            return 0
        if not pend:
            print("no watch keyframes await a caption.")
            return 0

        try:
            from .adapters import mac_vision
            captioner = (mac_vision.vlm_captioner(args.model) if args.model
                         else mac_vision.vlm_captioner())
        except (ImportError, RuntimeError) as e:
            p.error(f"recaption: {e}")

        def _on(mid, old, cap):
            if args.json:
                print(json.dumps({"id": mid, "old": old, "caption": cap}))
            else:
                print(f"  #{mid:<5} {cap}")

        if not args.json:
            print(f"captioning {len(pend)} keyframe(s) with a local VLM…")
        applied = recaption.recaption(store, captioner, limit=args.limit,
                                      on_caption=_on)
        print(f"{len(applied)} caption(s) written.")
        return 0

    elif args.cmd == "look":
        from . import senses
        try:
            from .adapters import mac_vision
            captioner = (mac_vision.vlm_captioner(args.model) if args.model
                         else mac_vision.vlm_captioner())
            caption = senses.glance(
                store, args.source, captioner, remember=args.remember,
                keep_keyframe=args.keep_keyframe, project=args.project)
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            p.error(f"look: {e}")
        if args.json:
            print(json.dumps({"source": args.source, "caption": caption,
                              "remembered": args.remember}))
        else:
            print(caption or "(nothing to describe)")
        return 0

    elif args.cmd == "usage":
        from .budget import machine_usage
        u = machine_usage()
        if args.json:
            print(json.dumps(u, indent=2))
        else:
            for st in u["stores"]:
                mems = f"{st['memories']} memories" if st["memories"] is not None else "?"
                print(f"{st['label']:<24} {st['mb']:>8.3f} MB  ({mems})  {st['path']}")
            cap = (f"  (machine cap {u['machine_budget_mb']} MB, policy "
                   f"'{u['machine_policy']}'"
                   + (", OVER" if u["over_budget"] else "") + ")"
                   if u["machine_budget_mb"] else "  (no machine-wide cap)")
            print(f"{'TOTAL':<24} {u['total_mb']:>8.3f} MB{cap}")
            if u.get("machine_budget_defaulted"):
                print("note: the machine cap is the INSTALL DEFAULT (20% of "
                      "free disk, max 2 GB) — review it: "
                      "`config machine_budget_mb <MB> --shared` (or 'off')")

    elif args.cmd == "tokens":
        if args.billed:
            from .billed import format_report, report
            r = report(store, source=(args.transcripts or None),
                       since_days=args.since_days)
        else:
            from .tokens import format_report, report
            r = report(store)
        print(json.dumps(r, indent=2, default=str) if args.json
              else format_report(r))

    elif args.cmd == "value":
        from . import value
        r = value.report(store, transcripts=(args.transcripts or None),
                         memory_md=args.memory_md, memory_dir=args.memory_dir)
        print(json.dumps(r, indent=2, default=str) if args.json
              else value.format_report(r))

    elif args.cmd == "tools":
        import json as _json

        from .adapters.mcp_server import (TOOLS, active_tools, set_tool_enabled,
                                          tool_tier, tools_disabled)
        from .tokens import estimate_tokens
        if args.profile == "full":
            for t in TOOLS:
                set_tool_enabled(store, t["name"], True)
            print("profile 'full' — all tools enabled")
        elif args.profile == "minimal":
            for t in TOOLS:
                set_tool_enabled(store, t["name"], False)  # core ignores disable
            print("profile 'minimal' — core tools only")
        elif args.action in ("enable", "disable"):
            if not args.name:
                p.error(f"`tools {args.action}` needs a tool name")
            print(set_tool_enabled(store, args.name, args.action == "enable"))

        off = tools_disabled(store)
        active = active_tools(store)
        active_tok = estimate_tokens(_json.dumps(active))
        all_tok = estimate_tokens(_json.dumps(TOOLS))
        print(f"\nMCP tools for this store — {len(active)}/{len(TOOLS)} advertised "
              f"(~{active_tok} of ~{all_tok} tok):\n")
        for t in TOOLS:
            name = t["name"]
            on = name not in off
            tier = tool_tier(name)
            cost = estimate_tokens(_json.dumps(t))
            mark = "on " if on else "OFF"
            lock = {"core": "core", "sense": "sns "}.get(tier, "opt ")
            if args.full:
                print(f"  [{mark}] {lock} ~{cost:>3}t  {name}")
                print(f"          {t['description']}")
            else:
                desc = (t["description"][:54] + "…") if len(t["description"]) > 55 \
                    else t["description"]
                print(f"  [{mark}] {lock} ~{cost:>3}t  {name:<16} {desc}")
        print("\nMemory tools are ON by default; the live senses ('sns' — look / "
              "feel / see / recaption) are OFF until you opt in, since they reach "
              "a camera / the battery / a local model:\n  fornixdb tools enable "
              "<name>    (turn a sense on)\n  fornixdb tools disable <name>   "
              "(trim an optional tool)\n  fornixdb tools --profile minimal\n"
              "Core tools cannot be disabled. There is no universal token limit — "
              "but some devices (small on-device models) have little prompt space, "
              "so trimming may be REQUIRED there. Changes take effect on the next "
              "MCP session/restart.")

    elif args.cmd == "topics":
        rows = store.conn.execute(
            """SELECT t.name, count(mt.memory_id) c FROM topic t
               LEFT JOIN memory_topic mt ON mt.topic_id = t.id
               GROUP BY t.id ORDER BY c DESC""".strip()
        ).fetchall()
        for r in rows:
            print(f"{r['c']:>5}  {r['name']}")

    elif args.cmd == "stats":
        print(json.dumps(store.stats(), indent=2, default=str))

    elif args.cmd == "field":
        from .field import format_field_debug
        lit = {int(x) for x in (args.lit or "").split(",") if x.strip().isdigit()}
        print(format_field_debug(store, args.thought,
                                 active_project=args.project, episode_ids=lit))

    elif args.cmd == "field-stats":
        from .field import field_log_path_for
        from .field_stats import format_report, load_beats, summarize
        path = args.log or field_log_path_for(store)
        summary = summarize(load_beats(path))
        if args.json:
            summary["log_path"] = str(path) if path else None
            print(json.dumps(summary, indent=2))
        else:
            print(format_report(summary, str(path) if path else None))

    elif args.cmd == "floor-stats":
        from .floor_stats import (format_report, load_records,
                                  outcomes_from_store, summarize)
        from .proactive import floor_log_path_for
        path = args.log or floor_log_path_for(store)
        records = load_records(path)
        ids = {r.get("id") for r in records if r.get("decision") == "surfaced"}
        if args.transcripts:
            # honest outcome: was each PUSH actually referenced downstream, from
            # the transcripts — not the lifetime-recall_count proxy.
            from .usefulness_scan import outcomes_from_scan, scan
            outcomes = outcomes_from_scan(scan(args.transcripts))
        else:
            outcomes = outcomes_from_store(store, ids)
        summary = summarize(records, outcomes)
        summary["log_path"] = str(path) if path else None
        summary["outcome_source"] = "transcripts" if args.transcripts else "store_counts"
        if args.json:
            print(json.dumps(summary, indent=2, default=str))
        else:
            print(format_report(summary))

    elif args.cmd == "usefulness-scan":
        from .usefulness_scan import format_report as us_report
        from .usefulness_scan import referenced_counts_from_scan, scan
        result = scan(args.transcripts)
        if args.apply:
            counts = referenced_counts_from_scan(result)
            credited = store.record_referenced(counts)
            result["applied"] = {"memories_scanned": len(counts), "credited": credited}
        n_suppressed = len(store.proactive_suppressed())
        result["proactive_suppressed"] = n_suppressed
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(us_report(result))
            if args.apply:
                print(f"\napplied: use-credit written for {result['applied']['credited']} "
                      f"memory(ies) referenced downstream "
                      f"(of {result['applied']['memories_scanned']} pushed).")
            print(f"\nproactive-suppressed: {n_suppressed} memory(ies) muted from the "
                  "push channels (`fornixdb suppress --scan` to refresh, --list to see).")

    elif args.cmd == "suppress":
        from . import suppress as sup
        if args.undo is not None:
            n = store.clear_proactive_suppression([args.undo], "cli_undo")
            msg = {"undo": args.undo, "cleared": n}
            print(json.dumps(msg) if args.json else
                  (f"redeemed #{args.undo} — it can push again."
                   if n else f"#{args.undo} was not suppressed; nothing to undo."))
        elif args.list_:
            rows = store.proactive_suppressed()
            if args.json:
                print(json.dumps(rows, indent=2, default=str))
            else:
                if not rows:
                    print("no memories are proactive-suppressed.")
                else:
                    print(f"proactive-suppressed memories ({len(rows)}):")
                    for r in rows:
                        print(f"  #{r['id']:<5} pushed {r['suppressed_pushed']}, "
                              f"referenced {r['suppressed_referenced']}  "
                              f"since {(r['proactive_suppressed_at'] or '')[:10]}  "
                              f"{(r['gist'] or '')[:60]}")
        else:   # --scan (default)
            report = sup.scan_and_apply(store, args.transcripts, apply=args.apply)
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print(sup.format_report(report))

    elif args.cmd == "reproject":
        from . import reproject as rp
        if args.undo:
            res = rp.undo(store)
            print(json.dumps(res) if args.json
                  else f"restored {res['restored']} project label(s).")
            return 0
        result = rp.propose(store, min_margin=args.min_margin, suspect=args.suspect)
        applied = None
        if args.apply and result["proposals"]:
            try:
                applied = rp.apply_proposals(store, result["proposals"])
            except FrozenStoreError as e:
                print(f"refused: {e}", file=sys.stderr)
                return 1
        if args.json:
            print(json.dumps({**result, "applied": applied}, indent=2, default=str))
        else:
            print(rp.format_report(result, applied=applied))

    elif args.cmd == "import-markdown":
        target = stores[-1][1] if (args.shared and len(stores) > 1) else store
        try:
            if args.frontmatter:
                from .adapters.markdown_import import import_directory
                result = import_directory(target, args.path, project=args.project)
            else:
                from .adapters.markdown_doc import import_path
                result = import_path(target, args.path, project=args.project)
        except FrozenStoreError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"imported {result['imported']}, skipped {result['skipped']}, "
                  f"links {result['links']}")

    elif args.cmd == "export-markdown":
        target = stores[-1][1] if (args.shared and len(stores) > 1) else store
        from pathlib import Path

        from .adapters.markdown_export import (export_directory,
                                               export_document)
        sel = dict(project=args.project, kind=args.kind,
                   include_superseded=args.include_superseded, query=args.query,
                   when=args.when, since=args.since, until=args.until)
        try:
            if args.document is not None:
                out_file = args.document or str(
                    Path(args.out_dir) / "FornixDB-export.md")
                result = export_document(target, out_file, **sel)
                loc = result["file"]
            else:
                result = export_directory(target, args.out_dir,
                                          index_name=args.index_name, **sel)
                loc = result["dir"]
        except ValueError as e:           # e.g. an unreadable --when phrase
            print(f"couldn't export: {e}")
            return 2
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"exported {result['exported']} memories to {loc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
