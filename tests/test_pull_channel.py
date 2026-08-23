"""The pull channel (L1): measuring the half of memory use that was invisible.

A memory reaches the model two ways — pushed (a hook injects it, paid for every
session whether used or not) or PULLED (the agent runs a reader verb itself,
paid for only when asked for). Only pushes were measured, so the honesty layer
credited the expensive, less-referenced channel with the whole question.
"""
import json
import tempfile
import unittest
from pathlib import Path

from fornixdb.usefulness_scan import (BLOCK_MARKER, PULL_CHANNEL, attribute,
                                      iter_events, scan)


def _push(ids, event="UserPromptSubmit"):
    text = (f"[FornixDB · {BLOCK_MARKER} — surfaced by topic]\n"
            + "\n".join(f"#{i} 2026-08-01 sem  a gist" for i in ids))
    return {"type": "attachment",
            "attachment": {"content": text, "hookEvent": event}}


def _pull(ids, extra=""):
    text = "\n".join(f"#{i}     2026-08-01  sem (proj)  a gist here" for i in ids)
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "content": text + extra}]}}


def _cite(ids):
    return {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "as noted in " + " ".join(f"#{i}" for i in ids)}]}}


def _write(tmp, events):
    p = Path(tmp) / "session.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return str(p)


class TestPullDetection(unittest.TestCase):

    def test_a_reader_result_is_a_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            evs = list(iter_events(_write(tmp, [_pull([12, 34])])))
            self.assertEqual(len(evs), 1)
            kind, ids, chan, chars = evs[0]
            self.assertEqual(kind, "pull")
            self.assertEqual(ids, {12, 34})
            self.assertEqual(chan, PULL_CHANNEL)
            self.assertGreater(chars, 0)

    def test_a_write_result_is_not_a_pull(self):
        # "stored #123" and a supersede line must never count as a read
        with tempfile.TemporaryDirectory() as tmp:
            ev = {"type": "user", "message": {"content": [
                {"type": "tool_result",
                 "content": "stored #123\n#120 superseded by #123 (kept)"}]}}
            self.assertEqual(list(iter_events(_write(tmp, [ev]))), [])

    def test_a_tool_result_carrying_a_push_block_is_skipped(self):
        # a hook can append its injected block to a tool result; those ids are
        # pushes and are already counted from the attachment
        with tempfile.TemporaryDirectory() as tmp:
            evs = list(iter_events(_write(tmp, [
                _pull([7], extra=f"\n[FornixDB · {BLOCK_MARKER}]\n#9 2026-08-01 sem x")])))
            self.assertEqual(evs, [])

    def test_sidechain_pulls_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = _pull([5])
            ev["isSidechain"] = True
            self.assertEqual(list(iter_events(_write(tmp, [ev]))), [])

    def test_tool_result_content_may_be_a_block_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = {"type": "user", "message": {"content": [
                {"type": "tool_result", "content": [
                    {"type": "text", "text": "#42     2026-08-01  sem  a gist"}]}]}}
            evs = list(iter_events(_write(tmp, [ev])))
            self.assertEqual(evs[0][1], {42})


class TestPullAttribution(unittest.TestCase):

    def test_a_citation_after_a_pull_credits_the_pull(self):
        pm, pc = attribute([("pull", {1}, PULL_CHANNEL, 100), ("cite", {1}, None)])
        self.assertEqual(pm[1]["pull_referenced"], 1)
        self.assertEqual(pm[1]["referenced"], 0)
        self.assertEqual(pc[PULL_CHANNEL]["referenced"], 1)

    def test_an_uncited_pull_is_an_impression_only(self):
        pm, pc = attribute([("pull", {1}, PULL_CHANNEL, 100)])
        self.assertEqual(pm[1]["pull_impressions"], 1)
        self.assertEqual(pm[1]["pull_referenced"], 0)
        self.assertEqual(pc[PULL_CHANNEL]["referenced"], 0)

    def test_a_pull_after_a_push_takes_the_credit(self):
        # the pull is what put the memory in front of the model at use time
        pm, pc = attribute([("push", {1}, "UserPromptSubmit", 50),
                            ("pull", {1}, PULL_CHANNEL, 100),
                            ("cite", {1}, None)])
        self.assertEqual(pm[1]["referenced"], 0)
        self.assertEqual(pm[1]["pull_referenced"], 1)
        self.assertEqual(pc["L3"]["impressions"], 1)
        self.assertEqual(pc["L3"]["referenced"], 0)

    def test_a_push_after_a_pull_takes_the_credit(self):
        pm, _ = attribute([("pull", {1}, PULL_CHANNEL, 100),
                           ("push", {1}, "UserPromptSubmit", 50),
                           ("cite", {1}, None)])
        self.assertEqual(pm[1]["referenced"], 1)
        self.assertEqual(pm[1]["pull_referenced"], 0)

    def test_push_only_figures_are_unaffected_by_pull_traffic(self):
        # the suppression and floor joins read these; a busy pull channel must
        # not move them
        pm, _ = attribute([("push", {1}, "UserPromptSubmit", 50),
                           ("cite", {1}, None),
                           ("pull", {2}, PULL_CHANNEL, 100),
                           ("cite", {2}, None)])
        self.assertEqual(pm[1]["impressions"], 1)
        self.assertEqual(pm[1]["referenced"], 1)
        self.assertEqual(pm[2]["impressions"], 0)
        self.assertEqual(pm[2]["referenced"], 0)

    def test_one_citation_settles_one_delivery(self):
        pm, _ = attribute([("pull", {1}, PULL_CHANNEL, 10),
                           ("cite", {1}, None), ("cite", {1}, None)])
        self.assertEqual(pm[1]["pull_referenced"], 1)


class TestScanReportsBothChannels(unittest.TestCase):

    def test_totals_and_cost_are_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, [_push([1]), _cite([1]), _pull([2]), _cite([2])])
            r = scan(tmp)
            self.assertEqual(r["impressions"], 1)
            self.assertEqual(r["referenced"], 1)
            self.assertEqual(r["pull_impressions"], 1)
            self.assertEqual(r["pull_referenced"], 1)
            self.assertEqual(r["pull_rate"], 1.0)
            self.assertGreater(r["pulled_tokens"], 0)
            self.assertGreater(r["injected_tokens"], 0)

    def test_pull_cost_is_not_folded_into_injected_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, [_pull([2] * 1)])
            r = scan(tmp)
            self.assertEqual(r["injected_tokens"], 0)
            self.assertGreater(r["pulled_tokens"], 0)

    def test_channel_rows_do_not_acquire_per_memory_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, [_push([1]), _pull([2])])
            r = scan(tmp)
            for ch, c in r["by_channel"].items():
                self.assertNotIn("pull_impressions", c, ch)

    def test_tokens_per_reference_is_none_without_a_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, [_pull([2])])
            r = scan(tmp)
            self.assertIsNone(r["by_channel"][PULL_CHANNEL]["tokens_per_reference"])

    def test_tokens_per_reference_is_priced_when_referenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, [_pull([2]), _cite([2])])
            r = scan(tmp)
            self.assertGreater(
                r["by_channel"][PULL_CHANNEL]["tokens_per_reference"], 0)


if __name__ == "__main__":
    unittest.main()


class TestWriteJoinsStayPushOnly(unittest.TestCase):
    """Use-credit and the floor outcome join feed RANKING. Measuring the pull
    channel must not quietly re-tune it — that is a separate decision.
    """

    def _scan(self, tmp):
        _write(tmp, [_push([1]), _cite([1]), _pull([2]), _cite([2])])
        return scan(tmp)

    def test_use_credit_covers_pushed_ids_only(self):
        from fornixdb.usefulness_scan import referenced_counts_from_scan
        with tempfile.TemporaryDirectory() as tmp:
            counts = referenced_counts_from_scan(self._scan(tmp))
            self.assertEqual(counts, {1: 1})   # #2 was pulled, never pushed

    def test_a_pulled_memory_never_has_its_credit_reset(self):
        # the regression this guards: a pull-only id landing in the map would be
        # written a 0, clearing credit an earlier push had earned
        from fornixdb.usefulness_scan import referenced_counts_from_scan
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, [_pull([77]), _cite([77])])
            self.assertNotIn(77, referenced_counts_from_scan(scan(tmp)))

    def test_floor_outcomes_cover_pushed_ids_only(self):
        from fornixdb.usefulness_scan import outcomes_from_scan
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(outcomes_from_scan(self._scan(tmp)), {1: "useful"})
