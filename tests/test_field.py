"""L5 parallel multi-domain activation — the field and its settling."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"  # deterministic: keyword recall, no model

from fornixdb.core import MemoryStore
from fornixdb.field import (DEFAULT_MAX_CHARS, field_block, field_recall,
                            format_field_debug, run_field, settle)
from fornixdb.multistore import set_config
from fornixdb.proactive import HEADER, row_line


def iso(days_ago: int) -> str:
    return (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")


class FieldBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = MemoryStore(db_path=Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()


class TestDomainScoping(FieldBase):
    def test_domains_scope_by_kind_and_time(self):
        sem = self.s.store("The octopus logo is the FornixDB mascot",
                           kind="semantic")
        # episodic gists need >=4 content words or the low-information filter
        # (correctly) drops them from every push path, the field included
        recent = self.s.store("Rendered the octopus logo mascot artwork",
                              kind="episodic", event_time=iso(2))
        deep = self.s.store("First octopus logo render attempt months back",
                            kind="episodic", event_time=iso(90))
        fee = self.s.store("Never put a hair streak on the octopus logo",
                           kind="feedback")
        fr = run_field(self.s, "octopus logo render")
        self.assertEqual([r["id"] for r in fr.by_domain.get("knowledge", [])], [sem])
        self.assertEqual([r["id"] for r in fr.by_domain.get("recent", [])], [recent])
        self.assertEqual([r["id"] for r in fr.by_domain.get("deep", [])], [deep])
        self.assertEqual([r["id"] for r in fr.by_domain.get("guidance", [])], [fee])

    def test_context_domain_needs_active_project(self):
        self.s.store("Widget budget spreadsheet lives in Documents",
                     kind="semantic", project="widgets")
        fr = run_field(self.s, "widget budget spreadsheet")
        self.assertNotIn("context", fr.by_domain)
        fr = run_field(self.s, "widget budget spreadsheet",
                       active_project="widgets")
        self.assertEqual(len(fr.by_domain["context"]), 1)

    def test_parallel_domains_config_trims(self):
        self.s.store("The octopus logo is the FornixDB mascot", kind="semantic")
        self.s.store("Octopus logo styling feedback noted", kind="feedback")
        set_config(self.s, "parallel_domains", "knowledge")
        fr = run_field(self.s, "octopus logo")
        self.assertEqual(set(fr.by_domain), {"knowledge"})

    def test_excluded_ids_stay_out(self):
        sem = self.s.store("The octopus logo is the FornixDB mascot",
                           kind="semantic")
        fr = run_field(self.s, "octopus logo", exclude_ids={sem})
        self.assertNotIn(sem, fr.rows)


class TestSettling(FieldBase):
    def test_cross_domain_cluster_beats_lone_hit(self):
        # a knowledge row and a recent row linked together = a corroborated
        # pattern; an unlinked loner matching the same words must not outrank it
        k = self.s.store("Seam freeze fix: soft seam seed kills Wan motion",
                         kind="semantic")
        r = self.s.store("Chased the seam freeze in shot three all afternoon",
                         kind="episodic", event_time=iso(1))
        self.s.store("Seam allowance on the jacket pattern", kind="semantic")
        self.s.link(r, k)
        fr = run_field(self.s, "seam freeze fix")
        st = settle(self.s, fr)
        self.assertTrue(st.settled)
        self.assertEqual({m["id"] for m in st.rows} & {k, r}, {k, r})
        self.assertIsNotNone(st.direction)
        self.assertTrue(st.direction.startswith("settled: "))

    def test_shared_topic_clusters_across_domains(self):
        k = self.s.store("Mortar spec: Ardex X 3 Plus is the wrong product",
                         kind="semantic", topics=["pool"])
        r = self.s.store("Stop-work called on the mortar issue",
                         kind="episodic", event_time=iso(3), topics=["pool"])
        fr = run_field(self.s, "mortar issue")
        st = settle(self.s, fr)
        self.assertTrue(st.settled)
        self.assertEqual({m["id"] for m in st.rows}, {k, r})

    def test_disjoint_singletons_degrade_to_l4(self):
        self.s.store("Render pipeline notes mention the word budget",
                     kind="semantic")
        self.s.store("Budget meeting ran long", kind="episodic",
                      event_time=iso(1))
        fr = run_field(self.s, "budget")
        st = settle(self.s, fr)
        self.assertFalse(st.settled)
        self.assertIsNone(st.direction)
        self.assertTrue(st.rows)  # best single hits still emitted

    def test_empty_field_abstains(self):
        self.s.store("The octopus logo is the FornixDB mascot", kind="semantic")
        block, ids = field_recall(self.s, "quarterly tax escrow schedule")
        self.assertIsNone(block)
        self.assertEqual(ids, [])


class TestNeighborhood(FieldBase):
    def test_lit_neighbor_joins_the_pattern(self):
        # the neighbor shares no words with the thought — only a link to a row
        # a query domain vouched for. It must ride in via the cluster.
        k = self.s.store("Seam freeze fix: soft seam seed kills Wan motion",
                         kind="semantic")
        n = self.s.store("Chain-extension keeps the character consistent",
                         kind="semantic")
        self.s.link(k, n)
        fr = run_field(self.s, "seam freeze fix")
        self.assertIn("neighborhood", fr.by_domain)
        st = settle(self.s, fr)
        self.assertTrue(st.settled)
        self.assertIn(n, [m["id"] for m in st.rows])

    def test_pure_neighborhood_cannot_win(self):
        # nothing matches the thought; a lit episode id spreads to a neighbor —
        # activation alone must not produce a block
        a = self.s.store("Chain-extension keeps the character consistent",
                         kind="semantic")
        b = self.s.store("VRT lipsync pass runs after the chain render",
                         kind="semantic")
        self.s.link(a, b)
        block, ids = field_recall(self.s, "quarterly tax escrow schedule",
                                  episode_ids={a})
        self.assertIsNone(block)
        self.assertEqual(ids, [])


class TestBlock(FieldBase):
    def _settled(self):
        k = self.s.store("Seam freeze fix: soft seam seed kills Wan motion",
                         kind="semantic")
        r = self.s.store("Chased the seam freeze in shot three",
                         kind="episodic", event_time=iso(1))
        self.s.link(r, k)
        return k, r

    def test_block_has_header_direction_and_gists(self):
        self._settled()
        block, ids = field_recall(self.s, "seam freeze fix")
        self.assertIsNotNone(block)
        lines = block.splitlines()
        self.assertEqual(lines[0], HEADER)
        self.assertTrue(lines[1].startswith("settled: "))
        self.assertEqual(len(ids), len(lines) - 2)

    def test_budget_trims_whole_lines(self):
        self._settled()
        fr = run_field(self.s, "seam freeze fix")
        st = settle(self.s, fr)
        full = field_block(st, DEFAULT_MAX_CHARS)
        tight = field_block(st, len(full) - 5)
        self.assertLess(len(tight), len(full))
        self.assertEqual(tight.splitlines()[0], HEADER)
        # trimmed down to a header+direction husk with no memory line = no block
        self.assertIsNone(field_block(st, len(HEADER) + len(st.direction) + 2))

    def test_dissent_line_is_config_gated(self):
        self._settled()
        # an on-topic loner outside the winning cluster = the minority report
        # (it must match every thought token — AND-mode recall admits no less)
        self.s.store("Seam freeze fix attempted on the jacket pattern instead",
                     kind="semantic")
        block, _ = field_recall(self.s, "seam freeze fix")
        self.assertNotIn("tension:", block)          # default off
        set_config(self.s, "parallel_dissent", "on")
        block, ids = field_recall(self.s, "seam freeze fix")
        self.assertIn("tension:", block)
        self.assertEqual(len(ids), len(block.splitlines()) - 2)

    def test_debug_view_renders(self):
        self._settled()
        out = format_field_debug(self.s, "seam freeze fix")
        self.assertIn("knowledge", out)
        self.assertIn("settled: True", out)
        self.assertIn("block:", out)

    def test_floor_log_marks_emitted_rows_surfaced_on_l5(self):
        import json
        k, r = self._settled()
        set_config(self.s, "floor_log", "on")
        _, ids = field_recall(self.s, "seam freeze fix")
        log = Path(self.tmp.name) / "floor_log.jsonl"
        self.assertTrue(log.exists())
        recs = [json.loads(l) for l in log.read_text().splitlines()]
        self.assertTrue(all(rec["channel"] == "L5" for rec in recs))
        surfaced = {rec["id"] for rec in recs if rec["decision"] == "surfaced"}
        self.assertEqual(surfaced, set(ids))


class TestBeatLog(FieldBase):
    def _beats(self):
        import json
        log = Path(self.tmp.name) / "field_log.jsonl"
        if not log.exists():
            return []
        return [json.loads(l) for l in log.read_text().splitlines()]

    def _seed_pattern(self):
        k = self.s.store("Seam freeze fix: soft seam seed kills Wan motion",
                         kind="semantic")
        r = self.s.store("Chased the seam freeze in shot three all afternoon",
                         kind="episodic", event_time=iso(1))
        self.s.link(r, k)
        return k, r

    def test_settled_beat_records_winner_glue_and_shadow(self):
        k, r = self._seed_pattern()
        # a loner that matches every token = the dissent shadow candidate
        self.s.store("Seam freeze fix attempted on the jacket pattern instead",
                     kind="semantic")
        set_config(self.s, "floor_log", "on")   # dissent stays OFF
        _, ids = field_recall(self.s, "seam freeze fix")
        beats = self._beats()
        self.assertEqual(len(beats), 1)
        b = beats[0]
        self.assertTrue(b["settled"])
        self.assertEqual(set(b["winner"]), {k, r})
        self.assertEqual(b["emitted"], ids)
        self.assertGreaterEqual(b["link_glue"], 1)
        self.assertIsNotNone(b["dissent_shadow"])   # shadow logged...
        self.assertFalse(b["dissent_emitted"])      # ...while the line is off
        self.assertIn("ms", b)

    def test_dissent_emitted_tracks_the_block_not_the_computation(self):
        # The whole point of the corrected flag: dissent_emitted is TRUE only
        # when the tension line actually reached the injected block, so a
        # downstream reference measurement counts host exposures, not shadows.
        k, r = self._seed_pattern()
        did = self.s.store(
            "Seam freeze fix attempted on the jacket pattern instead",
            kind="semantic")
        set_config(self.s, "floor_log", "on")
        set_config(self.s, "parallel_dissent", "on")
        _, ids = field_recall(self.s, "seam freeze fix")
        b = self._beats()[0]
        self.assertIn(did, b["emitted"])            # the tension id reached the block
        self.assertTrue(b["dissent_emitted"])       # ...so it counts as emitted
        self.assertEqual(b["dissent_shadow"], did)

    def test_tension_line_is_reserved_from_the_budget_trim(self):
        # the minority report must survive a budget that exactly fits the winning
        # cluster — under the old append-last/pop-first order it was trimmed away.
        self._seed_pattern()
        self.s.store("Seam freeze fix attempted on the jacket pattern instead",
                     kind="semantic")
        set_config(self.s, "parallel_dissent", "on")
        fr = run_field(self.s, "seam freeze fix")
        st = settle(self.s, fr)
        self.assertIsNotNone(st.dissent)             # dissent on + shadow exists
        winners = "\n".join([HEADER, st.direction]
                            + [row_line(m) for m in st.rows])
        block = field_block(st, len(winners))        # no room left for tension
        self.assertIn("tension:", block)             # ...yet it survives (reserved)
        self.assertGreater(len(block), len(winners))  # riding one line on top

    def test_degraded_and_abstained_beats_are_logged(self):
        self.s.store("Render pipeline notes mention the word budget",
                     kind="semantic")
        set_config(self.s, "floor_log", "on")
        field_recall(self.s, "budget for the render pipeline")   # degraded
        field_recall(self.s, "quarterly tax escrow schedule")    # abstains
        beats = self._beats()
        self.assertEqual(len(beats), 2)
        self.assertFalse(beats[0]["settled"])
        self.assertTrue(beats[0]["emitted"])
        self.assertEqual(beats[1]["emitted"], [])

    def test_no_beat_log_when_switch_off(self):
        self._seed_pattern()
        field_recall(self.s, "seam freeze fix")
        self.assertEqual(self._beats(), [])

    def test_field_stats_report(self):
        from fornixdb.field_stats import format_report, load_beats, summarize
        self._seed_pattern()
        set_config(self.s, "floor_log", "on")
        field_recall(self.s, "seam freeze fix")
        path = Path(self.tmp.name) / "field_log.jsonl"
        s = summarize(load_beats(path))
        self.assertEqual((s["beats"], s["settled"]), (1, 1))
        out = format_report(s, str(path))
        self.assertIn("settled=1", out)
        self.assertIn("winner glue", out)

    def test_configured_domains_survives_hand_edited_meta(self):
        # set_config validates writes, but a hand-edited meta table (or a
        # legacy literal "off") can still reach the read side — a beat must
        # degrade to the full default set, never die or silently mis-scope
        from fornixdb.field import DOMAINS, configured_domains

        def raw_set(value):
            self.s.conn.execute(
                "INSERT INTO meta(key, value) VALUES ('parallel_domains', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (value,))
            self.s.conn.commit()

        for stale in ("off", "ALL", "none", "knowlege,typo"):
            raw_set(stale)
            self.assertEqual(configured_domains(self.s), list(DOMAINS), stale)
        raw_set("knowledge")
        self.assertEqual([d.id for d in configured_domains(self.s)],
                         ["knowledge"])

    def test_settle_survives_hand_edited_parallel_limit(self):
        from fornixdb.field import DEFAULT_LIMIT
        self._seed_pattern()
        for bad in ("banana", "0", "-3"):
            self.s.conn.execute(
                "INSERT INTO meta(key, value) VALUES ('parallel_limit', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (bad,))
            self.s.conn.commit()
            fr = run_field(self.s, "seam freeze fix")
            st = settle(self.s, fr)          # must not raise on the hot path
            self.assertLessEqual(len(st.rows), DEFAULT_LIMIT)

    def test_field_stats_buckets_are_disjoint(self):
        # a settled beat whose gists were all session-deduped emits nothing —
        # it is settled_quiet, not an abstention; the old subtraction produced
        # a NEGATIVE degraded count on the live log (settled ∩ not-emitted)
        from fornixdb.field_stats import summarize
        beats = [
            {"settled": True, "emitted": [1, 2]},    # settled_emitted
            {"settled": True, "emitted": []},        # settled_quiet
            {"settled": False, "emitted": [3]},      # degraded (L4 fallback push)
            {"settled": False, "emitted": []},       # abstained
        ]
        s = summarize(beats)
        self.assertEqual((s["settled"], s["settled_emitted"], s["settled_quiet"],
                          s["degraded"], s["abstained"]), (2, 1, 1, 1, 1))
        self.assertEqual(s["settled_emitted"] + s["settled_quiet"]
                         + s["degraded"] + s["abstained"], s["beats"])
        self.assertGreaterEqual(s["degraded"], 0)


class TestCadenceSeam(FieldBase):
    """L5 rides the L4 metronome: the dial changes the gather, not the beat."""

    def _seed_pattern(self):
        k = self.s.store("Seam freeze fix: soft seam seed kills Wan motion",
                         kind="semantic")
        r = self.s.store("Chased the seam freeze in shot three all afternoon",
                         kind="episodic", event_time=iso(1))
        self.s.link(r, k)
        return k, r

    def test_pulse_routes_through_field_when_dial_on(self):
        from fornixdb.cadence import Episode, pulse
        k, r = self._seed_pattern()
        set_config(self.s, "parallel_recall", "on")
        ep = Episode()
        block = pulse(self.s, "chasing the seam freeze fix in the render", ep)
        self.assertIsNotNone(block)
        self.assertIn("settled: ", block)              # the field, not a plain pulse
        self.assertEqual(ep.pulse_count, 1)
        self.assertEqual(ep.pulsed_ids & {k, r}, {k, r})
        # the metronome still owns the beat: an unmoved thought debounces
        self.assertIsNone(
            pulse(self.s, "chasing the seam freeze fix in the render", ep))

    def test_pulse_unchanged_when_dial_off(self):
        from fornixdb.cadence import Episode, pulse
        set_config(self.s, "parallel_recall", "off")  # override the 0.5.0 default
        self._seed_pattern()
        block = pulse(self.s, "chasing the seam freeze fix in the render",
                      Episode())
        self.assertIsNotNone(block)
        self.assertNotIn("settled: ", block)           # plain L4 block


if __name__ == "__main__":
    unittest.main()
