"""Markdown↔store staleness correlation (the dream-time cross-check).

Covers the forward-marker block parser, the scan's gates (mtime, closure
language, cosine, distinct-pair accept, missing file row), the dream/propose
wiring (native_dir + config gate), and the persisted-flags → brief-line loop
with its cheap revalidation.
"""
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

os.environ["FORNIXDB_VECTORS"] = "off"   # embeddings are inserted by hand

from fornixdb import markdown_stale
from fornixdb.consolidate import dream, propose
from fornixdb.core import MemoryStore
from fornixdb.multistore import get_config, set_config
from fornixdb.vectors import to_blob

MODEL = "test-model"
OLD = (datetime.now() - timedelta(days=9)).replace(microsecond=0)
NEWER = (datetime.now() - timedelta(days=2)).replace(microsecond=0)


def _store(tmp):
    return MemoryStore(db_path=Path(tmp) / "t.db")


def _embed(store, mem_id, vec):
    store.conn.execute(
        "INSERT OR REPLACE INTO embedding(memory_id, chunk, model, dim, vector) "
        "VALUES (?, 0, ?, ?, ?)", (mem_id, MODEL, len(vec), to_blob(vec)))
    store.conn.commit()


def _md_file(dirpath, name, body, mtime=None):
    p = Path(dirpath) / f"{name}.md"
    p.write_text(f"---\nname: {name}\ndescription: {name} topic file\n---\n\n"
                 f"{body}\n", encoding="utf-8")
    if mtime is not None:
        os.utime(p, (mtime.timestamp(), mtime.timestamp()))
    return p


def _seed(store, mddir, *, epi_gist="Session: the demo was filmed and posted",
          epi_time=NEWER, file_mtime=OLD, file_vec=(1.0, 0.0, 0.0),
          epi_vec=(0.95, 0.05, 0.0),
          body="## PICKUP\nFilm the demo, then post it to X."):
    """One markdown file with a PICKUP block + one later closure session row,
    vectors aligned — the guitar-demo shape. Returns (file_id, epi_id)."""
    path = _md_file(mddir, "project_demo", body, mtime=file_mtime)
    fid = store.store("demo topic file", path.read_text(), kind="semantic",
                      name="project_demo")
    _embed(store, fid, list(file_vec))
    eid = store.store(epi_gist, "transcript gist", kind="episodic",
                      recorded_time=epi_time.isoformat())
    _embed(store, eid, list(epi_vec))
    return fid, eid


class TestForwardBlocks(unittest.TestCase):
    def test_pickup_block_ends_at_blank_line(self):
        blocks = markdown_stale.forward_blocks(
            "intro text\n\n## PICKUP\nfilm the demo\npost to X\n\nunrelated tail")
        self.assertEqual(len(blocks), 1)
        self.assertIn("film the demo", blocks[0])
        self.assertNotIn("unrelated tail", blocks[0])

    def test_block_ends_at_next_heading(self):
        blocks = markdown_stale.forward_blocks(
            "NEXT: owner runs A10\nthen ship\n# History\nold stuff")
        self.assertEqual(len(blocks), 1)
        self.assertNotIn("old stuff", blocks[0])

    def test_marker_vocabulary(self):
        for marker in ("PICKUP", "Next steps", "TODO", "RESUME HERE",
                       "open items", "still open", "next up", "pending"):
            self.assertTrue(markdown_stale.forward_blocks(f"x\n{marker}: y\n"),
                            marker)

    def test_prose_next_without_punctuation_does_not_fire(self):
        self.assertEqual(markdown_stale.forward_blocks(
            "the next release went well\nall shipped\n"), [])

    def test_marker_inside_a_wikilink_slug_does_not_fire(self):
        self.assertEqual(markdown_stale.forward_blocks(
            "- [[project-video-series-pickup]] — current state of the series\n"), [])

    def test_frontmatter_title_does_not_fire(self):
        # `name: Session Pickup — …` is a title, not an open-work claim
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            fid, eid = _seed(s, md, body="history only, all shipped long ago")
            p = Path(md) / "project_demo.md"
            p.write_text("---\nname: project_demo\ndescription: Session Pickup"
                         " — resume here\n---\n\nhistory only\n",
                         encoding="utf-8")
            os.utime(p, (OLD.timestamp(), OLD.timestamp()))
            self.assertEqual(markdown_stale.scan(s, md), [])

    def test_no_markers(self):
        self.assertEqual(markdown_stale.forward_blocks(
            "everything here is history\nnothing forward-looking\n"), [])


class TestScan(unittest.TestCase):
    def test_overtaken_pickup_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            fid, eid = _seed(s, md)
            flags = markdown_stale.scan(s, md)
            self.assertEqual(len(flags), 1)
            f = flags[0]
            self.assertEqual(f["file_id"], fid)
            self.assertEqual(f["overtaken_by"], eid)
            self.assertIn("PICKUP", f["marker"])
            self.assertGreaterEqual(f["cosine"], 0.5)

    def test_extended_closure_vocabulary_fires(self):
        # "filmed and posted" carries none of _CLOSURE_RE's code words
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md, epi_gist="7/14: demo already filmed + posted to X")
            self.assertEqual(len(markdown_stale.scan(s, md)), 1)

    def test_file_edited_after_the_session_never_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md, file_mtime=datetime.now())
            self.assertEqual(markdown_stale.scan(s, md), [])

    def test_no_closure_language_never_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md, epi_gist="Session: discussed the demo plan further")
            self.assertEqual(markdown_stale.scan(s, md), [])

    def test_dissimilar_vectors_never_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md, epi_vec=(0.0, 1.0, 0.0))
            self.assertEqual(markdown_stale.scan(s, md), [])

    def test_distinct_link_accepts_the_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            fid, eid = _seed(s, md)
            s.link(fid, eid, relation="distinct")
            self.assertEqual(markdown_stale.scan(s, md), [])

    def test_superseded_episodic_row_is_never_evidence(self):
        # retired stays retired: a superseded session row must not flag a file
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            fid, eid = _seed(s, md)
            newer = s.store("corrected session note", "x", kind="episodic")
            s.supersede(eid, newer)
            self.assertEqual(markdown_stale.scan(s, md), [])

    def test_superseded_file_row_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            fid, eid = _seed(s, md)
            replacement = s.store("replacement topic row", "x", kind="semantic")
            s.supersede(fid, replacement)
            # the successor has no vector, so nothing qualifies
            self.assertEqual(markdown_stale.scan(s, md), [])

    def test_file_without_imported_row_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md)
            _md_file(md, "project_orphan", "## PICKUP\nnever imported",
                     mtime=OLD)
            flags = markdown_stale.scan(s, md)
            self.assertEqual([f["name"] for f in flags], ["project_demo"])

    def test_index_files_and_marker_free_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md, body="history only, nothing forward-looking")
            (md / "MEMORY.md").write_text("- PICKUP everywhere\n")
            self.assertEqual(markdown_stale.scan(s, md), [])

    def test_missing_directory_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            self.assertEqual(markdown_stale.scan(s, Path(tmp) / "nope"), [])


class TestDreamWiring(unittest.TestCase):
    def test_propose_without_native_dir_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            self.assertEqual(propose(s)["markdown_stale"], [])

    def test_config_gate_off_disables(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md)
            set_config(s, "native_dir", str(md))
            set_config(s, "dream_markdown_stale", "off")
            self.assertEqual(propose(s)["markdown_stale"], [])

    def test_dream_counts_narrative_and_persisted_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md)
            set_config(s, "native_dir", str(md))
            rep = dream(s)
            self.assertEqual(rep["counts"]["markdown_stale"], 1)
            self.assertIn("markdown note", rep["narrative"])
            persisted = json.loads(get_config(s, markdown_stale.FLAGS_KEY))
            self.assertEqual(len(persisted), 1)
            self.assertEqual(persisted[0]["name"], "project_demo")

    def test_clean_scan_clears_persisted_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            path = _md_file(md, "project_demo", "## PICKUP\nfilm it", mtime=OLD)
            set_config(s, "native_dir", str(md))
            set_config(s, markdown_stale.FLAGS_KEY, json.dumps(
                [{"path": str(path), "edited": OLD.isoformat(),
                  "file": "project_demo.md", "file_id": 1, "overtaken_by": 2,
                  "marker": "## PICKUP", "epi_time": "2026-01-01"}]))
            dream(s)  # no imported row -> empty scan -> cleared
            self.assertEqual((get_config(s, markdown_stale.FLAGS_KEY) or ""), "")


class TestBriefLine(unittest.TestCase):
    def test_line_appears_and_clears_on_file_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            _seed(s, md)
            set_config(s, "native_dir", str(md))
            dream(s)
            line = markdown_stale.brief_line(s)
            self.assertIsNotNone(line)
            self.assertIn("project_demo.md", line)
            self.assertIn("markdown may be stale", line)
            # the natural fix: the user rewrites (touches) the file
            time.sleep(1.1)  # mtime is second-granular
            now = time.time()
            os.utime(Path(md) / "project_demo.md", (now, now))
            self.assertIsNone(markdown_stale.brief_line(s))

    def test_line_clears_on_distinct_accept(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            fid, eid = _seed(s, md)
            set_config(s, "native_dir", str(md))
            dream(s)
            self.assertIsNotNone(markdown_stale.brief_line(s))
            s.link(fid, eid, relation="distinct")
            self.assertIsNone(markdown_stale.brief_line(s))

    def test_line_clears_when_evidence_row_is_superseded_between_dreams(self):
        # flags persist between dreams; a row retired AFTER the scan must
        # never be re-cited by the brief line
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            fid, eid = _seed(s, md)
            set_config(s, "native_dir", str(md))
            dream(s)
            self.assertIsNotNone(markdown_stale.brief_line(s))
            newer = s.store("corrected session note", "x", kind="episodic")
            s.supersede(eid, newer)
            self.assertIsNone(markdown_stale.brief_line(s))

    def test_line_clears_when_file_row_is_superseded_between_dreams(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, md = _store(tmp), Path(tmp) / "mem"
            md.mkdir()
            fid, eid = _seed(s, md)
            set_config(s, "native_dir", str(md))
            dream(s)
            self.assertIsNotNone(markdown_stale.brief_line(s))
            replacement = s.store("replacement topic row", "x", kind="semantic")
            s.supersede(fid, replacement)
            self.assertIsNone(markdown_stale.brief_line(s))

    def test_no_flags_no_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(markdown_stale.brief_line(_store(tmp)))

    def test_malformed_flags_never_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            set_config(s, markdown_stale.FLAGS_KEY, "{not json")
            self.assertIsNone(markdown_stale.brief_line(s))
            set_config(s, markdown_stale.FLAGS_KEY, json.dumps([{"no": "keys"}]))
            self.assertIsNone(markdown_stale.brief_line(s))


if __name__ == "__main__":
    unittest.main()
