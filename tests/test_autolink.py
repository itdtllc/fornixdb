"""Auto-link at store time (§15.2 item 5): [[name]] wikilinks in a memory's
content become real 'relates' edges when it is stored, and the near-duplicate
note offers linking as well as superseding."""

import tempfile
import unittest
from pathlib import Path

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.adapters.mcp_server import FornixMCP


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


def links_of(store, mem_id):
    return {r["related_id"] for r in store.conn.execute(
        "SELECT related_id FROM memory_link WHERE memory_id = ? AND relation = 'relates'",
        (mem_id,))}


class TestStoreTimeWikilinks(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def test_wikilink_to_existing_memory_becomes_relates_edge(self):
        target = self.s.store("disk cap rules", "the 2GB ceiling", name="disk-cap")
        src = self.s.store("budget note", "see [[disk-cap]] for the ceiling")
        self.assertEqual(links_of(self.s, src), {target})

    def test_unknown_wikilink_is_skipped_silently(self):
        src = self.s.store("a note", "refers to [[does-not-exist-yet]]")
        self.assertEqual(links_of(self.s, src), set())  # intent marker, not an error

    def test_self_reference_is_skipped(self):
        # name resolves to the row being stored -> must not self-link
        mid = self.s.store("recursive", "I am [[me]]", name="me")
        self.assertEqual(links_of(self.s, mid), set())

    def test_link_in_gist_also_resolves(self):
        t = self.s.store("target", "body", name="t")
        src = self.s.store("points at [[t]] in the gist", "plain detail")
        self.assertEqual(links_of(self.s, src), {t})

    def test_multiple_and_deduped(self):
        a = self.s.store("a", "x", name="aa")
        b = self.s.store("b", "y", name="bb")
        src = self.s.store("multi", "[[aa]] and [[bb]] and [[aa]] again")
        self.assertEqual(links_of(self.s, src), {a, b})

    def test_resolves_to_live_successor_after_supersede(self):
        old = self.s.store("v1", "first", name="topic")
        new = self.s.store("v2", "second")
        self.s.supersede(old, new)  # name handle moves to the successor
        src = self.s.store("ref", "cites [[topic]]")
        self.assertEqual(links_of(self.s, src), {new})

    def test_backfill_helper_links_pre_existing_content(self):
        # a memory stored before its target existed: link_wikilinks back-fills it
        src = self.s.store("early", "mentions [[late]]")
        self.assertEqual(links_of(self.s, src), set())
        late = self.s.store("late arrival", "body", name="late")
        self.assertEqual(self.s.link_wikilinks(src, "mentions [[late]]"), [late])
        self.assertEqual(links_of(self.s, src), {late})


class TestRememberSurface(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.srv = FornixMCP(db_path=Path(self.tmp.name) / "m.db", shared=False)

    def tearDown(self):
        self.srv.store.close()
        self.tmp.cleanup()

    def test_remember_reports_autolinked_wikilinks(self):
        self.srv.remember(title="disk-cap", content="the 2GB ceiling rule")
        out = self.srv.remember(title="budget", content="follows [[disk-cap]] closely")
        self.assertIn("linked #", out)

    def test_link_tool_connects_two_memories(self):
        a = self.srv.remember(title="one", content="first memory")
        b = self.srv.remember(title="two", content="second memory")
        aid = int(a.split("#")[1].split()[0])
        bid = int(b.split("#")[1].split()[0])
        out = self.srv.link(str(aid), str(bid))
        self.assertIn("linked", out)
        self.assertEqual(links_of(self.srv.store, aid), {bid})

    def test_link_tool_rejects_self_link(self):
        a = self.srv.remember(title="solo", content="only memory")
        aid = int(a.split("#")[1].split()[0])
        self.assertIn("itself", self.srv.link(str(aid), str(aid)))


class TestRememberMany(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.srv = FornixMCP(db_path=Path(self.tmp.name) / "m.db", shared=False)

    def tearDown(self):
        self.srv.store.close()
        self.tmp.cleanup()

    def _count(self):
        return self.srv.store.conn.execute(
            "SELECT count(*) c FROM memory WHERE superseded_time IS NULL").fetchone()["c"]

    def test_batch_stores_all_items(self):
        out = self.srv.remember_many(items=[
            {"title": "a", "content": "first thing"},
            {"title": "b", "content": "second thing"},
            {"content": "third, untitled"},
        ])
        self.assertEqual(self._count(), 3)
        self.assertEqual(out.count("stored #"), 3)

    def test_empty_is_handled(self):
        self.assertIn("nothing to store", self.srv.remember_many(items=[]))

    def test_item_without_content_is_skipped_not_fatal(self):
        out = self.srv.remember_many(items=[
            {"title": "ok", "content": "has content"},
            {"title": "bad"},  # no content
        ])
        self.assertEqual(self._count(), 1)
        self.assertIn("skipped", out)

    def test_batch_honors_update_and_autolink(self):
        self.srv.remember(title="anchor", content="original anchor text")
        out = self.srv.remember_many(items=[
            {"title": "anchor", "content": "revised anchor text"},   # update -> supersede
            {"title": "ref", "content": "points at [[anchor]]"},     # auto-link
        ])
        self.assertIn("supersedes", out)
        self.assertIn("linked #", out)


if __name__ == "__main__":
    unittest.main()
