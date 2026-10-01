"""The brief's "recent sessions" list: bounded to the past, and shared fairly
across projects so one auto-capturing consumer cannot fill it alone."""

import unittest
from datetime import datetime, timedelta

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.multistore import multi_brief


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


def ago(hours):
    return (datetime.now() - timedelta(hours=hours)).isoformat()


class TestBriefRecent(unittest.TestCase):
    def setUp(self):
        self.s = mem_store()

    def tearDown(self):
        self.s.close()

    def _epi(self, gist, project, hours):
        return self.s.store(gist, kind="episodic", project=project,
                            event_time=ago(hours))

    def test_future_rows_excluded(self):
        past = self._epi("yesterday's session", "alpha", 24)
        self._epi("reminder scheduled next week", "alpha", -24 * 7)
        ids = [r["id"] for r in self.s.brief()["recent"]]
        self.assertEqual(ids, [past])

    def test_chatty_project_capped_and_counted(self):
        for i in range(20):
            self._epi(f"scan chatter {i}", "busyfeed", i * 0.1)
        quiet = self._epi("real work session", "quietproj", 30)
        b = self.s.brief()
        projects = [r["project"] for r in b["recent"]]
        self.assertEqual(projects.count("busyfeed"), 2)
        self.assertIn(quiet, [r["id"] for r in b["recent"]])
        self.assertEqual(b["recent_folded"], {"busyfeed": 18})
        self.assertNotIn("_rn", b["recent"][0])

    def test_newest_first_after_cap(self):
        older = self._epi("older work", "quietproj", 10)
        newer = self._epi("newer chatter", "busyfeed", 1)
        ids = [r["id"] for r in self.s.brief()["recent"]]
        self.assertEqual(ids, [newer, older])

    def test_project_filter_is_uncapped(self):
        for i in range(5):
            self._epi(f"scan chatter {i}", "busyfeed", i)
        b = self.s.brief(project="busyfeed")
        self.assertEqual(len(b["recent"]), 5)
        self.assertEqual(b["recent_folded"], {})

    def test_cap_zero_disables(self):
        for i in range(5):
            self._epi(f"scan chatter {i}", "busyfeed", i)
        b = self.s.brief(recent_per_project=0)
        self.assertEqual(len(b["recent"]), 5)
        self.assertEqual(b["recent_folded"], {})

    def test_unlabelled_rows_share_one_bucket(self):
        for i in range(4):
            self._epi(f"untagged {i}", None, i)
        b = self.s.brief()
        self.assertEqual(len(b["recent"]), 2)
        self.assertEqual(b["recent_folded"], {"-": 2})

    def test_multi_brief_merges_folded_counts(self):
        for i in range(4):
            self._epi(f"scan chatter {i}", "busyfeed", i)
        peer = mem_store()
        try:
            for i in range(3):
                peer.store(f"peer chatter {i}", kind="episodic",
                           project="busyfeed", event_time=ago(i))
            b = multi_brief([(None, self.s), ("shared", peer)])
            self.assertEqual(b["recent_folded"], {"busyfeed": 3})
        finally:
            peer.close()


if __name__ == "__main__":
    unittest.main()
