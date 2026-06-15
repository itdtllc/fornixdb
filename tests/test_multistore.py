import unittest

from fornixdb.core import MemoryStore
from fornixdb.db import connect
from fornixdb.multistore import (capture_mode, get_config, multi_brief,
                                 multi_recall, multi_timeline, resolve_ref,
                                 set_config)


def two_stores():
    mine = MemoryStore(conn=connect(":memory:"))
    shared = MemoryStore(conn=connect(":memory:"))
    return [("", mine), ("shared", shared)]


class TestMultiStore(unittest.TestCase):
    def setUp(self):
        self.stores = two_stores()
        self.mine = self.stores[0][1]
        self.shared = self.stores[1][1]
        self.a = self.mine.store("my own working note about renders")
        self.b = self.shared.store("owner prefers concise renders summary")

    def test_recall_merges_and_tags(self):
        rows = multi_recall(self.stores, "renders", embedder=False)
        self.assertEqual(len(rows), 2)
        tags = {r["_store"] for r in rows}
        self.assertEqual(tags, {"", "shared"})

    def test_recall_dedupes_same_fact_across_stores(self):
        # the same fact in the agent store AND the shared tier answers once;
        # the kept copy names its twin (nothing silently hidden)
        gist = "Owner prefers concise answers in the morning"
        mine_id = self.mine.store(gist, salience=0.9)
        shared_id = self.shared.store(gist, salience=0.2)
        rows = multi_recall(self.stores, "concise answers morning", embedder=False)
        hits = [r for r in rows if r["gist"] == gist]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], mine_id)  # higher-scored copy survives
        self.assertEqual(hits[0]["also_in"], [f"shared:{shared_id}"])

    def test_recall_keeps_same_store_repeats(self):
        # two identical gists WITHIN one store are distinct events — both stay
        self.mine.store("Ran the consolidation pass")
        self.mine.store("Ran the consolidation pass")
        rows = multi_recall(self.stores, "consolidation pass", embedder=False)
        self.assertEqual(
            len([r for r in rows if "consolidation" in r["gist"]]), 2)

    def test_recall_keeps_different_facts_across_stores(self):
        rows = multi_recall(self.stores, "renders", embedder=False)
        self.assertEqual(len(rows), 2)  # setUp's two render rows differ — both stay

    def test_timeline_merges(self):
        rows = multi_timeline(self.stores, "2000", "2999")
        self.assertEqual(len(rows), 2)

    def test_brief_merges(self):
        b = multi_brief(self.stores)
        self.assertEqual(len(b["salient"]), 2)

    def test_resolve_ref(self):
        store, ref = resolve_ref(self.stores, f"shared:{self.b}")
        self.assertIs(store, self.shared)
        self.assertEqual(ref, str(self.b))
        store, ref = resolve_ref(self.stores, str(self.a))
        self.assertIs(store, self.mine)

    def test_config_roundtrip_and_validation(self):
        self.assertEqual(capture_mode(self.mine), "suggest")  # default
        set_config(self.mine, "capture_mode", "auto")
        self.assertEqual(capture_mode(self.mine), "auto")
        self.assertEqual(get_config(self.mine, "capture_mode"), "auto")
        with self.assertRaises(ValueError):
            set_config(self.mine, "capture_mode", "yolo")


if __name__ == "__main__":
    unittest.main()
