"""Per-session meta keys: recognized by doctor, collected by the GC.

One set of these lands per session (which memories it was pushed, where its
cadence stood, which project it pinned) and nothing ever removed them, so on a
lived-in store they outnumbered real configuration roughly twenty to one.
"""
import unittest
from datetime import datetime, timedelta

from fornixdb.consolidate import META_GC_KEEP_DAYS, meta_gc
from fornixdb.core import FrozenStoreError, MemoryStore
from fornixdb.db import SESSION_SCOPED_META_PREFIXES, connect
from fornixdb.multistore import get_config, set_config


def mem_store():
    return MemoryStore(conn=connect(":memory:"))


def _iso(days_ago):
    return (datetime.now() - timedelta(days=days_ago)).replace(
        microsecond=0).isoformat()


class TestMetaGC(unittest.TestCase):

    def _session(self, store, sid, days_ago, ended=True):
        store.record_session(sid, started=_iso(days_ago),
                             ended=_iso(days_ago) if ended else None)

    def _keys_for(self, store, sid):
        for prefix in SESSION_SCOPED_META_PREFIXES:
            set_config(store, f"{prefix}{sid}", "x")

    def test_collects_keys_of_a_long_ended_session(self):
        s = mem_store()
        self._session(s, "old-session", days_ago=META_GC_KEEP_DAYS + 5)
        self._keys_for(s, "old-session")
        res = meta_gc(s, apply=True)
        self.assertEqual(res["deleted"], len(SESSION_SCOPED_META_PREFIXES))
        for prefix in SESSION_SCOPED_META_PREFIXES:
            self.assertIsNone(get_config(s, f"{prefix}old-session"))

    def test_keeps_keys_of_a_recently_ended_session(self):
        s = mem_store()
        self._session(s, "recent", days_ago=1)
        self._keys_for(s, "recent")
        res = meta_gc(s, apply=True)
        self.assertEqual(res["deleted"], 0)
        self.assertEqual(get_config(s, f"proactive_injected_recent"), "x")

    def test_never_touches_a_session_with_no_row(self):
        # session rows are written when a session ENDS, so a key with no row
        # belongs to a session that is STILL RUNNING — collecting it would
        # re-push memories that session has already been shown
        s = mem_store()
        self._keys_for(s, "live-right-now")
        res = meta_gc(s, apply=True)
        self.assertEqual(res["deleted"], 0)
        self.assertEqual(res["live_or_unrecorded"],
                         len(SESSION_SCOPED_META_PREFIXES))
        self.assertEqual(get_config(s, "cadence_turn_live-right-now"), "x")

    def test_dry_run_reports_without_deleting(self):
        s = mem_store()
        self._session(s, "old", days_ago=META_GC_KEEP_DAYS + 1)
        self._keys_for(s, "old")
        res = meta_gc(s, apply=False)
        self.assertEqual(len(res["collectable"]),
                         len(SESSION_SCOPED_META_PREFIXES))
        self.assertEqual(res["deleted"], 0)
        self.assertEqual(get_config(s, "cadence_turn_old"), "x")

    def test_durable_configuration_is_never_collected(self):
        s = mem_store()
        self._session(s, "old", days_ago=META_GC_KEEP_DAYS + 1)
        set_config(s, "vectors", "on")
        set_config(s, "decay_semantic", "0.01")
        meta_gc(s, apply=True)
        self.assertEqual(get_config(s, "vectors"), "on")
        self.assertEqual(get_config(s, "decay_semantic"), "0.01")

    def test_keep_days_is_honored(self):
        s = mem_store()
        self._session(s, "five-days", days_ago=5)
        self._keys_for(s, "five-days")
        self.assertEqual(meta_gc(s, keep_days=10, apply=False)["collectable"], [])
        self.assertTrue(meta_gc(s, keep_days=2, apply=False)["collectable"])

    def test_frozen_store_refuses(self):
        s = mem_store()
        self._session(s, "old", days_ago=META_GC_KEEP_DAYS + 1)
        self._keys_for(s, "old")
        s.conn.execute("INSERT OR REPLACE INTO meta (key, value) "
                       "VALUES ('frozen', '1')")
        s.conn.commit()
        s.__dict__.pop("_frozen_cache", None)
        with self.assertRaises(FrozenStoreError):
            meta_gc(s, apply=True)


if __name__ == "__main__":
    unittest.main()
