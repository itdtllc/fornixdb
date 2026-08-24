"""`usage --forget` — dropping a store from the machine registry for good.

The point of these tests is the DURABILITY. Every connect re-registers, so a
forget that only deletes the registry entry is undone the moment anything opens
that file again. That is the same failure shape as a redemption reversed by the
next suppression scan, and it is the case worth guarding."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from fornixdb.cli import main as cli_main

from fornixdb.budget import machine_usage
from fornixdb.core import MemoryStore
from fornixdb.db import (REGISTRY_ENV, _read_path_list, forget_store,
                         registry_ignore_path, registry_path, unforget_store)


class RegistryForgetTests(unittest.TestCase):
    def setUp(self):
        # The registry skips temp-dir stores, so the STORES must live outside
        # tempdir for registration to happen at all; only the registry itself
        # is redirected. A dedicated home keeps the real machine registry out
        # of reach of the test.
        self.home = Path(tempfile.mkdtemp())
        self.reg = self.home / "reg" / "fornix-stores.json"
        self._old = os.environ.get(REGISTRY_ENV)
        os.environ[REGISTRY_ENV] = str(self.reg)
        self.dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        if self._old is None:
            os.environ.pop(REGISTRY_ENV, None)
        else:
            os.environ[REGISTRY_ENV] = self._old
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self.home, ignore_errors=True)

    def _store(self, name):
        path = self.dir / name
        MemoryStore(str(path)).close()
        return str(path.resolve())

    def _registered(self):
        return _read_path_list(registry_path())

    def test_store_registers_on_open(self):
        p = self._store("a.db")
        self.assertIn(p, self._registered())

    def test_forget_removes_from_registry(self):
        p = self._store("a.db")
        self.assertTrue(forget_store(p))
        self.assertNotIn(p, self._registered())
        self.assertIn(p, _read_path_list(registry_ignore_path()))

    def test_forget_survives_reopening_the_store(self):
        """The bug this verb exists for: re-registration on every connect."""
        p = self._store("a.db")
        forget_store(p)
        MemoryStore(p).close()          # an incidental open
        self.assertNotIn(p, self._registered())

    def test_unforget_lets_it_register_again(self):
        p = self._store("a.db")
        forget_store(p)
        self.assertTrue(unforget_store(p))
        MemoryStore(p).close()
        self.assertIn(p, self._registered())

    def test_forget_is_idempotent(self):
        p = self._store("a.db")
        self.assertTrue(forget_store(p))
        self.assertFalse(forget_store(p))

    def test_unforget_of_a_path_never_forgotten_is_a_no_op(self):
        self.assertFalse(unforget_store(str(self.dir / "nope.db")))

    def test_forget_never_touches_the_file(self):
        p = self._store("a.db")
        before = Path(p).stat().st_size
        forget_store(p)
        self.assertTrue(Path(p).exists())
        self.assertEqual(Path(p).stat().st_size, before)

    def test_forgetting_an_absent_path_is_recorded(self):
        """A store deleted before being forgotten must still stay excluded if
        a file of that name reappears — otherwise the entry silently returns."""
        ghost = str((self.dir / "ghost.db").resolve())
        self.assertTrue(forget_store(ghost))
        MemoryStore(ghost).close()
        self.assertNotIn(ghost, self._registered())

    def test_machine_usage_excludes_a_forgotten_store(self):
        keep, drop = self._store("keep.db"), self._store("drop.db")
        both = machine_usage()
        self.assertEqual({s["path"] for s in both["stores"]} & {keep, drop},
                         {keep, drop})
        forget_store(drop)
        after = machine_usage()
        paths = {s["path"] for s in after["stores"]}
        self.assertIn(keep, paths)
        self.assertNotIn(drop, paths)
        self.assertIn(drop, after["forgotten"])
        self.assertLess(after["total_mb"], both["total_mb"])

    def test_usage_excludes_a_path_registered_by_an_older_version(self):
        """Filtering happens at read time too, so a registry written before
        this existed is corrected without needing a rewrite."""
        p = self._store("a.db")
        forget_store(p)
        self.reg.write_text(json.dumps([p], indent=1))   # as an old version left it
        self.assertNotIn(p, {s["path"] for s in machine_usage()["stores"]})

    def test_damaged_ignore_file_does_not_break_opening_a_store(self):
        p = self._store("a.db")
        forget_store(p)
        registry_ignore_path().write_text("{ not json")
        MemoryStore(p).close()          # must not raise
        self.assertIn(p, self._registered())   # unreadable = nothing forgotten

    def test_ignore_file_is_owner_only(self):
        if os.name == "nt":
            self.skipTest("POSIX mode bits")
        forget_store(self._store("a.db"))
        self.assertEqual(registry_ignore_path().stat().st_mode & 0o077, 0)

    def test_registry_off_makes_forget_a_no_op(self):
        os.environ[REGISTRY_ENV] = "off"
        self.assertFalse(forget_store(str(self.dir / "a.db")))
        self.assertIsNone(registry_ignore_path())


class ForgetCliTests(RegistryForgetTests):
    """Through the CLI, not the library. The first cut of this verb passed every
    library test and still raised UnboundLocalError the moment it was run for
    real, because a function-local `from pathlib import Path` further down the
    dispatcher made `Path` local to the whole function. Library-level tests
    cannot see that class of fault; these run the argv path."""

    def _cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(["--db", self._store("cli.db"), *argv])
        return rc, out.getvalue() + err.getvalue()

    def test_usage_forget_runs_and_reports(self):
        p = self._store("drop.db")
        rc, text = self._cli("usage", "--forget", p)
        self.assertEqual(rc, 0)
        self.assertIn("forgotten", text)
        self.assertNotIn(p, self._registered())

    def test_usage_forget_twice_says_nothing_changed(self):
        p = self._store("drop.db")
        self._cli("usage", "--forget", p)
        _, text = self._cli("usage", "--forget", p)
        self.assertIn("nothing changed", text)

    def test_usage_unforget_runs(self):
        p = self._store("drop.db")
        self._cli("usage", "--forget", p)
        rc, text = self._cli("usage", "--unforget", p)
        self.assertEqual(rc, 0)
        self.assertIn("counted again", text)

    def test_usage_still_works_with_no_flags(self):
        self._store("a.db")
        rc, text = self._cli("usage")
        self.assertEqual(rc, 0)
        self.assertIn("TOTAL", text)

    def test_forgotten_line_names_the_path(self):
        p = self._store("drop.db")
        self._cli("usage", "--forget", p)
        _, text = self._cli("usage")
        self.assertIn("1 forgotten", text)
        self.assertIn(p, text)


if __name__ == "__main__":
    unittest.main()
