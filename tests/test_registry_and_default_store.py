"""Registry filename branding + no-side-effect store creation (decision
2026-06-21). The machine registry default must be FornixDB-branded
('fornix-stores.json', not 'stores.json'), a legacy registry auto-migrates,
and a command must never materialize a store merely by falling back to the
default path — only `init` (or --create) creates the default store, so a
read-only command can't litter ~/.fornixdb."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from fornixdb.cli import main
from fornixdb.db import _migrate_legacy_registry, registry_path


def _run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class _IsolatedHome:
    """Point HOME at a temp dir and clear the FornixDB env overrides so the
    default-path logic resolves under the temp dir, not the real home."""

    _ENV = ("HOME", "FORNIXDB_DB", "FORNIXDB_REGISTRY", "FORNIXDB_SHARED_DB")

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = {k: os.environ.get(k) for k in self._ENV}
        os.environ["HOME"] = self._tmp.name
        for k in self._ENV[1:]:
            os.environ.pop(k, None)
        return Path(self._tmp.name)

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()


class TestRegistryBranding(unittest.TestCase):
    def test_default_registry_is_fornix_branded(self):
        with _IsolatedHome():
            reg = registry_path()
            self.assertIsNotNone(reg)
            self.assertEqual(reg.name, "fornix-stores.json")

    def test_off_disables_registry(self):
        saved = os.environ.get("FORNIXDB_REGISTRY")
        os.environ["FORNIXDB_REGISTRY"] = "off"
        try:
            self.assertIsNone(registry_path())
        finally:
            if saved is None:
                os.environ.pop("FORNIXDB_REGISTRY", None)
            else:
                os.environ["FORNIXDB_REGISTRY"] = saved

    def test_legacy_registry_is_migrated(self):
        with tempfile.TemporaryDirectory() as d:
            legacy = Path(d) / "stores.json"
            legacy.write_text('["/some/store.db"]')
            reg = Path(d) / "fornix-stores.json"
            _migrate_legacy_registry(reg)
            self.assertTrue(reg.exists())
            self.assertFalse(legacy.exists())
            self.assertEqual(reg.read_text(), '["/some/store.db"]')

    def test_migration_never_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as d:
            legacy = Path(d) / "stores.json"
            legacy.write_text('["/legacy.db"]')
            reg = Path(d) / "fornix-stores.json"
            reg.write_text('["/current.db"]')
            _migrate_legacy_registry(reg)
            self.assertEqual(reg.read_text(), '["/current.db"]')  # untouched
            self.assertTrue(legacy.exists())                      # left in place


class TestNoSideEffectStoreCreation(unittest.TestCase):
    def test_readonly_command_does_not_create_default_store(self):
        with _IsolatedHome() as home:
            rc, _out, err = _run_cli("recall", "anything")
            self.assertEqual(rc, 0)
            self.assertIn("no FornixDB store", err)
            self.assertFalse((home / ".fornixdb" / "fornix.db").exists())
            self.assertFalse((home / ".fornixdb" / "fornix-stores.json").exists())

    def test_doctor_does_not_create_default_store(self):
        with _IsolatedHome() as home:
            rc, _out, _err = _run_cli("doctor")
            self.assertEqual(rc, 0)
            self.assertFalse((home / ".fornixdb" / "fornix.db").exists())
            self.assertFalse((home / ".fornixdb" / "stores.json").exists())

    def test_init_creates_default_store(self):
        with _IsolatedHome() as home:
            rc, _out, _err = _run_cli("init")
            self.assertEqual(rc, 0)
            self.assertTrue((home / ".fornixdb" / "fornix.db").exists())

    def test_create_flag_allows_creation(self):
        with _IsolatedHome() as home:
            rc, _out, _err = _run_cli("--create", "recall", "anything")
            self.assertEqual(rc, 0)
            self.assertTrue((home / ".fornixdb" / "fornix.db").exists())

    def test_explicit_db_path_still_creates(self):
        with _IsolatedHome(), tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "explicit.db")
            rc, _out, _err = _run_cli("--db", db, "--no-shared", "init")
            self.assertEqual(rc, 0)
            self.assertTrue(Path(db).exists())


if __name__ == "__main__":
    unittest.main()
