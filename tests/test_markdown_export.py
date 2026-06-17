r"""Regression tests for the export filename sanitizer.

The bug: `_safe_filename` only stripped POSIX path separators (`/ \ #`), so a
memory name containing a Windows-illegal character — most dangerously a colon,
e.g. "site phase 3: RE…" — survived into the path. On NTFS that colon opens an
Alternate Data Stream: the visible file is 0 bytes and the real content is
written to a hidden stream that normal tooling never reads back. These tests
pin the fix so the regression can't return."""

import re
import unittest

from fornixdb.adapters.markdown_export import _safe_filename

# Every character Windows forbids in a filename.
WINDOWS_ILLEGAL = set('<>:"/\\|?*') | {chr(c) for c in range(0x20)}


class TestSafeFilename(unittest.TestCase):
    def test_colon_never_survives(self):
        # the data-loss case: a colon must not reach the path (NTFS ADS).
        out = _safe_filename("site phase 3: RE offerings")
        self.assertNotIn(":", out)

    def test_strips_every_windows_illegal_char(self):
        for ch in WINDOWS_ILLEGAL:
            out = _safe_filename(f"a{ch}b")
            self.assertNotIn(ch, out, f"{ch!r} survived sanitization")

    def test_hash_still_stripped(self):
        # behavior preserved from the original POSIX-only regex.
        self.assertNotIn("#", _safe_filename("topic #1"))

    def test_output_has_no_illegal_chars(self):
        name = 'we<lcome>: "notes"/draft\\v2|final?*#1'
        out = _safe_filename(name)
        self.assertFalse(WINDOWS_ILLEGAL & set(out))
        self.assertNotIn("#", out)

    def test_safe_name_is_left_intact(self):
        self.assertEqual(_safe_filename("retirement-estimator-notes"),
                         "retirement-estimator-notes")

    def test_all_unsafe_falls_back(self):
        # nothing usable left -> a stable default, never an empty filename.
        self.assertEqual(_safe_filename(":::"), "memory")
        self.assertEqual(_safe_filename(""), "memory")


if __name__ == "__main__":
    unittest.main()
