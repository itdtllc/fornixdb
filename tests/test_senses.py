"""The multimodal APIs are declared intent: every entry point must exist,
say TBD honestly, and refuse to pretend it works."""

import unittest

from fornixdb import senses
from fornixdb.core import MemoryStore
from fornixdb.db import connect


class TestSensesAreHonestStubs(unittest.TestCase):
    def test_every_sense_raises_tbd(self):
        s = MemoryStore(conn=connect(":memory:"))
        for call in (lambda: senses.see(s, "/tmp/x.jpg"),
                     lambda: senses.watch(s, "camera:0"),
                     lambda: senses.hear(s, "mic:0"),
                     lambda: senses.feel(s, {"force": 1.2}, sensor="gripper")):
            with self.assertRaises(NotImplementedError) as ctx:
                call()
            self.assertIn("TBD", str(ctx.exception))
        s.close()


if __name__ == "__main__":
    unittest.main()
