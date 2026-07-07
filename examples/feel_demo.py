"""feel() end-to-end, in a throwaway store — proprioception as memory.

There is now a first-class command for this:

    fornixdb feel                 # capture the Mac's power state right now
    fornixdb feel "lid closed" --sensor lid
    fornixdb feel --live --seconds 60      # change-gated loop; unplug to see it

This script shows the same machinery a level down, so you can watch each commit
and the recall that follows. It never touches your real store.

    python examples/feel_demo.py            # scripted stream — instant,
                                            # deterministic: first / change /
                                            # heartbeat, then recall
    python examples/feel_demo.py --live     # your REAL battery for ~40s; unplug
                                            # or replug the charger to trigger a
                                            # change commit
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from fornixdb import feelloop
from fornixdb.adapters import mac_proprioception as mp
from fornixdb.core import MemoryStore
from fornixdb.db import connect


def show_commit(ev):
    print(f"  COMMIT  #{ev.memory_id:<4} {ev.reason:<9} {ev.gist}")


def scripted(store):
    """A hand-made reading stream with a fake clock — no hardware, no wait."""
    WALL0 = datetime(2026, 7, 7, 14, 0, 0)
    AC = {"source": "AC", "state": "charged", "percent": 80}
    BATT = {"source": "battery", "state": "discharging", "percent": 80}
    BATT_LOW = {"source": "battery", "state": "discharging", "percent": 70}
    stream = [
        (0.0,  AC),        # first commit (no past yet)
        (3.0,  AC),        # unchanged -> held (no memory)
        (6.0,  BATT),      # unplugged: source+state changed -> CHANGE commit
        (9.0,  BATT),      # unchanged -> held
        (12.0, BATT),      # 6s quiet since last commit -> HEARTBEAT
        (15.0, BATT_LOW),  # charge bucket 80 -> 70 -> CHANGE commit
    ]
    print("scripted stream (fake clock, fake readings):")
    return feelloop.run_feel(
        store, iter(stream), sensor="power", start_wall=WALL0,
        heartbeat_seconds=6, on_commit=show_commit)


def live(store):
    """Your real battery, sampled every 3s for ~40s. Unplug/replug to see a
    change commit; sitting still yields a heartbeat."""
    print("watching your REAL battery for ~40s at 3s intervals —")
    print("  >>> UNPLUG or REPLUG your charger now to trigger a change <<<\n")
    frames = mp.battery_frames(interval_seconds=3.0, percent_step=5)
    return feelloop.run_feel(
        store, frames, sensor="power", heartbeat_seconds=18,
        max_seconds=40, on_commit=show_commit)


def main():
    db = str(Path(tempfile.mkdtemp()) / "demo.db")
    store = MemoryStore(conn=connect(db))

    run = live if "--live" in sys.argv else scripted
    events = run(store)

    print(f"\n{len(events)} memories committed. rows now in the store:")
    for gist, et, src in store.conn.execute(
            "SELECT gist, event_time, source FROM memory "
            "WHERE source = 'senses:feel' ORDER BY id"):
        print(f"  {et}  {src}  {gist}")

    print('\nrecall "when did the laptop go on battery?" ->')
    for m in store.recall("when did the laptop go on battery", limit=3):
        print(f"  #{m['id']}  {m['gist']}")

    store.close()
    print(f"\n(throwaway store was {db} — your real fornix.db was untouched)")


if __name__ == "__main__":
    main()
