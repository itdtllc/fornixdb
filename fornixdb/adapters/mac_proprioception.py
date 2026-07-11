"""macOS proprioception adapter — a Mac laptop's own power/battery state as
`feel()` readings. A reference host adapter for the feel-loop (`fornixdb.feelloop`).

Generic to any Mac laptop: it encodes the *shape* of `pmset` output, never any
one machine's state. The live values it reports (your charge, whether you are on
mains) exist only in the caller's store — nothing machine-specific is baked in
here, so this file is safe to share as an example other Mac users can fork.

Two layers, mirroring the adapter pattern elsewhere in this package:

  parse_batt(text) -> dict   PURE — parse a captured `pmset -g batt` string.
                             Testable with no subprocess.
  read_battery()   -> dict   thin wrapper that shells out to `pmset` and parses.
  battery_frames(...)        a `(monotonic_ts, reading)` generator for run_feel,
                             coarsened so field-change gating fires on real
                             transitions, not minute-to-minute drift.

`pmset -g batt` prints a source line and a battery line, e.g.

    Now drawing from 'AC Power'
     -InternalBattery-0 (id=...)\t80%; AC attached; not charging present: true

or, unplugged:

    Now drawing from 'Battery Power'
     -InternalBattery-0 (id=...)\t75%; discharging; 3:42 remaining present: true

The transition-worthy fields are `source` (AC vs battery) and `state`
(charging / discharging / charged / not charging); `percent` drifts continuously
and `remaining` is minute-level noise, so `battery_frames` buckets the percent
(default 10% steps) and drops `remaining`. Pass the raw dict straight to
`run_feel` with `ignore_fields={"percent", "remaining"}` instead if you prefer.
"""
from __future__ import annotations

import re
import subprocess
import time
from typing import Callable, Iterator

__all__ = ["parse_batt", "read_battery", "battery_frames",
           "parse_battery_temp", "pick_cpu_temp", "read_temperature"]

_SOURCE_RE = re.compile(r"Now drawing from '([^']+)'")
_PERCENT_RE = re.compile(r"(\d+)%")
_REMAINING_RE = re.compile(r"(\d+):(\d\d)\s+remaining")

# Canonical charge verbs, longest/most-specific first so a substring match
# ("charging" inside "discharging") never wins over the real word.
_CHARGE_WORDS = ("finishing charge", "not charging", "discharging",
                 "charging", "charged")


def parse_batt(text: str) -> dict:
    """Parse `pmset -g batt` output into a reading. Pure — no subprocess.

    Returns keys source ("AC" | "battery" | None), percent (int | None),
    state (a canonical charge word | None), remaining ("H:MM" | None; None for
    a 0:00 / no-estimate reading). Missing fields come back None rather than
    raising, so a truncated or unusual `pmset` layout degrades gracefully.
    """
    reading: dict = {"source": None, "percent": None,
                     "state": None, "remaining": None}

    m = _SOURCE_RE.search(text)
    if m:
        reading["source"] = "AC" if "AC" in m.group(1) else "battery"

    m = _PERCENT_RE.search(text)
    if m:
        reading["percent"] = int(m.group(1))

    low = text.lower()
    for word in _CHARGE_WORDS:
        if word in low:
            reading["state"] = word
            break

    m = _REMAINING_RE.search(text)
    if m and (int(m.group(1)) or int(m.group(2))):   # skip 0:00 (charged/idle)
        reading["remaining"] = f"{int(m.group(1))}:{m.group(2)}"

    return reading


def read_battery(*, timeout: float = 5.0) -> dict:
    """Run `pmset -g batt` and parse it. The thin subprocess wrapper around
    `parse_batt`; raises if `pmset` is missing (non-Mac) or times out."""
    out = subprocess.run(["pmset", "-g", "batt"],
                         capture_output=True, text=True,
                         timeout=timeout).stdout
    return parse_batt(out)


# ---- temperature — the machine's warmth as a feeling -------------------------
#
# Two thermometers, both sudo-free:
#   • CPU die sensors via the IOHID event system (Apple Silicon exposes them
#     as HID services; the same source `macmon`/`smctemp` read). Private API,
#     reached with ctypes — degrades to None anywhere it isn't available.
#   • The battery's own thermistor from `ioreg` (AppleSmartBattery
#     "Temperature", hundredths of °C) — works on any Mac laptop.

_BATT_TEMP_RE = re.compile(r'"Temperature"\s*=\s*(\d+)')


def parse_battery_temp(text: str) -> float | None:
    """Battery temperature in °C from `ioreg -rn AppleSmartBattery` output.
    Pure — no subprocess. None when the key is absent (desktops, truncation)
    or implausible."""
    m = _BATT_TEMP_RE.search(text)
    if not m:
        return None
    c = int(m.group(1)) / 100.0
    return round(c, 1) if 0.0 < c < 100.0 else None


def pick_cpu_temp(sensors: list[tuple[str, float]]) -> float | None:
    """The hottest die temperature from named HID sensor readings. Pure.

    Prefers sensors named like the die ("tdie…"); otherwise falls back to the
    hottest plausible reading. Calibration references ("tcal") report a fixed
    ~52° regardless of load and are never real warmth, so they are excluded."""
    plausible = [(n, t) for n, t in sensors
                 if 0.0 < t < 120.0 and "cal" not in n.lower()]
    die = [t for n, t in plausible if "tdie" in n.lower()]
    pool = die or [t for _, t in plausible]
    return round(max(pool), 1) if pool else None


def _hid_temperatures() -> list[tuple[str, float]]:
    """All temperature sensors the IOHID event system exposes, as
    (product_name, celsius). Empty list when the private API is unavailable
    (Intel Macs, hardened runtimes, future OS changes) — never raises."""
    try:
        import ctypes
        import ctypes.util

        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")

        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFNumberCreate.restype = ctypes.c_void_p
        cf.CFNumberCreate.argtypes = [
            ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
        cf.CFDictionaryCreate.restype = ctypes.c_void_p
        cf.CFDictionaryCreate.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_long] + [ctypes.c_void_p] * 2)
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
        cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]

        iokit.IOHIDEventSystemClientCreate.restype = ctypes.c_void_p
        iokit.IOHIDEventSystemClientCreate.argtypes = [ctypes.c_void_p]
        iokit.IOHIDEventSystemClientSetMatching.restype = None
        iokit.IOHIDEventSystemClientSetMatching.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p]
        iokit.IOHIDEventSystemClientCopyServices.restype = ctypes.c_void_p
        iokit.IOHIDEventSystemClientCopyServices.argtypes = [ctypes.c_void_p]
        iokit.IOHIDServiceClientCopyProperty.restype = ctypes.c_void_p
        iokit.IOHIDServiceClientCopyProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p]
        iokit.IOHIDServiceClientCopyEvent.restype = ctypes.c_void_p
        iokit.IOHIDServiceClientCopyEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        iokit.IOHIDEventGetFloatValue.restype = ctypes.c_double
        iokit.IOHIDEventGetFloatValue.argtypes = [ctypes.c_void_p, ctypes.c_int64]

        UTF8 = 0x08000100

        def cfstr(s: str):
            return cf.CFStringCreateWithCString(None, s.encode(), UTF8)

        def cfnum(n: int):
            v = ctypes.c_int64(n)
            return cf.CFNumberCreate(None, 4, ctypes.byref(v))  # SInt64

        # Temperature sensors live at AppleVendor usage page 0xff00, usage 5.
        keys = (ctypes.c_void_p * 2)(cfstr("PrimaryUsagePage"),
                                     cfstr("PrimaryUsage"))
        vals = (ctypes.c_void_p * 2)(cfnum(0xFF00), cfnum(5))
        match = cf.CFDictionaryCreate(None, keys, vals, 2, None, None)

        client = iokit.IOHIDEventSystemClientCreate(None)
        if not client:
            return []
        iokit.IOHIDEventSystemClientSetMatching(client, match)
        services = iokit.IOHIDEventSystemClientCopyServices(client)
        if not services:
            return []

        KIOHIDEventTypeTemperature = 15
        product = cfstr("Product")
        buf = ctypes.create_string_buffer(256)
        out: list[tuple[str, float]] = []
        for i in range(cf.CFArrayGetCount(services)):
            svc = cf.CFArrayGetValueAtIndex(services, i)
            ev = iokit.IOHIDServiceClientCopyEvent(
                svc, KIOHIDEventTypeTemperature, 0, 0)
            if not ev:
                continue
            celsius = iokit.IOHIDEventGetFloatValue(
                ev, KIOHIDEventTypeTemperature << 16)  # event field base
            name = "?"
            ref = iokit.IOHIDServiceClientCopyProperty(svc, product)
            if ref and cf.CFStringGetCString(ref, buf, 256, UTF8):
                name = buf.value.decode(errors="replace")
            out.append((name, celsius))
        return out
    except Exception:
        return []


def read_temperature(*, timeout: float = 5.0,
                     hid: Callable[[], list[tuple[str, float]]] | None = None,
                     run: Callable[..., "subprocess.CompletedProcess"] | None = None,
                     ) -> dict:
    """The machine's warmth: {"cpu_c": 38.5 | None, "battery_c": 30.7 | None}.
    Either thermometer may be absent (Intel HID, no battery) — its key is then
    None, never an exception. `hid`/`run` are injectable for tests."""
    sensors = (hid or _hid_temperatures)()
    reading: dict = {"cpu_c": pick_cpu_temp(sensors), "battery_c": None}
    try:
        out = (run or subprocess.run)(
            ["ioreg", "-rn", "AppleSmartBattery"],
            capture_output=True, text=True, timeout=timeout).stdout
        reading["battery_c"] = parse_battery_temp(out or "")
    except Exception:
        pass
    return reading


def _coarsen(reading: dict, percent_step: int) -> dict:
    """A gating-friendly view: bucket `percent`, drop `remaining`, so
    field-change gating fires on source/state transitions and ~step-sized
    charge moves rather than every minute."""
    out = {"source": reading.get("source"), "state": reading.get("state")}
    p = reading.get("percent")
    if isinstance(p, int):
        out["percent"] = (p // percent_step) * percent_step if percent_step > 1 else p
    return out


def battery_frames(*, interval_seconds: float = 60.0,
                   percent_step: int = 10,
                   count: int | None = None,
                   reader: Callable[[], dict] = read_battery,
                   clock: Callable[[], float] = time.monotonic,
                   sleep: Callable[[float], None] = time.sleep,
                   ) -> Iterator[tuple[float, dict]]:
    """Yield `(monotonic_ts, reading)` for `run_feel`, sampling every
    `interval_seconds`. Readings are coarsened (`percent` bucketed to
    `percent_step`, `remaining` dropped); pass `percent_step=1` to keep every
    point. Runs forever unless `count` is given. `reader`/`clock`/`sleep` are
    injectable so the loop is testable without a real Mac or real waiting."""
    n = 0
    while True:
        yield clock(), _coarsen(reader(), percent_step)
        n += 1
        if count is not None and n >= count:
            return
        sleep(interval_seconds)
