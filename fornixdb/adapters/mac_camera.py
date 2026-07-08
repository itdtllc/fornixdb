"""macOS frame-source adapters — turn a camera, the screen, or a video file
into the `(monotonic_ts, frame)` iterator `fornixdb.watchloop.run_watch`
consumes. A reference host adapter for the watch loop.

Three sources, one shape. Each yields `(clock(), frame)` where `frame` is
encoded JPEG bytes; the loop persists a frame to disk only when the salience
gate commits it, so uncommitted frames never leave RAM.

  camera_frames()   the default webcam via OpenCV (`cv2`, an optional [mac]
                    extra). macOS shows a camera-permission prompt the first
                    time — the OS indicator is ground truth that capture is on.
  screen_frames()   the main display via `screencapture` — zero dependency,
                    zero permission-to-import, useful to demo the whole
                    pipeline without a camera.
  file_frames(path) a video file via OpenCV, for tests and verification runs.

`open_stream(source)` maps a CLI source string to the right generator plus a
stable label for the memory gist.

Everything is generic to any Mac — only the *shape* of capture is encoded, no
machine-specific state. `grab`, `clock`, and `sleep` are injectable so the
rate-limiting and dispatch logic is unit-testable with no camera, no screen,
and no real waiting, the way the other adapters take injectable readers.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator

__all__ = ["camera_frames", "screen_frames", "file_frames", "open_stream"]

Frame = bytes
Grab = Callable[[], "Frame | None"]      # returns the next frame, or None at end


def _stream(grab: Grab, *, rate_hz: float, count: int | None,
            clock: Callable[[], float], sleep: Callable[[float], None],
            ) -> Iterator[tuple[float, Frame]]:
    """Pace a `grab()` into `(timestamp, frame)` pairs at ~rate_hz. Stops when
    grab returns None (source ended) or `count` frames have been yielded."""
    interval = 1.0 / rate_hz if rate_hz and rate_hz > 0 else 0.0
    n = 0
    while count is None or n < count:
        frame = grab()
        if frame is None:
            return
        yield clock(), frame
        n += 1
        if count is not None and n >= count:
            return
        if interval:
            sleep(interval)


def _cv2_grabber(source) -> tuple[Grab, Callable[[], None]]:
    """A `(grab, close)` pair over an OpenCV capture (webcam index or file
    path). cv2 is imported lazily — only the real camera/file path needs it."""
    try:
        import cv2                                 # optional [mac] extra
    except ImportError as e:                       # pragma: no cover - env-dep
        raise ImportError(
            "camera/file watch sources need OpenCV — install the optional Mac "
            "extras: pip install 'fornixdb[mac]'  (the 'screen' source needs "
            "no dependency at all)") from e

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"could not open video source {source!r} — a webcam may be in use "
            "by another app, or macOS has not granted camera permission yet")

    def grab() -> Frame | None:
        ok, frame = cap.read()
        if not ok:
            return None
        ok2, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok2 else None

    return grab, cap.release


def _screencapture_grabber() -> tuple[Grab, Callable[[], None] | None]:
    """A `grab` that shells out to `screencapture` for one JPEG of the main
    display. Zero dependency; the temp file is read and removed each frame so
    only committed keyframes ever persist."""
    import os
    import subprocess
    import tempfile

    def grab() -> Frame | None:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            subprocess.run(["screencapture", "-x", "-t", "jpg", path],
                           check=True, capture_output=True, timeout=10)
            with open(path, "rb") as f:
                data = f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return data or None

    return grab, None


def _from_grabber(grab: Grab | None, factory, *, rate_hz: float,
                  count: int | None, clock, sleep) -> Iterator[tuple[float, Frame]]:
    """Shared body: build the default grabber when none is injected, stream it,
    and always release the underlying capture when the generator closes."""
    close = None
    if grab is None:
        grab, close = factory()
    try:
        yield from _stream(grab, rate_hz=rate_hz, count=count,
                           clock=clock, sleep=sleep)
    finally:
        if close is not None:
            close()


def camera_frames(*, device: int = 0, rate_hz: float = 2.0,
                  count: int | None = None, grab: Grab | None = None,
                  clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep,
                  ) -> Iterator[tuple[float, Frame]]:
    """Yield JPEG frames from the webcam at ~rate_hz (default 2). Runs until
    the stream ends or `count` frames; the loop's max_seconds usually stops it.
    Inject `grab` to test without a camera."""
    return _from_grabber(grab, lambda: _cv2_grabber(device),
                         rate_hz=rate_hz, count=count, clock=clock, sleep=sleep)


def screen_frames(*, rate_hz: float = 1.0, count: int | None = None,
                  grab: Grab | None = None,
                  clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep,
                  ) -> Iterator[tuple[float, Frame]]:
    """Yield JPEG frames of the main display at ~rate_hz (default 1) via
    `screencapture` — no extra dependency. Inject `grab` to test without a
    real screen."""
    return _from_grabber(grab, _screencapture_grabber,
                         rate_hz=rate_hz, count=count, clock=clock, sleep=sleep)


def file_frames(path: str, *, rate_hz: float = 2.0, count: int | None = None,
                grab: Grab | None = None,
                clock: Callable[[], float] = time.monotonic,
                sleep: Callable[[float], None] = time.sleep,
                ) -> Iterator[tuple[float, Frame]]:
    """Yield JPEG frames decoded from a video file (OpenCV). For tests and
    owner-run verification of the loop against recorded footage."""
    return _from_grabber(grab, lambda: _cv2_grabber(path),
                         rate_hz=rate_hz, count=count, clock=clock, sleep=sleep)


def open_stream(source: str, *, rate_hz: float | None = None,
                count: int | None = None,
                ) -> tuple[Iterator[tuple[float, Frame]], str]:
    """Resolve a watch source string to `(frames, source_label)`.

    "camera" → the default webcam; "screen" → the main display; anything else
    is treated as a video-file path for playback. The label is what the memory
    gist reads ("watch[camera]: scene change")."""
    if source == "camera":
        return camera_frames(rate_hz=rate_hz or 2.0, count=count), "camera"
    if source == "screen":
        return screen_frames(rate_hz=rate_hz or 1.0, count=count), "screen"
    if not Path(source).expanduser().is_file():
        raise FileNotFoundError(
            f"watch source {source!r} is neither 'camera'/'screen' nor a video "
            "file that exists")
    return file_frames(source, rate_hz=rate_hz or 2.0, count=count), "file"
