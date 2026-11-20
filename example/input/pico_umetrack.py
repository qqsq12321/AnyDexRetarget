"""PICO stereo-camera hand tracking (UmeTrack) from the sibling ``Hand`` repo.

Unlike ``pico4`` (PICO system hand tracking via the SDK), this input runs
``Hand/tools/start_pico_on_demand_filtered.sh``: the headset streams its
stereo cameras to the PC, UmeTrack estimates both hands, and the result is
position (80 ms) / rotation (60 ms) filtered.  The launcher writes one JSON
record per processed frame to ``<session>/landmarks.jsonl`` and flushes it,
so this module tails that file.

Two operating modes:

  live (default):
    Starts the Hand launcher as a child process with ``PICO_LIVE_OUTPUT``
    pointing at a fresh session directory, then tails its landmarks.jsonl.
    The launcher (and the headset app) is stopped when this process exits.

  replay (``--file path/to/landmarks.jsonl``):
    Replays a recorded session, paced by the capture timestamps.

Each record's ``hands.<left|right>.keypoints_21.landmarks_xyz_mm`` is already
in MediaPipe 21-point order, in the left camera's OpenCV frame (x right,
y down, z forward, right-handed) and millimetres.  It is converted to metres
and made wrist-relative; the Retargeter estimates the wrist frame itself.

Configs live in ``config/<optimizer>/pico_umetrack/`` (initialised from pico4).

Usage:
    python teleop_sim.py --input pico_umetrack --robot linker_l20 --hand right
    python teleop_sim.py --input pico_umetrack --file /path/to/landmarks.jsonl --robot linker_l20
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HAND_ROOT = str(Path(__file__).resolve().parents[3] / "Hand")
LAUNCHER = "tools/start_pico_on_demand_filtered.sh"
DEFAULT_DURATION_S = 600
DEFAULT_PORT = 8765  # launcher default (PICO_LIVE_PORT)
MAX_DURATION_S = 3600  # launcher limit

_POLL_S = 0.005
_STOP_GRACE_S = 20.0  # launcher cleanup stops the headset app and collects reports


def _parse_hand(hand: dict) -> Optional[np.ndarray]:
    if not isinstance(hand, dict):
        return None
    keypoints = hand.get("keypoints_21")
    if not isinstance(keypoints, dict):
        return None
    points = keypoints.get("landmarks_xyz_mm")
    try:
        result = np.asarray(points, dtype=np.float32).reshape(21, 3) / 1000.0
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(result)):
        return None
    result -= result[0].copy()
    return result


def _parse_record(line: str) -> Optional[dict]:
    """Return ``{"t": seconds, "left": ..., "right": ...}`` or None."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    hands = record.get("hands") or {}
    if not isinstance(hands, dict):
        hands = {}
    timestamp_ns = record.get("timestamp_ns")
    return {
        "t": float(timestamp_ns) * 1e-9 if isinstance(timestamp_ns, (int, float)) else None,
        "left": _parse_hand(hands.get("left")),
        "right": _parse_hand(hands.get("right")),
    }


class PicoUmeTrack:
    """PICO stereo-camera UmeTrack hand tracking input device.

    Args:
        hand_root: Path to the ``Hand`` repository (live mode).
        port: Local TCP port the headset streams to via ``adb reverse`` (live mode).
        duration: Capture duration in seconds passed to the launcher (live mode).
        headless: Run the launcher without its stereo preview window (live mode).
        file: Recorded ``landmarks.jsonl`` to replay instead of going live.
        playback_speed: Replay speed multiplier.
        loop: Loop the replay.
    """

    def __init__(
        self,
        hand_root: str = DEFAULT_HAND_ROOT,
        port: int = DEFAULT_PORT,
        duration: int = DEFAULT_DURATION_S,
        headless: bool = False,
        file: Optional[str] = None,
        playback_speed: float = 1.0,
        loop: bool = True,
    ) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._left_hand: Optional[np.ndarray] = None
        self._right_hand: Optional[np.ndarray] = None
        self._last_update: float = 0.0
        self._finished = False
        self._process: Optional[subprocess.Popen] = None

        if file:
            self._records = self._load_recording(Path(file))
            self._playback_speed = playback_speed
            self._loop = loop
            self._thread = threading.Thread(target=self._run_replay, daemon=True)
        else:
            self._jsonl = self._start_launcher(Path(hand_root), port, duration, headless)
            self._thread = threading.Thread(target=self._run_tail, daemon=True)
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────

    def get_fingers_data(self) -> dict:
        empty = np.zeros((21, 3), dtype=np.float32)
        with self._lock:
            left = self._left_hand.copy() if self._left_hand is not None else empty.copy()
            right = self._right_hand.copy() if self._right_hand is not None else empty.copy()
        return {"left_fingers": left, "right_fingers": right}

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._stop_launcher()

    # ── live mode ─────────────────────────────────────────────────────────

    def _start_launcher(self, hand_root: Path, port: int, duration: int, headless: bool) -> Path:
        launcher = hand_root / LAUNCHER
        if not launcher.is_file():
            raise FileNotFoundError(
                f"Hand launcher not found: {launcher} (set --pico-umetrack-root)")
        if not 1 <= duration <= MAX_DURATION_S:
            raise ValueError(f"duration must be 1..{MAX_DURATION_S} seconds, got {duration}")

        session = time.strftime("%Y%m%d_%H%M%S") + "_anydex"
        output = hand_root / "output" / "pico_live" / session
        output.mkdir(parents=True, exist_ok=False)
        log_path = output / "anydex_launcher.log"

        env = {**os.environ, "PICO_LIVE_OUTPUT": str(output), "PICO_LIVE_PORT": str(port)}
        mode = "headless" if headless else "preview"
        # Own session so a plain exit can stop the whole launcher tree; the
        # launcher's EXIT trap stops inference and the headset app.
        self._process = subprocess.Popen(
            ["bash", str(launcher), str(duration), mode, "anydex"],
            cwd=hand_root, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, text=True, bufsize=1,
        )
        atexit.register(self._stop_launcher)
        threading.Thread(target=self._pump_log, args=(log_path,), daemon=True).start()
        print(f"  PICO UmeTrack: session {output}")
        print(f"  PICO UmeTrack: launcher log {log_path}")
        return output / "landmarks.jsonl"

    def _pump_log(self, log_path: Path) -> None:
        """Mirror launcher output to its log; echo it until the first frame."""
        assert self._process is not None and self._process.stdout is not None
        with open(log_path, "w", encoding="utf-8") as log:
            for line in self._process.stdout:
                log.write(line)
                log.flush()
                if self._last_update == 0.0:
                    print(f"  [pico_umetrack] {line.rstrip()}", flush=True)
        code = self._process.wait()
        if not self._stop.is_set():
            print(f"  [pico_umetrack] launcher exited with status {code}; see {log_path}", flush=True)
            # No more frames will arrive; the teleop loops stop on _finished.
            self._finished = True

    def _stop_launcher(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        # SIGINT to the launcher's bash runs its cleanup trap; escalate if stuck.
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=_STOP_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
                process.wait(timeout=3.0)
                return
            except (subprocess.TimeoutExpired, ProcessLookupError):
                continue

    def _run_tail(self) -> None:
        while not self._jsonl.exists():
            if self._stop.is_set() or self._finished:
                return
            self._stop.wait(0.2)
        with open(self._jsonl, "r", encoding="utf-8") as f:
            pending = ""
            while not self._stop.is_set():
                chunk = f.readline()
                if not chunk:
                    if self._finished:
                        return
                    self._stop.wait(_POLL_S)
                    continue
                pending += chunk
                if not pending.endswith("\n"):
                    continue  # partial line; the writer flushes whole records
                parsed = _parse_record(pending)
                pending = ""
                if parsed is not None:
                    self._publish(parsed)

    # ── replay mode ───────────────────────────────────────────────────────

    @staticmethod
    def _load_recording(path: Path) -> list[dict]:
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if not path.is_file():
            raise FileNotFoundError(f"Recording file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            records = [r for r in map(_parse_record, f) if r is not None and r["t"] is not None]
        if not records:
            raise ValueError(f"No timestamped records in {path}")
        print(f"Loaded PICO UmeTrack recording: {path}")
        print(f"  Total frames: {len(records)}  Duration: {records[-1]['t'] - records[0]['t']:.2f}s")
        return records

    def _run_replay(self) -> None:
        records = self._records
        while not self._stop.is_set():
            start = time.monotonic()
            base = records[0]["t"]
            for record in records:
                delay = (record["t"] - base) / self._playback_speed - (time.monotonic() - start)
                if delay > 0 and self._stop.wait(delay):
                    return
                self._publish(record)
            if not self._loop:
                self._finished = True
                return

    def _publish(self, parsed: dict) -> None:
        # A hand missing from a frame is dropped, not held: the launcher does
        # not reuse stale skeletons either, and the teleop loop keeps the last
        # robot target when a hand is absent.
        with self._lock:
            self._left_hand = parsed["left"]
            self._right_hand = parsed["right"]
            self._last_update = time.monotonic()
