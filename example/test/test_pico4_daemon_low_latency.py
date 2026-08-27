"""Regression checks for low-latency Pico 4 TCP relay sockets."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
import unittest

INPUT_ROOT = Path(__file__).resolve().parents[1] / "input"
if str(INPUT_ROOT) not in sys.path:
    sys.path.insert(0, str(INPUT_ROOT))

from pico4_daemon import _configure_low_latency_tcp


class _FakeSocket:
    def __init__(self) -> None:
        self.options: list[tuple[int, int, int]] = []

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.options.append((level, option, value))


class Pico4DaemonLowLatencyTest(unittest.TestCase):
    def test_disables_nagle_and_requests_quick_ack(self) -> None:
        sock = _FakeSocket()

        _configure_low_latency_tcp(sock)

        self.assertIn(
            (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
            sock.options,
        )
        if hasattr(socket, "TCP_QUICKACK"):
            self.assertIn(
                (socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1),
                sock.options,
            )


if __name__ == "__main__":
    unittest.main()
