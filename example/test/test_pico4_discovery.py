"""Regression checks for Pico 4 discovery across multiple host interfaces."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from unittest import TestCase, mock

INPUT_ROOT = Path(__file__).resolve().parents[1] / "input"
if str(INPUT_ROOT) not in sys.path:
    sys.path.insert(0, str(INPUT_ROOT))

import pico4


class Pico4DiscoveryTest(TestCase):
    def test_local_ips_include_all_active_physical_interfaces(self) -> None:
        interfaces = [
            ("wlp109s0f0", "10.10.10.129", "10.10.10.255"),
            ("enxe65446de354b", "192.168.250.222", "192.168.250.255"),
        ]
        loopback_info = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.1.1", 0)),
        ]

        with (
            mock.patch.object(pico4, "_get_local_ipv4_interfaces", return_value=interfaces),
            mock.patch.object(socket, "getaddrinfo", return_value=loopback_info),
        ):
            self.assertEqual(
                pico4._get_local_ips(),
                ["10.10.10.129", "192.168.250.222"],
            )

    def test_broadcast_targets_use_each_interfaces_real_broadcast(self) -> None:
        interfaces = [
            ("wlp109s0f0", "10.10.10.129", "10.10.10.255"),
            ("enxe65446de354b", "192.168.250.222", "192.168.250.255"),
        ]

        with mock.patch.object(
            pico4,
            "_get_local_ipv4_interfaces",
            return_value=interfaces,
        ):
            self.assertEqual(
                pico4._get_broadcast_targets(),
                [
                    ("10.10.10.129", "10.10.10.255"),
                    ("192.168.250.222", "192.168.250.255"),
                ],
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
