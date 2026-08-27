"""Regression checks for the Inspire RH56DFTP RS485 protocol."""

from __future__ import annotations

import sys
import types
from pathlib import Path
import unittest
from unittest.mock import patch

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import numpy as np

from output.real.drivers_inspire import (
    InspireSerialOutput,
    _build_read_packet,
    _build_write_packet,
    _parse_response,
)


def _response(command: int, address: int, data: bytes, hand_id: int = 1) -> bytes:
    frame = bytearray([
        0x90,
        0xEB,
        hand_id,
        len(data) + 3,
        command,
        address & 0xFF,
        (address >> 8) & 0xFF,
    ])
    frame.extend(data)
    frame.append(sum(frame[2:]) & 0xFF)
    return bytes(frame)


class _FakeSerialPort:
    def __init__(self) -> None:
        self.rx = bytearray()
        self.writes: list[bytes] = []
        self.is_open = True

    @property
    def in_waiting(self) -> int:
        return len(self.rx)

    def reset_input_buffer(self) -> None:
        self.rx.clear()

    def write(self, packet: bytes) -> int:
        packet = bytes(packet)
        self.writes.append(packet)
        command = packet[4]
        address = packet[5] | (packet[6] << 8)
        if command == 0x11 and address == 1000:
            self.rx.extend(_response(command, address, b"\x01"))
        elif command == 0x12 and address == 1486:
            self.rx.extend(_response(command, address, b"\x01"))
        return len(packet)

    def read(self, size: int) -> bytes:
        if not self.rx:
            return b""
        size = min(size, len(self.rx))
        chunk = bytes(self.rx[:size])
        del self.rx[:size]
        return chunk

    def close(self) -> None:
        self.is_open = False


class InspireProtocolTest(unittest.TestCase):
    def test_official_read_angle_frame(self) -> None:
        packet = _build_read_packet(1, 1546, 12)
        self.assertEqual(packet.hex(" "), "eb 90 01 04 11 0a 06 0c 32")

    def test_official_write_angle_frame(self) -> None:
        data = bytes.fromhex("64 00 64 00 64 00 64 00 d0 07 00 00")
        packet = _build_write_packet(1, 1486, data)
        self.assertEqual(
            packet.hex(" "),
            "eb 90 01 0f 12 ce 05 64 00 64 00 64 00 64 00 d0 07 00 00 5c",
        )

    def test_parse_official_read_response(self) -> None:
        frame = bytes.fromhex(
            "90 eb 01 0f 11 0a 06 "
            "64 00 64 00 64 00 64 00 d0 07 00 00 98"
        )
        data = _parse_response(frame, hand_id=1, command=0x11, address=1546)
        values = [
            int.from_bytes(data[index:index + 2], "little", signed=True)
            for index in range(0, len(data), 2)
        ]
        self.assertEqual(values, [100, 100, 100, 100, 2000, 0])

    def test_reject_bad_checksum(self) -> None:
        frame = bytearray(_response(0x11, 1000, b"\x01"))
        frame[-1] ^= 0xFF
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            _parse_response(bytes(frame), hand_id=1, command=0x11, address=1000)

    def test_constructor_handshake_and_checked_send(self) -> None:
        fake_port = _FakeSerialPort()
        serial_module = types.SimpleNamespace(
            Serial=lambda port_name, baudrate, timeout: fake_port
        )
        with patch.dict(sys.modules, {"serial": serial_module}):
            output = InspireSerialOutput("/dev/fake", hand_id=1)
            output.send(np.zeros(12, dtype=np.float64), [])

        self.assertEqual(fake_port.writes[0], _build_read_packet(1, 1000, 1))
        self.assertEqual(fake_port.writes[1][4:7], bytes.fromhex("12 ce 05"))
        self.assertEqual(fake_port.writes[1][7:19], bytes.fromhex("e8 03") * 6)
        self.assertEqual(output.send_count, 1)


if __name__ == "__main__":
    unittest.main()
