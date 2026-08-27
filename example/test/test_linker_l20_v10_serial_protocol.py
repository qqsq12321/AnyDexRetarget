"""Regression checks for the LinkerHand L20 V10 RS485 profile."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from output.real.drivers_l20 import LinkerL20V10SerialOutput


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _read_response(slave_id: int, values: list[int]) -> bytes:
    data = bytearray([slave_id, 0x04, len(values) * 2])
    for value in values:
        data.extend([value >> 8, value & 0xFF])
    return bytes(data) + _crc16(data).to_bytes(2, "little")


def _write_response(packet: bytes) -> bytes:
    data = packet[:6]
    return data + _crc16(data).to_bytes(2, "little")


class _FakeSerialPort:
    def __init__(self, *, respond: bool = True) -> None:
        self.respond = respond
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
        if self.respond and packet[:6] == bytes.fromhex("29 04 00 00 00 1e"):
            self.rx.extend(_read_response(41, [128] * 30))
        elif self.respond and packet[1] == 0x10:
            self.rx.extend(_write_response(packet))
        return len(packet)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        size = min(size, len(self.rx))
        data = bytes(self.rx[:size])
        del self.rx[:size]
        return data

    def close(self) -> None:
        self.is_open = False


class LinkerL20V10SerialProtocolTest(unittest.TestCase):
    def test_qpos_mapping_uses_standard_l20_register_layout(self) -> None:
        joint_names = [
            "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
            "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
            "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
            "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
            "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
            "thumb_mcp", "thumb_ip",
        ]
        output = LinkerL20V10SerialOutput(dry_run=True)

        registers = output._qpos_to_registers(
            np.zeros(len(joint_names), dtype=np.float64),
            joint_names,
        )

        self.assertEqual(len(registers), 30)
        self.assertEqual(registers[15:25], [0] * 10)
        self.assertNotEqual(registers[25:30], [0] * 5)

    def test_left_hand_reverses_mcp_roll_from_previous_mapping(self) -> None:
        joint_names = [
            "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
            "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
            "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
            "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
            "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
            "thumb_mcp", "thumb_ip",
        ]
        right = LinkerL20V10SerialOutput(dry_run=True, hand_side="right")
        left = LinkerL20V10SerialOutput(dry_run=True, hand_side="left")

        for register, joint_name in enumerate(
            (
                "index_mcp_roll",
                "middle_mcp_roll",
                "ring_mcp_roll",
                "pinky_mcp_roll",
            ),
            start=6,
        ):
            with self.subTest(joint_name=joint_name):
                qpos = np.zeros(len(joint_names), dtype=np.float64)
                qpos[joint_names.index(joint_name)] = -0.23
                self.assertEqual(
                    right._qpos_to_registers(qpos, joint_names)[register],
                    0,
                )
                self.assertEqual(
                    left._qpos_to_registers(qpos, joint_names)[register],
                    0,
                )

    def test_first_write_slews_active_registers_from_hardware_position(self) -> None:
        fake_port = _FakeSerialPort()
        serial_module = types.SimpleNamespace(
            EIGHTBITS=8,
            PARITY_NONE="N",
            STOPBITS_ONE=1,
            Serial=lambda **kwargs: fake_port,
        )
        joint_names = [
            "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
            "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
            "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
            "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
            "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
            "thumb_mcp", "thumb_ip",
        ]

        with patch.dict(sys.modules, {"serial": serial_module}):
            output = LinkerL20V10SerialOutput(
                "/dev/fake",
                command_hz=0,
                max_register_step=3,
            )
            output.send(np.zeros(len(joint_names)), joint_names)

        values = [
            int.from_bytes(fake_port.writes[1][index:index + 2], "big")
            for index in range(7, 67, 2)
        ]
        active = values[:15] + values[25:30]
        self.assertTrue(all(abs(value - 128) <= 3 for value in active))
        self.assertEqual(values[15:25], [0] * 10)

    def test_constructor_uses_read_only_position_handshake(self) -> None:
        fake_port = _FakeSerialPort()
        serial_module = types.SimpleNamespace(
            EIGHTBITS=8,
            PARITY_NONE="N",
            STOPBITS_ONE=1,
            Serial=lambda **kwargs: fake_port,
        )

        with patch.dict(sys.modules, {"serial": serial_module}):
            output = LinkerL20V10SerialOutput("/dev/fake")

        self.assertEqual(len(fake_port.writes), 1)
        self.assertEqual(fake_port.writes[0].hex(" "), "29 04 00 00 00 1e 76 2a")
        output.close()

    def test_constructor_closes_port_when_handshake_times_out(self) -> None:
        fake_port = _FakeSerialPort(respond=False)
        serial_module = types.SimpleNamespace(
            EIGHTBITS=8,
            PARITY_NONE="N",
            STOPBITS_ONE=1,
            Serial=lambda **kwargs: fake_port,
        )

        with (
            patch.dict(sys.modules, {"serial": serial_module}),
            self.assertRaisesRegex(RuntimeError, "did not respond"),
        ):
            LinkerL20V10SerialOutput("/dev/fake")

        self.assertFalse(fake_port.is_open)


if __name__ == "__main__":
    unittest.main()
