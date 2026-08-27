"""Regression checks for the LinkerHand L20 Modbus RTU protocol."""

from __future__ import annotations

import struct
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from output.real.drivers_linker_l20 import (
    LinkerL20SerialOutput,
    _build_read_request,
    _build_write_multiple_request,
    _linker_l20_retarget_to_sdk,
    _parse_read_response,
    _sdk_to_registers,
)


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _with_crc(data: bytes) -> bytes:
    return data + _crc16(data).to_bytes(2, "little")


def _read_response(slave_id: int, function: int, values: list[int]) -> bytes:
    payload = struct.pack(">" + "H" * len(values), *values)
    return _with_crc(bytes([slave_id, function, len(payload)]) + payload)


def _write_response(slave_id: int, address: int, count: int) -> bytes:
    return _with_crc(struct.pack(">BBHH", slave_id, 0x10, address, count))


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
        function = packet[1]
        address = int.from_bytes(packet[2:4], "big")
        count = int.from_bytes(packet[4:6], "big")
        if function == 0x04 and address == 200 and count == 6:
            self.rx.extend(_read_response(1, 0x04, [1, 0, 2, 0, 1, 0]))
        elif function == 0x04 and address == 0 and count == 30:
            self.rx.extend(_read_response(1, 0x04, [100] * 30))
        elif function == 0x10 and address == 0 and count == 30:
            self.rx.extend(_write_response(1, address, count))
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


class LinkerL20ProtocolTest(unittest.TestCase):
    def test_build_official_version_read_request(self) -> None:
        self.assertEqual(
            _build_read_request(1, 0x04, 200, 6).hex(" "),
            "01 04 00 c8 00 06 f1 f6",
        )

    def test_build_write_multiple_request(self) -> None:
        packet = _build_write_multiple_request(1, 0, [0x12, 0x34])
        self.assertEqual(
            packet.hex(" "),
            "01 10 00 00 00 02 04 00 12 00 34 52 7d",
        )

    def test_parse_read_response_and_reject_bad_crc(self) -> None:
        frame = _read_response(1, 0x04, [1, 0, 2, 0, 1, 0])
        self.assertEqual(
            _parse_read_response(frame, slave_id=1, function=0x04, count=6),
            [1, 0, 2, 0, 1, 0],
        )
        bad_frame = bytearray(frame)
        bad_frame[-1] ^= 0xFF
        with self.assertRaisesRegex(RuntimeError, "CRC"):
            _parse_read_response(
                bytes(bad_frame), slave_id=1, function=0x04, count=6
            )

    def test_sdk_vector_expands_to_modbus_register_order(self) -> None:
        sdk = list(range(20))
        registers = _sdk_to_registers(sdk)
        self.assertEqual(len(registers), 30)
        self.assertEqual(registers[0:5], sdk[10:15])
        self.assertEqual(registers[5:10], sdk[5:10])
        self.assertEqual(registers[10:15], sdk[0:5])
        self.assertEqual(registers[15:25], [0] * 10)
        self.assertEqual(registers[25:30], sdk[15:20])

    def test_retarget_mapping_is_name_based_and_respects_joint_limits(self) -> None:
        joint_names = [
            "index_mcp_roll",
            "index_mcp_pitch",
            "index_pip",
            "index_dip",
            "middle_mcp_roll",
            "middle_mcp_pitch",
            "middle_pip",
            "middle_dip",
            "pinky_mcp_roll",
            "pinky_mcp_pitch",
            "pinky_pip",
            "pinky_dip",
            "ring_mcp_roll",
            "ring_mcp_pitch",
            "ring_pip",
            "ring_dip",
            "thumb_cmc_roll",
            "thumb_cmc_yaw",
            "thumb_cmc_pitch",
            "thumb_mcp",
            "thumb_ip",
        ]
        qpos = np.zeros(len(joint_names), dtype=np.float64)
        qpos[joint_names.index("index_mcp_pitch")] = 1.4
        qpos[joint_names.index("index_mcp_roll")] = 0.26
        qpos[joint_names.index("index_pip")] = 1.08

        sdk = _linker_l20_retarget_to_sdk(qpos, joint_names, "right")

        self.assertEqual(len(sdk), 20)
        self.assertEqual(sdk[1], 0)
        self.assertEqual(sdk[6], 255)
        self.assertEqual(sdk[16], 0)
        self.assertEqual(sdk[11:15], [0, 0, 0, 0])

    def test_retarget_mapping_accepts_generic_piecewise_joint_calibration(self) -> None:
        joint_names = [
            "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
            "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
            "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
            "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
            "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
            "thumb_mcp", "thumb_ip",
        ]
        qpos = np.zeros(len(joint_names), dtype=np.float64)
        qpos[joint_names.index("thumb_cmc_roll")] = 0.73
        mapping = {
            "default": {
                "thumb_cmc_roll": {
                    "input": [0.14, 0.73, 1.44],
                    "output": [255, 128, 0],
                },
            },
        }

        sdk = _linker_l20_retarget_to_sdk(
            qpos,
            joint_names,
            "left",
            joint_command_mapping=mapping,
        )

        self.assertEqual(sdk[5], 128)

    def test_constructor_handshake_precedes_first_position_write(self) -> None:
        fake_port = _FakeSerialPort()
        serial_module = types.SimpleNamespace(
            Serial=lambda *args, **kwargs: fake_port
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
            output = LinkerL20SerialOutput("/dev/fake", command_hz=0)
            output.send(np.zeros(21, dtype=np.float64), joint_names)

        self.assertEqual(fake_port.writes[0], _build_read_request(1, 0x04, 200, 6))
        self.assertEqual(fake_port.writes[1][1], 0x10)
        self.assertEqual(fake_port.writes[1][2:6], bytes.fromhex("00 00 00 1e"))
        self.assertEqual(output.send_count, 1)

    def test_first_position_write_is_slew_limited_from_hardware_position(self) -> None:
        fake_port = _FakeSerialPort()
        serial_module = types.SimpleNamespace(Serial=lambda *args, **kwargs: fake_port)
        joint_names = [
            "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
            "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
            "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
            "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
            "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
            "thumb_mcp", "thumb_ip",
        ]

        with patch.dict(sys.modules, {"serial": serial_module}):
            output = LinkerL20SerialOutput(
                "/dev/fake",
                command_hz=0,
                max_register_step=3,
            )
            output.send(np.zeros(21, dtype=np.float64), joint_names)

        self.assertEqual(fake_port.writes[1], _build_read_request(1, 0x04, 0, 30))
        position_write = fake_port.writes[2]
        values = list(struct.unpack(">" + "H" * 30, position_write[7:-2]))
        self.assertTrue(all(abs(value - 100) <= 3 for value in values))


if __name__ == "__main__":
    unittest.main()
