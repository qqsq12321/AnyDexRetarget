"""Regression checks for the official LinkerHand L20 CAN protocol."""

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

from output.real.drivers_linker_l20_can import (  # noqa: E402
    LinkerL20CanOutput,
    _build_position_payloads,
)


class _FakeMessage:
    def __init__(
        self,
        *,
        arbitration_id: int,
        data,
        is_extended_id: bool = False,
    ) -> None:
        self.arbitration_id = arbitration_id
        self.data = bytearray(data)
        self.is_extended_id = is_extended_id


class _FakeBus:
    def __init__(self, *, respond: bool = True) -> None:
        self.respond = respond
        self.sent: list[_FakeMessage] = []
        self.rx: list[_FakeMessage] = []
        self.closed = False

    def send(self, message: _FakeMessage) -> None:
        self.sent.append(message)
        frame_property = message.data[0]
        if self.respond and frame_property in (0xC0, 0xC1, 0xC2):
            self.rx.append(
                _FakeMessage(
                    arbitration_id=message.arbitration_id,
                    data=[frame_property, 1, 2, 3],
                )
            )

    def recv(self, timeout: float):
        del timeout
        return self.rx.pop(0) if self.rx else None

    def shutdown(self) -> None:
        self.closed = True


def _fake_can_module(bus: _FakeBus):
    return types.SimpleNamespace(
        Message=_FakeMessage,
        interface=types.SimpleNamespace(Bus=lambda **kwargs: bus),
    )


class LinkerL20CanProtocolTest(unittest.TestCase):
    def test_position_vector_splits_into_official_can_frames(self) -> None:
        sdk = list(range(20))

        self.assertEqual(
            _build_position_payloads(sdk),
            [
                bytes([0x13, 10, 11, 12, 13, 14]),
                bytes([0x14, 15, 16, 17, 18, 19]),
                bytes([0x11, 0, 1, 2, 3, 4]),
                bytes([0x12, 5, 6, 7, 8, 9]),
            ],
        )

    def test_handshake_precedes_first_position_frames(self) -> None:
        bus = _FakeBus()
        joint_names = [
            "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
            "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
            "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
            "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
            "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
            "thumb_mcp", "thumb_ip",
        ]

        with patch.dict(sys.modules, {"can": _fake_can_module(bus)}):
            output = LinkerL20CanOutput("can0", hand_side="right", command_hz=0)
            output.send(np.zeros(21, dtype=np.float64), joint_names)
            output.close()

        self.assertEqual([message.data[0] for message in bus.sent[:3]], [0xC0, 0xC1, 0xC2])
        self.assertEqual([message.data[0] for message in bus.sent[3:]], [0x13, 0x14, 0x11, 0x12])
        self.assertTrue(all(message.arbitration_id == 0x28 for message in bus.sent))
        self.assertTrue(bus.closed)

    def test_failed_handshake_closes_bus_without_position_write(self) -> None:
        bus = _FakeBus(respond=False)

        with patch.dict(sys.modules, {"can": _fake_can_module(bus)}):
            with self.assertRaisesRegex(RuntimeError, "No Linker L20 CAN response"):
                LinkerL20CanOutput("can0", hand_side="right")

        self.assertEqual([message.data[0] for message in bus.sent], [0xC0])
        self.assertTrue(bus.closed)


if __name__ == "__main__":
    unittest.main()
