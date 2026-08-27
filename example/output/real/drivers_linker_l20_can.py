"""Official LinkerHand L20 output driver over SocketCAN."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .base import HandOutput
from .drivers_linker_l20 import _linker_l20_retarget_to_sdk
from .joint_command_mapping import resolve_joint_command_mapping

_LINKER_L20_RIGHT_CAN_ID = 0x28
_LINKER_L20_LEFT_CAN_ID = 0x27
_LINKER_L20_DEFAULT_BITRATE = 1_000_000
_LINKER_L20_INTER_FRAME_S = 0.002
_LINKER_L20_DEVICE_INFO_PROPERTIES = (0xC0, 0xC1, 0xC2)


def _build_position_payloads(sdk_values: Sequence[int]) -> list[bytes]:
    """Split the official 20-value L20 pose into four standard CAN frames."""
    values = [int(value) for value in sdk_values]
    if len(values) != 20:
        raise ValueError(f"Linker L20 SDK vector must have 20 values, got {len(values)}")
    if any(not 0 <= value <= 255 for value in values):
        raise ValueError("Linker L20 SDK values must be in [0, 255]")
    return [
        bytes([0x13, *values[10:15]]),
        bytes([0x14, *values[15:20]]),
        bytes([0x11, *values[0:5]]),
        bytes([0x12, *values[5:10]]),
    ]


class LinkerL20CanOutput(HandOutput):
    """Control a standard LinkerHand L20 over a Linux SocketCAN interface."""

    def __init__(
        self,
        channel: str = "can0",
        *,
        hand_side: str = "right",
        bitrate: int = _LINKER_L20_DEFAULT_BITRATE,
        can_id: int | None = None,
        command_hz: float = 80.0,
        response_timeout: float = 0.2,
        joint_command_mapping: dict | None = None,
    ) -> None:
        try:
            import can
        except ImportError as exc:
            raise ImportError(
                "python-can is required for Linker L20 CAN control. "
                "Install it with `pip install python-can`."
            ) from exc

        self._can = can
        self._channel = str(channel)
        self._bitrate = int(bitrate)
        self._hand_side = hand_side.lower()
        if self._hand_side not in {"left", "right"}:
            raise ValueError(f"Linker L20 hand side must be left or right, got {hand_side}")
        default_id = (
            _LINKER_L20_RIGHT_CAN_ID
            if self._hand_side == "right"
            else _LINKER_L20_LEFT_CAN_ID
        )
        self._can_id = default_id if can_id is None else int(can_id)
        if not 0 <= self._can_id <= 0x7FF:
            raise ValueError(f"Linker L20 CAN ID must be an 11-bit ID, got {can_id}")
        if command_hz < 0:
            raise ValueError(f"Linker L20 command_hz must be non-negative, got {command_hz}")
        if response_timeout <= 0:
            raise ValueError(
                f"Linker L20 response_timeout must be positive, got {response_timeout}"
            )
        self._command_interval = 0.0 if command_hz == 0 else 1.0 / command_hz
        self._response_timeout = float(response_timeout)
        self._joint_command_mapping = resolve_joint_command_mapping(
            joint_command_mapping,
            self._hand_side,
        )
        self._last_command_time = 0.0
        self.send_count = 0

        self._bus = can.interface.Bus(
            channel=self._channel,
            interface="socketcan",
            bitrate=self._bitrate,
        )
        try:
            self.device_info = {
                frame_property: self._request_device_info(frame_property)
                for frame_property in _LINKER_L20_DEVICE_INFO_PROPERTIES
            }
        except Exception:
            self._bus.shutdown()
            raise
        print(
            f"Connected to Linker L20 CAN on {self._channel} "
            f"@ {self._bitrate} bit/s (id=0x{self._can_id:02X}, "
            f"device_info={self.device_info})."
        )

    def _send_payload(self, payload: bytes) -> None:
        message = self._can.Message(
            arbitration_id=self._can_id,
            data=payload,
            is_extended_id=False,
        )
        self._bus.send(message)

    def _request_device_info(self, frame_property: int) -> bytes:
        self._send_payload(bytes([frame_property, 0]))
        deadline = time.monotonic() + self._response_timeout
        while time.monotonic() < deadline:
            message = self._bus.recv(timeout=max(0.0, deadline - time.monotonic()))
            if message is None:
                break
            data = bytes(message.data)
            if (
                message.arbitration_id == self._can_id
                and data
                and data[0] == frame_property
            ):
                return data[1:]
        raise RuntimeError(
            f"No Linker L20 CAN response on {self._channel} for "
            f"property 0x{frame_property:02X}; check USB-CAN, can0 state, "
            "1 Mbps bitrate, hand power and CANH/CANL wiring"
        )

    def send(self, qpos: np.ndarray, joint_names: list[str]) -> None:
        now = time.monotonic()
        if self._command_interval and now - self._last_command_time < self._command_interval:
            return
        sdk_values = _linker_l20_retarget_to_sdk(
            qpos,
            joint_names,
            self._hand_side,
            self._joint_command_mapping,
        )
        for payload in _build_position_payloads(sdk_values):
            self._send_payload(payload)
            time.sleep(_LINKER_L20_INTER_FRAME_S)
        self._last_command_time = time.monotonic()
        self.send_count += 1

    def close(self) -> None:
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None
            print(
                f"Closed Linker L20 CAN on {self._channel} "
                f"(id=0x{self._can_id:02X})."
            )


__all__ = ["LinkerL20CanOutput", "_build_position_payloads"]
