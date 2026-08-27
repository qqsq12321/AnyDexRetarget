"""Direct RS485 protocol driver for Inspire RH56DFTP/RH56DFX hands."""

import time

import numpy as np

from .base import HandOutput

_INSPIRE_CHANNEL_INDICES = [4, 6, 2, 0, 9, 8]
_INSPIRE_CHANNEL_MAX_RAD = [1.47, 1.47, 1.47, 1.47, 0.6, 1.308]
_INSPIRE_CHANNEL_INVERT = [True, True, True, True, True, True]
_INSPIRE_SERIAL_RESPONSE_TIMEOUT_S = 0.1
_INSPIRE_READ_REGISTER = 0x11
_INSPIRE_WRITE_REGISTER = 0x12
_INSPIRE_HAND_ID_ADDRESS = 1000
_INSPIRE_ANGLE_SET_ADDRESS = 1486


def _checksum(frame_without_checksum: bytes | bytearray) -> int:
    return sum(frame_without_checksum[2:]) & 0xFF


def _build_read_packet(hand_id: int, address: int, register_length: int) -> bytes:
    if not 1 <= hand_id <= 254:
        raise ValueError(f"Inspire hand ID must be in [1, 254], got {hand_id}")
    if not 0 <= address <= 0xFFFF:
        raise ValueError(f"Invalid Inspire register address: {address}")
    if not 1 <= register_length <= 0xFF:
        raise ValueError(f"Invalid Inspire register length: {register_length}")

    packet = bytearray([
        0xEB,
        0x90,
        hand_id,
        0x04,
        _INSPIRE_READ_REGISTER,
        address & 0xFF,
        (address >> 8) & 0xFF,
        register_length,
    ])
    packet.append(_checksum(packet))
    return bytes(packet)


def _build_write_packet(hand_id: int, address: int, data: bytes) -> bytes:
    if not 1 <= hand_id <= 254:
        raise ValueError(f"Inspire hand ID must be in [1, 254], got {hand_id}")
    if not 0 <= address <= 0xFFFF:
        raise ValueError(f"Invalid Inspire register address: {address}")
    if not 1 <= len(data) <= 0xFC:
        raise ValueError(f"Invalid Inspire write length: {len(data)}")

    packet = bytearray([
        0xEB,
        0x90,
        hand_id,
        len(data) + 3,
        _INSPIRE_WRITE_REGISTER,
        address & 0xFF,
        (address >> 8) & 0xFF,
    ])
    packet.extend(data)
    packet.append(_checksum(packet))
    return bytes(packet)


def _parse_response(
    frame: bytes,
    *,
    hand_id: int,
    command: int,
    address: int,
) -> bytes:
    if len(frame) < 9:
        raise RuntimeError(f"Short Inspire response: {frame.hex(' ')}")
    if frame[:2] != b"\x90\xEB":
        raise RuntimeError(f"Invalid Inspire response header: {frame.hex(' ')}")
    expected_length = frame[3] + 5
    if len(frame) != expected_length:
        raise RuntimeError(
            f"Invalid Inspire response length: expected {expected_length}, got {len(frame)}"
        )
    if frame[2] != hand_id:
        raise RuntimeError(f"Unexpected Inspire hand ID: expected {hand_id}, got {frame[2]}")
    if frame[4] != command:
        raise RuntimeError(
            f"Unexpected Inspire response command: expected 0x{command:02X}, "
            f"got 0x{frame[4]:02X}"
        )
    response_address = frame[5] | (frame[6] << 8)
    if response_address != address:
        raise RuntimeError(
            f"Unexpected Inspire response address: expected {address}, got {response_address}"
        )
    if frame[-1] != _checksum(frame[:-1]):
        raise RuntimeError(f"Invalid Inspire response checksum: {frame.hex(' ')}")
    return frame[7:-1]


def _inspire_retarget_to_real(retarget_output: np.ndarray) -> list[int]:
    """Map 12-D Inspire retarget output (rad) to 6 serial control channels."""
    output = np.asarray(retarget_output, dtype=np.float64)
    if output.shape[0] < 12:
        raise ValueError(f"Expected at least 12 Inspire joints, got shape {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError("Invalid Inspire retarget output: contains NaN/Inf")

    result: list[int] = []
    for idx, max_rad, invert in zip(
        _INSPIRE_CHANNEL_INDICES,
        _INSPIRE_CHANNEL_MAX_RAD,
        _INSPIRE_CHANNEL_INVERT,
    ):
        value = float(np.clip(output[idx] / max_rad, 0.0, 1.0))
        if invert:
            value = 1.0 - value
        result.append(int(value * 2000))
    return result


class InspireSerialOutput(HandOutput):
    """Direct serial controller for Inspire RH56DFTP/RH56DFX hands."""

    def __init__(self, port_name: str, baudrate: int = 115200, hand_id: int = 1):
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for Inspire hand control. "
                "Install it with `pip install pyserial`."
            ) from exc

        self._port = serial.Serial(port_name, baudrate, timeout=0.01)
        self._hand_id = int(hand_id)
        self._port_name = port_name
        self._baudrate = int(baudrate)
        self.send_count = 0
        try:
            self._port.reset_input_buffer()
            reported_id = self.read_register(_INSPIRE_HAND_ID_ADDRESS, 1)[0]
            if reported_id != self._hand_id:
                raise RuntimeError(
                    f"Inspire hand reported ID {reported_id}, expected {self._hand_id}"
                )
        except Exception:
            self._port.close()
            raise
        print(
            f"Connected to Inspire hand serial at {port_name} @ {baudrate} baud "
            f"(id={reported_id})."
        )

    def _read_response(self) -> bytes:
        deadline = time.time() + _INSPIRE_SERIAL_RESPONSE_TIMEOUT_S
        input_bytes = bytearray()
        while time.time() < deadline:
            chunk = self._port.read(self._port.in_waiting or 1)
            if chunk:
                input_bytes += chunk
                header_index = input_bytes.find(b"\x90\xEB")
                if header_index > 0:
                    del input_bytes[:header_index]
                if len(input_bytes) >= 4:
                    expected_length = input_bytes[3] + 5
                    if len(input_bytes) >= expected_length:
                        return bytes(input_bytes[:expected_length])
        return bytes(input_bytes)

    def read_register(self, address: int, register_length: int) -> bytes:
        self._port.write(_build_read_packet(self._hand_id, address, register_length))
        frame = self._read_response()
        data = _parse_response(
            frame,
            hand_id=self._hand_id,
            command=_INSPIRE_READ_REGISTER,
            address=address,
        )
        if len(data) != register_length:
            raise RuntimeError(
                f"Inspire register {address} returned {len(data)} bytes, "
                f"expected {register_length}"
            )
        return data

    def write_register(self, address: int, data: bytes) -> None:
        self._port.write(_build_write_packet(self._hand_id, address, data))
        frame = self._read_response()
        result = _parse_response(
            frame,
            hand_id=self._hand_id,
            command=_INSPIRE_WRITE_REGISTER,
            address=address,
        )
        if result != b"\x01":
            raise RuntimeError(
                f"Inspire register write {address} was rejected: {result.hex(' ')}"
            )

    @staticmethod
    def _encode_channels(channels: list[int]) -> list[int]:
        if len(channels) != 6:
            raise ValueError(f"Inspire hand expects 6 channels, got {len(channels)}")
        return [int(np.clip(round(ch / 2.0), 0, 1000)) for ch in channels]

    def send(self, qpos, joint_names):
        channels = _inspire_retarget_to_real(qpos)
        encoded = self._encode_channels(channels)
        data = bytearray()
        for angle in encoded:
            data.append(angle & 0xFF)
            data.append((angle >> 8) & 0xFF)
        self.write_register(_INSPIRE_ANGLE_SET_ADDRESS, bytes(data))
        self.send_count += 1

    def close(self):
        if self._port.is_open:
            self._port.close()
            print(f"Closed Inspire hand serial at {self._port_name} @ {self._baudrate} baud.")


__all__ = [
    "InspireSerialOutput",
    "_build_read_packet",
    "_build_write_packet",
    "_inspire_retarget_to_real",
    "_parse_response",
]
