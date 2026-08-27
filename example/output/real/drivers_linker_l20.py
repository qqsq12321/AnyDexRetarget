"""Modbus RTU output driver for LinkerHand L20 V1.0/V1.1 hands."""

from __future__ import annotations

import struct
import time
from collections.abc import Sequence

import numpy as np

from .base import HandOutput
from .joint_command_mapping import map_joint_command, resolve_joint_command_mapping

_LINKER_L20_READ_HOLDING = 0x03
_LINKER_L20_READ_INPUT = 0x04
_LINKER_L20_WRITE_MULTIPLE = 0x10
_LINKER_L20_POSITION_ADDRESS = 0
_LINKER_L20_VERSION_ADDRESS = 200
_LINKER_L20_REGISTER_COUNT = 30
_LINKER_L20_INTER_FRAME_S = 0.003

_SDK_SOURCE_JOINTS = (
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "thumb_cmc_roll",
    "index_mcp_roll",
    "middle_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_yaw",
    None,
    None,
    None,
    None,
    "thumb_mcp",
    "index_pip",
    "middle_pip",
    "ring_pip",
    "pinky_pip",
)

_RIGHT_MIN_RAD = np.array(
    [
        0, 0, 0, 0, 0,
        -0.297, -0.26, -0.26, -0.26, -0.26,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ],
    dtype=np.float64,
)
_RIGHT_MAX_RAD = np.array(
    [
        0.87, 1.4, 1.4, 1.4, 1.4,
        0.683, 0.26, 0.26, 0.26, 0.26,
        1.78, 0, 0, 0, 0,
        1.29, 1.08, 1.08, 1.08, 1.08,
    ],
    dtype=np.float64,
)
_RIGHT_INVERT = np.array(
    [
        True, True, True, True, True,
        True, False, False, False, False,
        True, False, False, False, False,
        True, True, True, True, True,
    ],
    dtype=bool,
)

_LEFT_MIN_RAD = _RIGHT_MIN_RAD.copy()
_LEFT_MIN_RAD[10] = 0.122
_LEFT_MAX_RAD = _RIGHT_MAX_RAD.copy()
_LEFT_INVERT = np.array(
    [
        True, True, True, True, True,
        True, True, True, True, True,
        True, False, False, False, False,
        True, True, True, True, True,
    ],
    dtype=bool,
)


def _crc16(data: bytes | bytearray) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _append_crc(frame: bytes | bytearray) -> bytes:
    data = bytes(frame)
    return data + _crc16(data).to_bytes(2, "little")


def _validate_slave_id(slave_id: int) -> int:
    slave_id = int(slave_id)
    if not 1 <= slave_id <= 247:
        raise ValueError(f"Linker L20 slave ID must be in [1, 247], got {slave_id}")
    return slave_id


def _build_read_request(
    slave_id: int,
    function: int,
    address: int,
    count: int,
) -> bytes:
    slave_id = _validate_slave_id(slave_id)
    if function not in (_LINKER_L20_READ_HOLDING, _LINKER_L20_READ_INPUT):
        raise ValueError(f"Unsupported Linker L20 read function: 0x{function:02X}")
    if not 0 <= address <= 0xFFFF:
        raise ValueError(f"Invalid Linker L20 register address: {address}")
    if not 1 <= count <= 125:
        raise ValueError(f"Invalid Linker L20 register count: {count}")
    return _append_crc(struct.pack(">BBHH", slave_id, function, address, count))


def _build_write_multiple_request(
    slave_id: int,
    address: int,
    values: Sequence[int],
) -> bytes:
    slave_id = _validate_slave_id(slave_id)
    values = [int(value) for value in values]
    if not values or len(values) > 123:
        raise ValueError(f"Invalid Linker L20 write count: {len(values)}")
    if not 0 <= address <= 0xFFFF:
        raise ValueError(f"Invalid Linker L20 register address: {address}")
    if any(not 0 <= value <= 0xFFFF for value in values):
        raise ValueError("Linker L20 register values must be in [0, 65535]")

    payload = struct.pack(">" + "H" * len(values), *values)
    header = struct.pack(
        ">BBHHB",
        slave_id,
        _LINKER_L20_WRITE_MULTIPLE,
        address,
        len(values),
        len(payload),
    )
    return _append_crc(header + payload)


def _validate_response_crc(frame: bytes) -> None:
    if len(frame) < 5:
        raise RuntimeError(f"Short Linker L20 response: {frame.hex(' ')}")
    expected = _crc16(frame[:-2])
    actual = int.from_bytes(frame[-2:], "little")
    if actual != expected:
        raise RuntimeError(f"Invalid Linker L20 response CRC: {frame.hex(' ')}")


def _validate_response_header(
    frame: bytes,
    *,
    slave_id: int,
    function: int,
) -> None:
    _validate_response_crc(frame)
    if frame[0] != slave_id:
        raise RuntimeError(
            f"Unexpected Linker L20 slave ID: expected {slave_id}, got {frame[0]}"
        )
    if frame[1] == function | 0x80:
        raise RuntimeError(
            f"Linker L20 Modbus exception for 0x{function:02X}: code 0x{frame[2]:02X}"
        )
    if frame[1] != function:
        raise RuntimeError(
            f"Unexpected Linker L20 function: expected 0x{function:02X}, "
            f"got 0x{frame[1]:02X}"
        )


def _parse_read_response(
    frame: bytes,
    *,
    slave_id: int,
    function: int,
    count: int,
) -> list[int]:
    _validate_response_header(frame, slave_id=slave_id, function=function)
    expected_bytes = count * 2
    if frame[2] != expected_bytes or len(frame) != expected_bytes + 5:
        raise RuntimeError(
            f"Invalid Linker L20 read length: byte_count={frame[2]}, "
            f"frame_length={len(frame)}, expected_count={count}"
        )
    return list(struct.unpack(">" + "H" * count, frame[3:-2]))


def _parse_write_response(
    frame: bytes,
    *,
    slave_id: int,
    address: int,
    count: int,
) -> None:
    _validate_response_header(
        frame,
        slave_id=slave_id,
        function=_LINKER_L20_WRITE_MULTIPLE,
    )
    if len(frame) != 8:
        raise RuntimeError(f"Invalid Linker L20 write response: {frame.hex(' ')}")
    response_address, response_count = struct.unpack(">HH", frame[2:6])
    if response_address != address or response_count != count:
        raise RuntimeError(
            "Unexpected Linker L20 write acknowledgement: "
            f"address={response_address}, count={response_count}"
        )


def _sdk_to_registers(sdk_values: Sequence[int]) -> list[int]:
    values = [int(value) for value in sdk_values]
    if len(values) != 20:
        raise ValueError(f"Linker L20 SDK vector must have 20 values, got {len(values)}")
    if any(not 0 <= value <= 255 for value in values):
        raise ValueError("Linker L20 SDK values must be in [0, 255]")

    registers = [0] * _LINKER_L20_REGISTER_COUNT
    registers[0:5] = values[10:15]
    registers[5:10] = values[5:10]
    registers[10:15] = values[0:5]
    registers[25:30] = values[15:20]
    return registers


def _linker_l20_retarget_to_sdk(
    qpos: np.ndarray,
    joint_names: Sequence[str],
    hand_side: str,
    joint_command_mapping: dict | None = None,
) -> list[int]:
    qpos = np.asarray(qpos, dtype=np.float64)
    names = [str(name) for name in joint_names]
    if qpos.ndim != 1 or len(qpos) != len(names):
        raise ValueError(
            "Linker L20 qpos and joint_names must be matching 1-D arrays: "
            f"qpos.shape={qpos.shape}, len(joint_names)={len(names)}"
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Invalid Linker L20 retarget output: contains NaN/Inf")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate Linker L20 retarget joint names")

    hand_side = hand_side.lower()
    if hand_side == "right":
        minimums, maximums, invert = (
            _RIGHT_MIN_RAD,
            _RIGHT_MAX_RAD,
            _RIGHT_INVERT,
        )
    elif hand_side == "left":
        minimums, maximums, invert = _LEFT_MIN_RAD, _LEFT_MAX_RAD, _LEFT_INVERT
    else:
        raise ValueError(f"Linker L20 hand side must be left or right, got {hand_side}")
    calibrated_joints = resolve_joint_command_mapping(
        joint_command_mapping,
        hand_side,
    )

    index_by_name = {name: index for index, name in enumerate(names)}
    missing = [name for name in _SDK_SOURCE_JOINTS if name and name not in index_by_name]
    if missing:
        raise ValueError(f"Linker L20 retarget output is missing joints: {missing}")

    sdk_values: list[int] = []
    for index, name in enumerate(_SDK_SOURCE_JOINTS):
        if name is None:
            sdk_values.append(0)
            continue
        if name in calibrated_joints:
            command = map_joint_command(
                float(qpos[index_by_name[name]]),
                calibrated_joints[name],
            )
            sdk_values.append(int(np.clip(round(command), 0, 255)))
            continue
        span = maximums[index] - minimums[index]
        normalized = float(
            np.clip((qpos[index_by_name[name]] - minimums[index]) / span, 0.0, 1.0)
        )
        if invert[index]:
            normalized = 1.0 - normalized
        sdk_values.append(round(normalized * 255.0))
    return sdk_values


class LinkerL20SerialOutput(HandOutput):
    """Direct LinkerHand L20 controller over RS485 Modbus RTU."""

    def __init__(
        self,
        port_name: str,
        *,
        hand_side: str = "right",
        baudrate: int = 115200,
        slave_id: int = 1,
        command_hz: float = 80.0,
        max_register_step: int = 0,
        joint_command_mapping: dict | None = None,
    ) -> None:
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for Linker L20 control. "
                "Install it with `pip install pyserial`."
            ) from exc

        self._port_name = port_name
        self._baudrate = int(baudrate)
        self._slave_id = _validate_slave_id(slave_id)
        self._hand_side = hand_side.lower()
        if self._hand_side not in {"left", "right"}:
            raise ValueError(f"Linker L20 hand side must be left or right, got {hand_side}")
        if command_hz < 0:
            raise ValueError(f"Linker L20 command_hz must be non-negative, got {command_hz}")
        if max_register_step < 0:
            raise ValueError(
                "Linker L20 max_register_step must be non-negative, "
                f"got {max_register_step}"
            )
        self._command_interval = 0.0 if command_hz == 0 else 1.0 / command_hz
        self._max_register_step = int(max_register_step)
        self._joint_command_mapping = resolve_joint_command_mapping(
            joint_command_mapping,
            self._hand_side,
        )
        self._last_registers: list[int] | None = None
        self._last_command_time = 0.0
        self._last_frame_time = 0.0
        self.send_count = 0

        self._port = serial.Serial(
            port_name,
            self._baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.05,
            write_timeout=0.05,
        )
        try:
            versions = self.read_input_registers(_LINKER_L20_VERSION_ADDRESS, 6)
            if self._max_register_step:
                self._last_registers = self.read_input_registers(
                    _LINKER_L20_POSITION_ADDRESS,
                    _LINKER_L20_REGISTER_COUNT,
                )
        except Exception:
            self._port.close()
            raise
        print(
            f"Connected to Linker L20 serial at {port_name} @ {baudrate} baud "
            f"(slave={self._slave_id}, versions={versions})."
        )

    def _wait_for_bus(self) -> None:
        remaining = _LINKER_L20_INTER_FRAME_S - (time.monotonic() - self._last_frame_time)
        if remaining > 0:
            time.sleep(remaining)

    def _read_frame(self) -> bytes:
        header = self._port.read(2)
        if len(header) < 2:
            return header
        if header[1] & 0x80:
            return header + self._port.read(3)
        if header[1] in (_LINKER_L20_READ_HOLDING, _LINKER_L20_READ_INPUT):
            byte_count = self._port.read(1)
            if len(byte_count) != 1:
                return header + byte_count
            return header + byte_count + self._port.read(byte_count[0] + 2)
        if header[1] == _LINKER_L20_WRITE_MULTIPLE:
            return header + self._port.read(6)
        return header + self._port.read(self._port.in_waiting)

    def _transact(self, request: bytes) -> bytes:
        self._wait_for_bus()
        self._port.reset_input_buffer()
        self._port.write(request)
        frame = self._read_frame()
        self._last_frame_time = time.monotonic()
        if not frame:
            raise RuntimeError(
                f"No Linker L20 response from {self._port_name}; "
                "check power, RS485 mode, A/B wiring, baudrate and slave ID"
            )
        return frame

    def read_input_registers(self, address: int, count: int) -> list[int]:
        request = _build_read_request(
            self._slave_id,
            _LINKER_L20_READ_INPUT,
            address,
            count,
        )
        frame = self._transact(request)
        return _parse_read_response(
            frame,
            slave_id=self._slave_id,
            function=_LINKER_L20_READ_INPUT,
            count=count,
        )

    def write_registers(self, address: int, values: Sequence[int]) -> None:
        request = _build_write_multiple_request(self._slave_id, address, values)
        frame = self._transact(request)
        _parse_write_response(
            frame,
            slave_id=self._slave_id,
            address=address,
            count=len(values),
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
        target_registers = _sdk_to_registers(sdk_values)
        command_registers = target_registers
        if self._max_register_step and self._last_registers is not None:
            lower = np.asarray(self._last_registers) - self._max_register_step
            upper = np.asarray(self._last_registers) + self._max_register_step
            command_registers = np.clip(target_registers, lower, upper).astype(int).tolist()
        self.write_registers(_LINKER_L20_POSITION_ADDRESS, command_registers)
        self._last_registers = command_registers
        self._last_command_time = time.monotonic()
        self.send_count += 1

    def close(self) -> None:
        if self._port.is_open:
            self._port.close()
            print(
                f"Closed Linker L20 serial at {self._port_name} "
                f"@ {self._baudrate} baud."
            )


__all__ = [
    "LinkerL20SerialOutput",
    "_build_read_request",
    "_build_write_multiple_request",
    "_linker_l20_retarget_to_sdk",
    "_parse_read_response",
    "_sdk_to_registers",
]
