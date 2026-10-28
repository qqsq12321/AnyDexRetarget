"""RS485 Modbus driver for Linker L20 hand."""

import time

import numpy as np

from .base import HandOutput

_L20_PORT = "/dev/ttyUSB0"
_L20_BAUDRATE = 460800
_L20_RIGHT_SLAVE_ID = 0x29
_L20_LEFT_SLAVE_ID = 0x2A
_L20_SPEED = 200
_L20_CURRENT_LIMIT = 200
_L20_CLEAR_FAULTS = False
_L20_OPEN_ON_EXIT = False
_L20_PRINT_REGISTERS = False
_L20_DRY_RUN = False
_L20_NEUTRAL_ROLL = 128

# L20 V10 active-joint order used by the GEORT real-hand mapper.
_L20_ACTIVE_JOINT_ORDER = (
    "THUMB_CMC_YAW",
    "THUMB_CMC_ROLL",
    "THUMB_CMC_PITCH",
    "THUMB_MCP",
    "INDEX_MCP_ROLL",
    "INDEX_MCP_PITCH",
    "INDEX_PIP",
    "MIDDLE_MCP_ROLL",
    "MIDDLE_MCP_PITCH",
    "MIDDLE_PIP",
    "RING_MCP_ROLL",
    "RING_MCP_PITCH",
    "RING_PIP",
    "PINKY_MCP_ROLL",
    "PINKY_MCP_PITCH",
    "PINKY_PIP",
)

# Effective limits from the L20 V10 URDF. PIP is constrained by DIP mimic 0.89.
_L20_JOINT_LIMITS = {
    "THUMB_CMC_YAW": (0.0, 1.57),
    "THUMB_CMC_ROLL": (0.0, 1.39),
    "THUMB_CMC_PITCH": (0.0, 0.83),
    "THUMB_MCP": (0.0, 1.25),
    "INDEX_MCP_ROLL": (-0.23, 0.23),
    "INDEX_MCP_PITCH": (0.0, 1.22),
    "INDEX_PIP": (0.0, 1.7415730337078652),
    "MIDDLE_MCP_ROLL": (-0.23, 0.23),
    "MIDDLE_MCP_PITCH": (0.0, 1.22),
    "MIDDLE_PIP": (0.0, 1.7415730337078652),
    "RING_MCP_ROLL": (-0.23, 0.23),
    "RING_MCP_PITCH": (0.0, 1.22),
    "RING_PIP": (0.0, 1.7415730337078652),
    "PINKY_MCP_ROLL": (-0.23, 0.23),
    "PINKY_MCP_PITCH": (0.0, 1.22),
    "PINKY_PIP": (0.0, 1.7415730337078652),
}

# MODBUS direction is opposite to URDF qpos for these joints. Finger MCP roll
# is intentionally not inverted, matching GEORT and the L20 register protocol.
_L20_INVERTED_JOINTS = frozenset({
    "THUMB_CMC_YAW",
    "THUMB_CMC_ROLL",
    "THUMB_CMC_PITCH",
    "THUMB_MCP",
    "INDEX_MCP_PITCH",
    "INDEX_PIP",
    "MIDDLE_MCP_PITCH",
    "MIDDLE_PIP",
    "RING_MCP_PITCH",
    "RING_PIP",
    "PINKY_MCP_PITCH",
    "PINKY_PIP",
})

_L20_FOUR_FINGER_MCP_PITCH_JOINTS = frozenset({
    "INDEX_MCP_PITCH",
    "MIDDLE_MCP_PITCH",
    "RING_MCP_PITCH",
    "PINKY_MCP_PITCH",
})
_L20_FOUR_FINGER_MCP_PITCH_OFFSET_RAD = 0.0

# Extra real-hand qpos offsets applied before qpos->register normalization.
_L20_JOINT_OFFSETS_RAD = np.array([
    0.0, 0.15, 0.0, 0.0,
    0.0, -0.15235988, 0.0,
    0.0, -0.15235988, 0.0,
    0.0, -0.15235988, 0.0,
    0.0, -0.15235988, 0.0,
], dtype=np.float64)
_L20_JOINT_OFFSET_BY_SUFFIX = dict(zip(_L20_ACTIVE_JOINT_ORDER, _L20_JOINT_OFFSETS_RAD))

# Measured simulation-qpos -> normalized real-hand command alignment from GEORT.
_L20_QPOS_ALIGNMENT_POINTS = {
    "THUMB_CMC_ROLL": (
        (0.13962634015954636, 0.0),
        (0.7299065850398866, 0.5),
        (1.4423598775598299, 1.0),
    ),
    "THUMB_CMC_PITCH": (
        (0.05235987755982989, 0.0),
        (0.4499065850398866, 0.5),
        (0.8823598775598299, 1.0),
    ),
    "THUMB_MCP": (
        (0.13962634015954636, 0.0),
        (0.67735987755983, 0.5),
        (1.25, 1.0),
    ),
}


def _clamp_uint8(value: float) -> int:
    return int(np.clip(int(round(value)), 0, 255))


class LinkerL20RS485:
    """Low-level RS485 controller for Linker L20 hand."""

    POSITION_START = 0
    POSITION_COUNT = 30
    SPEED_START = 30
    SPEED_COUNT = 30
    CLEAR_FAULT_START = 90
    CLEAR_FAULT_COUNT = 30
    CURRENT_LIMIT_START = 150
    CURRENT_LIMIT_COUNT = 30

    def __init__(
        self,
        port: str = _L20_PORT,
        baudrate: int = _L20_BAUDRATE,
        slave_id: int = _L20_RIGHT_SLAVE_ID,
        timeout: float = 0.05,
    ):
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for Linker L20 control. "
                "Install it with `pip install pyserial`."
            ) from exc

        self.slave_id = int(slave_id)
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        self._port_name = port
        self._baudrate = int(baudrate)

    @staticmethod
    def _crc16(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    def _send_request(self, payload: bytes, expected_len: int) -> bytes:
        crc = self._crc16(payload)
        frame = payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
        self._serial.reset_input_buffer()
        self._serial.write(frame)
        self._serial.flush()
        response = self._serial.read(expected_len)
        if len(response) != expected_len:
            raise TimeoutError(
                f"Incomplete Modbus response: expected {expected_len} bytes, got {len(response)} bytes."
            )
        body = response[:-2]
        crc_recv = response[-2] | (response[-1] << 8)
        if self._crc16(body) != crc_recv:
            raise RuntimeError("CRC mismatch in Modbus response.")
        if response[0] != self.slave_id:
            raise RuntimeError("Unexpected slave id in Modbus response.")
        if response[1] & 0x80:
            raise RuntimeError(f"Modbus exception code: {response[2]}")
        return response

    def write_registers(self, start_address: int, values: list[int]) -> None:
        registers = [_clamp_uint8(v) for v in values]
        register_count = len(registers)
        byte_count = register_count * 2
        payload = bytearray([
            self.slave_id,
            0x10,
            (start_address >> 8) & 0xFF,
            start_address & 0xFF,
            (register_count >> 8) & 0xFF,
            register_count & 0xFF,
            byte_count,
        ])
        for value in registers:
            payload.extend([0x00, value])
        self._send_request(bytes(payload), expected_len=8)

    def configure_motion_profile(self, speed: int, current_limit: int, clear_faults: bool) -> None:
        self.write_registers(self.SPEED_START, [speed] * self.SPEED_COUNT)
        self.write_registers(self.CURRENT_LIMIT_START, [current_limit] * self.CURRENT_LIMIT_COUNT)
        if clear_faults:
            self.write_registers(self.CLEAR_FAULT_START, [1] * self.CLEAR_FAULT_COUNT)
            time.sleep(0.02)
            self.write_registers(self.CLEAR_FAULT_START, [0] * self.CLEAR_FAULT_COUNT)

    def set_positions(self, registers: list[int]) -> None:
        if len(registers) != self.POSITION_COUNT:
            raise ValueError(f"L20 position command must contain 30 registers, got {len(registers)}.")
        self.write_registers(self.POSITION_START, registers)

    def close(self) -> None:
        if self._serial.is_open:
            self._serial.close()
            print(f"Closed Linker L20 serial at {self._port_name} @ {self._baudrate} baud.")


class LinkerL20Output(HandOutput):
    """Output L20 commands using GEORT's calibrated qpos-to-register mapping."""

    def __init__(
        self,
        port: str = _L20_PORT,
        baudrate: int = _L20_BAUDRATE,
        slave_id: int = _L20_RIGHT_SLAVE_ID,
        speed: int = _L20_SPEED,
        current_limit: int = _L20_CURRENT_LIMIT,
        clear_faults: bool = _L20_CLEAR_FAULTS,
        open_on_exit: bool = _L20_OPEN_ON_EXIT,
        print_registers: bool = _L20_PRINT_REGISTERS,
        dry_run: bool = _L20_DRY_RUN,
        transport: LinkerL20RS485 | None = None,
    ):
        self._slave_id = int(slave_id)
        self._owns_hand = transport is None
        self._open_on_exit = bool(open_on_exit)
        self._print_registers = bool(print_registers)
        self._dry_run = bool(dry_run)
        self._last_registers: list[int] | None = None
        self._last_joint_index_key: tuple[str, ...] | None = None
        self._joint_index_by_suffix: dict[str, int] = {}
        self.send_count = 0

        if self._dry_run:
            self._hand = None
            print("Linker L20 dry-run enabled: serial output is disabled.")
        else:
            self._hand = transport or LinkerL20RS485(
                port=port, baudrate=baudrate, slave_id=self._slave_id
            )
            self._hand.slave_id = self._slave_id
            print(
                f"Configuring Linker L20 at {port} @ {baudrate} "
                f"(slave_id={self._slave_id}, speed={speed}, current_limit={current_limit})..."
            )
            try:
                # A shared RS485 adapter serves both hands.  Select the target
                # slave before every configuration/request, not just at open.
                self._hand.slave_id = self._slave_id
                self._hand.configure_motion_profile(
                    speed=speed,
                    current_limit=current_limit,
                    clear_faults=clear_faults,
                )
            except Exception as exc:
                if self._owns_hand:
                    self._hand.close()
                raise RuntimeError(
                    "Linker L20 did not respond during startup. "
                    "Check hand power, RS485 wiring, port, baudrate, and slave id. "
                    f"Current constants: port={port}, baudrate={baudrate}, slave_id={self._slave_id}."
                ) from exc
            print(f"Connected to Linker L20 serial at {port} @ {baudrate} baud (slave_id={self._slave_id}).")

    def _ensure_joint_index(self, joint_names) -> None:
        key = tuple(str(name) for name in joint_names)
        if key == self._last_joint_index_key:
            return

        upper_names = [name.upper() for name in key]
        index_by_suffix = {}
        for suffix in _L20_ACTIVE_JOINT_ORDER:
            matches = [idx for idx, name in enumerate(upper_names) if name.endswith(suffix)]
            if not matches:
                raise ValueError(f"Missing Linker L20 joint in retarget output: *{suffix}")
            if len(matches) > 1:
                raise ValueError(f"Ambiguous Linker L20 joint suffix *{suffix}: {matches}")
            index_by_suffix[suffix] = matches[0]

        self._joint_index_by_suffix = index_by_suffix
        self._last_joint_index_key = key

    def _joint_normalized(self, qpos: np.ndarray, suffix: str) -> float:
        value = float(qpos[self._joint_index_by_suffix[suffix]])
        value += float(_L20_JOINT_OFFSET_BY_SUFFIX.get(suffix, 0.0))
        alignment = _L20_QPOS_ALIGNMENT_POINTS.get(suffix)
        if alignment is not None:
            sim_qpos, real_normalized = zip(*alignment)
            return float(np.interp(value, sim_qpos, real_normalized))

        lower, upper = _L20_JOINT_LIMITS[suffix]
        if upper <= lower:
            return 0.5
        if suffix in _L20_FOUR_FINGER_MCP_PITCH_JOINTS:
            value -= _L20_FOUR_FINGER_MCP_PITCH_OFFSET_RAD
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    @staticmethod
    def _normalized_register(suffix: str, normalized: float) -> int:
        ratio = float(np.clip(normalized, 0.0, 1.0))
        if suffix in _L20_INVERTED_JOINTS:
            ratio = 1.0 - ratio
        return _clamp_uint8(ratio * 255.0)

    def _qpos_to_registers(self, qpos: np.ndarray, joint_names) -> list[int]:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(qpos)):
            raise ValueError("Invalid Linker L20 retarget output: contains NaN/Inf")
        self._ensure_joint_index(joint_names)

        normalized = {
            suffix: self._joint_normalized(qpos, suffix)
            for suffix in _L20_ACTIVE_JOINT_ORDER
        }
        reg = lambda suffix: self._normalized_register(suffix, normalized[suffix])

        roll = [reg("THUMB_CMC_ROLL")] + [_L20_NEUTRAL_ROLL] * 4
        yaw = [
            reg("THUMB_CMC_YAW"),
            reg("INDEX_MCP_ROLL"),
            reg("MIDDLE_MCP_ROLL"),
            reg("RING_MCP_ROLL"),
            reg("PINKY_MCP_ROLL"),
        ]
        root1 = [
            reg("THUMB_CMC_PITCH"),
            reg("INDEX_MCP_PITCH"),
            reg("MIDDLE_MCP_PITCH"),
            reg("RING_MCP_PITCH"),
            reg("PINKY_MCP_PITCH"),
        ]
        distal = [
            reg("THUMB_MCP"),
            reg("INDEX_PIP"),
            reg("MIDDLE_PIP"),
            reg("RING_PIP"),
            reg("PINKY_PIP"),
        ]
        return roll + yaw + root1 + distal + list(distal) + list(distal)

    @staticmethod
    def open_palm_registers() -> list[int]:
        roll = [_L20_NEUTRAL_ROLL] * 5
        yaw = [128] * 5
        extended = [255] * 5
        return roll + yaw + extended + extended + extended + extended

    def open_palm(self) -> None:
        registers = self.open_palm_registers()
        if self._print_registers or self._dry_run:
            print(f"L20 open-palm registers: {registers}")
        if self._hand is not None:
            self._hand.slave_id = self._slave_id
            self._hand.set_positions(registers)
        self._last_registers = registers

    def send(self, qpos, joint_names):
        registers = self._qpos_to_registers(qpos, joint_names)
        if registers == self._last_registers:
            return

        if self._print_registers or self._dry_run:
            print(f"L20 registers: {registers}")
        if self._hand is not None:
            self._hand.slave_id = self._slave_id
            self._hand.set_positions(registers)
        self._last_registers = registers
        self.send_count += 1

    def close(self):
        if self._hand is None:
            return
        try:
            if self._open_on_exit:
                self._hand.slave_id = self._slave_id
                self._hand.set_positions(self.open_palm_registers())
                time.sleep(0.2)
        finally:
            if self._owns_hand:
                self._hand.close()
            self._hand = None


__all__ = ["LinkerL20Output", "LinkerL20RS485"]
