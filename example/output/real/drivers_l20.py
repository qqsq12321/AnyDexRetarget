"""Real-hand output drivers and post-processing mappings."""

import json
import socket
import time

import numpy as np

_INSPIRE_CHANNEL_INDICES = [4, 6, 2, 0, 9, 8]
_INSPIRE_CHANNEL_MAX_RAD = [1.47, 1.47, 1.47, 1.47, 0.6, 1.308]
_INSPIRE_CHANNEL_INVERT = [True, True, True, True, True, True]
_INSPIRE_SERIAL_RESPONSE_TIMEOUT_S = 0.1
_L20_PORT = "/dev/ttyUSB0"
_L20_BAUDRATE = 460800
_L20_RIGHT_SLAVE_ID = 0x29
_L20_LEFT_SLAVE_ID = 0x2A
_L20_SLAVE_ID = _L20_RIGHT_SLAVE_ID
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
# Tune these after retargeting changes; values are radians in _L20_ACTIVE_JOINT_ORDER.
_L20_JOINT_OFFSETS_RAD = np.array([
    0.0, 0.0, 0.0, 0.0,  # THUMB: CMC_YAW, CMC_ROLL, CMC_PITCH, MCP
    0.0, -0.15235988, 0.0,  # INDEX: MCP_ROLL, MCP_PITCH, PIP
    0.0, -0.15235988, 0.0,  # MIDDLE: MCP_ROLL, MCP_PITCH, PIP
    0.0, -0.15235988, 0.0,  # RING: MCP_ROLL, MCP_PITCH, PIP
    0.0, -0.15235988, 0.0,  # PINKY: MCP_ROLL, MCP_PITCH, PIP
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
    return int(np.clip(round(value), 0, 255))

class WujiOutput:
    """Output driver for Wuji Hand via wujihandpy."""

    def __init__(self):
        import wujihandpy
        self.hand = wujihandpy.Hand()
        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=False,
            filter=wujihandpy.filter.LowPass(cutoff_freq=5.0),
        )
        time.sleep(0.5)

    def send(self, qpos, joint_names):
        self.controller.set_joint_target_position(qpos.reshape(5, 4))

    def close(self):
        self.hand.write_joint_enabled(False)


class ShadowTCPOutput:
    """Output driver for Shadow Hand via TCP socket to docker_ros_bridge."""

    def __init__(self, docker_ip="localhost", port=5555):
        self.docker_ip = docker_ip
        self.port = port
        self.sock = self._connect()

    def _connect(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.docker_ip, self.port))
                print(f"Connected to Shadow Hand ROS bridge at {self.docker_ip}:{self.port}")
                return s
            except ConnectionRefusedError:
                print(f"Cannot connect to {self.docker_ip}:{self.port}, retrying in 2s...")
                time.sleep(2)

    def send(self, qpos, joint_names):
        msg = json.dumps({
            "joint_names": joint_names,
            "positions": qpos.tolist(),
        }) + "\n"
        try:
            self.sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("Connection lost, reconnecting...")
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = self._connect()
            self.sock.sendall(msg.encode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


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


class InspireSerialOutput:
    """Direct serial controller for Inspire RH56DFX hand."""

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
        print(f"Connected to Inspire hand serial at {port_name} @ {baudrate} baud.")

    def _read_response(self) -> bytes:
        deadline = time.time() + _INSPIRE_SERIAL_RESPONSE_TIMEOUT_S
        input_bytes = bytearray()
        while time.time() < deadline:
            chunk = self._port.read(self._port.in_waiting or 1)
            if chunk:
                input_bytes += chunk
            else:
                break
        return bytes(input_bytes)

    @staticmethod
    def _encode_channels(channels: list[int]) -> list[int]:
        if len(channels) != 6:
            raise ValueError(f"Inspire hand expects 6 channels, got {len(channels)}")
        return [int(np.clip(round(ch / 2.0), 0, 1000)) for ch in channels]

    def send(self, qpos, joint_names):
        channels = _inspire_retarget_to_real(qpos)
        encoded = self._encode_channels(channels)
        packet = bytearray([0xEB, 0x90, self._hand_id, 0x0F, 0x12, 0xCE, 0x05])
        for angle in encoded:
            packet.append(angle & 0xFF)
            packet.append((angle >> 8) & 0xFF)
        checksum = sum(packet[2:2 + 0x0F + 3])
        packet.append(checksum & 0xFF)
        self._port.write(packet)
        self._read_response()

    def close(self):
        if self._port.is_open:
            self._port.close()
            print(f"Closed Inspire hand serial at {self._port_name} @ {self._baudrate} baud.")


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
        port: str = "/dev/ttyUSB0",
        baudrate: int = 460800,
        slave_id: int = 41,
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
        payload = bytearray(
            [
                self.slave_id,
                0x10,
                (start_address >> 8) & 0xFF,
                start_address & 0xFF,
                (register_count >> 8) & 0xFF,
                register_count & 0xFF,
                byte_count,
            ]
        )
        for value in registers:
            payload.extend([0x00, value])
        self._send_request(bytes(payload), expected_len=8)

    def read_input_registers(self, start_address: int, count: int) -> list[int]:
        """Read `count` input registers (Modbus function code 0x04) starting at start_address."""
        payload = bytes(
            [
                self.slave_id,
                0x04,
                (start_address >> 8) & 0xFF,
                start_address & 0xFF,
                (count >> 8) & 0xFF,
                count & 0xFF,
            ]
        )
        response = self._send_request(payload, expected_len=3 + count * 2 + 2)
        byte_count = response[2]
        data = response[3 : 3 + byte_count]
        return [data[i + 1] for i in range(0, byte_count, 2)]

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


class LinkerL20V10SerialOutput:
    """Output L20 commands using GEORT's calibrated qpos-to-register mapping."""

    def __init__(
        self,
        port_name: str = _L20_PORT,
        hand_side: str = "right",
        baudrate: int = _L20_BAUDRATE,
        slave_id: int = _L20_SLAVE_ID,
        command_hz: float = 80.0,
        max_register_step: int = 0,
        open_on_exit: bool = _L20_OPEN_ON_EXIT,
        print_registers: bool = _L20_PRINT_REGISTERS,
        dry_run: bool = _L20_DRY_RUN,
    ):
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
        self._open_on_exit = bool(open_on_exit)
        self._print_registers = bool(print_registers)
        self._dry_run = bool(dry_run)
        self._command_interval = 0.0 if command_hz == 0 else 1.0 / command_hz
        self._max_register_step = int(max_register_step)
        self._last_command_time = 0.0
        self._last_registers: list[int] | None = None
        self._last_joint_index_key: tuple[str, ...] | None = None
        self._joint_index_by_suffix: dict[str, int] = {}

        if self._dry_run:
            self._hand = None
            print("Linker L20 dry-run enabled: serial output is disabled.")
        else:
            self._hand = LinkerL20RS485(
                port=port_name,
                baudrate=baudrate,
                slave_id=slave_id,
            )
            try:
                current_registers = self._hand.read_input_registers(
                    self._hand.POSITION_START,
                    self._hand.POSITION_COUNT,
                )
                if self._max_register_step:
                    self._last_registers = current_registers
            except Exception as exc:
                self._hand.close()
                raise RuntimeError(
                    "Linker L20 V10 did not respond to the read-only startup handshake. "
                    "Check hand power, RS485 wiring, port, baudrate, and slave id. "
                    f"Current settings: port={port_name}, baudrate={baudrate}, "
                    f"slave_id={slave_id}."
                ) from exc
            print(
                f"Connected to Linker L20 V10 serial at {port_name} "
                f"@ {baudrate} baud (slave_id={slave_id})."
            )

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

    def _normalized_register(self, suffix: str, normalized: float) -> int:
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
        return roll + yaw + root1 + [0] * 10 + distal

    @staticmethod
    def open_palm_registers() -> list[int]:
        roll = [_L20_NEUTRAL_ROLL] * 5
        yaw = [128] * 5
        extended = [255] * 5
        return roll + yaw + extended + [0] * 10 + extended

    def send(self, qpos, joint_names):
        now = time.monotonic()
        if self._command_interval and now - self._last_command_time < self._command_interval:
            return
        target_registers = self._qpos_to_registers(qpos, joint_names)
        registers = target_registers
        if self._max_register_step and self._last_registers is not None:
            lower = np.asarray(self._last_registers) - self._max_register_step
            upper = np.asarray(self._last_registers) + self._max_register_step
            registers = np.clip(target_registers, lower, upper).astype(int).tolist()
            registers[15:25] = [0] * 10
        if registers == self._last_registers:
            return

        if self._print_registers or self._dry_run:
            print(f"L20 registers: {registers}")
        if self._hand is not None:
            self._hand.set_positions(registers)
        self._last_registers = registers
        self._last_command_time = time.monotonic()

    def close(self):
        if self._hand is None:
            return
        try:
            if self._open_on_exit:
                self._hand.set_positions(self.open_palm_registers())
                time.sleep(0.2)
        finally:
            self._hand.close()


# Preserve the local reference driver's historical name.
LinkerL20Output = LinkerL20V10SerialOutput
