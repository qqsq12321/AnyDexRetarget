"""CLI integration checks for LinkerHand L20 CAN and Pico v2."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import teleop_real


def test_l20_can_transport_builds_can_output() -> None:
    sentinel = object()
    with patch.object(teleop_real, "LinkerL20CanOutput", return_value=sentinel) as can:
        output = teleop_real._create_linker_l20_output(
            transport="can",
            hand_side="right",
            port="/dev/ttyUSB0",
            baudrate=115200,
            slave_id=1,
            can_channel="can0",
            can_bitrate=1_000_000,
            can_id=None,
            command_hz=80.0,
            rs485_profile="standard",
        )

    assert output is sentinel
    can.assert_called_once_with(
        channel="can0",
        hand_side="right",
        bitrate=1_000_000,
        can_id=None,
        command_hz=80.0,
    )


def test_l20_v10_rs485_profile_builds_v10_output() -> None:
    sentinel = object()
    with patch.object(
        teleop_real,
        "LinkerL20V10SerialOutput",
        return_value=sentinel,
    ) as serial:
        output = teleop_real._create_linker_l20_output(
            transport="rs485",
            hand_side="right",
            port="/dev/ttyUSB0",
            baudrate=460800,
            slave_id=41,
            can_channel="can0",
            can_bitrate=1_000_000,
            can_id=None,
            command_hz=80.0,
            rs485_profile="v10",
            max_register_step=3,
        )

    assert output is sentinel
    serial.assert_called_once_with(
        port_name="/dev/ttyUSB0",
        hand_side="right",
        baudrate=460800,
        slave_id=41,
        command_hz=80.0,
        max_register_step=3,
    )


def test_l20_standard_rs485_passes_register_slew_limit() -> None:
    sentinel = object()
    mapping = {"default": {"thumb_cmc_roll": {"input": [0, 1], "output": [255, 0]}}}
    with patch.object(
        teleop_real,
        "LinkerL20SerialOutput",
        return_value=sentinel,
    ) as serial:
        output = teleop_real._create_linker_l20_output(
            transport="rs485",
            hand_side="right",
            port="/dev/ttyUSB0",
            baudrate=460800,
            slave_id=42,
            can_channel="can0",
            can_bitrate=1_000_000,
            can_id=None,
            command_hz=50.0,
            rs485_profile="standard",
            max_register_step=2,
            joint_command_mapping=mapping,
        )

    assert output is sentinel
    serial.assert_called_once_with(
        port_name="/dev/ttyUSB0",
        hand_side="right",
        baudrate=460800,
        slave_id=42,
        command_hz=50.0,
        max_register_step=2,
        joint_command_mapping=mapping,
    )


def test_pico_l20_v2_default_config_path() -> None:
    path = teleop_real._resolve_default_config_path(
        robot="linker_l20",
        optimizer="adaptive",
        input_device_type="pico4",
        retarget_version="v2",
    )

    assert path == "config/adaptive/pico4/pico4_linker_l20_v2.yaml"
