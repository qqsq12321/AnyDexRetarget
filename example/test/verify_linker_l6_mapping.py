#!/usr/bin/env python3
"""Verify L6 assets, independent controls, mimic joints, FK and all configs.

Run from the repository root with the simulation environment activated:
    python example/test/verify_linker_l6_mapping.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "example"))

from anydexretarget import Retargeter
from output.sim.mujoco_output import (
    ROBOT_HAND_CONFIGS,
    map_retarget_qpos,
    retarget_to_mujoco_target,
    validate_mujoco_actuator_mapping,
)

ASSETS = PROJECT_ROOT / "assets/linker_l6"
ACTUATED_JOINTS = (
    "thumbCMC", "thumbMCP", "indexMCP", "middleMCP", "ringMCP", "littleMCP",
)
MIMIC_MULTIPLIERS = {
    "thumbPIP": ("thumbMCP", 0.611762),
    "indexPIP": ("indexMCP", 0.976341),
    "middlePIP": ("middleMCP", 1.001797),
    "ringPIP": ("ringMCP", 0.982213),
    "littlePIP": ("littleMCP", 0.974884),
}


def hand_keypoints(side: str, flexion: float) -> np.ndarray:
    """Produce a non-degenerate synthetic MediaPipe-format open/curl pose."""
    points = np.zeros((21, 3), dtype=np.float64)
    points[1:5] = [
        [0.0, 0.027, 0.028], [0.008, 0.043, 0.047],
        [0.016 + 0.015 * flexion, 0.051, 0.068 - 0.01 * flexion],
        [0.022 + 0.025 * flexion, 0.053, 0.085 - 0.03 * flexion],
    ]
    for start, y, z, lengths in (
        (5, 0.028, 0.065, (0.035, 0.024, 0.018)),
        (9, 0.008, 0.070, (0.039, 0.026, 0.019)),
        (13, -0.012, 0.066, (0.036, 0.025, 0.018)),
        (17, -0.031, 0.058, (0.028, 0.020, 0.016)),
    ):
        points[start] = [0.0, y, z]
        for segment, length in enumerate(lengths, 1):
            angle = flexion * (0.7 + 0.5 * segment)
            points[start + segment] = points[start + segment - 1] + [
                length * np.sin(angle), 0.0, length * np.cos(angle),
            ]
    if side == "left":
        points[:, 1] *= -1
    return points


def _check_mimics(values: dict, side: str, tolerance: float) -> None:
    for name, (source, multiplier) in MIMIC_MULTIPLIERS.items():
        offset = 0.612611 if side == "right" and name == "thumbPIP" else 0.0
        expected = multiplier * values[source] + offset
        np.testing.assert_allclose(values[name], expected, atol=tolerance, rtol=0)


def _joint_addresses(model) -> dict[str, int]:
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id): int(address)
        for joint_id, address in enumerate(model.jnt_qposadr)
    }


def verify_side(side: str) -> None:
    hand_cfg = ROBOT_HAND_CONFIGS["linker_l6"]
    config = PROJECT_ROOT / "example/config/adaptive/mediapipe/mediapipe_linker_l6.yaml"
    optimizer = Retargeter.from_yaml(str(config), side).optimizer
    robot = optimizer.robot
    names = list(robot.dof_joint_names)
    assert optimizer.num_joints == 11 and optimizer.num_opt_vars == 6
    assert len(optimizer.mimic_map) == 5 and len(set(names)) == 11
    independent_names = {names[index] for index in optimizer.independent_indices}
    assert independent_names == set(ACTUATED_JOINTS)
    neutral = optimizer.neutral_qpos
    np.testing.assert_allclose(
        optimizer.expand_to_full_qpos(neutral[optimizer.independent_indices]), neutral,
        atol=1e-12, rtol=0,
    )

    urdf = ASSETS / f"{side}_linker/{side}_hand.urdf"
    for mesh in ET.parse(urdf).findall(".//mesh"):
        path = Path(mesh.get("filename"))
        assert not path.is_absolute() and (urdf.parent / path).is_file(), path
    mjcf = Path(hand_cfg["model_path"](side))
    for mesh in ET.parse(mjcf).findall("asset/mesh"):
        path = Path(mesh.get("file"))
        assert not path.is_absolute() and (mjcf.parent / path).is_file(), path

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    assert (model.nq, model.nu, model.neq, model.nmesh) == (11, 6, 5, 12)
    assert tuple(validate_mujoco_actuator_mapping(model, hand_cfg)) == ACTUATED_JOINTS
    addresses = _joint_addresses(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert home_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    np.testing.assert_allclose(
        data.qpos, [neutral[names.index(name)] for name in addresses], atol=1e-12, rtol=0,
    )

    for channel, name in enumerate(ACTUATED_JOINTS):
        pulse = np.zeros(11)
        pulse[names.index(name)] = 0.25
        expected = np.zeros(6)
        expected[channel] = 0.25
        np.testing.assert_allclose(map_retarget_qpos(pulse, hand_cfg, names), expected)
        # Deliberately permute Pinocchio order to guard against positional maps.
        np.testing.assert_allclose(map_retarget_qpos(pulse[::-1], hand_cfg, names[::-1]), expected)

        mujoco.mj_resetDataKeyframe(model, data, home_id)
        target = data.ctrl.copy()
        target[channel] += 0.2
        data.ctrl[:] = target
        for _ in range(1000):
            mujoco.mj_step(model, data)
        actual = [data.qpos[addresses[joint]] for joint in ACTUATED_JOINTS]
        np.testing.assert_allclose(actual, target, atol=1e-4, rtol=0)
        _check_mimics({name: data.qpos[index] for name, index in addresses.items()}, side, 1e-5)

    for fraction in (0.0, 0.5, 1.0):
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        target = model.actuator_ctrlrange[:, 0] + fraction * np.diff(model.actuator_ctrlrange, axis=1)[:, 0]
        data.ctrl[:] = target
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        actual = [data.qpos[addresses[joint]] for joint in ACTUATED_JOINTS]
        np.testing.assert_allclose(actual, target, atol=1e-4, rtol=0)
        _check_mimics({name: data.qpos[index] for name, index in addresses.items()}, side, 1e-5)

    # Check every link, including the fixed base offset and five fixed tips.
    link_names = [link.get("name") for link in ET.parse(urdf).findall("link")]
    max_position_error = max_rotation_error = 0.0
    poses = [neutral]
    for index in optimizer.independent_indices:
        qpos = neutral.copy()
        qpos[index] += 0.3
        poses.append(optimizer.expand_to_full_qpos(qpos[optimizer.independent_indices]))
    rng = np.random.default_rng(6)
    for _ in range(8):
        independent = optimizer.independent_indices
        poses.append(optimizer.expand_to_full_qpos(rng.uniform(
            optimizer.opt_lower_bounds[independent], optimizer.opt_upper_bounds[independent],
        )))
    for pose in poses:
        robot.compute_forward_kinematics(pose)
        for name, index in addresses.items():
            data.qpos[index] = pose[names.index(name)]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for name in link_names:
            pin_pose = robot.get_link_pose(robot.get_link_index(name))
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert body_id >= 0, name
            position_error = np.linalg.norm(pin_pose[:3, 3] - data.xpos[body_id])
            rotation = pin_pose[:3, :3].T @ data.xmat[body_id].reshape(3, 3)
            rotation_error = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
            max_position_error = max(max_position_error, position_error)
            max_rotation_error = max(max_rotation_error, rotation_error)
    assert max_position_error < 1e-6, max_position_error
    assert max_rotation_error < 1e-5, max_rotation_error
    print(f"[{side}] assets, six controls, five mimics, limits and FK passed "
          f"(position={max_position_error:.3e} m, rotation={max_rotation_error:.3e} rad)")


def verify_configs() -> None:
    configs = sorted((PROJECT_ROOT / "example/config").glob("*/*/*_linker_l6.yaml"))
    kinds = ("adaptive", "vector", "dexpilot")
    inputs = ("avp", "mediapipe", "noitom", "pico4", "pico_umetrack", "quest3")
    expected = {PROJECT_ROOT / f"example/config/{kind}/{source}/{source}_linker_l6.yaml"
                for kind in kinds for source in inputs}
    assert set(configs) == expected, ("Missing configs", expected - set(configs),
                                     "Unexpected configs", set(configs) - expected)
    hand_cfg = ROBOT_HAND_CONFIGS["linker_l6"]
    for side in ("right", "left"):
        for config in configs:
            retargeter = Retargeter.from_yaml(str(config), side)
            optimizer = retargeter.optimizer
            assert optimizer.robot.model.name == f"Linker{side.title()}Hand"
            outputs = []
            for flexion in (0.0, 0.5, 1.0):
                keypoints = hand_keypoints(side, flexion)
                qpos = retargeter.retarget(keypoints)
                assert qpos.shape == (11,) and np.all(np.isfinite(qpos))
                assert np.all(qpos >= optimizer.opt_lower_bounds - 1e-6)
                assert np.all(qpos <= optimizer.opt_upper_bounds + 1e-6)
                _check_mimics(dict(zip(optimizer.robot.dof_joint_names, qpos)), side, 1e-6)
                outputs.append(qpos)
                target = retarget_to_mujoco_target(
                    {f"{side}_fingers": keypoints}, side, retargeter, hand_cfg, 6,
                )
                assert target.shape == (6,) and np.all(np.isfinite(target))
            assert np.linalg.norm(outputs[-1] - outputs[0]) > 0.01, config
        print(f"[{side}] all {len(configs)} configs solved open/curl inputs with feasible coupled output")


def main() -> None:
    for side in ("right", "left"):
        verify_side(side)
    verify_configs()
    print("Linker L6 simulation verification PASSED")


if __name__ == "__main__":
    main()
