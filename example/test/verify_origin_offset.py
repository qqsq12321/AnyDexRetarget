#!/usr/bin/env python3
"""Regression checks for optimizer wrist alignment (no viewer or live input).

Run from the repository root:
    python example/test/verify_origin_offset.py
"""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for search_path in (PROJECT_ROOT, PROJECT_ROOT / "example", Path(__file__).resolve().parent):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from anydexretarget import Retargeter
from debug_skeleton import (
    build_pinch_scaled_skeleton,
    build_raw_skeleton,
    build_scaled_skeleton,
    estimate_pinocchio_to_mujoco_transform,
    get_robot_fk_skeleton,
)
from debug_skeleton_vector import build_vector_scaled_skeleton
from verify_linker_l6_mapping import ASSETS, _check_mimics, hand_keypoints

KINDS = ("adaptive", "vector", "dexpilot")
OFFSET = np.array([0.009, -0.013, 0.02])


def load_config(kind, robot_type="linker_l6"):
    path = PROJECT_ROOT / f"example/config/{kind}/pico4/pico4_{robot_type}.yaml"
    with path.open() as stream:
        config = yaml.safe_load(stream)
    config["robot"].pop("origin_offset", None)
    config["retarget"].pop("target_offset", None)
    return config


def optimizer_with_offset(config, side, offset):
    config = deepcopy(config)
    config["robot"]["origin_offset"] = offset
    return Retargeter(config, side).optimizer


def point_layout(optimizer, kind):
    if kind == "vector":
        return (optimizer._kv_computed_link_names, optimizer._kv_computed_link_indices,
                optimizer._kv_computed_link_offsets)
    indices = optimizer.computed_link_ids if kind == "dexpilot" else optimizer.computed_link_indices
    return optimizer.computed_link_names, indices, optimizer.computed_link_offsets


def objective_for(optimizer, kind, keypoints):
    if kind == "adaptive":
        alphas = optimizer._compute_pinch_alpha(keypoints)
        targets = optimizer._compute_adaptive_targets(keypoints, alphas)
        return optimizer._get_objective_analytical(*targets, alphas, None)
    if kind == "vector":
        targets = optimizer._compute_target_vectors(keypoints)
        contact = optimizer._compute_contact_alpha(keypoints) if optimizer._has_contact else None
        return optimizer._get_objective(targets, None, contact)
    targets = optimizer._extract_target_vectors(keypoints)
    optimizer._update_projection_state(targets)
    references = optimizer._compute_reference_vectors(targets)

    def objective(x, grad):
        cost, derivative = optimizer._compute_objective_and_gradient(x, references)
        if grad.size:
            grad[:] = derivative
        return cost

    return objective


def verify_zero_default():
    for robot_type in ("linker_l6", "shadow_hand"):
        for side in ("right", "left"):
            for kind in KINDS:
                config = load_config(kind, robot_type)
                default = Retargeter(deepcopy(config), side).optimizer
                zero = optimizer_with_offset(config, side, [0.0, 0.0, 0.0])
                np.testing.assert_array_equal(default.origin_offset, np.zeros(3))
                for flexion in (0.0, 0.5, 1.0):
                    keypoints = hand_keypoints(side, flexion)
                    q_default = default.solve(keypoints)
                    q_zero = zero.solve(keypoints)
                    np.testing.assert_array_equal(q_default, q_zero)
                    assert default.compute_cost(q_default, keypoints) == zero.compute_cost(q_zero, keypoints)
    print("Missing and explicit zero offsets are identical for L6/Shadow, both hands, all optimizers")


def verify_actual_offset():
    for side in ("right", "left"):
        for kind in KINDS:
            config = load_config(kind)
            original = optimizer_with_offset(config, side, [0.0, 0.0, 0.0])
            shifted = optimizer_with_offset(config, side, OFFSET)
            qpos = original.neutral_qpos.copy()
            names, indices, offsets = point_layout(original, kind)
            points = original.robot.compute_points_batch(qpos, indices, offsets)
            shifted_names, shifted_indices, shifted_offsets = point_layout(shifted, kind)
            assert names == shifted_names
            actual = shifted.robot.compute_points_batch(qpos, shifted_indices, shifted_offsets)
            pose = shifted.robot.get_link_pose(shifted.robot.get_link_index(shifted.origin_link_name))
            delta = pose[:3, :3] @ OFFSET
            expected = points.copy()
            expected[[name == original.origin_link_name for name in names]] += delta
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
            np.testing.assert_allclose(shifted.get_origin_position(), pose[:3, 3] + delta, atol=1e-14)
            np.testing.assert_allclose(
                shifted.finger_root_vectors, original.finger_root_vectors - delta * 100, atol=1e-12,
            )

            keypoints = hand_keypoints(side, 0.7)
            cost0 = original.compute_cost(qpos, keypoints)
            cost1 = shifted.compute_cost(qpos, keypoints)
            assert abs(cost1 - cost0) > 1e-3, (side, kind, cost0, cost1)
            q0 = original.solve(keypoints)
            q1 = shifted.solve(keypoints)
            assert np.max(np.abs(q1 - q0)) > 1e-3, (side, kind)
            assert np.all(q1 >= shifted.opt_lower_bounds - 1e-6)
            assert np.all(q1 <= shifted.opt_upper_bounds + 1e-6)
            _check_mimics(dict(zip(shifted.robot.dof_joint_names, q1)), side, 1e-6)
    print("Nonzero offset changes the real loss and solved joints; physical FK, limits and mimics stay valid")


def verify_gradients():
    max_error = 0.0
    for side in ("right", "left"):
        for kind in KINDS:
            # A moving wrist frame also checks the offset's rotational Jacobian.
            for origin_link in ("base_link", "thumb_prox_base"):
                config = load_config(kind)
                config["robot"]["origin_link"] = origin_link
                if kind == "vector":
                    for vector in config["retarget"]["key_vectors"]:
                        vector["origin"] = origin_link
                optimizer = optimizer_with_offset(config, side, OFFSET)
                lower = optimizer.opt_lower_bounds[optimizer.independent_indices]
                upper = optimizer.opt_upper_bounds[optimizer.independent_indices]
                x = lower + 0.42 * (upper - lower)
                objective = objective_for(optimizer, kind, hand_keypoints(side, 0.7))
                analytical = np.empty_like(x)
                objective(x, analytical)
                numerical = np.empty_like(x)
                for index in range(len(x)):
                    step = np.zeros_like(x)
                    step[index] = 1e-6
                    numerical[index] = (objective(x + step, np.empty(0))
                                        - objective(x - step, np.empty(0))) / 2e-6
                np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-5,
                                           err_msg=f"{side} {kind} {origin_link}")
                max_error = max(max_error, float(np.max(np.abs(analytical - numerical))))
                qpos = optimizer.expand_to_full_qpos(x)
                pose = optimizer.robot.get_link_pose(optimizer.robot.get_link_index(origin_link))
                np.testing.assert_allclose(
                    optimizer.get_origin_position(), pose[:3, 3] + pose[:3, :3] @ OFFSET,
                    atol=1e-14, rtol=0,
                )
    print(f"Offset/mimic analytical gradients match finite differences (max error {max_error:.3e})")


def verify_skeleton_alignment():
    import mujoco

    for side in ("right", "left"):
        for kind in KINDS:
            config = load_config(kind)
            original = optimizer_with_offset(config, side, [0.0, 0.0, 0.0])
            shifted = optimizer_with_offset(config, side, OFFSET)
            keypoints = hand_keypoints(side, 0.7)
            qpos = original.neutral_qpos.copy()
            red0, _ = get_robot_fk_skeleton(original, qpos)
            red1, _ = get_robot_fk_skeleton(shifted, qpos)
            np.testing.assert_allclose(red1[0], red0[0] + OFFSET, atol=1e-14)
            np.testing.assert_array_equal(red1[1:], red0[1:])
            builders = [build_raw_skeleton, build_pinch_scaled_skeleton, build_scaled_skeleton]
            if kind == "vector":
                builders.append(build_vector_scaled_skeleton)
            for builder in builders:
                points0, edges0 = builder(keypoints, original)
                points1, edges1 = builder(keypoints, shifted)
                np.testing.assert_allclose(points1, points0 + OFFSET, atol=1e-14)
                np.testing.assert_allclose(points1[0], shifted.get_origin_position(), atol=1e-14)
                assert edges0 == edges1
            model = mujoco.MjModel.from_xml_path(str(ASSETS / f"linker_l6_{side}_mujoco.xml"))
            data = mujoco.MjData(model)
            r0, t0, _ = estimate_pinocchio_to_mujoco_transform(original, model, data)
            r1, t1, _ = estimate_pinocchio_to_mujoco_transform(shifted, model, data)
            np.testing.assert_array_equal(r1, r0)
            np.testing.assert_array_equal(t1, t0)
    print("Skeleton roots/targets use the optimizer reference; physical MuJoCo alignment is unchanged")


def verify_invalid_offsets():
    config = load_config("adaptive")
    invalid = (None, 0.02, [0, 0], [0, 0, 0, 0], [[0, 0, 0]],
               [0, np.nan, 0], [0, 0, np.inf], ["invalid", 0, 0], {"z": 0.02})
    for value in invalid:
        try:
            optimizer_with_offset(config, "right", value)
        except ValueError as exc:
            assert "robot.origin_offset" in str(exc), str(exc)
        else:
            raise AssertionError(f"Invalid offset was accepted: {value!r}")
    print("Invalid origin offsets fail with a clear configuration error")


def main():
    verify_zero_default()
    verify_actual_offset()
    verify_gradients()
    verify_skeleton_alignment()
    verify_invalid_offsets()
    print("Retargeting origin offset verification PASSED")


if __name__ == "__main__":
    main()
