#!/usr/bin/env python3
"""Verify whole-target alignment without a viewer or live input.

Run from the repository root:
    python example/test/verify_target_offset.py
"""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for search_path in (PROJECT_ROOT, PROJECT_ROOT / "example", Path(__file__).resolve().parent):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from anydexretarget import Retargeter
from debug_skeleton import (
    build_pinch_scaled_skeleton,
    build_raw_skeleton,
    build_scaled_skeleton,
    get_robot_fk_skeleton,
)
from debug_skeleton_vector import build_vector_scaled_skeleton
from verify_linker_l6_mapping import _check_mimics, hand_keypoints
from verify_origin_offset import KINDS, load_config, objective_for, point_layout

OFFSET = np.array([0.009, -0.013, 0.02])


def optimizer_with_offset(config, side, offset):
    config = deepcopy(config)
    config["retarget"]["target_offset"] = offset
    return Retargeter(config, side).optimizer


def verify_zero_default():
    for robot_type in ("linker_l6", "shadow_hand"):
        for side in ("right", "left"):
            for kind in KINDS:
                config = load_config(kind, robot_type)
                default = Retargeter(deepcopy(config), side).optimizer
                zero = optimizer_with_offset(config, side, [0.0, 0.0, 0.0])
                np.testing.assert_array_equal(default.target_offset, np.zeros(3))
                for flexion in (0.0, 0.5, 1.0):
                    keypoints = hand_keypoints(side, flexion)
                    q0 = default.solve(keypoints)
                    q1 = zero.solve(keypoints)
                    np.testing.assert_array_equal(q0, q1)
                    assert default.compute_cost(q0, keypoints) == zero.compute_cost(q1, keypoints)
    print("Missing/zero target offsets preserve L6/Shadow results for both hands and all optimizers")


def verify_adaptive_geometry():
    for side in ("right", "left"):
        config = load_config("adaptive")
        config["retarget"]["lateral_scaling"] = 0.7
        config["retarget"]["pinch_scaling"] = 1.7
        config["retarget"]["segment_scaling"] = {
            name: [0.8 + i * 0.1, 1.2, 1.4, 0.9]
            for i, name in enumerate(("thumb", "index", "middle", "ring", "pinky"))
        }
        original = optimizer_with_offset(config, side, [0, 0, 0])
        shifted = optimizer_with_offset(config, side, OFFSET)
        for flexion in (0.0, 0.7, 1.0):
            keypoints = hand_keypoints(side, flexion)
            before = keypoints.copy()
            alphas = original._compute_pinch_alpha(keypoints)
            np.testing.assert_array_equal(alphas, shifted._compute_pinch_alpha(keypoints))
            for weights in (alphas, np.full(original.num_fingers, original.pinch_alpha_max)):
                tip0, dirs0, full0 = original._compute_adaptive_targets(keypoints, weights)
                tip1, dirs1, full1 = shifted._compute_adaptive_targets(keypoints, weights)
                np.testing.assert_allclose(tip1, tip0 + OFFSET * 100, atol=1e-12)
                np.testing.assert_allclose(full1, full0 + OFFSET * 100, atol=1e-12)
                np.testing.assert_array_equal(dirs1, dirs0)
                translated = shifted._compute_adaptive_targets(keypoints + [0.1, -0.2, 0.3], weights)
                np.testing.assert_allclose(translated[0], tip1, atol=1e-12)
                np.testing.assert_allclose(translated[2], full1, atol=1e-12)
            np.testing.assert_array_equal(keypoints, before)
            green0, edges0 = build_scaled_skeleton(keypoints, original)
            green1, edges1 = build_scaled_skeleton(keypoints, shifted)
            np.testing.assert_allclose(green1, green0 + OFFSET, atol=1e-14)
            assert edges0 == edges1
            for start, end in edges1:
                np.testing.assert_allclose(
                    green1[end] - green1[start], green0[end] - green0[start], atol=1e-14,
                )
            indices = shifted.MP_PIP_INDICES + shifted.MP_DIP_INDICES + shifted.MP_TIP_INDICES
            actual_targets = shifted._compute_full_hand_vectors(keypoints, shifted.segment_scaling_full,
                                                                shifted.lateral_scaling, shifted.lateral_axis)
            np.testing.assert_allclose(
                (green1[indices] - shifted.get_origin_position()) * 100, actual_targets, atol=1e-12,
            )
    print("All green points translate rigidly after scaling; real full-hand/pinch targets match")


def verify_vector_and_dexpilot_targets():
    for side in ("right", "left"):
        config = load_config("vector")
        first, second = deepcopy(config["retarget"]["key_vectors"][:2])
        config["retarget"]["scaling_factor"] = 1.7
        config["retarget"]["key_vectors"] = [first, second, {
            "origin": first["task"], "task": second["task"],
            "origin_kp": first["task_kp"], "task_kp": second["task_kp"],
        }, {
            "origin": first["task"], "task": first["origin"],
            "origin_kp": first["task_kp"], "task_kp": first["origin_kp"],
        }]
        original = optimizer_with_offset(config, side, [0, 0, 0])
        shifted = optimizer_with_offset(config, side, OFFSET)
        keypoints = hand_keypoints(side, 0.7)
        vectors0 = original._compute_target_vectors(keypoints)
        vectors1 = shifted._compute_target_vectors(keypoints)
        expected_delta = np.array([1, 1, 0, -1])[:, None] * OFFSET * 100
        np.testing.assert_allclose(vectors1, vectors0 + expected_delta, atol=1e-12)

        config = load_config("dexpilot")
        config["retarget"]["scaling"] = 1.7
        original = optimizer_with_offset(config, side, [0, 0, 0])
        shifted = optimizer_with_offset(config, side, OFFSET)
        keypoints[8] = keypoints[4] + [0.001, 0, 0]
        raw0 = original._extract_target_vectors(keypoints)
        raw1 = shifted._extract_target_vectors(keypoints)
        np.testing.assert_array_equal(raw1, raw0)
        original._update_projection_state(raw0)
        shifted._update_projection_state(raw1)
        np.testing.assert_array_equal(original.projected, shifted.projected)
        references0 = original._compute_reference_vectors(raw0)
        references1 = shifted._compute_reference_vectors(raw1)
        np.testing.assert_array_equal(references1[:original.num_proj], references0[:original.num_proj])
        np.testing.assert_allclose(references1[original.num_proj:],
                                   references0[original.num_proj:] + OFFSET * 100, atol=1e-12)
    print("Wrist targets shift once; finger-to-finger vectors and pinch projections stay unchanged")


def verify_fk_and_solver():
    for side in ("right", "left"):
        for kind in KINDS:
            config = load_config(kind)
            original = optimizer_with_offset(config, side, [0, 0, 0])
            shifted = optimizer_with_offset(config, side, OFFSET)
            qpos = original.neutral_qpos.copy()
            red0, _ = get_robot_fk_skeleton(original, qpos)
            red1, _ = get_robot_fk_skeleton(shifted, qpos)
            np.testing.assert_array_equal(red1, red0)
            np.testing.assert_array_equal(shifted.origin_offset, np.zeros(3))
            np.testing.assert_array_equal(shifted.finger_root_vectors, original.finger_root_vectors)
            names0, ids0, offsets0 = point_layout(original, kind)
            names1, ids1, offsets1 = point_layout(shifted, kind)
            assert names0 == names1 and list(ids0) == list(ids1)
            np.testing.assert_array_equal(offsets1, offsets0)

            keypoints = hand_keypoints(side, 0.7)
            raw0, _ = build_raw_skeleton(keypoints, original)
            raw1, _ = build_raw_skeleton(keypoints, shifted)
            np.testing.assert_array_equal(raw1, raw0)
            builders = [build_scaled_skeleton, build_pinch_scaled_skeleton]
            if kind == "vector":
                builders.append(build_vector_scaled_skeleton)
            for builder in builders:
                points0, edges0 = builder(keypoints, original)
                points1, edges1 = builder(keypoints, shifted)
                np.testing.assert_allclose(points1, points0 + OFFSET, atol=1e-14)
                assert edges0 == edges1
            cost0 = original.compute_cost(qpos, keypoints)
            cost1 = shifted.compute_cost(qpos, keypoints)
            assert abs(cost1 - cost0) > 1e-3, (side, kind, cost0, cost1)
            q0 = original.solve(keypoints)
            q1 = shifted.solve(keypoints)
            assert np.max(np.abs(q1 - q0)) > 1e-3, (side, kind)
            assert np.all(q1 >= shifted.opt_lower_bounds - 1e-6)
            assert np.all(q1 <= shifted.opt_upper_bounds + 1e-6)
            _check_mimics(dict(zip(shifted.robot.dof_joint_names, q1)), side, 1e-6)
    print("Robot FK/reference stays unchanged at equal q; target offset changes real losses/solutions")


def verify_gradients():
    max_error = 0.0
    for side in ("right", "left"):
        for kind in KINDS:
            optimizer = optimizer_with_offset(load_config(kind), side, OFFSET)
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
                                       err_msg=f"{side} {kind}")
            max_error = max(max_error, float(np.max(np.abs(analytical - numerical))))
    print(f"Target-offset gradients match finite differences (max error {max_error:.3e})")


def verify_live_config():
    for kind, builder in (("adaptive", build_scaled_skeleton),
                          ("vector", build_vector_scaled_skeleton)):
        path = PROJECT_ROOT / f"example/config/{kind}/pico4/pico4_linker_l6.yaml"
        for side in ("right", "left"):
            retargeter = Retargeter.from_yaml(str(path), side)
            shifted = retargeter.optimizer
            offset = np.asarray(retargeter.config["retarget"]["target_offset"], dtype=float)
            np.testing.assert_array_equal(shifted.origin_offset, np.zeros(3))
            np.testing.assert_array_equal(shifted.target_offset, offset)
            config = deepcopy(retargeter.config)
            config["retarget"]["target_offset"] = [0, 0, 0]
            original = Retargeter(config, side)
            raw = hand_keypoints(side, 0.7)
            q0, verbose0 = original.retarget_verbose(raw, apply_filter=False)
            q1, verbose1 = retargeter.retarget_verbose(raw, apply_filter=False)
            np.testing.assert_array_equal(verbose1["mediapipe_kp"], verbose0["mediapipe_kp"])
            green0, _ = builder(verbose0["mediapipe_kp"], original.optimizer)
            green1, _ = builder(verbose1["mediapipe_kp"], shifted)
            np.testing.assert_allclose(green1, green0 + offset, atol=1e-14)
            if kind == "vector":
                targets0 = original.optimizer._compute_target_vectors(verbose0["mediapipe_kp"])
                targets1 = shifted._compute_target_vectors(verbose1["mediapipe_kp"])
                np.testing.assert_allclose(targets1, targets0 + offset * 100, atol=1e-12)
            print(
                f"{kind} {side} live YAML/pipeline translates green targets by {offset.tolist()} m "
                f"(joint change {np.max(np.abs(q1 - q0)):.4f} rad)"
            )


def verify_invalid_offsets():
    invalid = (None, 0.02, [0, 0], [0, 0, 0, 0], [[0, 0, 0]],
               [0, np.nan, 0], [0, 0, np.inf], ["invalid", 0, 0], {"z": 0.02})
    for kind in KINDS:
        for value in invalid:
            try:
                optimizer_with_offset(load_config(kind), "right", value)
            except ValueError as exc:
                assert "retarget.target_offset" in str(exc), str(exc)
            else:
                raise AssertionError(f"Invalid target offset was accepted: {value!r}")
    print("Invalid target offsets fail with a clear configuration error")


def main():
    verify_zero_default()
    verify_adaptive_geometry()
    verify_vector_and_dexpilot_targets()
    verify_fk_and_solver()
    verify_gradients()
    verify_live_config()
    verify_invalid_offsets()
    print("Whole-target offset verification PASSED")


if __name__ == "__main__":
    main()
