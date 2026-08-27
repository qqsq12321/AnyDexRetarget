from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from anydexretarget.mediapipe import apply_mediapipe_transformations
from anydexretarget.optimizer import (
    AdaptiveLPFilterV2,
    AdaptiveOptimizerAnalyticalV2,
    BaseOptimizer,
)
from anydexretarget.retarget import Retargeter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "example/config/adaptive/pico4/pico4_inspire_hand.yaml"
CONFIG_V2_PATH = PROJECT_ROOT / "example/config/adaptive/pico4/pico4_inspire_hand_v2.yaml"
RECORDING_PATH = PROJECT_ROOT / "example/data/avp1.pkl"


def _v2_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["optimizer"]["type"] = "AdaptiveOptimizerAnalyticalV2"
    config["retarget"]["contact_v2"] = {
        "weight": 160.0,
        "huber_delta": 0.5,
        "enabled_fingers": ["index", "middle"],
        "min_beta": 0.8,
        "target_distances_cm": {
            "index": 0.2,
            "middle": 1.25,
            "ring": 3.25,
            "pinky": 5.3,
        },
    }
    return config


def _direct_control_config() -> dict:
    config = _v2_config()
    config["retarget"]["contact_v2"]["enabled_fingers"] = [
        "index",
        "middle",
        "ring",
        "pinky",
    ]
    config["retarget"]["direct_control_v2"] = {
        "finger_blend": 1.0,
        "thumb_blend": 1.0,
        "finger_curl_open_rad": 0.1,
        "finger_curl_closed_rad": 2.8,
        "thumb_yaw_open_ratio": -2.05,
        "thumb_yaw_opposed_ratio": 0.1,
        "thumb_curl_open_rad": 0.13,
        "thumb_curl_closed_rad": 1.69,
    }
    return config


def _synthetic_open_hand() -> np.ndarray:
    keypoints = np.zeros((21, 3), dtype=np.float64)
    roots = {
        1: (-2.0, 0.0, 0.0),
        5: (-1.5, 0.0, 0.0),
        9: (-0.5, 0.0, 0.0),
        13: (0.5, 0.0, 0.0),
        17: (1.5, 0.0, 0.0),
    }
    for root_index, root in roots.items():
        root = np.asarray(root, dtype=np.float64)
        for offset in range(4):
            keypoints[root_index + offset] = root + np.array([0.0, 0.0, offset])
    return keypoints


def _recorded_pose(frame_index: int) -> np.ndarray:
    # This is a trusted, version-controlled project fixture.
    with RECORDING_PATH.open("rb") as handle:
        frames = pickle.load(handle)
    raw_keypoints = np.asarray(frames[frame_index]["right_fingers"], dtype=np.float64)
    keypoints = apply_mediapipe_transformations(raw_keypoints, "right")
    rotation = Rotation.from_euler("xyz", [-7.0, 0.0, -20.0], degrees=True)
    return keypoints @ rotation.as_matrix().T


def _tip_distance_mm(optimizer, qpos: np.ndarray, finger_index: int) -> float:
    positions = optimizer.robot.compute_points_batch(
        qpos,
        optimizer.computed_link_indices,
        optimizer.computed_link_offsets,
    )
    thumb = positions[optimizer.task_indices[0]]
    finger = positions[optimizer.task_indices[finger_index]]
    return float(np.linalg.norm(finger - thumb) * 1000.0)


def test_v2_optimizer_is_available_from_factory() -> None:
    optimizer = BaseOptimizer.from_config(_v2_config())

    assert isinstance(optimizer, AdaptiveOptimizerAnalyticalV2)


def test_pico_inspire_v2_config_selects_v2_optimizer_and_filter() -> None:
    retargeter = Retargeter.from_yaml(str(CONFIG_V2_PATH), hand_side="right")

    assert isinstance(retargeter.optimizer, AdaptiveOptimizerAnalyticalV2)
    assert isinstance(retargeter.lp_filter, AdaptiveLPFilterV2)


def test_pico_inspire_v2_config_enables_all_pinch_partners() -> None:
    retargeter = Retargeter.from_yaml(str(CONFIG_V2_PATH), hand_side="right")

    assert retargeter.optimizer.contact_enabled_partners_v2 == (1, 2, 3, 4)


def test_v2_direct_targets_keep_single_finger_control_independent() -> None:
    optimizer = AdaptiveOptimizerAnalyticalV2(_direct_control_config())
    keypoints = _synthetic_open_hand()
    keypoints[6] = [-1.5, 0.0, 1.0]
    keypoints[7] = [-1.5, 1.0, 1.0]
    keypoints[8] = [-1.5, 1.0, 0.0]

    target = optimizer._compute_direct_targets_v2(keypoints)

    assert target[0] > 1.3
    np.testing.assert_allclose(target[[2, 4, 6]], 0.0, atol=1e-6)


def test_v2_direct_thumb_yaw_tracks_lateral_motion() -> None:
    optimizer = AdaptiveOptimizerAnalyticalV2(_direct_control_config())
    open_hand = _synthetic_open_hand()
    open_hand[4] = [-6.15, 0.0, 3.0]
    opposed_hand = open_hand.copy()
    opposed_hand[4] = [0.3, 0.0, 3.0]

    open_target = optimizer._compute_direct_targets_v2(open_hand)
    opposed_target = optimizer._compute_direct_targets_v2(opposed_hand)

    assert open_target[8] < 0.1
    assert opposed_target[8] > 1.2


def test_v2_non_pinch_output_uses_independent_direct_channels() -> None:
    optimizer = AdaptiveOptimizerAnalyticalV2(_direct_control_config())
    keypoints = _synthetic_open_hand()
    keypoints[6] = [-1.5, 0.0, 1.0]
    keypoints[7] = [-1.5, 1.0, 1.0]
    keypoints[8] = [-1.5, 1.0, 0.0]

    expected = optimizer._compute_direct_targets_v2(keypoints)
    result = optimizer.solve(keypoints)

    np.testing.assert_allclose(
        result[optimizer.independent_indices],
        expected[optimizer.independent_indices],
        atol=1e-6,
    )


def test_v2_contact_gradient_matches_finite_difference() -> None:
    optimizer = AdaptiveOptimizerAnalyticalV2(_v2_config())
    optimizer.set_timing_enabled(False)
    keypoints = _recorded_pose(240)
    alphas = optimizer._compute_pinch_alpha(keypoints)
    optimizer._prepare_contact_v2(keypoints, alphas)

    qpos = optimizer.expand_to_full_qpos(
        (optimizer.opt_lower_bounds[optimizer.independent_indices]
         + optimizer.opt_upper_bounds[optimizer.independent_indices]) / 2.0
    )
    positions = optimizer.robot.compute_points_batch(
        qpos, optimizer.computed_link_indices, optimizer.computed_link_offsets
    ) * 100.0
    jacobians = optimizer.robot.compute_all_jacobians_batch_with_offsets(
        qpos, optimizer.computed_link_indices, optimizer.computed_link_offsets
    ) * 100.0
    _, analytical = optimizer._contact_loss_and_grad_v2(qpos, positions, jacobians)

    epsilon = 1e-6
    for joint_index in optimizer.independent_indices:
        q_plus = qpos.copy()
        q_minus = qpos.copy()
        q_plus[joint_index] += epsilon
        q_minus[joint_index] -= epsilon

        def contact_loss(q: np.ndarray) -> float:
            points = optimizer.robot.compute_points_batch(
                q, optimizer.computed_link_indices, optimizer.computed_link_offsets
            ) * 100.0
            jac = optimizer.robot.compute_all_jacobians_batch_with_offsets(
                q, optimizer.computed_link_indices, optimizer.computed_link_offsets
            ) * 100.0
            return optimizer._contact_loss_and_grad_v2(q, points, jac)[0]

        numerical = (contact_loss(q_plus) - contact_loss(q_minus)) / (2.0 * epsilon)
        assert np.isclose(analytical[joint_index], numerical, rtol=2e-3, atol=2e-3)


def test_v2_contact_ignores_disabled_ring_pinch() -> None:
    optimizer = AdaptiveOptimizerAnalyticalV2(_v2_config())
    keypoints = np.zeros((21, 3), dtype=np.float64)

    optimizer._prepare_contact_v2(
        keypoints,
        np.array([1.0, 0.0, 0.0, 1.0, 0.0]),
    )

    assert optimizer._contact_partner_v2 is None


def test_v2_contact_waits_for_an_unambiguous_pinch() -> None:
    optimizer = AdaptiveOptimizerAnalyticalV2(_v2_config())
    keypoints = np.zeros((21, 3), dtype=np.float64)

    optimizer._prepare_contact_v2(
        keypoints,
        np.array([0.75, 0.75, 0.0, 0.0, 0.0]),
    )

    assert optimizer._contact_partner_v2 is None


def test_v2_contact_rejects_two_equally_close_partners() -> None:
    config = _direct_control_config()
    config["retarget"]["contact_v2"]["dominance_margin"] = 0.15
    optimizer = AdaptiveOptimizerAnalyticalV2(config)
    keypoints = np.zeros((21, 3), dtype=np.float64)

    optimizer._prepare_contact_v2(
        keypoints,
        np.array([1.0, 1.0, 0.95, 0.0, 0.0]),
    )

    assert optimizer._contact_partner_v2 is None


def test_v2_contact_strength_ramps_after_activation_threshold() -> None:
    optimizer = AdaptiveOptimizerAnalyticalV2(_v2_config())
    keypoints = np.zeros((21, 3), dtype=np.float64)

    optimizer._prepare_contact_v2(
        keypoints,
        np.array([0.9, 0.9, 0.0, 0.0, 0.0]),
    )

    assert np.isclose(optimizer._contact_beta_v2, 0.5)


def test_v2_closes_recorded_index_pinch() -> None:
    keypoints = _recorded_pose(240)
    v1 = BaseOptimizer.from_yaml(str(CONFIG_PATH), hand_side="right")
    v2 = AdaptiveOptimizerAnalyticalV2(_v2_config())
    v1.set_timing_enabled(False)
    v2.set_timing_enabled(False)

    distance_v1 = _tip_distance_mm(v1, v1.solve(keypoints), 1)
    distance_v2 = _tip_distance_mm(v2, v2.solve(keypoints), 1)

    assert distance_v1 > 20.0
    assert distance_v2 < 4.0
    assert distance_v2 < distance_v1 * 0.2


def test_v2_projects_recorded_middle_pinch_to_reachable_distance() -> None:
    keypoints = _recorded_pose(1480)
    v1 = BaseOptimizer.from_yaml(str(CONFIG_PATH), hand_side="right")
    v2 = AdaptiveOptimizerAnalyticalV2(_v2_config())
    v1.set_timing_enabled(False)
    v2.set_timing_enabled(False)

    distance_v1 = _tip_distance_mm(v1, v1.solve(keypoints), 2)
    distance_v2 = _tip_distance_mm(v2, v2.solve(keypoints), 2)

    assert distance_v1 > 20.0
    assert distance_v2 < 14.0
    assert distance_v2 < distance_v1 * 0.6


def test_pico_inspire_v2_config_closes_all_recorded_pinch_partners() -> None:
    samples = {
        1519: (1, 5.0),
        1474: (2, 13.0),
        1426: (3, 33.0),
        1370: (4, 53.0),
    }
    for frame_index, (finger_index, maximum_distance_mm) in samples.items():
        optimizer = BaseOptimizer.from_yaml(str(CONFIG_V2_PATH), hand_side="right")
        optimizer.set_timing_enabled(False)

        distance = _tip_distance_mm(
            optimizer,
            optimizer.solve(_recorded_pose(frame_index)),
            finger_index,
        )

        assert distance < maximum_distance_mm


def test_adaptive_filter_v2_bypasses_motion_and_smooths_micro_jitter() -> None:
    filter_v2 = AdaptiveLPFilterV2(
        alpha=0.5,
        bypass_ratio=0.008,
        joint_ranges=np.array([1.0, 2.0]),
    )

    np.testing.assert_allclose(filter_v2.next(np.array([0.0, 0.0])), [0.0, 0.0])
    np.testing.assert_allclose(
        filter_v2.next(np.array([0.020, 0.010])),
        [0.020, 0.005],
    )


def test_adaptive_filter_v2_passes_thumb_channels_without_lag() -> None:
    filter_v2 = AdaptiveLPFilterV2(
        alpha=0.5,
        bypass_ratio=0.008,
        joint_ranges=np.ones(3),
        passthrough_mask=np.array([False, True, True]),
    )
    filter_v2.next(np.zeros(3))

    output = filter_v2.next(np.array([0.004, 0.004, -0.004]))

    np.testing.assert_allclose(output, [0.002, 0.004, -0.004])


def test_retargeter_v2_filter_preserves_mimic_constraints() -> None:
    retargeter = Retargeter.from_yaml(str(CONFIG_V2_PATH), hand_side="right")
    optimizer = retargeter.optimizer
    independent = np.zeros(optimizer.num_opt_vars, dtype=np.float64)
    first = optimizer.expand_to_full_qpos(independent)
    retargeter._apply_filter_v2(first)

    independent[optimizer.independent_indices.tolist().index(9)] = 0.01
    second = retargeter._apply_filter_v2(
        optimizer.expand_to_full_qpos(independent)
    )

    for mimic_index, (source_index, multiplier, offset) in optimizer.mimic_map.items():
        assert np.isclose(
            second[mimic_index],
            second[source_index] * multiplier + offset,
        )
