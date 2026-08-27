"""Adaptive hand retargeting with explicit fingertip contact loss."""

from __future__ import annotations

import numpy as np

from .analytical_optimizer import AdaptiveOptimizerAnalytical
from .utils import M_TO_CM, huber_loss_grad_np, huber_loss_np


class AdaptiveOptimizerAnalyticalV2(AdaptiveOptimizerAnalytical):
    """Combine configured independent-joint tracking with pinch closure."""

    _NON_THUMB_NAMES = ("index", "middle", "ring", "pinky")

    def __init__(self, config: dict):
        super().__init__(config)
        contact_config = config.get("retarget", {}).get("contact_v2", {})
        self.contact_weight_v2 = float(contact_config.get("weight", 160.0))
        self.contact_huber_delta_v2 = float(
            contact_config.get("huber_delta", 0.5)
        )
        self.contact_min_beta_v2 = float(contact_config.get("min_beta", 0.8))
        self.contact_dominance_margin_v2 = float(
            contact_config.get("dominance_margin", 0.0)
        )

        distance_config = contact_config.get("target_distances_cm", {})
        defaults = {"index": 0.2, "middle": 1.25, "ring": 3.25, "pinky": 5.3}
        active_names = self._NON_THUMB_NAMES[: self.num_fingers - 1]
        enabled_names = contact_config.get("enabled_fingers", active_names)
        self.contact_enabled_partners_v2 = tuple(
            index + 1 for index, name in enumerate(active_names) if name in enabled_names
        )
        self.contact_target_distances_cm_v2 = np.array(
            [float(distance_config.get(name, defaults[name])) for name in active_names],
            dtype=np.float64,
        )

        self._contact_partner_v2: int | None = None
        self._contact_beta_v2 = 0.0
        self._contact_target_cm_v2 = 0.0

        direct_config = config.get("retarget", {}).get("direct_control_v2", {})
        self.direct_control_enabled_v2 = bool(direct_config)
        self.direct_finger_blend_v2 = float(direct_config.get("finger_blend", 0.0))
        self.direct_thumb_blend_v2 = float(direct_config.get("thumb_blend", 0.0))
        self.direct_pinch_finger_blend_v2 = float(
            direct_config.get("pinch_finger_blend", 0.0)
        )
        self.direct_pinch_thumb_blend_v2 = float(
            direct_config.get("pinch_thumb_blend", 0.0)
        )
        self.direct_finger_curl_open_v2 = float(
            direct_config.get("finger_curl_open_rad", 0.1)
        )
        self.direct_finger_curl_closed_v2 = float(
            direct_config.get("finger_curl_closed_rad", 2.8)
        )
        self.direct_thumb_yaw_open_v2 = float(
            direct_config.get("thumb_yaw_open_ratio", -2.05)
        )
        self.direct_thumb_yaw_opposed_v2 = float(
            direct_config.get("thumb_yaw_opposed_ratio", 0.1)
        )
        self.direct_thumb_curl_open_v2 = float(
            direct_config.get("thumb_curl_open_rad", 0.13)
        )
        self.direct_thumb_curl_closed_v2 = float(
            direct_config.get("thumb_curl_closed_rad", 1.69)
        )
        self.direct_only_v2 = bool(direct_config.get("direct_only", True))
        self._direct_joint_indices_v2 = self._resolve_direct_joint_indices_v2(
            direct_config.get("joint_names", {})
        )

    def _resolve_direct_joint_indices_v2(
        self,
        configured_names: dict,
    ) -> dict[str, tuple[int, ...]]:
        indices = {}
        for joint_index in range(1, self.robot.model.njoints):
            joint = self.robot.model.joints[joint_index]
            if joint.nq > 0:
                indices[self.robot.model.names[joint_index]] = joint.idx_q
        defaults = {
            "index": "index_proximal_joint",
            "middle": "middle_proximal_joint",
            "ring": "ring_proximal_joint",
            "pinky": "pinky_proximal_joint",
            "thumb_yaw": "thumb_proximal_yaw_joint",
            "thumb_pitch": "thumb_proximal_pitch_joint",
        }
        resolved = {}
        for role, default_name in defaults.items():
            joint_names = configured_names.get(role, default_name)
            if isinstance(joint_names, str):
                joint_names = [joint_names]
            resolved_indices = tuple(
                indices[joint_name]
                for joint_name in joint_names
                if joint_name in indices
            )
            if resolved_indices:
                resolved[role] = resolved_indices
        return resolved

    @staticmethod
    def _segment_angle_v2(first: np.ndarray, second: np.ndarray) -> float:
        first_norm = first / (np.linalg.norm(first) + 1e-8)
        second_norm = second / (np.linalg.norm(second) + 1e-8)
        return float(np.arccos(np.clip(first_norm @ second_norm, -1.0, 1.0)))

    @staticmethod
    def _normalized_range_v2(value: float, lower: float, upper: float) -> float:
        return float(np.clip((value - lower) / (upper - lower + 1e-8), 0.0, 1.0))

    def _compute_direct_targets_v2(self, keypoints: np.ndarray) -> np.ndarray:
        """Map independent human finger intent onto configured robot joints."""
        targets = np.zeros(self.num_joints, dtype=np.float64)
        finger_landmarks = {
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }
        for name, landmarks in finger_landmarks.items():
            qpos_indices = self._direct_joint_indices_v2.get(name, ())
            if not qpos_indices:
                continue
            mcp, pip, dip, tip = landmarks
            proximal = keypoints[pip] - keypoints[mcp]
            intermediate = keypoints[dip] - keypoints[pip]
            distal = keypoints[tip] - keypoints[dip]
            curl = self._segment_angle_v2(proximal, intermediate)
            curl += self._segment_angle_v2(intermediate, distal)
            amount = self._normalized_range_v2(
                curl,
                self.direct_finger_curl_open_v2,
                self.direct_finger_curl_closed_v2,
            )
            for qpos_index in qpos_indices:
                targets[qpos_index] = (
                    self.opt_lower_bounds[qpos_index]
                    + amount
                    * (
                        self.opt_upper_bounds[qpos_index]
                        - self.opt_lower_bounds[qpos_index]
                    )
                )

        thumb_yaw_indices = self._direct_joint_indices_v2.get("thumb_yaw", ())
        if thumb_yaw_indices:
            palm_lateral = keypoints[17] - keypoints[5]
            palm_width = float(np.linalg.norm(palm_lateral))
            lateral_unit = palm_lateral / (palm_width + 1e-8)
            thumb_lateral_ratio = float(
                ((keypoints[4] - keypoints[0]) @ lateral_unit)
                / (palm_width + 1e-8)
            )
            amount = self._normalized_range_v2(
                thumb_lateral_ratio,
                self.direct_thumb_yaw_open_v2,
                self.direct_thumb_yaw_opposed_v2,
            )
            for thumb_yaw_index in thumb_yaw_indices:
                targets[thumb_yaw_index] = (
                    self.opt_lower_bounds[thumb_yaw_index]
                    + amount
                    * (
                        self.opt_upper_bounds[thumb_yaw_index]
                        - self.opt_lower_bounds[thumb_yaw_index]
                    )
                )

        thumb_pitch_indices = self._direct_joint_indices_v2.get("thumb_pitch", ())
        if thumb_pitch_indices:
            proximal = keypoints[2] - keypoints[1]
            intermediate = keypoints[3] - keypoints[2]
            distal = keypoints[4] - keypoints[3]
            curl = self._segment_angle_v2(proximal, intermediate)
            curl += self._segment_angle_v2(intermediate, distal)
            amount = self._normalized_range_v2(
                curl,
                self.direct_thumb_curl_open_v2,
                self.direct_thumb_curl_closed_v2,
            )
            for thumb_pitch_index in thumb_pitch_indices:
                targets[thumb_pitch_index] = (
                    self.opt_lower_bounds[thumb_pitch_index]
                    + amount
                    * (
                        self.opt_upper_bounds[thumb_pitch_index]
                        - self.opt_lower_bounds[thumb_pitch_index]
                    )
                )

        return self.expand_to_full_qpos(targets[self.independent_indices])

    def _blend_direct_targets_v2(
        self,
        qpos: np.ndarray,
        direct_targets: np.ndarray,
    ) -> np.ndarray:
        if not self.direct_control_enabled_v2:
            return qpos

        output = np.asarray(qpos, dtype=np.float64).copy()
        pinch_partner = self._contact_partner_v2
        partner_name = (
            self._NON_THUMB_NAMES[pinch_partner - 1]
            if pinch_partner is not None
            else None
        )
        for name in self._NON_THUMB_NAMES:
            qpos_indices = self._direct_joint_indices_v2.get(name, ())
            if not qpos_indices:
                continue
            blend = (
                self.direct_pinch_finger_blend_v2
                if name == partner_name
                else self.direct_finger_blend_v2
            )
            for qpos_index in qpos_indices:
                output[qpos_index] = (
                    (1.0 - blend) * output[qpos_index]
                    + blend * direct_targets[qpos_index]
                )

        thumb_blend = (
            self.direct_pinch_thumb_blend_v2
            if pinch_partner is not None
            else self.direct_thumb_blend_v2
        )
        for name in ("thumb_yaw", "thumb_pitch"):
            qpos_indices = self._direct_joint_indices_v2.get(name, ())
            if not qpos_indices:
                continue
            for qpos_index in qpos_indices:
                output[qpos_index] = (
                    (1.0 - thumb_blend) * output[qpos_index]
                    + thumb_blend * direct_targets[qpos_index]
                )
        return self.expand_to_full_qpos(output[self.independent_indices])

    def _prepare_contact_v2(
        self,
        mediapipe_keypoints: np.ndarray,
        alphas: np.ndarray,
    ) -> None:
        """Select one unambiguous pinch partner and prepare its closure target."""
        self._contact_partner_v2 = None
        self._contact_beta_v2 = 0.0
        self._contact_target_cm_v2 = 0.0
        if (
            self.num_fingers < 2
            or len(alphas) < 2
            or not self.contact_enabled_partners_v2
        ):
            return

        ranked_partners = sorted(
            self.contact_enabled_partners_v2,
            key=lambda index: float(alphas[index]),
            reverse=True,
        )
        partner = ranked_partners[0]
        if len(ranked_partners) > 1:
            runner_up = ranked_partners[1]
            if (
                float(alphas[runner_up]) >= self.contact_min_beta_v2
                and float(alphas[partner] - alphas[runner_up])
                < self.contact_dominance_margin_v2
            ):
                return
        raw_beta = float(
            np.clip(
                alphas[partner] / (self.pinch_alpha_max + 1e-8),
                0.0,
                1.0,
            )
        )
        if raw_beta < self.contact_min_beta_v2:
            return
        beta = float(
            np.clip(
                (raw_beta - self.contact_min_beta_v2)
                / (1.0 - self.contact_min_beta_v2 + 1e-8),
                0.0,
                1.0,
            )
        )

        thumb_tip = mediapipe_keypoints[self.MP_TIP_INDICES[0]]
        mp_finger = self.mp_finger_indices[partner]
        finger_tip = mediapipe_keypoints[self.MP_TIP_INDICES[mp_finger]]
        human_distance_cm = float(np.linalg.norm(finger_tip - thumb_tip) * M_TO_CM)
        reachable_distance_cm = self.contact_target_distances_cm_v2[partner - 1]
        scaled_human_distance_cm = max(
            human_distance_cm * self.pinch_scaling,
            reachable_distance_cm,
        )

        self._contact_partner_v2 = partner
        self._contact_beta_v2 = beta
        self._contact_target_cm_v2 = (
            (1.0 - beta) * scaled_human_distance_cm
            + beta * reachable_distance_cm
        )

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        self._prepare_contact_v2(keypoints, self._compute_pinch_alpha(keypoints))
        direct_targets = self._compute_direct_targets_v2(keypoints)
        if (
            self.direct_control_enabled_v2
            and self.direct_only_v2
            and self._contact_partner_v2 is None
        ):
            self.last_qpos = direct_targets.astype(np.float64)
            return direct_targets.astype(np.float32)
        result = super().solve(keypoints, last_qpos)
        result = self._blend_direct_targets_v2(result, direct_targets)
        self.last_qpos = result.astype(np.float64)
        return result.astype(np.float32)

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        self._prepare_contact_v2(keypoints, self._compute_pinch_alpha(keypoints))
        return super().compute_cost(qpos, keypoints)

    def _contact_loss_and_grad_v2(
        self,
        qpos: np.ndarray,
        positions: np.ndarray,
        jacobians: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return the scalar closure loss and its analytical qpos gradient."""
        del qpos
        grad = np.zeros(self.num_joints, dtype=np.float64)
        partner = self._contact_partner_v2
        if partner is None:
            return 0.0, grad

        thumb_index = self.task_indices[0]
        finger_index = self.task_indices[partner]
        relative_tip = positions[finger_index] - positions[thumb_index]
        distance_cm = float(np.linalg.norm(relative_tip))
        residual_cm = distance_cm - self._contact_target_cm_v2
        effective_weight = self.contact_weight_v2 * self._contact_beta_v2

        loss = effective_weight * float(
            huber_loss_np(
                np.array([residual_cm]), self.contact_huber_delta_v2
            )[0]
        )
        distance_direction = relative_tip / (distance_cm + 1e-8)
        relative_jacobian = jacobians[finger_index] - jacobians[thumb_index]
        residual_grad = float(
            huber_loss_grad_np(
                np.array([residual_cm]), self.contact_huber_delta_v2
            )[0]
        )
        grad += (
            effective_weight
            * residual_grad
            * (distance_direction @ relative_jacobian)
        )
        return loss, grad

    def _extra_loss_and_grad_v2(
        self,
        qpos: np.ndarray,
        positions: np.ndarray,
        jacobians: np.ndarray,
        alphas: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        del alphas
        return self._contact_loss_and_grad_v2(qpos, positions, jacobians)
