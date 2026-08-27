"""Adaptive optimizer with analytical gradients for hand retargeting."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base_optimizer import BaseOptimizer
from .utils import M_TO_CM, TimingStats, huber_loss_grad_np, huber_loss_np


class AdaptiveOptimizerAnalytical(BaseOptimizer):
    """Adaptive optimizer with analytical (hand-written) gradients.

    Same loss function as AdaptiveOptimizer but uses hand-written gradients
    instead of autograd for faster performance.
    """

    def __init__(self, config: dict):
        """Initialize AdaptiveOptimizerAnalytical."""
        super().__init__(config)

        # Initialize timing stats
        self._timing = TimingStats()
        self._enable_timing = True

        retarget_config = config.get('retarget', {})

        # TipDirVec parameters
        self.huber_delta_dir = retarget_config.get('huber_delta_dir', 0.5)
        self.w_pos = retarget_config.get('w_pos', 1.0)
        self.w_dir = retarget_config.get('w_dir', 10.0)
        self.project_tip_dir = retarget_config.get('project_tip_dir', False)
        self.pinch_scaling = float(retarget_config.get('pinch_scaling', 1.0))
        self.pinch_alpha_max = float(retarget_config.get('alpha', self.PINCH_ALPHA_MAX))

        # FullHandVec parameters
        self.w_full_hand = retarget_config.get('w_full_hand', 1.0)
        self.lateral_scaling = float(retarget_config.get('lateral_scaling', 1.0))
        self.lateral_axis = int(retarget_config.get('lateral_axis', 1))
        segment_scaling_config = retarget_config.get('segment_scaling', {})
        all_finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        # Select finger names based on num_fingers
        if self.num_fingers == 4:
            finger_names = ['thumb', 'index', 'middle', 'ring']
        else:
            finger_names = all_finger_names
        nf = self.num_fingers
        # For optimization: (nf, 3) - PIP, DIP, TIP only
        self.segment_scaling = np.ones((nf, 3), dtype=np.float64)
        # For visualization: (nf, 4) - MCP, PIP, DIP, TIP (full version)
        self.segment_scaling_full = np.ones((nf, 4), dtype=np.float64)
        for i, finger_name in enumerate(finger_names):
            if finger_name in segment_scaling_config:
                scales = np.array(segment_scaling_config[finger_name])
                if len(scales) == 4:
                    # 4-param format: [MCP, PIP, DIP, TIP]
                    self.segment_scaling_full[i] = scales
                    self.segment_scaling[i] = scales[1:4]  # PIP, DIP, TIP for optimization
                elif len(scales) == 3:
                    # 3-param format: [PIP, DIP, TIP] - assume MCP scale = 1.0
                    self.segment_scaling_full[i] = np.array([1.0, scales[0], scales[1], scales[2]])
                    self.segment_scaling[i] = scales

        # Pinch thresholds (for non-thumb fingers)
        pinch_config = retarget_config.get('pinch_thresholds', {})
        if self.num_fingers == 4:
            non_thumb_fingers = ['index', 'middle', 'ring']
        else:
            non_thumb_fingers = ['index', 'middle', 'ring', 'pinky']
        self.d1 = np.array([
            pinch_config.get(f, {}).get('d1', 2.0) for f in non_thumb_fingers
        ], dtype=np.float64)
        self.d2 = np.array([
            pinch_config.get(f, {}).get('d2', 4.0) for f in non_thumb_fingers
        ], dtype=np.float64)

        # Add link1 points for finger plane computation while preserving
        # possibly duplicated frame names with different local offsets.
        self.link1_indices = []
        extra_indices = []
        extra_names = []
        extra_offsets = []
        for name in self.link1_names:
            self.link1_indices.append(len(self.computed_link_indices) + len(extra_indices))
            extra_indices.append(self.robot.get_link_index(name))
            extra_names.append(name)
            extra_offsets.append(np.zeros(3, dtype=np.float64))

        self.computed_link_indices.extend(extra_indices)
        self.computed_link_names.extend(extra_names)
        self.computed_link_offsets = np.concatenate(
            [self.computed_link_offsets, np.asarray(extra_offsets, dtype=np.float64)],
            axis=0,
        )

    # Cap alpha below 1.0 so FullHandVec never vanishes completely: at full
    # pinch the blend still keeps (1 - PINCH_ALPHA_MAX) of the posture
    # objective, which is what stops the intermediate joints from drifting
    # away from the input skeleton once the fingertip objective is satisfied.
    PINCH_ALPHA_MAX = 0.7

    def _compute_pinch_alpha(self, mediapipe_keypoints: np.ndarray) -> np.ndarray:
        """Compute alpha weights for each finger."""
        thumb_tip = mediapipe_keypoints[self.MP_TIP_INDICES[0]]
        # Use only the non-thumb finger indices that this hand has
        non_thumb_mp_indices = self.mp_finger_indices[1:]  # skip thumb
        finger_tips = np.array([mediapipe_keypoints[self.MP_TIP_INDICES[i]] for i in non_thumb_mp_indices])
        distances = np.linalg.norm(finger_tips - thumb_tip, axis=1) * M_TO_CM
        alphas_nt = np.clip(
            (self.d2 - distances) / (self.d2 - self.d1 + 1e-8),
            0.0,
            self.pinch_alpha_max,
        )
        alpha_thumb = np.max(alphas_nt)
        return np.concatenate([[alpha_thumb], alphas_nt])

    def _compute_adaptive_targets(
        self,
        mediapipe_keypoints: np.ndarray,
        alphas: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build TipDirVec and FullHandVec targets.

        FullHandVec keeps the per-finger chain scaling used for the full-hand
        posture target.  During a detected pinch, only the thumb and the active
        pinch finger switch their tip-position target toward a uniformly scaled
        wrist->tip vector so opposition contacts are not broken by per-finger
        segment scaling.
        """
        nf = self.num_fingers
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, self.segment_scaling_full,
            self.lateral_scaling, self.lateral_axis,
        )
        target_tip_vectors = target_full_hand_vectors[2 * nf:].copy()

        if alphas is not None and self.pinch_scaling != 1.0 and nf > 1:
            partner = int(np.argmax(alphas[1:]) + 1)
            alpha = float(alphas[partner])
            if alpha > 0.0:
                beta = min(alpha / (self.pinch_alpha_max + 1e-8), 1.0)
                pinch_tip_vectors = self._compute_tip_vectors(
                    mediapipe_keypoints, self.pinch_scaling
                )
                for i in (0, partner):
                    target_tip_vectors[i] = (
                        (1.0 - beta) * target_tip_vectors[i]
                        + beta * pinch_tip_vectors[i]
                    )

        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        return target_tip_vectors, target_tip_dirs, target_full_hand_vectors

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for joint angles."""
        if self._enable_timing:
            t_total_start = time.perf_counter()
            t_preprocess_start = time.perf_counter()
            self._timing.start_frame()

        mediapipe_keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        if mediapipe_keypoints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {mediapipe_keypoints.shape}")

        reg_qpos = self._get_reg_qpos(last_qpos)
        init_qpos = self._get_init_qpos(last_qpos)

        alphas = self._compute_pinch_alpha(mediapipe_keypoints)
        (
            target_tip_vectors,
            target_tip_dirs,
            target_full_hand_vectors,
        ) = self._compute_adaptive_targets(mediapipe_keypoints, alphas)

        if self._enable_timing:
            self._timing.preprocess_ms += (time.perf_counter() - t_preprocess_start) * 1000
            t_nlopt_start = time.perf_counter()

        objective_fn = self._get_objective_analytical(
            target_tip_vectors, target_tip_dirs, target_full_hand_vectors, alphas, reg_qpos
        )
        result = self._run_optimization(objective_fn, init_qpos)

        if self._enable_timing:
            self._timing.nlopt_ms += (time.perf_counter() - t_nlopt_start) * 1000
            self._timing.total_ms += (time.perf_counter() - t_total_start) * 1000
            self._timing.call_count += 1
            self._timing.end_frame(self.opt.get_numevals())

        return result

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute cost for given joint angles."""
        alphas = self._compute_pinch_alpha(mediapipe_keypoints)
        (
            target_tip_vectors,
            target_tip_dirs,
            target_full_hand_vectors,
        ) = self._compute_adaptive_targets(mediapipe_keypoints, alphas)
        loss, _ = self._loss_and_grad_analytical(
            qpos, target_tip_vectors, target_tip_dirs, target_full_hand_vectors, alphas, None
        )
        return float(loss)



    def _loss_and_grad_analytical(
        self,
        qpos: np.ndarray,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        alphas: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """Compute loss and gradient analytically."""
        qpos = np.asarray(qpos, dtype=np.float64)

        # Forward kinematics
        if self._enable_timing:
            t_fk_start = time.perf_counter()

        positions = self.robot.compute_points_batch(
            qpos,
            self.computed_link_indices,
            self.computed_link_offsets,
        ) * M_TO_CM

        if self._enable_timing:
            self._timing.fk_ms += (time.perf_counter() - t_fk_start) * 1000
            t_jac_start = time.perf_counter()

        # Get Jacobians (num_links, 3, nq) - already in world frame
        Js = self.robot.compute_all_jacobians_batch_with_offsets(
            qpos,
            self.computed_link_indices,
            self.computed_link_offsets,
        ) * M_TO_CM

        if self._enable_timing:
            self._timing.jacobian_ms += (time.perf_counter() - t_jac_start) * 1000
            t_grad_start = time.perf_counter()

        nf = self.num_fingers

        # Extract positions
        origin_pos = positions[self.origin_indices]  # (nf, 3)
        task_pos = positions[self.task_indices]  # (nf, 3)
        link3_pos = positions[self.link3_indices]  # (nf, 3)
        link4_pos = positions[self.link4_indices]  # (nf, 3)
        wrist_pos = positions[self.origin_indices[0]]  # (3,)

        # Get Jacobians for each link type
        J_origin = Js[self.origin_indices]  # (nf, 3, nq)
        J_task = Js[self.task_indices]  # (nf, 3, nq)
        J_link3 = Js[self.link3_indices]  # (nf, 3, nq)
        J_link4 = Js[self.link4_indices]  # (nf, 3, nq)
        J_wrist = Js[self.origin_indices[0]]  # (3, nq)

        total_loss = 0.0
        total_grad = np.zeros(self.num_joints, dtype=np.float64)

        # === Tip Position Loss ===
        robot_tip_vec = task_pos - origin_pos  # (nf, 3)
        diff_pos = robot_tip_vec - target_tip_vectors  # (nf, 3)
        dist_pos = np.linalg.norm(diff_pos, axis=1)  # (nf,)
        loss_tip_pos = huber_loss_np(dist_pos, self.huber_delta)  # (nf,)

        huber_grad_pos = huber_loss_grad_np(dist_pos, self.huber_delta)  # (nf,)
        diff_normed_pos = diff_pos / (dist_pos[:, None] + 1e-8)  # (nf, 3)
        for i in range(nf):
            grad_coeff = alphas[i] * self.w_pos * huber_grad_pos[i]
            J_diff = J_task[i] - J_origin[i]  # (3, nq)
            total_grad += grad_coeff * (diff_normed_pos[i] @ J_diff)

        # === Tip Direction Loss ===
        robot_tip_dir_vec = task_pos - link4_pos  # (nf, 3)
        robot_tip_dir_norm = np.linalg.norm(robot_tip_dir_vec, axis=1, keepdims=True)  # (nf, 1)
        robot_tip_dirs = robot_tip_dir_vec / (robot_tip_dir_norm + 1e-8)  # (nf, 3)

        diff_dir = robot_tip_dirs - target_tip_dirs  # (nf, 3)
        dist_dir = np.linalg.norm(diff_dir, axis=1)  # (nf,)
        loss_tip_dir = huber_loss_np(dist_dir, self.huber_delta_dir)  # (nf,)

        huber_grad_dir = huber_loss_grad_np(dist_dir, self.huber_delta_dir)  # (nf,)
        diff_normed_dir = diff_dir / (dist_dir[:, None] + 1e-8)  # (nf, 3)
        for i in range(nf):
            grad_coeff = alphas[i] * self.w_dir * huber_grad_dir[i]
            u = robot_tip_dirs[i]  # (3,)
            n = robot_tip_dir_norm[i, 0]  # scalar
            J_norm = (np.eye(3) - np.outer(u, u)) / (n + 1e-8)  # (3, 3)
            J_diff = J_task[i] - J_link4[i]  # (3, nq)
            total_grad += grad_coeff * (diff_normed_dir[i] @ J_norm @ J_diff)

        # === Full Hand Vec Loss ===
        robot_pip_vec = link3_pos - wrist_pos  # (nf, 3)
        robot_dip_vec = link4_pos - wrist_pos  # (nf, 3)
        robot_tip_vec_full = task_pos - wrist_pos  # (nf, 3)

        target_pip = target_full_hand_vectors[:nf]
        target_dip = target_full_hand_vectors[nf:2*nf]
        target_tip = target_full_hand_vectors[2*nf:3*nf]

        diff_pip = robot_pip_vec - target_pip
        diff_dip = robot_dip_vec - target_dip
        diff_tip = robot_tip_vec_full - target_tip

        dist_pip = np.linalg.norm(diff_pip, axis=1)
        dist_dip = np.linalg.norm(diff_dip, axis=1)
        dist_tip = np.linalg.norm(diff_tip, axis=1)

        loss_pip = huber_loss_np(dist_pip, self.huber_delta)
        loss_dip = huber_loss_np(dist_dip, self.huber_delta)
        loss_tip_full = huber_loss_np(dist_tip, self.huber_delta)
        loss_full_hand = (loss_pip + loss_dip + loss_tip_full) / 3.0  # (nf,)

        huber_grad_pip = huber_loss_grad_np(dist_pip, self.huber_delta)
        huber_grad_dip = huber_loss_grad_np(dist_dip, self.huber_delta)
        huber_grad_tip = huber_loss_grad_np(dist_tip, self.huber_delta)

        diff_normed_pip = diff_pip / (dist_pip[:, None] + 1e-8)
        diff_normed_dip = diff_dip / (dist_dip[:, None] + 1e-8)
        diff_normed_tip = diff_tip / (dist_tip[:, None] + 1e-8)

        for i in range(nf):
            grad_coeff = (1.0 - alphas[i]) * self.w_full_hand / 3.0
            total_grad += grad_coeff * huber_grad_pip[i] * (diff_normed_pip[i] @ (J_link3[i] - J_wrist))
            total_grad += grad_coeff * huber_grad_dip[i] * (diff_normed_dip[i] @ (J_link4[i] - J_wrist))
            total_grad += grad_coeff * huber_grad_tip[i] * (diff_normed_tip[i] @ (J_task[i] - J_wrist))

        # === Total Loss ===
        loss_tip_dir_vec = self.w_pos * loss_tip_pos + self.w_dir * loss_tip_dir
        loss_full = self.w_full_hand * loss_full_hand
        loss_per_finger = alphas * loss_tip_dir_vec + (1.0 - alphas) * loss_full
        total_loss = np.sum(loss_per_finger)

        extra_loss, extra_grad = self._extra_loss_and_grad_v2(
            qpos, positions, Js, alphas
        )
        total_loss += extra_loss
        total_grad += extra_grad

        # === Regularization ===
        if last_qpos is not None:
            total_loss += self.norm_delta * np.sum((qpos - last_qpos) ** 2)
            total_grad += 2.0 * self.norm_delta * (qpos - last_qpos)

        if self._enable_timing:
            self._timing.gradient_ms += (time.perf_counter() - t_grad_start) * 1000

        return total_loss, total_grad

    def _extra_loss_and_grad_v2(
        self,
        qpos: np.ndarray,
        positions: np.ndarray,
        jacobians: np.ndarray,
        alphas: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """V2 extension hook that leaves the v1 objective unchanged."""
        return 0.0, np.zeros(self.num_joints, dtype=np.float64)

    def get_timing_stats(self) -> TimingStats:
        """Get timing statistics."""
        return self._timing

    def reset_timing_stats(self):
        """Reset timing statistics."""
        self._timing.reset()

    def set_timing_enabled(self, enabled: bool):
        """Enable or disable timing instrumentation."""
        self._enable_timing = enabled

    def _get_objective_analytical(
        self,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        alphas: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ):
        """Create NLopt objective function with analytical gradient.

        The objective operates on independent joints only (num_opt_vars).
        Mimic joints are expanded internally before FK.
        """
        target_tip_vectors = np.asarray(target_tip_vectors, dtype=np.float64)
        target_tip_dirs = np.asarray(target_tip_dirs, dtype=np.float64)
        target_full_hand_vectors = np.asarray(target_full_hand_vectors, dtype=np.float64)
        alphas = np.asarray(alphas, dtype=np.float64)
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float64)

        def objective(x, grad_out):
            opt_vars = np.asarray(x, dtype=np.float64)
            # Expand independent vars to full qpos
            full_qpos = self.expand_to_full_qpos(opt_vars)

            loss, full_grad = self._loss_and_grad_analytical(
                full_qpos, target_tip_vectors, target_tip_dirs, target_full_hand_vectors, alphas, last_qpos
            )
            if grad_out.size > 0:
                # Map gradient back to independent joints
                grad_out[:] = self.map_gradient_to_independent(full_grad)
            # Record iteration loss for plotting
            if self._enable_timing:
                self._timing.record_iter_loss(float(loss))
            return float(loss)

        return objective
