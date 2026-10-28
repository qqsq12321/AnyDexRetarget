"""Key-vector-based retargeting optimizer.

Reference: dex-retargeting VectorOptimizer, released with AnyTeleop
(Qin et al., RSS 2023, arXiv:2307.04577)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_optimizer import BaseOptimizer
from .utils import M_TO_CM, TimingStats, huber_loss_np, huber_loss_grad_np


class KeyVectorOptimizer(BaseOptimizer):
    """Key-vector-based retargeting optimizer.

    Reimplementation of the classic dex-retargeting VectorOptimizer: each key
    vector is an (origin_link, task_link) pair on the robot, matched to a
    (origin_kp, task_kp) MediaPipe keypoint pair, and scaled by a single
    global factor (not per-vector).

    Loss:
        L(q) = (1/N) * Σᵢ Huber(‖[FK(task_i)-FK(origin_i)] - scaling_factor*(mp[task_kp_i]-mp[origin_kp_i])‖)
             + norm_delta * ‖q - q_prev‖²
             + w_contact * alpha_partner * ‖FK(thumb_tip) - FK(partner_tip)‖²   (optional, see retarget.contact)

    Config:
        - retarget.scaling_factor: global scale applied to every human vector
          (default 1.0). E.g. a robot hand 1.2x larger than a human hand uses 1.2.
        - retarget.key_vectors (required), each entry:
            - origin: robot origin link name
            - task:   robot task link name
            - origin_kp: MediaPipe origin keypoint index
            - task_kp:   MediaPipe task keypoint index
        - retarget.w_contact / retarget.contact (optional): pinch contact loss.
          Mirrors AdaptiveOptimizerAnalytical's pinch alpha + contact term so a
          pinch is not just "both tips near their target" but the thumb and the
          closest finger are explicitly pulled together once they are within
          ``d2`` of each other. Only the single closest non-thumb finger
          contributes at a time (same one-contact-at-a-time design as
          adaptive). Config shape:
              contact:
                thumb_tip_link: <robot link at the thumb tip>
                thumb_tip_kp:   <MediaPipe thumb tip index, usually 4>
                thumb_tip_offset: [x, y, z]   # optional, local offset (meters)
                fingers:
                  - {tip_link: <robot link>, tip_kp: <MediaPipe index>,
                     tip_offset: [x, y, z], d1: 2.0, d2: 4.0}
                  - ...
          d1/d2 are in cm: alpha ramps from 0 at distance d2 to 1 at distance d1.
          tip_offset (optional, default zero) places the contact point at a
          local offset from the link frame -- e.g. the GEORT distal link
          frames used by linker_l20 sit at the DIP/IP joint, not the visible
          fingertip surface, so the same offsets BaseOptimizer uses for
          tip_links (see ROBOT_CONFIGS['linker_l20']['tip_offsets']) should be
          repeated here for a physically accurate pinch.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self._timing = TimingStats()
        self._enable_timing = True

        retarget_config = config.get('retarget', {})
        robot_config = config.get('robot', {})
        robot_type = robot_config.get('type', 'shadow_hand')

        kv_config = retarget_config.get('key_vectors')
        if not kv_config:
            raise ValueError(
                "retarget.key_vectors is required for KeyVectorOptimizer. "
                "Define a list of {origin, task, origin_kp, task_kp} entries."
            )

        origin_names_raw = [kv['origin'] for kv in kv_config]
        task_names_raw   = [kv['task']   for kv in kv_config]

        # Apply same left-hand name replacement as BaseOptimizer
        if robot_type == 'shadow_hand' and self.hand_side == 'left':
            def _replace(name):
                return name.replace('rh_', 'lh_')
            origin_names = [_replace(n) for n in origin_names_raw]
            task_names   = [_replace(n) for n in task_names_raw]
        elif robot_type == 'unitree_dex5_hand' and self.hand_side == 'left':
            def _replace(name):
                if name == 'base_link00':
                    return 'base_link00L'
                return f"{name[:-1]}L" if name.endswith('R') else name
            origin_names = [_replace(n) for n in origin_names_raw]
            task_names   = [_replace(n) for n in task_names_raw]
        elif robot_type in ('linker_l20', 'sharpa_hand', 'gaia_hand20') and self.hand_side == 'left':
            def _replace(name):
                return name.replace('right_', 'left_')
            origin_names = [_replace(n) for n in origin_names_raw]
            task_names   = [_replace(n) for n in task_names_raw]
        elif robot_type == 'inspire_hand' and self.hand_side == 'left':
            def _replace(name):
                return name.replace('right_', 'left_')
            origin_names = [_replace(n) for n in origin_names_raw]
            task_names   = [_replace(n) for n in task_names_raw]
        else:
            origin_names = origin_names_raw
            task_names   = task_names_raw

        # Deduplicate points by (frame name, local offset). A link referenced
        # by multiple vectors (e.g. the robot palm/wrist as every vector's
        # origin) is only resolved once for FK/Jacobian computation. The same
        # link name with a different offset (e.g. a contact point placed at
        # the fingertip surface instead of the link origin) is a distinct
        # point. key_vectors entries never carry an offset -- that mirrors
        # the classic VectorOptimizer exactly -- only optional contact points
        # (below) may.
        point_names = []
        point_offsets = []
        point_indices = {}

        def add_point(name: str, offset=None) -> int:
            off = np.zeros(3, dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64)
            key = (name, *off.tolist())
            if key not in point_indices:
                point_indices[key] = len(point_names)
                point_names.append(name)
                point_offsets.append(off)
            return point_indices[key]

        self._kv_origin_indices = np.asarray(
            [add_point(name) for name in origin_names], dtype=int,
        )
        self._kv_task_indices = np.asarray(
            [add_point(name) for name in task_names], dtype=int,
        )

        self._origin_kp_indices = np.array([kv['origin_kp'] for kv in kv_config], dtype=int)
        self._task_kp_indices   = np.array([kv['task_kp']   for kv in kv_config], dtype=int)
        self.num_vectors        = len(kv_config)
        self.scaling_factor     = float(retarget_config.get('scaling_factor', 1.0))

        # Optional pinch contact loss (disabled unless retarget.contact is set).
        self.w_contact = float(retarget_config.get('w_contact', 0.0))
        contact_config = retarget_config.get('contact')
        self._has_contact = bool(contact_config) and self.w_contact > 0.0
        if self._has_contact:
            thumb_link = contact_config['thumb_tip_link']
            if robot_type in ('linker_l20', 'sharpa_hand', 'gaia_hand20', 'inspire_hand') and self.hand_side == 'left':
                thumb_link = thumb_link.replace('right_', 'left_')
            elif robot_type == 'shadow_hand' and self.hand_side == 'left':
                thumb_link = thumb_link.replace('rh_', 'lh_')
            elif robot_type == 'unitree_dex5_hand' and self.hand_side == 'left':
                thumb_link = (
                    'base_link00L' if thumb_link == 'base_link00'
                    else (f"{thumb_link[:-1]}L" if thumb_link.endswith('R') else thumb_link)
                )
            self._contact_thumb_kp = int(contact_config['thumb_tip_kp'])
            thumb_offset = contact_config.get('thumb_tip_offset')
            self._contact_thumb_link_idx = add_point(thumb_link, thumb_offset)

            finger_cfgs = contact_config.get('fingers', [])
            if not finger_cfgs:
                raise ValueError(
                    "retarget.contact.fingers must list at least one "
                    "{tip_link, tip_kp, d1, d2} entry"
                )
            finger_links = [f['tip_link'] for f in finger_cfgs]
            if robot_type in ('linker_l20', 'sharpa_hand', 'gaia_hand20', 'inspire_hand') and self.hand_side == 'left':
                finger_links = [n.replace('right_', 'left_') for n in finger_links]
            elif robot_type == 'shadow_hand' and self.hand_side == 'left':
                finger_links = [n.replace('rh_', 'lh_') for n in finger_links]
            elif robot_type == 'unitree_dex5_hand' and self.hand_side == 'left':
                def _replace_suffix(name):
                    if name == 'base_link00':
                        return 'base_link00L'
                    return f"{name[:-1]}L" if name.endswith('R') else name
                finger_links = [_replace_suffix(n) for n in finger_links]

            self._contact_finger_link_indices = np.asarray(
                [
                    add_point(n, f.get('tip_offset'))
                    for n, f in zip(finger_links, finger_cfgs)
                ],
                dtype=int,
            )
            self._contact_finger_kp = np.array([int(f['tip_kp']) for f in finger_cfgs], dtype=int)
            self._contact_d1 = np.array([float(f.get('d1', 2.0)) for f in finger_cfgs], dtype=np.float64)
            self._contact_d2 = np.array([float(f.get('d2', 4.0)) for f in finger_cfgs], dtype=np.float64)

        # add_point calls above (key_vectors + optional contact points) are
        # all done; finalize the unique-link bookkeeping used for FK/Jacobian.
        self._kv_computed_link_names = point_names
        self._kv_computed_link_indices = [
            self.robot.get_link_index(name) for name in point_names
        ]
        self._kv_computed_link_offsets = np.asarray(point_offsets, dtype=np.float64)

    # ------------------------------------------------------------------
    # Target vector computation
    # ------------------------------------------------------------------

    def _compute_target_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute scaled target vectors from MediaPipe keypoints.

        Args:
            keypoints: (21, 3) MediaPipe keypoints in wrist frame (meters)

        Returns:
            (N, 3) target vectors in cm
        """
        origin_kp = keypoints[self._origin_kp_indices]  # (N, 3)
        task_kp   = keypoints[self._task_kp_indices]    # (N, 3)
        vecs = (task_kp - origin_kp) * self.scaling_factor * M_TO_CM
        return vecs.astype(np.float64)

    # ------------------------------------------------------------------
    # Pinch contact (optional)
    # ------------------------------------------------------------------

    def _compute_contact_alpha(self, keypoints: np.ndarray) -> tuple[int, float]:
        """Pick the single closest non-thumb finger and its pinch alpha.

        Mirrors AdaptiveOptimizerAnalytical._compute_pinch_alpha: alpha ramps
        from 0 at distance ``d2`` to 1 at distance ``d1`` (cm), and only the
        finger closest to the thumb contributes -- mimicking a human pinch,
        which only ever touches one finger to the thumb at a time.

        Returns:
            (finger_index, alpha) where finger_index indexes into
            ``self._contact_finger_link_indices`` / ``_contact_finger_kp``.
        """
        thumb_tip = keypoints[self._contact_thumb_kp]
        finger_tips = keypoints[self._contact_finger_kp]
        distances = np.linalg.norm(finger_tips - thumb_tip, axis=1) * M_TO_CM
        alphas = np.clip(
            (self._contact_d2 - distances) / (self._contact_d2 - self._contact_d1 + 1e-8),
            0.0, 1.0,
        )
        partner = int(np.argmax(alphas))
        return partner, float(alphas[partner])

    # ------------------------------------------------------------------
    # Loss and analytical gradient
    # ------------------------------------------------------------------

    def _loss_and_grad(
        self,
        qpos: np.ndarray,
        target_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
        contact: Optional[tuple[int, float]] = None,
    ) -> tuple[float, np.ndarray]:
        """Compute loss and analytical gradient.

        Args:
            qpos: (num_joints,) full joint positions
            target_vectors: (N, 3) scaled target vectors in cm
            last_qpos: (num_joints,) previous qpos for regularization, or None
            contact: optional (finger_index, alpha) from
                ``_compute_contact_alpha``, enabling the pinch contact term.

        Returns:
            (loss, full_grad) where full_grad has shape (num_joints,)
        """
        qpos = np.asarray(qpos, dtype=np.float64)

        # FK: world positions for all unique links. key_vectors points always
        # carry a zero offset (classic VectorOptimizer behavior); only
        # optional contact points may have a non-zero tip_offset.
        positions = self.robot.compute_points_batch(
            qpos,
            self._kv_computed_link_indices,
            self._kv_computed_link_offsets,
        ) * M_TO_CM  # (num_unique_links, 3)

        # Jacobians: (num_unique_links, 3, nq)
        Js = self.robot.compute_all_jacobians_batch_with_offsets(
            qpos,
            self._kv_computed_link_indices,
            self._kv_computed_link_offsets,
        ) * M_TO_CM

        # Per-vector positions and Jacobians
        origin_pos = positions[self._kv_origin_indices]  # (N, 3)
        task_pos   = positions[self._kv_task_indices]    # (N, 3)
        J_origin   = Js[self._kv_origin_indices]         # (N, 3, nq)
        J_task     = Js[self._kv_task_indices]           # (N, 3, nq)

        # Residuals
        robot_vec = task_pos - origin_pos                # (N, 3)
        diff      = robot_vec - target_vectors           # (N, 3)
        dist      = np.linalg.norm(diff, axis=1)         # (N,)

        # Mean Huber loss
        total_loss = float(np.mean(huber_loss_np(dist, self.huber_delta)))

        # Analytical gradient via chain rule
        huber_grad  = huber_loss_grad_np(dist, self.huber_delta)  # (N,)
        diff_normed = diff / (dist[:, None] + 1e-8)               # (N, 3)

        total_grad = np.zeros(self.num_joints, dtype=np.float64)
        N = self.num_vectors
        for i in range(N):
            J_diff = J_task[i] - J_origin[i]                      # (3, nq)
            total_grad += (huber_grad[i] / N) * (diff_normed[i] @ J_diff)

        # Pinch contact: pull the thumb tip and the closest finger tip
        # together once they are within d2 of each other (alpha > 0).
        if contact is not None and self._has_contact:
            partner, alpha = contact
            if alpha > 0.0:
                thumb_pos = positions[self._contact_thumb_link_idx]
                finger_pos = positions[self._contact_finger_link_indices[partner]]
                J_thumb = Js[self._contact_thumb_link_idx]
                J_finger = Js[self._contact_finger_link_indices[partner]]

                contact_diff = thumb_pos - finger_pos
                contact_weight = self.w_contact * alpha
                total_loss += contact_weight * float(np.dot(contact_diff, contact_diff))
                total_grad += 2.0 * contact_weight * (contact_diff @ (J_thumb - J_finger))

        # Regularization: penalize large joint velocity
        if last_qpos is not None:
            delta = qpos - last_qpos
            total_loss += self.norm_delta * float(np.sum(delta ** 2))
            total_grad += 2.0 * self.norm_delta * delta

        return total_loss, total_grad

    # ------------------------------------------------------------------
    # NLopt objective factory
    # ------------------------------------------------------------------

    def _get_objective(
        self,
        target_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
        contact: Optional[tuple[int, float]] = None,
    ):
        """Create NLopt objective function with analytical gradient.

        The objective operates on independent joints only (num_opt_vars).
        Mimic joints are expanded internally before FK, and the gradient is
        mapped back via chain rule.
        """
        target_vectors = np.asarray(target_vectors, dtype=np.float64)
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float64)

        def objective(x: np.ndarray, grad_out: np.ndarray) -> float:
            opt_vars  = np.asarray(x, dtype=np.float64)
            full_qpos = self.expand_to_full_qpos(opt_vars)

            loss, full_grad = self._loss_and_grad(full_qpos, target_vectors, last_qpos, contact)

            if grad_out.size > 0:
                grad_out[:] = self.map_gradient_to_independent(full_grad)

            if self._enable_timing:
                self._timing.record_iter_loss(float(loss))

            return float(loss)

        return objective

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for joint angles given MediaPipe keypoints."""
        if self._enable_timing:
            self._timing.start_frame()

        mediapipe_keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        if mediapipe_keypoints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {mediapipe_keypoints.shape}")

        target_vectors = self._compute_target_vectors(mediapipe_keypoints)
        contact = self._compute_contact_alpha(mediapipe_keypoints) if self._has_contact else None
        reg_qpos  = self._get_reg_qpos(last_qpos)
        init_qpos = self._get_init_qpos(last_qpos)

        objective_fn = self._get_objective(target_vectors, reg_qpos, contact)
        result = self._run_optimization(objective_fn, init_qpos)

        if self._enable_timing:
            self._timing.end_frame(self.opt.get_numevals())

        return result

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute loss for given joint angles."""
        mediapipe_keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        target_vectors = self._compute_target_vectors(mediapipe_keypoints)
        contact = self._compute_contact_alpha(mediapipe_keypoints) if self._has_contact else None
        loss, _ = self._loss_and_grad(
            np.asarray(qpos, dtype=np.float64), target_vectors, None, contact
        )
        return float(loss)

    # ------------------------------------------------------------------
    # Timing (required by benchmark_frames.py)
    # ------------------------------------------------------------------

    def get_timing_stats(self) -> TimingStats:
        """Get timing statistics."""
        return self._timing

    def reset_timing_stats(self):
        """Reset timing statistics."""
        self._timing.reset()

    def set_timing_enabled(self, enabled: bool):
        """Enable or disable timing instrumentation."""
        self._enable_timing = enabled
