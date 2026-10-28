"""DexPilot retargeting optimizer.

Based on the DexPilot paper (https://arxiv.org/abs/1910.03135).
Uses adaptive projection of finger-to-finger distances to achieve stable grasping.
When fingers get close (below project_dist), the optimizer projects the distance
to a smaller target to encourage grasping. Supports 4-finger and 5-finger hands.
"""

from __future__ import annotations

import numpy as np

from .base_optimizer import BaseOptimizer
from .utils import huber_loss_np, huber_norm_gradient_np, M_TO_CM


class DexPilotOptimizer(BaseOptimizer):
    """DexPilot retargeting optimizer.

    Extends the original DexPilot (designed for 4-finger Allegro Hand)
    to support both 4-finger and 5-finger hands. Uses adaptive weighting
    and distance projection to facilitate stable grasping.
    """

    def __init__(self, config: dict):
        """Initialize DexPilot optimizer from configuration.

        Args:
            config: Configuration dict with DexPilot parameters
        """
        super().__init__(config)

        # DexPilot specific parameters
        retarget_config = config.get('retarget', {})
        robot_config = config.get('robot', {})

        # Projection parameters
        # DexPilot's projection parameters are conventionally specified in
        # metres (0.03 means 3 cm).  The rest of AnyDexRetarget's optimizer
        # math uses centimetres, so convert once at the configuration boundary.
        self.project_dist = float(retarget_config.get('project_dist', 0.03)) * M_TO_CM
        self.escape_dist = float(retarget_config.get('escape_dist', 0.05)) * M_TO_CM
        self.eta1 = float(retarget_config.get('eta1', 1e-4)) * M_TO_CM
        self.eta2 = float(retarget_config.get('eta2', 3e-2)) * M_TO_CM
        self.scaling = float(retarget_config.get('scaling', 1.0))
        if not np.all(np.isfinite([
            self.project_dist, self.escape_dist, self.eta1, self.eta2,
            self.scaling,
        ])):
            raise ValueError("DexPilot parameters must be finite")
        if not 0 < self.project_dist < self.escape_dist:
            raise ValueError("DexPilot requires 0 < project_dist < escape_dist")
        if self.eta1 < 0 or self.eta2 < 0 or self.scaling <= 0:
            raise ValueError("DexPilot eta values must be non-negative and scaling positive")

        # BaseOptimizer has already resolved robot defaults and left/right
        # link-name conversions.  Reuse those resolved fingertip names here.
        self.wrist_link_name = self.origin_link_name
        self.fingertip_link_names = self.task_link_names.copy()

        if self.num_fingers < 2 or self.num_fingers > 5:
            raise ValueError(
                f"DexPilot requires 2-5 fingers, got {self.num_fingers}"
            )

        # Build vector link structure
        self._build_dexpilot_links()

        # Initialize projection state
        self._init_projection_state()

    def _build_dexpilot_links(self):
        """Build link structure for DexPilot vectors.

        Creates vectors between:
        1. Finger-to-finger (all pairs)
        2. Wrist-to-fingertip (all fingers)
        """
        # A Linker L20 ``*_distal`` frame is located at the DIP/IP joint, not
        # at the physical fingertip.  BaseOptimizer resolves a local
        # ``task_offsets`` point for every fingertip, so DexPilot must build its
        # vectors from those points rather than from the bare frame origins.
        #
        # Keep ``self.task_link_names`` untouched: it is the public per-finger
        # tip list used by FK visualization.  Older code replaced it with the
        # expanded DexPilot vector endpoints, which made the red skeleton draw
        # repeated DIP links instead of the five fingertips.
        all_points = [
            (self.wrist_link_name, np.zeros(3, dtype=np.float64)),
            *[
                (name, np.asarray(offset, dtype=np.float64))
                for name, offset in zip(
                    self.fingertip_link_names, self.task_offsets
                )
            ],
        ]

        origin_points = []
        task_points = []

        # Finger-to-finger vectors (higher index to lower index).
        for i in range(1, self.num_fingers):
            for j in range(i + 1, self.num_fingers + 1):
                origin_points.append(all_points[j])
                task_points.append(all_points[i])

        # Wrist-to-fingertip vectors.
        for i in range(1, self.num_fingers + 1):
            origin_points.append(all_points[0])
            task_points.append(all_points[i])

        self.dexpilot_origin_link_names = [point[0] for point in origin_points]
        self.dexpilot_task_link_names = [point[0] for point in task_points]

        # Deduplicate by both frame and local point.  For L20 this retains one
        # physical-tip point per distal frame with its configured local offset.
        point_names = []
        point_offsets = []
        point_index = {}

        def add_point(point):
            name, offset = point
            offset = np.asarray(offset, dtype=np.float64)
            key = (name, *offset.tolist())
            if key not in point_index:
                point_index[key] = len(point_names)
                point_names.append(name)
                point_offsets.append(offset)
            return point_index[key]

        self.origin_link_indices = np.asarray(
            [add_point(point) for point in origin_points], dtype=np.int32
        )
        self.task_link_indices = np.asarray(
            [add_point(point) for point in task_points], dtype=np.int32
        )
        self.computed_link_names = point_names
        self.computed_link_offsets = np.asarray(point_offsets, dtype=np.float64)
        self.computed_link_ids = [
            self.robot.get_link_index(name) for name in self.computed_link_names
        ]

    def _init_projection_state(self):
        """Initialize DexPilot projection state and weights."""
        # Total finger-to-finger vectors (all pairs)
        # For 5 fingers (thumb + 4 others), this is C(5,2) = 10
        total_finger_pairs = self.num_fingers * (self.num_fingers - 1) // 2

        # S1: thumb-to-finger vectors (primary projection)
        # Thumb is finger 0, so S1 = vectors from thumb to fingers 1,2,3,4
        self.num_s1 = self.num_fingers - 1

        # S2: non-thumb finger-to-finger vectors (secondary projection)
        # Vectors between fingers 1,2,3,4 (not including thumb)
        self.num_s2 = (self.num_fingers - 1) * (self.num_fingers - 2) // 2

        # Total projection vectors (S1 + S2, should equal total_finger_pairs)
        self.num_proj = self.num_s1 + self.num_s2

        assert self.num_proj == total_finger_pairs, \
            f"Projection count mismatch: {self.num_proj} != {total_finger_pairs}"

        # Projection state (which vectors are currently projected)
        self.projected = np.zeros(self.num_proj, dtype=bool)

        # S2 projection indices (secondary projections based on primary)
        self.s2_origin_indices = []
        self.s2_task_indices = []
        for i in range(0, self.num_fingers - 2):
            for j in range(i + 1, self.num_fingers - 1):
                self.s2_origin_indices.append(j)
                self.s2_task_indices.append(i)

        # Projected distances for each vector type
        self.projected_dist = np.concatenate([
            np.full(self.num_fingers - 1, self.eta1),  # Thumb-to-finger
            np.full(self.num_s2, self.eta2)             # Other finger pairs
        ])

    def _update_projection_state(self, target_vectors: np.ndarray):
        """Update which vectors should be projected based on current distances.

        Args:
            target_vectors: Human hand vectors (N, 3)
        """
        # Compute distances for projection vectors (finger-to-finger only)
        target_dist = np.linalg.norm(target_vectors[:self.num_proj], axis=1)

        # S1 projection (thumb-to-finger): project if dist < project_dist
        self.projected[:self.num_s1][target_dist[:self.num_s1] < self.project_dist] = True
        self.projected[:self.num_s1][target_dist[:self.num_s1] > self.escape_dist] = False

        # S2 projection (finger-to-finger, non-thumb): project if both related S1 are projected
        # Note: num_s2 can be 0 for hands with few fingers
        if self.num_s2 > 0 and self.num_s1 < self.num_proj:
            s2_start = self.num_s1
            s2_end = self.num_proj
            s2_proj = np.logical_and(
                self.projected[:self.num_s1][self.s2_origin_indices],
                self.projected[:self.num_s1][self.s2_task_indices]
            )
            # Also check distance threshold
            if s2_end > s2_start:
                s2_proj = np.logical_and(
                    s2_proj, target_dist[s2_start:s2_end] <= self.project_dist
                )
            self.projected[s2_start:s2_end] = s2_proj

    def _compute_reference_vectors(self, target_vectors: np.ndarray) -> np.ndarray:
        """Compute reference vectors with DexPilot projection.

        Args:
            target_vectors: Raw human hand vectors (N, 3)

        Returns:
            Reference vectors with projection applied (N, 3)
        """
        # Scale all vectors
        scaled_vectors = target_vectors * self.scaling

        # For projected vectors, replace with projected version
        target_dist = np.linalg.norm(target_vectors[:self.num_proj], axis=1, keepdims=True)
        target_dir = target_vectors[:self.num_proj] / np.maximum(target_dist, 1e-8)

        # Projected vectors: direction * projected_dist
        projected_vectors = target_dir * self.projected_dist[:, None]

        # Choose between normal and projected
        reference_vectors = np.where(
            self.projected[:, None],
            projected_vectors,
            scaled_vectors[:self.num_proj]
        )

        # Append wrist-to-fingertip vectors (no projection)
        reference_vectors = np.concatenate([
            reference_vectors,
            scaled_vectors[self.num_proj:]
        ], axis=0)

        return reference_vectors

    def _compute_weights(self) -> np.ndarray:
        """Compute adaptive weights for DexPilot loss.

        Returns:
            Weight vector for all vectors
        """
        # Base weights for projection vectors
        normal_weight = np.ones(self.num_proj, dtype=np.float64)
        high_weight = np.concatenate([
            np.full(self.num_s1, 200.0),
            np.full(self.num_s2, 400.0)
        ])
        proj_weights = np.where(self.projected, high_weight, normal_weight)

        # Higher weight for wrist-to-fingertip vectors (better tracking)
        wrist_weights = np.full(self.num_fingers, self.num_proj + self.num_fingers, dtype=np.float64)

        return np.concatenate([proj_weights, wrist_weights])

    def _to_independent_qpos(
        self,
        qpos: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """Normalize a full or independent qpos to the optimizer variable space."""
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape == (self.num_joints,):
            return qpos[self.independent_indices].copy()
        if qpos.shape == (self.num_opt_vars,):
            return qpos.copy()
        raise ValueError(
            f"{name} must have shape ({self.num_joints},) (full) or "
            f"({self.num_opt_vars},) (independent), got {qpos.shape}"
        )

    def _compute_objective_and_gradient(
        self,
        qpos_independent: np.ndarray,
        reference_vectors: np.ndarray,
        regularization_qpos: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Compute the DexPilot objective in the independent-joint space.

        Projection state and reference-vector construction are intentionally
        performed once per input frame by ``solve``/``compute_cost``.  This
        function receives the already projected reference vectors so NLopt
        evaluations cannot accidentally project or scale them a second time.
        """
        qpos_independent = self._to_independent_qpos(
            qpos_independent, name="qpos_independent"
        )
        reference_vectors = np.asarray(reference_vectors, dtype=np.float64)
        expected_vectors = self.num_proj + self.num_fingers
        if reference_vectors.shape != (expected_vectors, 3):
            raise ValueError(
                "reference_vectors must have shape "
                f"({expected_vectors}, 3), got {reference_vectors.shape}"
            )

        weights = self._compute_weights()

        # Forward kinematics at the actual target points.  On Linker L20 the
        # distal frame origin is the DIP/IP joint; ``computed_link_offsets``
        # moves it to the physical fingertip surface.  Positions and Jacobians
        # must use the exact same point for a consistent analytical gradient.
        qpos_full = self.expand_to_full_qpos(qpos_independent)
        link_positions = self.robot.compute_points_batch(
            qpos_full, self.computed_link_ids, self.computed_link_offsets
        ) * M_TO_CM

        origin_pos = link_positions[self.origin_link_indices]
        task_pos = link_positions[self.task_link_indices]
        robot_vectors = task_pos - origin_pos
        vec_diff = robot_vectors - reference_vectors
        vec_norm = np.linalg.norm(vec_diff, axis=1)

        huber_loss = huber_loss_np(vec_norm, delta=self.huber_delta)
        objective = float(np.mean(huber_loss * weights))

        # d rho(||e||) / d e, including the exact zero-residual case.
        grad_vec = (
            weights[:, None]
            * huber_norm_gradient_np(vec_diff, self.huber_delta)
            / len(robot_vectors)
        )

        jacobians = self.robot.compute_all_jacobians_batch_with_offsets(
            qpos_full, self.computed_link_ids, self.computed_link_offsets
        ) * M_TO_CM

        grad_link_pos = np.zeros(
            (len(self.computed_link_names), 3), dtype=np.float64
        )
        np.add.at(grad_link_pos, self.task_link_indices, grad_vec)
        np.add.at(grad_link_pos, self.origin_link_indices, -grad_vec)

        grad_full = np.zeros(self.num_joints, dtype=np.float64)
        for link_grad, jacobian in zip(grad_link_pos, jacobians):
            grad_full += link_grad @ jacobian
        grad_independent = self.map_gradient_to_independent(grad_full)

        # As in the reference implementation, regularize against the previous
        # solution.  Keep both the value and gradient in the same independent
        # variable space; self.last_qpos itself is a full qpos because mimic
        # joints are expanded by BaseOptimizer.
        if regularization_qpos is not None:
            regularization_qpos = self._to_independent_qpos(
                regularization_qpos, name="regularization_qpos"
            )
            delta = qpos_independent - regularization_qpos
            objective += self.norm_delta * float(delta @ delta)
            grad_independent += 2.0 * self.norm_delta * delta

        return objective, grad_independent

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve for a full robot qpos using DexPilot retargeting.

        ``last_qpos`` may be either a full qpos or an independent-joint vector.
        When omitted, the previous full solution maintained by BaseOptimizer is
        used for both warm start and temporal regularization.
        """
        target_vectors = self._extract_target_vectors(mediapipe_keypoints)
        self._update_projection_state(target_vectors)
        reference_vectors = self._compute_reference_vectors(target_vectors)

        previous_qpos = last_qpos if last_qpos is not None else self.last_qpos
        if previous_qpos is not None:
            init_qpos = self._to_independent_qpos(
                previous_qpos, name="last_qpos"
            )
            regularization_qpos = init_qpos.copy()
        else:
            init_qpos = self._get_init_qpos(None)
            regularization_qpos = None
        init_qpos = np.clip(
            init_qpos, self.independent_lower_bounds, self.independent_upper_bounds
        )

        def objective_fn(x: np.ndarray, grad: np.ndarray) -> float:
            obj, grad_val = self._compute_objective_and_gradient(
                x, reference_vectors, regularization_qpos
            )
            if grad.size > 0:
                grad[:] = grad_val
            return obj

        return self._run_optimization(objective_fn, init_qpos)

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute the current DexPilot objective for a full/independent qpos."""
        qpos_independent = self._to_independent_qpos(qpos, name="qpos")
        target_vectors = self._extract_target_vectors(mediapipe_keypoints)
        self._update_projection_state(target_vectors)
        reference_vectors = self._compute_reference_vectors(target_vectors)

        # If qpos is the just-computed solution, this temporal term is zero;
        # for arbitrary poses it remains relative to the previous full result.
        regularization_qpos = self.last_qpos
        cost, _ = self._compute_objective_and_gradient(
            qpos_independent, reference_vectors, regularization_qpos
        )
        return float(cost)

    def retarget(
        self,
        human_hand_positions: np.ndarray,
        warmstart_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compatibility alias for :meth:`solve`; always returns full qpos."""
        return self.solve(human_hand_positions, last_qpos=warmstart_qpos)

    def _extract_target_vectors(self, human_hand_positions: np.ndarray) -> np.ndarray:
        """Extract DexPilot vectors from human hand.

        Creates all finger-to-finger and wrist-to-fingertip vectors.

        Args:
            human_hand_positions: (21, 3) MediaPipe keypoints in meters

        Returns:
            (N, 3) target vectors in centimeters
        """
        human_pos_cm = human_hand_positions * M_TO_CM

        # Wrist is always keypoint 0
        wrist_pos = human_pos_cm[0]

        # MediaPipe fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
        mp_tip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

        # Get fingertip positions for the number of fingers this robot has
        fingertip_positions = []
        for i in range(self.num_fingers):
            if i < len(mp_tip_indices):
                tip_idx = mp_tip_indices[i]
                fingertip_positions.append(human_pos_cm[tip_idx])
        fingertip_positions = np.array(fingertip_positions)  # (num_fingers, 3)

        vectors = []

        # Finger-to-finger vectors (higher to lower)
        for i in range(0, self.num_fingers - 1):
            for j in range(i + 1, self.num_fingers):
                vec = fingertip_positions[i] - fingertip_positions[j]
                vectors.append(vec)

        # Wrist-to-fingertip vectors
        for i in range(self.num_fingers):
            vec = fingertip_positions[i] - wrist_pos
            vectors.append(vec)

        return np.array(vectors, dtype=np.float64)
