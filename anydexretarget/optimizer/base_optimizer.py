"""Base class for hand retargeting optimizers."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import nlopt
import numpy as np
import yaml

from ..robot import RobotWrapper
from .robot_configs import ROBOT_CONFIGS as _ROBOT_CONFIGS
from .utils import (
    M_TO_CM,
    CM_TO_M,
    TimingStats,
    LPFilter,
    huber_loss_np,
    huber_loss_grad_np,
)


# Project root for asset path resolution
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent


class BaseOptimizer(ABC):
    """Base class for hand retargeting optimizers.

    All parameters are read from configuration dict (loaded from YAML).
    Supports multiple robot hands (Wuji Hand, Shadow Hand, etc.) via configuration.
    """

    # MediaPipe keypoint indices
    MP_ORIGIN_IDX = 0  # Wrist
    MP_MCP_INDICES = [1, 5, 9, 13, 17]  # Finger roots (thumb uses CMC=1)
    MP_TIP_INDICES = [4, 8, 12, 16, 20]  # Fingertips
    MP_PIP_INDICES = [2, 6, 10, 14, 18]  # PIP joints (thumb uses MCP=2)
    MP_DIP_INDICES = [3, 7, 11, 15, 19]  # DIP joints

    ROBOT_CONFIGS = _ROBOT_CONFIGS

    def __init__(self, config: dict):
        """Initialize optimizer from configuration dict.

        Args:
            config: Configuration dict (typically loaded from YAML)
        """
        self.config = config

        # Extract optimizer config
        opt_config = config.get('optimizer', {})
        self.hand_side = opt_config.get('hand_side', 'right').lower()
        if self.hand_side not in ['right', 'left']:
            raise ValueError(f"hand_side must be 'right' or 'left', got {self.hand_side}")

        # Extract retarget config
        retarget_config = config.get('retarget', {})
        self.huber_delta = retarget_config.get('huber_delta', 2.0)
        self.norm_delta = retarget_config.get('norm_delta', 0.04)

        # Extract robot config
        robot_config = config.get('robot', {})
        robot_type = robot_config.get('type', 'shadow_hand')

        # Get robot-specific defaults
        if robot_type in self.ROBOT_CONFIGS:
            robot_defaults = self.ROBOT_CONFIGS[robot_type]
        else:
            robot_defaults = self.ROBOT_CONFIGS['shadow_hand']

        # Load URDF - support custom path or use default
        urdf_path = robot_config.get('urdf_path')
        if urdf_path:
            # Custom URDF path (absolute or relative to package root)
            urdf_path = Path(urdf_path)
            if not urdf_path.is_absolute():
                urdf_path = _PROJECT_ROOT / urdf_path
            if robot_type == 'inspire_hand' and self.hand_side == 'left':
                urdf_name = urdf_path.name
                if 'inspire_hand_right' in urdf_name:
                    urdf_path = urdf_path.with_name(urdf_name.replace('inspire_hand_right', 'inspire_hand_left'))
            urdf_path = str(urdf_path.resolve())
        else:
            # Default URDF path based on robot type and hand side
            urdf_subdir = robot_defaults['urdf_subdir']
            # Check for custom urdf_file config (e.g., for Menagerie with rh_/lh_ prefixes)
            urdf_file_config = robot_defaults.get('urdf_file')
            if urdf_file_config and isinstance(urdf_file_config, dict):
                urdf_filename = urdf_file_config.get(self.hand_side, f"{self.hand_side}.urdf")
            else:
                urdf_filename = f"{self.hand_side}.urdf"
            urdf_path = str((_PROJECT_ROOT / urdf_subdir / urdf_filename).resolve())

        self.robot = RobotWrapper(urdf_path)
        self.num_joints = self.robot.model.nq

        # Parse mimic joints from URDF
        self._parse_mimic_joints(urdf_path)

        # Setup NLopt optimizer (dimension = number of independent joints)
        self.num_opt_vars = len(self.independent_indices)
        self.opt = nlopt.opt(nlopt.LD_SLSQP, self.num_opt_vars)
        self.opt.set_maxeval(50)
        self.opt.set_ftol_abs(1e-4)

        # Apply joint limit overrides from config
        lower_bounds = self.robot.joint_limits[:, 0].copy()
        upper_bounds = self.robot.joint_limits[:, 1].copy()
        clamp_config = retarget_config.get('clamp_joint_lower', {})
        if clamp_config:
            for pattern, min_val in clamp_config.items():
                for ji in range(1, self.robot.model.njoints):
                    jname = self.robot.model.names[ji]
                    idx_q = self.robot.model.joints[ji].idx_q
                    nq = self.robot.model.joints[ji].nq
                    if nq > 0 and pattern in jname:
                        if lower_bounds[idx_q] < min_val:
                            lower_bounds[idx_q] = min_val

        self.opt_lower_bounds = lower_bounds
        self.opt_upper_bounds = upper_bounds
        # NLopt bounds only for independent joints
        self.opt.set_lower_bounds(lower_bounds[self.independent_indices].tolist())
        self.opt.set_upper_bounds(upper_bounds[self.independent_indices].tolist())

        # Link names - from config or robot defaults
        self.origin_link_name = robot_config.get('origin_link', robot_defaults['origin_link'])
        self.task_link_names = robot_config.get('tip_links', robot_defaults['tip_links'])
        self.link1_names = robot_config.get('link1_names', robot_defaults['link1_names'])
        self.link3_names = robot_config.get('link3_names', robot_defaults['link3_names'])
        self.link4_names = robot_config.get('link4_names', robot_defaults['link4_names'])
        self.task_offsets = self._resolve_link_offsets(
            robot_config.get('tip_offsets', robot_defaults.get('tip_offsets')),
            len(self.task_link_names),
        )
        self.link3_offsets = self._resolve_link_offsets(
            robot_config.get('link3_offsets', robot_defaults.get('link3_offsets')),
            len(self.link3_names),
        )
        self.link4_offsets = self._resolve_link_offsets(
            robot_config.get('link4_offsets', robot_defaults.get('link4_offsets')),
            len(self.link4_names),
        )
        neutral_qpos = robot_config.get('neutral_qpos', robot_defaults.get('neutral_qpos'))
        self.neutral_qpos = None if neutral_qpos is None else np.asarray(neutral_qpos, dtype=np.float64)

        # Number of fingers (4 for Allegro/Leap, 5 for others)
        self.num_fingers = robot_config.get('num_fingers', robot_defaults.get('num_fingers', 5))

        # For 4-finger hands, map MediaPipe 5-finger indices to 4-finger robot
        # MediaPipe: thumb=0, index=1, middle=2, ring=3, pinky=4
        # 4-finger: thumb=0, index=1, middle=2, ring=3 (pinky ignored)
        if self.num_fingers == 4:
            self.mp_finger_indices = [0, 1, 2, 3]  # Skip pinky
        else:
            self.mp_finger_indices = [0, 1, 2, 3, 4]  # All 5

        # Handle left/right hand prefix for shadow_hand
        # The config uses 'rh_' prefix by default, replace with 'lh_' for left hand
        if robot_type == 'shadow_hand' and self.hand_side == 'left':
            def replace_prefix(name):
                return name.replace('rh_', 'lh_')
            self.origin_link_name = replace_prefix(self.origin_link_name)
            self.task_link_names = [replace_prefix(n) for n in self.task_link_names]
            self.link1_names = [replace_prefix(n) for n in self.link1_names]
            self.link3_names = [replace_prefix(n) for n in self.link3_names]
            self.link4_names = [replace_prefix(n) for n in self.link4_names]
        elif robot_type == 'unitree_dex5_hand' and self.hand_side == 'left':
            def replace_suffix(name):
                if name == 'base_link00':
                    return 'base_link00L'
                return f"{name[:-1]}L" if name.endswith('R') else name
            self.origin_link_name = replace_suffix(self.origin_link_name)
            self.task_link_names = [replace_suffix(n) for n in self.task_link_names]
            self.link1_names = [replace_suffix(n) for n in self.link1_names]
            self.link3_names = [replace_suffix(n) for n in self.link3_names]
            self.link4_names = [replace_suffix(n) for n in self.link4_names]
        elif robot_type in ('sharpa_hand', 'gaia_hand20') and self.hand_side == 'left':
            def replace_prefix(name):
                return name.replace('right_', 'left_')
            self.origin_link_name = replace_prefix(self.origin_link_name)
            self.task_link_names = [replace_prefix(n) for n in self.task_link_names]
            self.link1_names = [replace_prefix(n) for n in self.link1_names]
            self.link3_names = [replace_prefix(n) for n in self.link3_names]
            self.link4_names = [replace_prefix(n) for n in self.link4_names]
        elif robot_type == 'linker_l20' and self.hand_side == 'left':
            def replace_prefix(name):
                return name.replace('right_', 'left_')
            self.origin_link_name = replace_prefix(self.origin_link_name)
            self.task_link_names = [replace_prefix(n) for n in self.task_link_names]
            self.link1_names = [replace_prefix(n) for n in self.link1_names]
            self.link3_names = [replace_prefix(n) for n in self.link3_names]
            self.link4_names = [replace_prefix(n) for n in self.link4_names]
        elif robot_type == 'inspire_hand' and self.hand_side == 'left':
            def replace_prefix(name):
                return name.replace('right_', 'left_')
            self.origin_link_name = replace_prefix(self.origin_link_name)
            self.task_link_names = [replace_prefix(n) for n in self.task_link_names]
            self.link1_names = [replace_prefix(n) for n in self.link1_names]
            self.link3_names = [replace_prefix(n) for n in self.link3_names]
            self.link4_names = [replace_prefix(n) for n in self.link4_names]

        # Build link indices
        self._build_link_indices()

        # Cache the robot's own finger roots, expressed in the optimizer frame.
        # These are mounting points fixed by the URDF, so one FK pass is enough.
        # FullHandVec anchors each finger chain here instead of at the human
        # wrist, which keeps the palm-length mismatch out of segment_scaling.
        self.finger_root_vectors = self._compute_finger_root_vectors()

        # Store last solution for warm start
        self.last_qpos = None

    def _compute_finger_root_vectors(self) -> np.ndarray:
        """Return origin->finger-root vectors at the neutral pose, in cm."""
        qpos = (
            self.neutral_qpos.copy()
            if self.neutral_qpos is not None
            else np.zeros(self.robot.model.nq, dtype=np.float64)
        )
        self.robot.compute_forward_kinematics(qpos)
        origin = self.robot.get_link_pose(
            self.robot.get_link_index(self.origin_link_name)
        )[:3, 3]
        roots = np.zeros((self.num_fingers, 3), dtype=np.float64)
        for i in range(min(self.num_fingers, len(self.link1_names))):
            link_id = self.robot.get_link_index(self.link1_names[i])
            roots[i] = self.robot.get_link_pose(link_id)[:3, 3] - origin
        return roots * M_TO_CM

    @staticmethod
    def _resolve_link_offsets(offsets_config, count: int) -> np.ndarray:
        """Normalize per-link local offsets to shape (count, 3)."""
        offsets = np.zeros((count, 3), dtype=np.float64)
        if offsets_config is None:
            return offsets

        arr = np.asarray(offsets_config, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError(f"Offset must have 3 values, got shape {arr.shape}")
            offsets[:] = arr.reshape(1, 3)
            return offsets
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Offsets must have shape (N, 3), got {arr.shape}")

        n = min(count, arr.shape[0])
        offsets[:n] = arr[:n]
        return offsets

    def _build_link_indices(self):
        """Build link indices for FK computation."""
        self.computed_link_names = []
        self.computed_link_indices = []
        self.computed_link_offsets = []

        def add_point(name: str, offset: np.ndarray | None = None) -> int:
            self.computed_link_names.append(name)
            self.computed_link_indices.append(self.robot.get_link_index(name))
            if offset is None:
                self.computed_link_offsets.append(np.zeros(3, dtype=np.float64))
            else:
                self.computed_link_offsets.append(np.asarray(offset, dtype=np.float64))
            return len(self.computed_link_indices) - 1

        origin_idx = add_point(self.origin_link_name)
        self.origin_indices = [origin_idx for _ in range(self.num_fingers)]
        self.task_indices = [
            add_point(name, offset)
            for name, offset in zip(self.task_link_names, self.task_offsets)
        ]
        self.link3_indices = [
            add_point(name, offset)
            for name, offset in zip(self.link3_names, self.link3_offsets)
        ]
        self.link4_indices = [
            add_point(name, offset)
            for name, offset in zip(self.link4_names, self.link4_offsets)
        ]
        self.computed_link_offsets = np.asarray(self.computed_link_offsets, dtype=np.float64)

    def _parse_mimic_joints(self, urdf_path: str):
        """Parse mimic joint relationships from URDF.

        Sets up:
            self.independent_indices: indices of independent joints in full qpos
            self.mimic_map: dict mapping mimic_qidx -> (source_qidx, multiplier, offset)
            self.has_mimic: whether any mimic joints exist
        """
        # Build joint name -> idx_q mapping from pinocchio model
        joint_name_to_qidx = {}
        for ji in range(1, self.robot.model.njoints):
            jname = self.robot.model.names[ji]
            nq = self.robot.model.joints[ji].nq
            if nq > 0:
                joint_name_to_qidx[jname] = self.robot.model.joints[ji].idx_q

        # Parse URDF XML for mimic tags
        self.mimic_map = {}  # mimic_qidx -> (source_qidx, multiplier, offset)
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for joint_elem in root.iter('joint'):
                mimic_elem = joint_elem.find('mimic')
                if mimic_elem is not None:
                    joint_name = joint_elem.get('name')
                    source_name = mimic_elem.get('joint')
                    multiplier = float(mimic_elem.get('multiplier', '1.0'))
                    offset = float(mimic_elem.get('offset', '0.0'))

                    if joint_name in joint_name_to_qidx and source_name in joint_name_to_qidx:
                        mimic_qidx = joint_name_to_qidx[joint_name]
                        source_qidx = joint_name_to_qidx[source_name]
                        self.mimic_map[mimic_qidx] = (source_qidx, multiplier, offset)
        except (ET.ParseError, FileNotFoundError):
            pass

        # Determine independent joint indices
        mimic_indices = set(self.mimic_map.keys())
        self.independent_indices = np.array(
            [i for i in range(self.num_joints) if i not in mimic_indices],
            dtype=np.int64
        )
        self.has_mimic = len(self.mimic_map) > 0

        if self.has_mimic:
            # Precompute gradient mapping: for each independent joint, which mimic joints depend on it?
            # mimic_dependents[source_qidx] = [(mimic_qidx, multiplier), ...]
            self._mimic_dependents = {}
            for mimic_qidx, (source_qidx, mult, offset) in self.mimic_map.items():
                if source_qidx not in self._mimic_dependents:
                    self._mimic_dependents[source_qidx] = []
                self._mimic_dependents[source_qidx].append((mimic_qidx, mult))

    def expand_to_full_qpos(self, opt_vars: np.ndarray) -> np.ndarray:
        """Expand independent joint values to full qpos with mimic constraints.

        Args:
            opt_vars: (num_opt_vars,) independent joint values

        Returns:
            full_qpos: (num_joints,) full joint vector with mimic values filled in
        """
        if not self.has_mimic:
            return opt_vars.copy()

        full_qpos = np.zeros(self.num_joints, dtype=np.float64)
        full_qpos[self.independent_indices] = opt_vars

        for mimic_qidx, (source_qidx, mult, offset) in self.mimic_map.items():
            full_qpos[mimic_qidx] = full_qpos[source_qidx] * mult + offset

        # Clip mimic joints to their limits
        full_qpos = np.clip(full_qpos, self.opt_lower_bounds, self.opt_upper_bounds)
        return full_qpos

    def map_gradient_to_independent(self, full_grad: np.ndarray) -> np.ndarray:
        """Map full gradient to independent joint gradient using chain rule.

        For mimic joint j with q_j = q_src * mult + offset:
            dL/dq_src += dL/dq_j * mult

        Args:
            full_grad: (num_joints,) gradient w.r.t. full qpos

        Returns:
            opt_grad: (num_opt_vars,) gradient w.r.t. independent joints
        """
        if not self.has_mimic:
            return full_grad.copy()

        # Start with direct gradients for independent joints
        mapped_grad = full_grad.copy()

        # Add chain rule contributions from mimic joints
        for source_qidx, deps in self._mimic_dependents.items():
            for mimic_qidx, mult in deps:
                mapped_grad[source_qidx] += mapped_grad[mimic_qidx] * mult

        return mapped_grad[self.independent_indices]

    @classmethod
    def from_yaml(cls, yaml_path: str, hand_side: str = None) -> "BaseOptimizer":
        """Create optimizer from YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file
            hand_side: Optional hand side override ('left' or 'right')

        Returns:
            Optimizer instance
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Override hand_side if provided
        if hand_side is not None:
            if 'optimizer' not in config:
                config['optimizer'] = {}
            config['optimizer']['hand_side'] = hand_side

        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: dict) -> "BaseOptimizer":
        """Create optimizer from configuration dict.

        Args:
            config: Configuration dict

        Returns:
            Optimizer instance
        """
        from .analytical_optimizer import AdaptiveOptimizerAnalytical

        opt_type = config.get('optimizer', {}).get('type', 'AdaptiveOptimizerAnalytical')

        if opt_type == 'AdaptiveOptimizerAnalytical':
            return AdaptiveOptimizerAnalytical(config)
        elif opt_type == 'AdaptiveOptimizerAnalyticalV2':
            from .analytical_optimizer_v2 import AdaptiveOptimizerAnalyticalV2
            return AdaptiveOptimizerAnalyticalV2(config)
        elif opt_type == 'KeyVectorOptimizer':
            from .key_vector_optimizer import KeyVectorOptimizer
            return KeyVectorOptimizer(config)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    @abstractmethod
    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for joint angles.

        Args:
            mediapipe_keypoints: (21, 3) MediaPipe keypoints in wrist frame
            last_qpos: Initial guess for optimization (warm start)

        Returns:
            qpos: (num_joints,) joint angles
        """
        pass

    @abstractmethod
    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute cost for given joint angles.

        Args:
            qpos: Joint angles
            mediapipe_keypoints: (21, 3) MediaPipe keypoints

        Returns:
            cost: Total loss value
        """
        pass

    # =========================================================================
    # Common helper methods (shared by subclasses)
    # =========================================================================

    def _get_init_qpos(self, last_qpos: Optional[np.ndarray]) -> np.ndarray:
        """Get initial qpos for optimization (independent joints only, clipped).

        Args:
            last_qpos: Optional last qpos from caller (full qpos)

        Returns:
            Initial values for independent joints (num_opt_vars,)
        """
        if last_qpos is not None:
            full_qpos = np.asarray(last_qpos, dtype=np.float64)
        elif self.last_qpos is not None:
            full_qpos = self.last_qpos
        elif self.neutral_qpos is not None:
            full_qpos = self.neutral_qpos
        else:
            full_qpos = (self.opt_lower_bounds + self.opt_upper_bounds) / 2.0

        full_qpos = np.clip(full_qpos, self.opt_lower_bounds, self.opt_upper_bounds)
        return full_qpos[self.independent_indices]

    def _get_reg_qpos(self, last_qpos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Get regularization qpos for norm_delta term (full qpos).

        Args:
            last_qpos: Optional last qpos from caller

        Returns:
            Regularization qpos (full) or None
        """
        if last_qpos is not None:
            return np.asarray(last_qpos, dtype=np.float64)
        elif self.last_qpos is not None:
            return self.last_qpos
        return None

    def _run_optimization(self, objective_fn, init_qpos: np.ndarray) -> np.ndarray:
        """Run NLopt optimization and update last_qpos.

        Args:
            objective_fn: NLopt objective function (operates on independent joints)
            init_qpos: Initial values for independent joints (num_opt_vars,)

        Returns:
            Optimized full qpos (num_joints,)
        """
        self.opt.set_min_objective(objective_fn)
        try:
            opt_result = self.opt.optimize(init_qpos.tolist())
            opt_vars = np.array(opt_result, dtype=np.float64)
        except RuntimeError as e:
            print(f"[{self.__class__.__name__}] Optimization failed: {e}")
            opt_vars = np.array(init_qpos, dtype=np.float64)

        # Expand to full qpos
        full_qpos = self.expand_to_full_qpos(opt_vars)
        self.last_qpos = full_qpos.astype(np.float64)
        return full_qpos.astype(np.float32)

    def _compute_tip_dirs(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute DIP->tip direction vectors (normalized).

        Args:
            keypoints: (21, 3) MediaPipe keypoints

        Returns:
            tip_dirs: (num_fingers, 3) normalized direction vectors
        """
        tip_dirs = []
        for fi in self.mp_finger_indices:
            dip_idx = self.MP_DIP_INDICES[fi]
            tip_idx = self.MP_TIP_INDICES[fi]
            dir_vec = keypoints[tip_idx] - keypoints[dip_idx]
            norm = np.linalg.norm(dir_vec)
            tip_dirs.append(dir_vec / (norm + 1e-8))
        return np.array(tip_dirs, dtype=np.float64)

    def _compute_tip_vectors(self, keypoints: np.ndarray, scaling: float = 1.0) -> np.ndarray:
        """Compute uniformly scaled origin->tip targets in cm."""
        keypoints_cm = np.asarray(keypoints, dtype=np.float64) * M_TO_CM
        wrist = keypoints_cm[self.MP_ORIGIN_IDX]
        tip_indices = [self.MP_TIP_INDICES[i] for i in self.mp_finger_indices]
        return (keypoints_cm[tip_indices] - wrist) * float(scaling)

    def _compute_full_hand_vectors(
        self,
        keypoints: np.ndarray,
        scaling: np.ndarray,
        lateral_scaling: float = 1.0,
        lateral_axis: int = 1,
    ) -> np.ndarray:
        """Compute origin->PIP/DIP/TIP targets by growing each finger chain.

        Every span from the wrist outwards is scaled independently, including
        the wrist->MCP one: the chain is grown as

            MCP = (MCP - wrist)  * s_mcp
            PIP = MCP + (PIP - MCP) * s_pip
            DIP = PIP + (DIP - PIP) * s_dip
            TIP = DIP + (TIP - DIP) * s_tip

        Nothing is anchored to the robot's own finger mounts.  Keeping the
        human's palm directions means opposition contacts survive scaling,
        while ``s_mcp`` still stretches the palm span to the robot's size.

        Args:
            keypoints: (21, 3) MediaPipe keypoints in meters
            scaling: (num_fingers, 4) per-finger wrist->MCP, MCP->PIP,
                PIP->DIP, DIP->TIP factors
            lateral_scaling: optional scale for the palm's lateral axis before
                wrist->MCP scaling.  Values below 1.0 reduce MCP abduction
                without changing the finger segment lengths.
            lateral_axis: palm coordinate axis to scale

        Returns:
            vectors: (num_fingers*3, 3) vectors in cm [PIP*N, DIP*N, TIP*N]
        """
        nf = self.num_fingers
        keypoints_cm = np.asarray(keypoints, dtype=np.float64) * M_TO_CM
        scaling = np.asarray(scaling, dtype=np.float64)
        if scaling.shape[1] != 4:
            raise ValueError(
                f"Expected (num_fingers, 4) scaling with an MCP column, got {scaling.shape}"
            )

        wrist = keypoints_cm[self.MP_ORIGIN_IDX]
        mcp_indices = [self.MP_MCP_INDICES[i] for i in self.mp_finger_indices]
        pip_indices = [self.MP_PIP_INDICES[i] for i in self.mp_finger_indices]
        dip_indices = [self.MP_DIP_INDICES[i] for i in self.mp_finger_indices]
        tip_indices = [self.MP_TIP_INDICES[i] for i in self.mp_finger_indices]

        palm = keypoints_cm[mcp_indices] - wrist
        if lateral_scaling != 1.0:
            palm[:, lateral_axis] *= lateral_scaling
        proximal = keypoints_cm[pip_indices] - keypoints_cm[mcp_indices]
        middle = keypoints_cm[dip_indices] - keypoints_cm[pip_indices]
        distal = keypoints_cm[tip_indices] - keypoints_cm[dip_indices]

        mcp_vectors = palm * scaling[:nf, 0:1]
        pip_vectors = mcp_vectors + proximal * scaling[:nf, 1:2]
        dip_vectors = pip_vectors + middle * scaling[:nf, 2:3]
        tip_vectors = dip_vectors + distal * scaling[:nf, 3:4]

        vectors = np.vstack([pip_vectors, dip_vectors, tip_vectors])
        return vectors.astype(np.float64)
