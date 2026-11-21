"""MuJoCo model selection and retarget-output post-processing."""

from pathlib import Path

import mujoco
import numpy as np

from anydexretarget import Retargeter

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ROBOT_HAND_CONFIGS = {
    "shadow_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "shadow_hand" / f"scene_{side}.xml"),
        "needs_menagerie_mapping": True,
        "base_quat": (0.7071068, 0, 0, 0.7071068),  # Rotate 90 deg around Z to align with MuJoCo model
    },
    "wuji_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "wuji_hand" / "right.xml"),
    },
    "gaia_hand20": {
        "model_path": lambda side: str(
            PROJECT_ROOT / "assets" / "gaia_hand20" / f"gaiahand20_{side}_mujoco.xml"
        ),
        # Pinocchio traverses the sibling chains alphabetically, while the
        # converted MJCF keeps the URDF chain order.
        "qpos_mapping": [
            16, 17, 18, 19,
            0, 1, 2, 3,
            8, 9, 10, 11,
            12, 13, 14, 15,
            4, 5, 6, 7,
        ],
        "qpos_servo_alpha": 0.2,
    },
    "allegro_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "allegro_hand" / "scene_right.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    },
    "inspire_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "inspire_hand" / "inspire_hand_right_mujoco.xml"),
        "qpos_mapping": [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5],
    },
    "ability_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "ability_hand" / "ability_hand_right_mujoco.xml"),
        "qpos_mapping": [8, 9, 0, 1, 2, 3, 6, 7, 4, 5],
    },
    "leap_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "leap_hand" / "leap_hand_right_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    },
    "svh_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "schunk_hand" / "schunk_svh_hand_right_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 13, 14, 15, 16, 9, 10, 11, 12, 4, 5, 6, 7, 17, 18, 19],
    },
    "linkerhand_l21": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "linkerhand_l21" / f"linkerhand_l21_{side}_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16],
        "qpos_servo_alpha": 0.2,
    },
    "linker_l6": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "linker_l6" / f"linker_l6_{side}_mujoco.xml"),
        "actuator_joint_names": [
            "thumbCMC", "thumbMCP", "indexMCP", "middleMCP", "ringMCP", "littleMCP",
        ],
    },
    "linker_l20": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "linker_l20" / f"linker_l20_{side}_mujoco.xml"),
        # This is the actuator order declared by the reference L20 MJCF.  It is
        # deliberately name based because Pinocchio qpos order is different.
        "actuator_joint_names": [
            "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
            "index_mcp_roll", "index_mcp_pitch", "index_pip",
            "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
            "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
            "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
        ],
    },
    "rohand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "rohand" / f"rohand_{side}_mujoco.xml"),
        "qpos_mapping": [3, 4, 1, 2, 0, 13, 14, 11, 12, 10, 18, 19, 16, 17, 15, 8, 9, 6, 7, 5, 20, 21, 23, 24, 22],
        "qpos_servo_alpha": 0.18,
        "base_quat": (0.7071068, 0, 0.7071068, 0),
    },
    "unitree_dex5_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "unitree_dex5_hand" / f"unitree_dex5_hand_{side}_mujoco.xml"),
        "qpos_mapping": [16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "qpos_servo_alpha": 0.2,
        "base_quat": (0.7071068, 0.7071068, 0, 0),
    },
    "sharpa_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "sharpa_hand" / f"{side}_sharpa_wave.xml"),
        "qpos_mapping": [17, 18, 19, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 8, 9, 10, 11, 12],
    },
}


def map_urdf_to_mujoco_menagerie(qpos: np.ndarray) -> np.ndarray:
    """Map URDF joint angles (22 DoF) to MuJoCo Menagerie actuators (20 DoF)."""
    ctrl = np.zeros(20, dtype=np.float32)
    ctrl[0] = 0.0
    ctrl[1] = 0.0
    ctrl[2] = qpos[17]
    ctrl[3] = qpos[18]
    ctrl[4] = qpos[19]
    ctrl[5] = qpos[20]
    ctrl[6] = qpos[21]
    ctrl[7] = qpos[0]
    ctrl[8] = qpos[1]
    ctrl[9] = qpos[2] + qpos[3]
    ctrl[10] = qpos[9]
    ctrl[11] = qpos[10]
    ctrl[12] = qpos[11] + qpos[12]
    ctrl[13] = qpos[13]
    ctrl[14] = qpos[14]
    ctrl[15] = qpos[15] + qpos[16]
    ctrl[16] = qpos[4]
    ctrl[17] = qpos[5]
    ctrl[18] = qpos[6]
    ctrl[19] = qpos[7] + qpos[8]
    return ctrl


def map_retarget_qpos(
    qpos: np.ndarray,
    hand_cfg: dict,
    joint_names=None,
) -> np.ndarray:
    """Map retarget output into the MuJoCo actuator order."""
    if hand_cfg.get("needs_menagerie_mapping"):
        return map_urdf_to_mujoco_menagerie(qpos)

    actuator_joint_names = hand_cfg.get("actuator_joint_names")
    if actuator_joint_names is not None:
        if joint_names is None:
            raise ValueError("joint_names are required for name-based actuator mapping")
        qpos = np.asarray(qpos)
        joint_names = [str(name) for name in joint_names]
        if qpos.ndim != 1 or len(qpos) != len(joint_names):
            raise ValueError(
                "Retarget qpos and joint_names must be matching 1-D arrays: "
                f"qpos.shape={qpos.shape}, len(joint_names)={len(joint_names)}"
            )
        normalized_names = [name.lower() for name in joint_names]
        duplicates = sorted(
            {name for name in normalized_names if normalized_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicate retarget joint names: {duplicates}")
        index_by_name = {name: idx for idx, name in enumerate(normalized_names)}
        missing = [name for name in actuator_joint_names if name.lower() not in index_by_name]
        if missing:
            raise ValueError(f"Retarget output is missing actuator joints: {missing}")
        indices = [index_by_name[name.lower()] for name in actuator_joint_names]
        return qpos[indices]

    if "qpos_mapping" in hand_cfg:
        return np.asarray(qpos)[hand_cfg["qpos_mapping"]]
    return np.asarray(qpos)


def retarget_to_mujoco_target(
    fingers_data: dict,
    hand_side: str,
    retargeter: Retargeter,
    hand_cfg: dict,
    target_len: int,
):
    """Retarget hand tracking input and map to MuJoCo control target."""
    fingers_pose = fingers_data[f"{hand_side}_fingers"]
    if np.allclose(fingers_pose, 0):
        return None

    qpos = retargeter.retarget(fingers_pose)

    target = map_retarget_qpos(
        qpos,
        hand_cfg,
        retargeter.optimizer.robot.dof_joint_names,
    )

    target = np.asarray(target, dtype=np.float32)
    if len(target) != target_len:
        if hand_cfg.get("actuator_joint_names") is not None:
            raise ValueError(
                "Name-based actuator mapping must exactly match MuJoCo control "
                f"size: mapped={len(target)}, target_len={target_len}"
            )
        buf = np.zeros(target_len, dtype=np.float32)
        n = min(len(target), target_len)
        buf[:n] = target[:n]
        target = buf
    return target


def apply_qpos_to_mujoco(model, data, qpos, hand_cfg, joint_names=None):
    """Apply retarget output qpos to MuJoCo model and step simulation."""
    ctrl = map_retarget_qpos(qpos, hand_cfg, joint_names)
    ctrl = np.asarray(ctrl, dtype=np.float32)

    qpos_servo_alpha = hand_cfg.get("qpos_servo_alpha")
    if qpos_servo_alpha is not None:
        n = min(len(ctrl), model.nq)
        data.qpos[:n] = ctrl[:n]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    elif model.nu > 0:
        n = min(len(ctrl), model.nu)
        data.ctrl[:n] = ctrl[:n]
        for _ in range(200):
            mujoco.mj_step(model, data)
    else:
        n = min(len(ctrl), model.nq)
        data.qpos[:n] = ctrl[:n]
        mujoco.mj_forward(model, data)

def mujoco_actuator_joint_names(model) -> list[str]:
    """Return the joint targeted by every MuJoCo actuator, in ctrl order.

    Linker L20 uses joint-transmission position actuators.  Reading the binding
    from ``actuator_trnid`` lets us verify the XML instead of assuming that its
    actuator declaration order has not changed.
    """
    names: list[str] = []
    joint_transmissions = {
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
    }
    for actuator_id in range(model.nu):
        trn_type = int(model.actuator_trntype[actuator_id])
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        ) or f"actuator[{actuator_id}]"
        if trn_type not in joint_transmissions:
            raise ValueError(
                f"{actuator_name!r} is not a joint actuator (trntype={trn_type})"
            )
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            raise ValueError(
                f"Cannot resolve target joint for actuator {actuator_name!r}"
            )
        names.append(joint_name)
    return names


def validate_mujoco_actuator_mapping(model, hand_cfg: dict) -> list[str]:
    """Validate configured name mapping against actual MuJoCo ctrl order.

    Returns the actual actuator joint names.  Hands without a name-based
    actuator mapping are left unchanged and return an empty list.
    """
    configured = hand_cfg.get("actuator_joint_names")
    if configured is None:
        return []

    actual = mujoco_actuator_joint_names(model)
    configured = [str(name) for name in configured]
    if configured != actual:
        rows = []
        for index in range(max(len(configured), len(actual))):
            expected = configured[index] if index < len(configured) else "<missing>"
            bound = actual[index] if index < len(actual) else "<missing>"
            marker = "OK" if expected == bound else "MISMATCH"
            rows.append(
                f"  ctrl[{index:02d}] configured={expected:<22} "
                f"xml={bound:<22} {marker}"
            )
        raise ValueError(
            "Configured actuator joint order does not match the loaded MuJoCo "
            "model:\n" + "\n".join(rows)
        )
    return actual
