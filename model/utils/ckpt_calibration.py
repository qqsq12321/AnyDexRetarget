"""Utilities for applying calibration MLP checkpoints at runtime."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from dataset import CalibrationDataset
from features import FINGER_NAMES, split_prediction
from network import CalibrationMLP, torch


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def predict_calibration(sample_path: str | Path, checkpoint_path: str | Path) -> tuple[np.ndarray, float]:
    """Predict ``segment_scaling`` and ``pinch_scaling`` from a saved sample."""
    if torch is None:
        raise RuntimeError("PyTorch is required to load calibration checkpoints")

    sample_path = resolve_project_path(sample_path)
    checkpoint_path = resolve_project_path(checkpoint_path)
    dataset = CalibrationDataset([sample_path])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = CalibrationMLP(int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = dataset.features.astype(np.float32)
    mean = ckpt.get("feature_mean")
    std = ckpt.get("feature_std")
    if mean is not None and std is not None:
        x = (x - mean) / std

    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32)).numpy()[0]
    return split_prediction(pred)


def apply_prediction_to_config(
    config: dict,
    segment_scaling: np.ndarray,
    pinch_scaling: float,
) -> dict:
    """Return a copied config with predicted calibration values applied."""
    updated = copy.deepcopy(config)
    retarget = updated.setdefault("retarget", {})
    retarget["segment_scaling"] = {
        name: [round(float(v), 4) for v in values]
        for name, values in zip(FINGER_NAMES, segment_scaling)
    }
    retarget["pinch_scaling"] = round(float(pinch_scaling), 4)
    return updated


def load_config_with_prediction(
    config_path: str | Path,
    sample_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[dict, np.ndarray, float]:
    """Load YAML config and apply checkpoint-predicted calibration in memory."""
    config_path = resolve_project_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    segment_scaling, pinch_scaling = predict_calibration(sample_path, checkpoint_path)
    return apply_prediction_to_config(config, segment_scaling, pinch_scaling), segment_scaling, pinch_scaling


def print_prediction(segment_scaling: np.ndarray, pinch_scaling: float) -> None:
    """Print predicted calibration values in YAML-ready form."""
    print("Predicted calibration:")
    print("  segment_scaling:")
    for name, values in zip(FINGER_NAMES, segment_scaling):
        print(f"    {name}: [{', '.join(f'{float(v):.4f}' for v in values)}]")
    print(f"  pinch_scaling: {float(pinch_scaling):.4f}")
