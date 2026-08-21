#!/usr/bin/env python3
"""Run calibration-parameter prediction for one saved sample."""

from __future__ import annotations

import argparse

import numpy as np

from dataset import CalibrationDataset
from features import FINGER_NAMES, split_prediction
from network import CalibrationMLP, torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict segment_scaling and pinch_scaling")
    parser.add_argument("--sample", required=True, help=".npz calibration sample")
    parser.add_argument("--checkpoint", required=True, help="Trained .pt checkpoint")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("PyTorch is required for prediction")

    dataset = CalibrationDataset([args.sample])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    segment, pinch = split_prediction(pred)
    print("segment_scaling:")
    for name, values in zip(FINGER_NAMES, segment):
        print(f"  {name}: [{', '.join(f'{v:.4f}' for v in values)}]")
    print(f"pinch_scaling: {pinch:.4f}")


if __name__ == "__main__":
    main()
