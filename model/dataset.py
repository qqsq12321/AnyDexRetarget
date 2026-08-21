"""Dataset helpers for calibration prediction samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .features import RobotGeometry, extract_features, target_vector
except ImportError:  # Allows running scripts from the model/ directory directly.
    from features import RobotGeometry, extract_features, target_vector


class CalibrationDataset:
    """In-memory dataset loaded from calibration .npz samples."""

    def __init__(self, sample_paths, robot_geometry_loader=None):
        self.sample_paths = [Path(p) for p in sample_paths]
        self.robot_geometry_loader = robot_geometry_loader
        self.features = []
        self.targets = []
        self.metadata = []
        self._load()

    def _load(self) -> None:
        for path in self.sample_paths:
            sample = np.load(path, allow_pickle=True)
            keypoints = sample["keypoints_open"]
            input_type = str(sample["input_type"])
            hand_side = str(sample["hand_side"])
            segment_scaling = sample["segment_scaling"]
            pinch_scaling = float(sample["pinch_scaling"])

            if "robot_segment_lengths" in sample and "robot_tip_reaches" in sample:
                robot_geometry = RobotGeometry(
                    segment_lengths=sample["robot_segment_lengths"],
                    tip_reaches=sample["robot_tip_reaches"],
                    root_positions=sample["robot_root_positions"] if "robot_root_positions" in sample else None,
                )
            elif self.robot_geometry_loader is not None:
                robot_geometry = self.robot_geometry_loader(str(sample["robot_type"]), hand_side)
            else:
                raise ValueError(
                    f"{path} does not include robot geometry; provide robot_geometry_loader"
                )

            self.features.append(extract_features(keypoints, robot_geometry, input_type, hand_side))
            self.targets.append(target_vector(segment_scaling, pinch_scaling))
            self.metadata.append({
                "path": str(path),
                "robot_type": str(sample["robot_type"]),
                "input_type": input_type,
                "hand_side": hand_side,
            })

        self.features = np.asarray(self.features, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def find_samples(data_dir: str | Path) -> list[Path]:
    """Return sorted .npz sample paths under a directory."""
    return sorted(Path(data_dir).glob("*.npz"))
