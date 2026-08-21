# Calibration Model

This directory contains the first version of a model-based calibration pipeline for predicting adaptive retargeting parameters directly.

## Goal

Predict the final YAML values for:

- `segment_scaling`: 5 fingers x 4 segments (`wrist->MCP`, `MCP->PIP`, `PIP->DIP`, `DIP->TIP`)
- `pinch_scaling`: one uniform scale for the active pinch pair

The retargeting optimizer is still used for final qpos solving. The model only predicts calibration parameters.

## Sample format

Training samples are `.npz` files with:

```python
keypoints_open        # (T, 21, 3), open-hand keypoint sequence
robot_type            # string
input_type            # string: mediapipe/noitom/quest3/avp/pico4
hand_side             # string: left/right
segment_scaling       # (5, 4), final tuned values
pinch_scaling         # scalar, final tuned value

# Optional but recommended, so training does not need to load URDFs:
robot_segment_lengths # (5, 4)
robot_tip_reaches     # (5,)
robot_root_positions  # (5, 3)
```

## Features

`features.py` extracts:

- human open-hand bone lengths, median/std over time
- human wrist-to-tip reaches, median/std over time
- palm/root spread distances, median/std over time
- robot segment lengths and tip reaches
- input-source one-hot
- hand-side one-hot

## Training

```bash
python model/train.py \
  --data-dir model/data/calibrated \
  --output model/checkpoints/calibration_mlp.pt
```

## Prediction

```bash
python model/predict.py \
  --sample model/data/calibrated/sample.npz \
  --checkpoint model/checkpoints/calibration_mlp.pt
```

The predictor prints YAML-ready `segment_scaling` and `pinch_scaling` values.
