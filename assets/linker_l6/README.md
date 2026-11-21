# Linker L6 (simulation only)

Both hands have 11 revolute joints: six independent joints and five coupled
PIP joints. The runtime identifier is `linker_l6`. No hardware driver is added.

## Imported Assets

The URDFs and their referenced meshes were imported from the local
`remote_control_robot` checkout at commit
`fb96562da8553cc43b109d3ee3009ce54c251994`, under:

```text
hand_mapping/retargeting/dex-retargeting/assets/robots/hands/linker_l6_hand/
```

The source checkout's `dex-retargeting/LICENSE` is included as `LICENSE`.
Visual GLB (right) and DAE (left) references use the source's equivalent STL
files instead; referenced rough-collision OBJ files are also retained. Meshes
are in metres. All paths resolve locally; the source checkout is not needed.

The left source `thumbPIP` lower bound (`0.612611`) conflicts with its mimic:
`thumbPIP = 0.611762 * thumbMCP`, with `thumbMCP` in `[0.235968, 0.99]`.
The imported left URDF widens that lower bound to `0.144356255616` so the
original coupling can move throughout the MCP range. The source transforms,
axes and side-specific mimic offsets are otherwise preserved. In particular,
the right thumb has a `0.612611` mimic offset, while the left thumb has zero
offset; the two source models are not exact mirrors. This simulation fix is
not a hardware limit calibration.

## Simulation Models

Regenerate the checked-in MJCFs with MuJoCo installed:

```bash
python assets/linker_l6/generate_mjcf.py
```

The generator uses MuJoCo's URDF importer to retain every fixed fingertip
frame and joint transform. It adds six position actuators, five polynomial
equality constraints, a valid `home` keyframe, and a simple viewing scene.
Actuator order is:

```text
thumbCMC thumbMCP indexMCP middleMCP ringMCP littleMCP
```

The source CAD inertias repeat assembly masses/centres of mass on individual
links. The MJCF therefore uses synthetic link masses (`0.03 kg`) and diagonal
inertias (`2e-5 kg m^2`) for pose tracking. Gravity, hand collisions and
self-collisions are disabled. These models are for kinematic retargeting and
visualization, not calibrated dynamics, contact/force evaluation, or hardware
control. The retained URDF collision meshes are not enabled in the MJCF.

## Usage

Run from the repository root:

```bash
python example/test/view_model.py --model linker_l6_right
python example/test/view_model.py --model linker_l6_left
python example/teleop_sim.py --robot linker_l6 --input pico4 --hand right
python example/teleop_sim.py --robot linker_l6 --optimizer vector --input replay --hand left
python example/test/verify_linker_l6_mapping.py
```

Adaptive, vector and DexPilot baseline YAMLs are provided for MediaPipe,
AVP, Pico 4, Quest 3 and Noitom. Video, camera, RealSense and replay use the
MediaPipe configs. These are starting parameters, not per-device or per-user
calibrations. Use the existing rotation/scaling/pinch calibration and skeleton
debug tools with `--robot linker_l6` to tune them. Six independent DOFs cannot
reproduce arbitrary finger abduction or independently bend each phalanx.

## Retargeting Target Alignment

`retarget.target_offset` translates the entire scaled human target skeleton
(green in `debug_skeleton.py`), not the robot FK skeleton (red). It is an
optional `[x, y, z]` in metres in the optimizer coordinate frame, applied
**after** input rotation, wrist normalization and segment/pinch scaling. It
defaults to `[0.0, 0.0, 0.0]`. The adaptive and vector Pico 4 L6 configs start with:

```yaml
robot:
  type: "linker_l6"

retarget:
  target_offset: [0.0, 0.0, 0.04]
```

This raises every green target point, including its wrist, by 4 cm along
L6's +Z axis without changing segment lengths or the robot's reference wrist.
The translated PIP/DIP/tip targets enter the real optimization loss; the
uniformly scaled pinch targets (yellow) receive the same translation. Tip
directions, pinch distances and finger-to-finger/contact vectors do not change.

Adaptive, vector and DexPilot support the parameter. For vector and DexPilot,
only wrist-anchored positional targets change; finger-to-finger vectors remain
translation invariant. Only the adaptive and vector Pico 4 L6 configs enable
this initial offset; other configs keep their zero default.

`robot.origin_offset` instead selects a local virtual wrist point on
`robot.origin_link`. It is left unset here: moving only that reference point
is not the same as translating the complete human target. Neither setting
changes the URDF/MJCF robot base.

The 4 cm value is a tuning starting point, not a measured calibration or a
guarantee of improved tracking. Adjust `target_offset` for your input/user and
restart the process after editing the YAML; configuration is not hot-reloaded.

Run the headless regression check with:

```bash
python example/test/verify_target_offset.py
```
