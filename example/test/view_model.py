"""Open a MuJoCo model with the interactive viewer (actuator sliders included).

Usage:
    python example/test/view_model.py                        # default: rohand right
    python example/test/view_model.py --model rohand_left
    python example/test/view_model.py --xml assets/rohand/rohand_right_mujoco.xml
"""

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRESETS = {
    "rohand_right": PROJECT_ROOT / "assets/rohand/rohand_right_mujoco.xml",
    "rohand_left":  PROJECT_ROOT / "assets/rohand/rohand_left_mujoco.xml",
    "linker_l6_right": PROJECT_ROOT / "assets/linker_l6/linker_l6_right_mujoco.xml",
    "linker_l6_left": PROJECT_ROOT / "assets/linker_l6/linker_l6_left_mujoco.xml",
    "shadow_right": PROJECT_ROOT / "assets/shadow_hand/scene_right.xml",
    "shadow_left":  PROJECT_ROOT / "assets/shadow_hand/scene_left.xml",
    "allegro":      PROJECT_ROOT / "assets/allegro_hand/scene_right.xml",
    "wuji":         PROJECT_ROOT / "assets/wuji_hand/right.xml",
}

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--model", default="rohand_right", choices=list(PRESETS), help="preset model name")
group.add_argument("--xml", help="direct path to a MuJoCo XML file")
args = parser.parse_args()

xml_path = Path(args.xml) if args.xml else PRESETS[args.model]
if not xml_path.exists():
    print(f"XML not found: {xml_path}", file=sys.stderr)
    sys.exit(1)

model = mujoco.MjModel.from_xml_path(str(xml_path))
data  = mujoco.MjData(model)
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if home_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, home_id)

# launch blocks until the window is closed; actuator sliders are in the right panel
mujoco.viewer.launch(model, data)
