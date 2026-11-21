#!/usr/bin/env python3
"""Build portable, simulation-only L6 models using MuJoCo's URDF importer."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco


ACTUATED_JOINTS = (
    "thumbCMC", "thumbMCP", "indexMCP", "middleMCP", "ringMCP", "littleMCP",
)


def generate(side: str, root_dir: Path) -> Path:
    root_dir = root_dir.resolve()
    urdf_path = root_dir / f"{side}_linker/{side}_hand.urdf"
    urdf = ET.parse(urdf_path).getroot()
    joints = [joint for joint in urdf.findall("joint") if joint.get("type") != "fixed"]
    independent = {joint.get("name") for joint in joints if joint.find("mimic") is None}
    if independent != set(ACTUATED_JOINTS) or len(joints) != 11:
        raise ValueError("Expected six independent and five mimic L6 joints")

    moving_links = {joint.find("child").get("link") for joint in joints}
    for link in urdf.findall("link"):
        for tag in ("collision", "inertial"):
            for element in link.findall(tag):
                link.remove(element)
        if link.get("name") in moving_links:
            # Source CAD inertias repeat the entire assembly's mass/COM. Use
            # synthetic inertias for stable pose visualization, not force tests.
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", {"value": "0.03"})
            ET.SubElement(inertial, "inertia", {
                "ixx": "0.00002", "iyy": "0.00002", "izz": "0.00002",
                "ixy": "0", "ixz": "0", "iyz": "0",
            })
        for index, visual in enumerate(link.findall("visual")):
            visual.set("name", f"{link.get('name')}_visual_{index}")
            mesh = visual.find("geometry/mesh")
            if mesh is not None:
                mesh.set("filename", str(urdf_path.parent / mesh.get("filename")))

    ET.SubElement(ET.SubElement(urdf, "mujoco"), "compiler", {
        "strippath": "false", "discardvisual": "false", "fusestatic": "false",
    })
    model = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    with tempfile.TemporaryDirectory() as temp_dir:
        converted = Path(temp_dir) / "converted.xml"
        mujoco.mj_saveLastXML(str(converted), model)
        mj = ET.parse(converted).getroot()

    mj.set("model", f"linker_l6_{side}")
    mj.find("compiler").set("meshdir", ".")
    for mesh in mj.findall("asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).relative_to(root_dir).as_posix())
    ET.SubElement(mj, "option", {
        "timestep": "0.002", "gravity": "0 0 0", "integrator": "implicitfast",
    })
    ET.SubElement(mj, "statistic", {"center": "0 0 0.11", "extent": "0.25"})
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "1280"})
    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", {"damping": "0.08", "armature": "0.0001"})
    ET.SubElement(default, "position", {"kp": "8", "kv": "0.3", "ctrllimited": "true"})
    for geom in mj.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    asset = mj.find("asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient", "rgb1": "0.3 0.5 0.7",
        "rgb2": "0 0 0", "width": "512", "height": "3072",
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane", "texuniform": "true",
        "texrepeat": "5 5", "reflectance": "0.2",
    })
    worldbody = mj.find("worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "light", {"pos": "0.5 0.5 1", "dir": "-0.5 -0.5 -1"})
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "pos": "0 0 -0.001", "size": "0 0 0.05",
        "material": "groundplane", "contype": "0", "conaffinity": "0",
    })

    by_name = {joint.get("name"): joint for joint in joints}
    # mj_saveLastXML rounds floats. Keep the original limits for actuator/FK
    # consistency at the ends of each independent and coupled joint range.
    for joint in mj.findall(".//body/joint"):
        limit = by_name[joint.get("name")].find("limit")
        joint.set("range", f"{limit.get('lower')} {limit.get('upper')}")
    actuator = ET.SubElement(mj, "actuator")
    neutral = {}
    for name in ACTUATED_JOINTS:
        limit = by_name[name].find("limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        neutral[name] = max(lower, min(0.0, upper))
        ET.SubElement(actuator, "position", {
            "name": f"{name}_actuator", "joint": name,
            "ctrlrange": f"{lower:.12g} {upper:.12g}",
        })
    equality = ET.SubElement(mj, "equality")
    for joint in joints:
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        multiplier = float(mimic.get("multiplier", "1"))
        offset = float(mimic.get("offset", "0"))
        neutral[joint.get("name")] = neutral[mimic.get("joint")] * multiplier + offset
        ET.SubElement(equality, "joint", {
            "name": f"{joint.get('name')}_mimic", "joint1": joint.get("name"),
            "joint2": mimic.get("joint"), "polycoef": f"{offset:.12g} {multiplier:.12g} 0 0 0",
            "solref": "0.004 1",
        })

    joint_order = [joint.get("name") for joint in mj.findall(".//body/joint")]
    ET.SubElement(ET.SubElement(mj, "keyframe"), "key", {
        "name": "home", "qpos": " ".join(f"{neutral[name]:.12g}" for name in joint_order),
        "ctrl": " ".join(f"{neutral[name]:.12g}" for name in ACTUATED_JOINTS),
    })
    output = root_dir / f"linker_l6_{side}_mujoco.xml"
    tree = ET.ElementTree(mj)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("right", "left", "both"), default="both")
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent
    for side in ("right", "left") if args.side == "both" else (args.side,):
        print(generate(side, root_dir))


if __name__ == "__main__":
    main()
