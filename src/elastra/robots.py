"""The two robot models: the ProtoMotions 29-DoF G1 and the HoST 23-DoF G1.

Both are third-party files (see ``THIRD_PARTY.md``) fetched by
``elastra-download-assets`` (:mod:`elastra.assets`).  This module only reads them:

* :func:`protomotions_scene_root` turns the ProtoMotions MJCF into the root element
  every surface scene is built on.  It adds the joint-limit settings of
  ``conf/robot/protomotions.yaml``, drops the IMU sensors and adds a light.
* :func:`host_robot_spec` loads the HoST URDF as an ``MjSpec`` with a free root
  joint, the form :mod:`elastra.scene` attaches a surface to.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from omegaconf import DictConfig

#: Rendering only.
LIGHT = {
    "pos": "2 0 5.0",
    "dir": "0 0 -1",
    "diffuse": "0.4 0.4 0.4",
    "specular": "0.1 0.1 0.1",
    "directional": "true",
}


def protomotions_mjcf(assets: Path) -> Path:
    return Path(assets) / "protomotions" / "g1_holo_compat.xml"


def protomotions_meshdir(assets: Path) -> str:
    return str((Path(assets) / "protomotions" / "meshes").resolve())


def protomotions_scene_root(
    assets: Path, robot_cfg: DictConfig, *, meshdir: str | None = None
) -> ET.Element:
    """The ProtoMotions G1 MJCF, ready for a surface to be added.

    * every hinge joint gets an early-activation ``margin`` and a stiffer
      ``solreflimit`` (``robot_cfg.joint_limits``): MuJoCo joint limits are soft,
      and the tracker was trained with hard articulation limits;
    * the IMU ``<sensor>`` block is removed (nothing reads it);
    * a light is added.

    The 14 foot contact pairs of the file still name the geom ``floor``; the
    surface builder points them at its own collision geom.
    """

    root = ET.parse(protomotions_mjcf(assets)).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", meshdir if meshdir is not None else protomotions_meshdir(assets))
    limits = robot_cfg.joint_limits
    special = {str(name): float(value) for name, value in limits.margin_rad_by_joint.items()}
    worldbody = root.find("worldbody")
    for joint in worldbody.iter("joint"):
        if joint.get("type", "hinge") != "hinge":
            continue
        margin = special.get(str(joint.get("name")), float(limits.margin_rad))
        joint.set("margin", f"{margin:g}")
        joint.set("solreflimit", str(limits.solreflimit))
    for sensor in root.findall("sensor"):
        root.remove(sensor)
    ET.SubElement(worldbody, "light", dict(LIGHT))
    return root


def host_urdf(assets: Path) -> Path:
    return Path(assets) / "host" / "g1_23dof.urdf"


def host_robot_spec(assets: Path) -> mujoco.MjSpec:
    """The HoST URDF as an ``MjSpec`` with a free joint on its root link.

    MuJoCo 3.4 reads a URDF through ``MjSpec.from_string``; the relative mesh
    directory of the file is pointed at the downloaded meshes.
    """

    path = host_urdf(assets).resolve()
    text = path.read_text(encoding="utf-8")
    token = 'meshdir="meshes"'
    if token not in text:
        raise RuntimeError(f"{path}: expected the HoST compiler meshdir {token!r}")
    text = text.replace(token, f'meshdir="{path.parent / "meshes"}"', 1)
    spec = mujoco.MjSpec.from_string(text)
    root_body = next(body for body in spec.bodies if body.name and body.name != "world")
    root_body.add_freejoint()
    return spec
