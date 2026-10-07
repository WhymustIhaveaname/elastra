"""Surfaces, and compiled robot-on-surface scenes.

A surface is written as ``mattress/<bed>``, ``trampoline/<name>`` or ``rigid``.
:func:`surface_xml` gives its MJCF with the ProtoMotions G1 on it; the HoST scene is
the same surface with that robot removed (:func:`strip_robot`) and the HoST URDF
attached through ``MjSpec`` (:func:`build_host_scene`).

A :class:`Scene` is one compiled model plus the index sets the simulation needs:
where the robot's joints and actuators are, which geoms collide for the robot,
which joints are surface nodes (the *support elements*), and the force the
trampoline applies to its nodes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from omegaconf import DictConfig

from elastra import mattress as mattress_module
from elastra import robots
from elastra import trampoline as trampoline_module
from elastra.config import assets_dir

KINDS = ("mattress", "trampoline", "rigid")
#: name prefixes of the surface elements written by mattress.py / trampoline.py
SUPPORT_JOINT_PREFIX = {
    "mattress": "mattress_slide_",
    "trampoline": "trampoline_slide_",
    "rigid": "mattress_slide_",
}
SUPPORT_GEOM_PREFIX = {
    "mattress": "mattress_geom_",
    "trampoline": "trampoline_geom_",
    "rigid": "mattress_geom_",
}


@dataclass(frozen=True)
class Surface:
    kind: str
    name: str

    @classmethod
    def parse(cls, text: str) -> Surface:
        text = str(text)
        if text == "rigid":
            return cls("rigid", "rigid")
        kind, _, name = text.partition("/")
        if kind not in ("mattress", "trampoline") or not name:
            raise ValueError(f"surface {text!r} is not mattress/<bed>, trampoline/<name> or rigid")
        return cls(kind, name)

    @property
    def label(self) -> str:
        return "rigid" if self.kind == "rigid" else f"{self.kind}/{self.name}"


def surface_xml(
    cfg: DictConfig, surface: Surface, *, timestep: float | None = None, meshdir: str | None = None
) -> str:
    """The surface's MJCF text with the ProtoMotions G1 on it."""

    assets = assets_dir(cfg)
    dt = float(cfg.sim.physics_dt_s if timestep is None else timestep)
    robot_cfg = cfg.robots.protomotions
    if surface.kind == "mattress":
        return mattress_module.mattress_scene(
            cfg.mattress, robot_cfg, assets, surface.name, timestep=dt, meshdir=meshdir
        )
    if surface.kind == "trampoline":
        return trampoline_module.trampoline_scene(
            cfg.trampoline, robot_cfg, assets, surface.name, timestep=dt, meshdir=meshdir
        )
    return mattress_module.rigid_scene(
        cfg.mattress, robot_cfg, assets, timestep=dt, meshdir=meshdir
    )


def strip_robot(xml_text: str, model_name: str) -> str:
    """The surface alone: the ProtoMotions robot and everything that refers to it removed.

    Removed: the ``pelvis`` subtree, the actuators, every tendon that spans a robot
    joint, every contact pair (all name a robot geom) and the mesh assets.
    """

    root = ET.fromstring(xml_text)
    worldbody = root.find("worldbody")
    robot = [body for body in worldbody.findall("body") if body.get("name") == "pelvis"]
    if len(robot) != 1:
        raise ValueError("expected exactly one robot root body named pelvis")
    robot_names = {
        element.get("name")
        for tag in ("geom", "joint", "site")
        for element in robot[0].iter(tag)
        if element.get("name")
    }
    worldbody.remove(robot[0])
    for tag in ("actuator", "asset"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    tendon = root.find("tendon")
    if tendon is not None:
        for entry in list(tendon):
            refs = {entry.get(a) for a in ("joint", "site", "geom", "tendon", "body")}
            refs |= {child.get(a) for child in entry for a in ("joint", "site", "geom", "body")}
            if refs & robot_names:
                tendon.remove(entry)
        if len(tendon) == 0:
            root.remove(tendon)
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)
    root.find("compiler").attrib.pop("meshdir", None)
    root.set("model", model_name)
    return ET.tostring(root, encoding="unicode")


@dataclass
class Scene:
    surface: Surface
    robot: str
    model: mujoco.MjModel
    robot_qpos: np.ndarray
    robot_dof: np.ndarray
    robot_nq: int
    torso_body: int
    pelvis_body: int
    robot_geoms: np.ndarray
    support_qpos: np.ndarray
    support_dof: np.ndarray
    support_geoms: np.ndarray
    support_flex: np.ndarray
    #: the force the surface applies to its own nodes, or None
    support_force: Any = None
    #: ProtoMotions only: actuator ids in controller joint order
    actuators: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    #: HoST only: PD gains, torque limits, action scale
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    torque_limits: np.ndarray | None = None
    action_scale: float = 1.0

    @property
    def live_surface(self) -> bool:
        """Whether the support elements move (False on rigid ground)."""

        return self.surface.kind != "rigid"


def _descends(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body != 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def robot_collision_geoms(model: mujoco.MjModel, root_body: str = "pelvis") -> np.ndarray:
    """Collidable geoms of the robot subtree, in model order."""

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    ids = [
        geom
        for geom in range(model.ngeom)
        if _descends(model, int(model.geom_bodyid[geom]), root)
        and (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))
    ]
    return np.asarray(ids, dtype=np.int64)


def _named(model: mujoco.MjModel, kind: mujoco.mjtObj, count: int, prefix: str) -> list[int]:
    return [i for i in range(count) if (mujoco.mj_id2name(model, kind, i) or "").startswith(prefix)]


def _support_sets(model: mujoco.MjModel, surface: Surface) -> dict[str, np.ndarray]:
    joints = _named(
        model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt, SUPPORT_JOINT_PREFIX[surface.kind]
    )
    geoms = _named(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom, SUPPORT_GEOM_PREFIX[surface.kind])
    flex = [
        f
        for f in range(model.nflex)
        if int(model.flex_contype[f]) or int(model.flex_conaffinity[f])
    ]
    joints_arr = np.asarray(joints, dtype=np.int64)
    return {
        "support_qpos": model.jnt_qposadr[joints_arr].astype(np.int64)
        if joints
        else np.zeros(0, np.int64),
        "support_dof": model.jnt_dofadr[joints_arr].astype(np.int64)
        if joints
        else np.zeros(0, np.int64),
        "support_geoms": np.asarray(geoms, dtype=np.int64),
        "support_flex": np.asarray(flex, dtype=np.int64),
    }


def freeze_contact_time_constants(model: mujoco.MjModel, reference_dt: float) -> None:
    """Write MuJoCo's ``solref[0] >= 2 dt`` safety floor into the model at ``reference_dt``.

    MuJoCo applies this floor at run time with the run's own physics step.  Writing
    it at the reference step keeps every contact and constraint time constant fixed
    when a run uses a smaller physics step (docs/validation.md), so only the
    integration step changes.  At the reference step the simulation is unchanged.
    """

    floor = 2.0 * float(reference_dt)
    for name in (
        "geom_solref",
        "pair_solref",
        "pair_solreffriction",
        "jnt_solref",
        "dof_solref",
        "eq_solref",
        "tendon_solref_lim",
        "tendon_solref_fri",
        "flex_solref",
    ):
        array = getattr(model, name)
        if array.size == 0:
            continue
        below = (array[:, 0] > 0.0) & (array[:, 0] < floor)
        array[below, 0] = floor


def _surface_force(cfg: DictConfig, surface: Surface) -> Any:
    if surface.kind == "trampoline":
        return trampoline_module.MembraneForce(cfg.trampoline, surface.name)
    return None


def protomotions_gains(assets: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    meta = yaml.safe_load((Path(assets) / "protomotions" / "unified_pipeline.yaml").read_text())
    return (
        list(meta["joint_names"]),
        np.asarray(meta["control"]["stiffness"], dtype=np.float64),
        np.asarray(meta["control"]["damping"], dtype=np.float64),
    )


def build_protomotions_scene(
    cfg: DictConfig, surface: Surface, *, xml: str | None = None, physics_dt: float | None = None
) -> Scene:
    """The ProtoMotions G1 on ``surface``, its tracker's PD control in the actuators.

    The tracker outputs PD targets; each actuator applies ``kp (target - q) - kd dq``
    with the tracker's gains and no control limit.  The passive stiffness, damping and
    friction loss of the actuated joints are removed, and every joint limit gets the
    solref of ``conf/robot/protomotions.yaml``.
    """

    assets = assets_dir(cfg)
    text = xml if xml is not None else surface_xml(cfg, surface)
    model = mujoco.MjModel.from_xml_string(text)
    model.opt.timestep = float(cfg.sim.physics_dt_s if physics_dt is None else physics_dt)
    model.opt.gravity[:] = list(cfg.sim.gravity_m_s2)
    joint_names, kp, kd = protomotions_gains(assets)
    joints = np.asarray([model.joint(name).id for name in joint_names], dtype=np.int64)
    actuators = []
    for joint in joints:
        match = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if match.size != 1:
            raise ValueError(f"joint {joint} has {match.size} actuators")
        actuators.append(int(match[0]))
    actuators_arr = np.asarray(actuators, dtype=np.int64)
    dofs = model.jnt_dofadr[joints].astype(np.int64)
    model.jnt_stiffness[joints] = 0.0
    model.dof_damping[dofs] = 0.0
    model.dof_frictionloss[dofs] = 0.0
    for actuator, gain, damping in zip(actuators_arr, kp, kd, strict=True):
        model.actuator_gainprm[actuator, 0] = float(gain)
        model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_biasprm[actuator, 0] = 0.0
        model.actuator_biasprm[actuator, 1] = -float(gain)
        model.actuator_biasprm[actuator, 2] = -float(damping)
        model.actuator_ctrllimited[actuator] = 0
    limits = cfg.robots.protomotions.joint_limits
    special = {str(k): float(v) for k, v in limits.margin_rad_by_joint.items()}
    for name, joint in zip(joint_names, joints, strict=True):
        model.jnt_margin[joint] = special.get(name, float(limits.margin_rad))
        model.jnt_solref[joint, 0] = float(limits.solref_time_constant_s)
        model.jnt_solref[joint, 1] = float(limits.solref_damping_ratio)
    freeze_contact_time_constants(model, float(cfg.sim.contact_reference_dt_s))
    return Scene(
        surface=surface,
        robot="protomotions",
        model=model,
        robot_qpos=model.jnt_qposadr[joints].astype(np.int64),
        robot_dof=dofs,
        robot_nq=7 + len(joint_names),
        torso_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
        pelvis_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
        robot_geoms=robot_collision_geoms(model),
        support_force=_surface_force(cfg, surface),
        actuators=actuators_arr,
        **_support_sets(model, surface),
    )


#: the mjOption fields the HoST scene takes from the surface scene
OPTION_FIELDS = (
    "timestep",
    "integrator",
    "solver",
    "iterations",
    "ls_iterations",
    "tolerance",
    "ls_tolerance",
    "impratio",
    "cone",
    "jacobian",
    "noslip_iterations",
    "noslip_tolerance",
    "density",
    "viscosity",
    "o_margin",
    "disableflags",
    "enableflags",
)


def _align_flex_bvh(model: mujoco.MjModel) -> None:
    """Work around a MuJoCo 3.4 ``MjSpec.attach`` defect for scenes with a flex.

    After ``attach`` the flex's bounding-volume hierarchy can start below
    ``nbvhstatic``, but the collision midphase reads a flex's boxes at
    ``node - nbvhstatic``; the collidable membrane then shrinks to a small patch.
    The flex's BVH block (its children are stored relative to ``flex_bvhadr``) is
    moved up to ``nbvhstatic``; nothing else in the model changes.
    """

    if model.nflex == 0:
        return
    if model.nflex != 1:
        raise RuntimeError("the flex BVH alignment is only defined for one flex")
    static = int(model.nbvhstatic)
    address = int(model.flex_bvhadr[0])
    size = int(model.flex_bvhnum[0])
    shift = static - address
    if shift == 0:
        return
    if shift < 0 or address + size + shift > int(model.nbvh):
        raise RuntimeError(f"unexpected flex BVH layout: adr {address} nbvhstatic {static}")
    child = model.bvh_child.reshape(-1, 2)
    nodeid = model.bvh_nodeid.reshape(-1)
    depth = model.bvh_depth.reshape(-1)
    source = slice(address, address + size)
    block = (child[source].copy(), nodeid[source].copy(), depth[source].copy())
    if (block[0] >= size).any():
        raise RuntimeError("flex BVH children are not stored relative to flex_bvhadr")
    child[source] = -1
    nodeid[source] = -1
    depth[source] = 0
    target = slice(address + shift, address + shift + size)
    child[target], nodeid[target], depth[target] = block
    model.flex_bvhadr[0] = address + shift


def host_joint_order(urdf_path: Path) -> tuple[list[str], list[dict[str, float]]]:
    """Actuated joints of a URDF in depth-first order (the order HoST's policies use)."""

    root = ET.parse(urdf_path).getroot()
    joints: dict[str, dict[str, Any]] = {}
    children: dict[str, list[str]] = {}
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        row = {
            "type": joint.get("type", "fixed"),
            "child": joint.find("child").get("link"),
            "lower": float(limit.get("lower", "0")) if limit is not None else 0.0,
            "upper": float(limit.get("upper", "0")) if limit is not None else 0.0,
            "effort": float(limit.get("effort", "0")) if limit is not None else 0.0,
        }
        joints[joint.get("name")] = row
        children.setdefault(joint.find("parent").get("link"), []).append(joint.get("name"))
        child_links.add(row["child"])
    roots = sorted({link.get("name") for link in root.findall("link")} - child_links)
    ordered: list[str] = []

    def walk(link: str) -> None:
        for name in children.get(link, []):
            if joints[name]["type"] not in ("fixed", "floating"):
                ordered.append(name)
            walk(joints[name]["child"])

    walk(roots[0])
    return ordered, [joints[name] for name in ordered]


def _by_substring(name: str, table: DictConfig) -> float:
    hits = [float(value) for key, value in table.items() if str(key) in name]
    if len(hits) != 1:
        raise ValueError(f"gain lookup for {name!r} matched {len(hits)} keys")
    return hits[0]


def build_host_scene(
    cfg: DictConfig, surface: Surface, *, xml: str | None = None, physics_dt: float | None = None
) -> Scene:
    """The HoST G1 on ``surface``: the surface scene without its robot, attached to the URDF.

    HoST applies its PD law as joint torques (``kp (a s) - kd dq``, clipped to the
    URDF effort limits) every physics step; there are no actuators.
    """

    assets = assets_dir(cfg)
    host_cfg = cfg.robots.host
    text = xml if xml is not None else surface_xml(cfg, surface)
    support = mujoco.MjSpec.from_string(strip_robot(text, "surface"))
    spec = robots.host_robot_spec(assets)
    frame = spec.worldbody.add_frame()
    spec.attach(support, prefix="", suffix="", frame=frame)
    for name in OPTION_FIELDS:
        if hasattr(support.option, name) and hasattr(spec.option, name):
            setattr(spec.option, name, getattr(support.option, name))
    spec.option.gravity = support.option.gravity
    model = spec.compile()
    if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("the HoST robot's free joint must come first")
    model.opt.timestep = float(cfg.sim.physics_dt_s if physics_dt is None else physics_dt)
    model.opt.gravity[:] = list(cfg.sim.gravity_m_s2)
    freeze_contact_time_constants(model, float(cfg.sim.contact_reference_dt_s))
    _align_flex_bvh(model)
    names, rows = host_joint_order(robots.host_urdf(assets))
    joints = np.asarray([model.joint(name).id for name in names], dtype=np.int64)
    robot_nv = 6 + len(names)
    model.dof_armature[6:robot_nv] = float(host_cfg.armature)
    return Scene(
        surface=surface,
        robot="host",
        model=model,
        robot_qpos=model.jnt_qposadr[joints].astype(np.int64),
        robot_dof=model.jnt_dofadr[joints].astype(np.int64),
        robot_nq=7 + len(names),
        torso_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
        pelvis_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
        robot_geoms=robot_collision_geoms(model),
        support_force=_surface_force(cfg, surface),
        kp=np.asarray([_by_substring(n, host_cfg.stiffness) for n in names]),
        kd=np.asarray([_by_substring(n, host_cfg.damping) for n in names]),
        torque_limits=np.asarray([row["effort"] for row in rows]),
        action_scale=float(host_cfg.action_scale),
        **_support_sets(model, surface),
    )


def build_scene(cfg: DictConfig, robot: str, surface: Surface | str, **kwargs: Any) -> Scene:
    surface = Surface.parse(surface) if isinstance(surface, str) else surface
    if robot == "host":
        return build_host_scene(cfg, surface, **kwargs)
    if robot == "protomotions":
        return build_protomotions_scene(cfg, surface, **kwargs)
    raise ValueError(f"unknown robot {robot!r}")
