"""The mattress: a grid of rigid cells on vertical slide joints over a rigid floor.

Each cell ``i`` is a box of mass ``m`` on a slide joint along ``-z`` (a positive
joint position is downward deflection ``x_i``) with stiffness ``k`` and damping ``c``.
Between cells there is one curvature term: for every interior cell a fixed tendon
of length ``(L x)_i = x_N + x_S + x_E + x_W - 4 x_i`` (the five-point Laplacian)
with stiffness ``b`` and damping ``d``.  The potential energy is

    U = k |x|^2 / 2 + b |L x|^2 / 2,     static stiffness K = k I + b L^T L,

and because every stencil row sums to zero, the cells' own weight sags the
unloaded bed uniformly by ``m g / k``: the surface the robot lands on is flat.

A bed is the five numbers ``k, b, c, d, m`` of ``conf/mattress/default.yaml``.  The cell
travel is bounded by the joint range and by a soft end-stop tendon per cell; the
robot-only floor below catches a robot that leaves the bed.

:func:`mattress_scene` writes the scene with the ProtoMotions G1 on it.  Element
order and number formatting follow the scenes the paper's results were computed
on (MuJoCo's contact order, and therefore every trajectory, depends on element
order), so the XML is identical to them up to element names and the last printed
digit of the cells' moment of inertia about z, which those scenes derived from an
unrounded cell mass.  A cell on a vertical slide joint never rotates, so its
moments of inertia do not enter the dynamics.
:func:`rigid_scene` is rigid ground: the same cell boxes, contact and floor, with the
boxes fixed to the world.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from elastra.robots import protomotions_scene_root

BED_FIELDS = ("k_n_per_m", "b_n_per_m", "c_n_s_per_m", "d_n_s_per_m", "cell_mass_kg")
#: centre, then the four neighbours.  Mathematics, not a parameter.
LAPLACIAN_COEFFICIENTS = (-4.0, 1.0, 1.0, 1.0, 1.0)

#: Rendering only: a static skin, a frame and a base plate around the bed.
SKIN_RGBA = "0.70 0.86 0.94 1"
SKIN_PITCH_M = 0.05
FRAME_RGBA = "0.30 0.16 0.08 1"
FLOOR_RGBA = "0.55 0.55 0.58 1"
FRAME_GEOMS = (
    ("frame_base", "0 0 -0.30", "1.1 1.03 0.04"),
    ("frame_x_neg", "-1.1 0 -0.025", "0.035 1.03 0.025"),
    ("frame_x_pos", "1.1 0 -0.025", "0.035 1.03 0.025"),
    ("frame_y_neg", "0 -1.03 -0.025", "1.1 0.035 0.025"),
    ("frame_y_pos", "0 1.03 -0.025", "1.1 0.035 0.025"),
)


def cell_name(kind: str, ix: int, iy: int) -> str:
    """``mattress_{kind}_x{ix}_y{iy}`` for kind in body / slide / geom / skin / endstop."""

    return f"mattress_{kind}_x{ix}_y{iy}"


def bed(cfg: DictConfig, name: str) -> dict[str, float]:
    """One bed's five numbers, exactly as the configuration gives them."""

    if name not in cfg.beds:
        raise KeyError(f"unknown bed {name!r}; known {sorted(cfg.beds)}")
    row = cfg.beds[name]
    return {field: float(row[field]) for field in BED_FIELDS}


def grid_shape(cfg: DictConfig) -> tuple[int, int]:
    return int(cfg.grid.nx), int(cfg.grid.ny)


def cell_centres(cfg: DictConfig) -> np.ndarray:
    """``(nx, ny, 2)`` cell centres; the grid is centred on the origin."""

    nx, ny = grid_shape(cfg)
    pitch = float(cfg.pitch_m)
    x = (np.arange(nx) - (nx - 1) / 2.0) * pitch
    y = (np.arange(ny) - (ny - 1) / 2.0) * pitch
    return np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)


def footprint_half_extents(cfg: DictConfig) -> tuple[float, float]:
    nx, ny = grid_shape(cfg)
    pitch = float(cfg.pitch_m)
    return nx * pitch / 2.0, ny * pitch / 2.0


def laplacian_stencils(nx: int, ny: int) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    """``(centre, [four neighbours])`` for every cell that has all four."""

    return [
        ((ix, iy), [(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)])
        for ix in range(1, nx - 1)
        for iy in range(1, ny - 1)
    ]


def box_inertia(mass: float, half: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = half
    return (
        mass * (y * y + z * z) / 3.0,
        mass * (x * x + z * z) / 3.0,
        mass * (x * x + y * y) / 3.0,
    )


def _g(value: float) -> str:
    return f"{float(value):g}"


def _cell_geom_attributes(cfg: DictConfig, ix: int, iy: int, pos: str) -> dict[str, str]:
    half_xy = float(cfg.pitch_m) / 2.0
    half_z = float(cfg.cell.half_height_m)
    contact = cfg.contact
    return {
        "name": cell_name("geom", ix, iy),
        "type": "box",
        "pos": pos,
        "size": f"{_g(half_xy)} {_g(half_xy)} {_g(half_z)}",
        "friction": str(contact.friction),
        "rgba": "0 0 0 0",
        "contype": "0",
        "conaffinity": "1",
        "priority": str(int(contact.priority)),
        "solref": str(contact.solref),
        "solimp": str(contact.solimp),
        "mass": "0",
    }


def _add_cells(worldbody: ET.Element, cfg: DictConfig, row: dict[str, float]) -> None:
    """One body per cell: slide joint, inertia, collision box (top at z = 0)."""

    nx, ny = grid_shape(cfg)
    centres = cell_centres(cfg)
    half_xy = float(cfg.pitch_m) / 2.0
    half_z = float(cfg.cell.half_height_m)
    mass = float(row["cell_mass_kg"])
    inertia = box_inertia(mass, (half_xy, half_xy, half_z))
    for ix in range(nx):
        for iy in range(ny):
            x, y = centres[ix, iy]
            body = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": cell_name("body", ix, iy),
                    "pos": f"{_g(x)} {_g(y)} 0",
                },
            )
            ET.SubElement(
                body,
                "joint",
                {
                    "name": cell_name("slide", ix, iy),
                    "type": "slide",
                    "axis": "0 0 -1",
                    "limited": "true",
                    "range": f"0 {_g(cfg.cell.travel_m)}",
                    "stiffness": f"{row['k_n_per_m']:.12g}",
                    "damping": f"{row['c_n_s_per_m']:.12g}",
                    "armature": "0",
                    "margin": f"{float(cfg.cell.joint_margin_m):.12g}",
                    "solreflimit": str(cfg.cell.solreflimit),
                },
            )
            ET.SubElement(
                body,
                "inertial",
                {
                    "pos": f"0 0 {_g(-half_z)}",
                    "mass": f"{mass:.12g}",
                    "diaginertia": " ".join(f"{value:.12g}" for value in inertia),
                },
            )
            ET.SubElement(body, "geom", _cell_geom_attributes(cfg, ix, iy, f"0 0 {_g(-half_z)}"))


def _add_fixed_cells(worldbody: ET.Element, cfg: DictConfig) -> None:
    """Rigid ground: the same cell boxes as static geoms of the world body."""

    nx, ny = grid_shape(cfg)
    centres = cell_centres(cfg)
    half_z = float(cfg.cell.half_height_m)
    for ix in range(nx):
        for iy in range(ny):
            x, y = centres[ix, iy]
            ET.SubElement(
                worldbody,
                "geom",
                _cell_geom_attributes(cfg, ix, iy, f"{_g(x)} {_g(y)} {_g(-half_z)}"),
            )


def _add_decoration_and_floor(worldbody: ET.Element, cfg: DictConfig) -> None:
    nx, ny = grid_shape(cfg)
    half_x, half_y = footprint_half_extents(cfg)
    travel = float(cfg.cell.travel_m)
    thickness = 2.0 * float(cfg.cell.half_height_m)
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "mattress_base_visual",
            "type": "box",
            "pos": f"0 0 {_g(-(travel + thickness))}",
            "size": f"{_g(half_x + 0.04)} {_g(half_y + 0.04)} 0.025",
            "rgba": "0 0 0 0",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    skin_pitch = SKIN_PITCH_M
    for ix in range(int(round(2.0 * half_x / skin_pitch))):
        for iy in range(int(round(2.0 * half_y / skin_pitch))):
            x = -half_x + (ix + 0.5) * skin_pitch
            y = -half_y + (iy + 0.5) * skin_pitch
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": cell_name("skin", ix, iy),
                    "type": "box",
                    "pos": f"{_g(round(x, 9))} {_g(round(y, 9))} -0.012",
                    "size": "0.027 0.027 0.012",
                    "rgba": SKIN_RGBA,
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
    for name, pos, size in FRAME_GEOMS:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"mattress_{name}",
                "type": "box",
                "pos": pos,
                "size": size,
                "rgba": FRAME_RGBA,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
    floor = cfg.floor
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": f"0 0 {_g(floor.z_m)}",
            "size": "0 0 0.05",
            "contype": "0",
            "conaffinity": "1",
            "priority": str(int(floor.priority)),
            "friction": str(floor.friction),
            "solref": str(floor.solref),
            "solimp": str(floor.solimp),
            "rgba": FLOOR_RGBA,
        },
    )


def _add_foot_pairs(root: ET.Element, cfg: DictConfig, robot_cfg: DictConfig) -> None:
    """Explicit contact pairs between the 14 foot capsules and every cell.

    The ProtoMotions model ships one pair per foot capsule against ``floor``; those
    14 pairs are pointed at the anchor cell and keep their place in the file, and
    the pairs with every other cell follow, foot by foot.
    """

    nx, ny = grid_shape(cfg)
    anchor = tuple(int(value) for value in cfg.foot_pair_anchor_cell)
    pair_cfg = robot_cfg.foot_pairs
    contact = root.find("contact")
    feet: list[str] = []
    for pair in contact.findall("pair"):
        if pair.get("geom2") != "floor":
            raise RuntimeError(f"unexpected contact pair {pair.attrib}")
        foot = str(pair.get("geom1"))
        feet.append(foot)
        pair.set("name", f"{foot.removesuffix('_collision')}_x{anchor[0]}_y{anchor[1]}")
        pair.set("geom2", cell_name("geom", *anchor))
        pair.set("solref", str(pair_cfg.solref))
        pair.set("solimp", str(pair_cfg.solimp))
    for foot in feet:
        for ix in range(nx):
            for iy in range(ny):
                if (ix, iy) == anchor:
                    continue
                ET.SubElement(
                    contact,
                    "pair",
                    {
                        "name": f"{foot.removesuffix('_collision')}_x{ix}_y{iy}",
                        "geom1": foot,
                        "geom2": cell_name("geom", ix, iy),
                        "solref": str(pair_cfg.solref),
                        "friction": str(pair_cfg.friction),
                        "solimp": str(pair_cfg.solimp),
                    },
                )


def _add_tendons(root: ET.Element, cfg: DictConfig, row: dict[str, float]) -> None:
    """One soft end stop per cell, then one curvature tendon per interior cell."""

    nx, ny = grid_shape(cfg)
    tendon = root.find("tendon")
    endstop = cfg.cell.endstop
    for ix in range(nx):
        for iy in range(ny):
            fixed = ET.SubElement(
                tendon,
                "fixed",
                {
                    "name": cell_name("endstop", ix, iy),
                    "limited": "true",
                    "range": f"{_g(endstop.lower_m)} {_g(cfg.cell.travel_m)}",
                    "margin": f"{_g(endstop.margin_m)}",
                    "solreflimit": str(endstop.solreflimit),
                },
            )
            ET.SubElement(fixed, "joint", {"joint": cell_name("slide", ix, iy), "coef": "1"})
    for index, (centre, neighbours) in enumerate(laplacian_stencils(nx, ny)):
        fixed = ET.SubElement(
            tendon,
            "fixed",
            {
                "name": f"mattress_curvature_{index}",
                "stiffness": f"{row['b_n_per_m']:.12g}",
                "damping": f"{row['d_n_s_per_m']:.12g}",
                "springlength": "0",
            },
        )
        for (ix, iy), coef in zip([centre, *neighbours], LAPLACIAN_COEFFICIENTS):
            ET.SubElement(
                fixed, "joint", {"joint": cell_name("slide", ix, iy), "coef": f"{coef:g}"}
            )


def _add_option(root: ET.Element, cfg: DictConfig, timestep: float) -> None:
    solver = cfg.solver
    ET.SubElement(
        root,
        "option",
        {
            "integrator": str(solver.integrator),
            "solver": str(solver.solver),
            "iterations": str(int(solver.iterations)),
            "tolerance": f"{float(solver.tolerance):g}",
            "timestep": f"{float(timestep):.12g}",
        },
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "800", "offheight": "720"})


def mattress_scene(
    cfg: DictConfig,
    robot_cfg: DictConfig,
    assets: Path,
    bed_name: str,
    *,
    timestep: float,
    meshdir: str | None = None,
) -> str:
    """The MJCF text of bed ``bed_name`` with the ProtoMotions G1 on it."""

    row = bed(cfg, bed_name)
    root = protomotions_scene_root(assets, robot_cfg, meshdir=meshdir)
    worldbody = root.find("worldbody")
    _add_cells(worldbody, cfg, row)
    _add_decoration_and_floor(worldbody, cfg)
    _add_foot_pairs(root, cfg, robot_cfg)
    _add_tendons(root, cfg, row)
    _add_option(root, cfg, timestep)
    return ET.tostring(root, encoding="unicode")


def rigid_scene(
    cfg: DictConfig,
    robot_cfg: DictConfig,
    assets: Path,
    *,
    timestep: float,
    meshdir: str | None = None,
) -> str:
    """Rigid ground: the mattress geometry and contact, every cell fixed to the world."""

    root = protomotions_scene_root(assets, robot_cfg, meshdir=meshdir)
    worldbody = root.find("worldbody")
    _add_fixed_cells(worldbody, cfg)
    _add_decoration_and_floor(worldbody, cfg)
    _add_foot_pairs(root, cfg, robot_cfg)
    _add_option(root, cfg, timestep)
    return ET.tostring(root, encoding="unicode")
