"""The trampoline: a circular membrane of nodes with a fixed rim.

A 25 x 25 grid of nodes, 0.1 m apart, covers [-1.2, 1.2]^2 m.  Nodes whose
centre lies inside the 1.2 m circle are *active*; an active node with a neighbour
outside the circle (or off the grid) is on the *rim*.  Interior (free) nodes slide
vertically (positive joint position = downward displacement ``x_i``); rim nodes
and the storage nodes outside the circle are held at zero by joint equality
constraints.  No node has a spring to the ground: the restoring force comes only
from the connections between neighbouring active nodes,

    f_i = sum_{j in N(i)} [ k_lin (x_i - x_j) + k_cub (x_i - x_j)^3 ],

which :class:`MembraneForce` applies to the free nodes as a generalized force before
every physics step.  Node damping ``c`` is MuJoCo joint damping.

The robot touches a two-dimensional MuJoCo flex built on the active nodes (its
vertices move with the node bodies); the flex's own elasticity is not used.

:func:`trampoline_scene` writes the scene with the ProtoMotions G1 on it.  Element
order and number formatting follow the scenes the paper's results were computed
on, so the XML is byte-identical to them up to element names.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from elastra.robots import protomotions_scene_root

#: Mass of a pinned node; it never moves, MuJoCo only needs it positive.
PINNED_NODE_MASS_KG = 1.0e-9
#: Rendering only.
FRAME_RGBA = "0.30 0.16 0.08 1"
SKIN_RGBA = "0.05 0.14 0.24 1"
SKIN_COUNT = 50
FRAME_RING_SEGMENTS = 48
FRAME_RING_RADIUS_M = 1.28
FRAME_LEGS = 6
FRAME_LEG_RADIUS_M = 1.1904


def node_name(kind: str, ix: int, iy: int) -> str:
    """``trampoline_{kind}_x{ix}_y{iy}`` for kind in node / slide / geom / skin / pin / endstop."""

    return f"trampoline_{kind}_x{ix}_y{iy}"


@dataclass(frozen=True)
class Topology:
    """The node grid and which nodes are active, on the rim, free or outside."""

    grid_size: int
    radius_m: float
    spacing_m: float
    positions_xy_m: np.ndarray  # (n, 2), flat index = ix * grid_size + iy
    active: np.ndarray
    rim: np.ndarray
    free: np.ndarray
    outside: np.ndarray
    edges: np.ndarray  # (E, 2) flat indices of neighbouring active nodes
    centre: int

    @property
    def pinned(self) -> np.ndarray:
        return self.rim | self.outside


def topology(grid_size: int, radius_m: float) -> Topology:
    if grid_size < 5 or grid_size % 2 != 1:
        raise ValueError("the membrane grid size must be odd and at least five")
    axis = np.linspace(-radius_m, radius_m, grid_size, dtype=np.float64)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    positions = np.stack([gx.ravel(), gy.ravel()], axis=1)
    active = np.linalg.norm(positions, axis=1) <= radius_m + 1.0e-12

    def flat(ix: int, iy: int) -> int:
        return ix * grid_size + iy

    rim = np.zeros(grid_size * grid_size, dtype=bool)
    edges: list[tuple[int, int]] = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            if not active[flat(ix, iy)]:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx_, ny_ = ix + dx, iy + dy
                if (
                    not (0 <= nx_ < grid_size and 0 <= ny_ < grid_size)
                    or not active[flat(nx_, ny_)]
                ):
                    rim[flat(ix, iy)] = True
                    break
    for ix in range(grid_size):
        for iy in range(grid_size):
            if not active[flat(ix, iy)]:
                continue
            for dx, dy in ((1, 0), (0, 1)):
                nx_, ny_ = ix + dx, iy + dy
                if nx_ < grid_size and ny_ < grid_size and active[flat(nx_, ny_)]:
                    edges.append((flat(ix, iy), flat(nx_, ny_)))
    centre = flat(grid_size // 2, grid_size // 2)
    return Topology(
        grid_size=grid_size,
        radius_m=float(radius_m),
        spacing_m=float(axis[1] - axis[0]),
        positions_xy_m=positions,
        active=active,
        rim=rim,
        free=active & ~rim,
        outside=~active,
        edges=np.asarray(edges, dtype=np.int64),
        centre=centre,
    )


def flex_triangles(topo: Topology) -> list[tuple[int, int, int]]:
    """Two triangles per grid square with four active corners, one with three."""

    n = topo.grid_size
    triangles: list[tuple[int, int, int]] = []
    for ix in range(n - 1):
        for iy in range(n - 1):
            corners = [ix * n + iy, (ix + 1) * n + iy, (ix + 1) * n + iy + 1, ix * n + iy + 1]
            live = [value for value in corners if topo.active[value]]
            if len(live) == 4:
                triangles += [(live[0], live[1], live[2]), (live[0], live[2], live[3])]
            elif len(live) == 3:
                triangles.append(tuple(live))
    return triangles


def topology_of(cfg: DictConfig) -> Topology:
    return topology(int(cfg.grid_size), float(cfg.radius_m))


def trampoline(cfg: DictConfig, name: str) -> dict[str, float]:
    if name not in cfg.trampolines:
        raise KeyError(f"unknown trampoline {name!r}; known {sorted(cfg.trampolines)}")
    row = cfg.trampolines[name]
    return {
        key: float(row[key])
        for key in (
            "edge_stiffness_n_per_m",
            "edge_cubic_stiffness_n_per_m3",
            "node_damping_n_s_per_m",
            "node_mass_kg",
        )
    }


def _box_inertia(mass: float, half: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = half
    return (
        mass * (y * y + z * z) / 3.0,
        mass * (x * x + z * z) / 3.0,
        mass * (x * x + y * y) / 3.0,
    )


def trampoline_scene(
    cfg: DictConfig,
    robot_cfg: DictConfig,
    assets: Path,
    name: str,
    *,
    timestep: float,
    meshdir: str | None = None,
) -> str:
    """The MJCF text of trampoline ``name`` with the ProtoMotions G1 on it."""

    row = trampoline(cfg, name)
    topo = topology_of(cfg)
    n = topo.grid_size
    root = protomotions_scene_root(assets, robot_cfg, meshdir=meshdir)
    worldbody = root.find("worldbody")
    half_xy = topo.spacing_m / 2.0
    half_z = float(cfg.node_half_height_m)
    travel = float(cfg.travel_m)
    for flat in range(n * n):
        ix, iy = divmod(flat, n)
        x, y = topo.positions_xy_m[flat]
        body = ET.SubElement(
            worldbody, "body", {"name": node_name("node", ix, iy), "pos": f"{x:.12g} {y:.12g} 0"}
        )
        ET.SubElement(
            body,
            "joint",
            {
                "name": node_name("slide", ix, iy),
                "type": "slide",
                "axis": "0 0 -1",
                "limited": "true",
                "range": f"0 {travel:.12g}",
                "stiffness": "0",
                "damping": f"{row['node_damping_n_s_per_m']:.12g}",
                "armature": "0",
                "margin": f"{float(cfg.joint_margin_m):g}",
                "solreflimit": str(cfg.solreflimit),
            },
        )
        mass = row["node_mass_kg"] if topo.free[flat] else PINNED_NODE_MASS_KG
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": f"0 0 {-half_z:.12g}",
                "mass": f"{mass:.12g}",
                "diaginertia": " ".join(
                    f"{v:.12g}" for v in _box_inertia(mass, (half_xy, half_xy, half_z))
                ),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": node_name("geom", ix, iy),
                "type": "box",
                "pos": f"0 0 {-half_z:.12g}",
                "size": f"{half_xy:.12g} {half_xy:.12g} {half_z:.12g}",
                "friction": str(cfg.flex.friction),
                "rgba": "0 0 0 0",
                # the node boxes never collide, the flex is the contact surface; they
                # carry the flex's contact parameters only for completeness
                "contype": "0",
                "conaffinity": "0",
                "priority": str(int(cfg.flex.priority)),
                "solref": str(cfg.flex.solref),
                "solimp": str(cfg.flex.solimp),
                "mass": "0",
            },
        )

    # rendering only: base plate, static skin, frame ring and legs
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "trampoline_base_visual",
            "type": "cylinder",
            "pos": f"0 0 {-travel - 0.05:.12g}",
            "size": f"{topo.radius_m + 0.04:.12g} 0.025",
            "rgba": "0 0 0 0",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    skins = SKIN_COUNT
    skin_pitch = 2.0 * topo.radius_m / skins
    for ix in range(skins):
        for iy in range(skins):
            x = -topo.radius_m + (ix + 0.5) * skin_pitch
            y = -topo.radius_m + (iy + 0.5) * skin_pitch
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": node_name("skin", ix, iy),
                    "type": "box",
                    "pos": f"{x:.10g} {y:.10g} -0.004",
                    "size": f"{skin_pitch * 0.56:.10g} {skin_pitch * 0.56:.10g} 0.004",
                    "rgba": SKIN_RGBA if math.hypot(x, y) <= topo.radius_m else "0 0 0 0",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
    for k in range(FRAME_RING_SEGMENTS):
        a0 = 2.0 * math.pi * k / FRAME_RING_SEGMENTS
        a1 = 2.0 * math.pi * (k + 1) / FRAME_RING_SEGMENTS
        r = FRAME_RING_RADIUS_M
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"trampoline_frame_ring_{k}",
                "type": "capsule",
                "fromto": (
                    f"{r * math.cos(a0):.10g} {r * math.sin(a0):.10g} -0.025 "
                    f"{r * math.cos(a1):.10g} {r * math.sin(a1):.10g} -0.025"
                ),
                "size": "0.035",
                "rgba": FRAME_RGBA,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
    for k in range(FRAME_LEGS):
        angle = 2.0 * math.pi * k / FRAME_LEGS
        r = FRAME_LEG_RADIUS_M
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"trampoline_leg_{k}",
                "type": "cylinder",
                "pos": f"{r * math.cos(angle):.10g} {r * math.sin(angle):.10g} -0.27",
                "size": "0.035 0.22",
                "rgba": FRAME_RGBA,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )

    # the robot's foot pairs were made for a rigid floor; the flex replaces them
    contact = root.find("contact")
    for pair in list(contact.findall("pair")):
        contact.remove(pair)

    tendon = root.find("tendon")
    for flat in range(n * n):
        ix, iy = divmod(flat, n)
        fixed = ET.SubElement(
            tendon,
            "fixed",
            {
                "name": node_name("endstop", ix, iy),
                "limited": "true",
                "range": f"{float(cfg.endstop.lower_m):.12g} {travel:.12g}",
                "margin": f"{float(cfg.endstop.margin_m):.12g}",
                "solreflimit": str(cfg.endstop.solreflimit),
            },
        )
        ET.SubElement(fixed, "joint", {"joint": node_name("slide", ix, iy), "coef": "1"})

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
            "cone": str(solver.cone),
        },
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})

    active = [int(v) for v in np.flatnonzero(topo.active)]
    vertex_of = {flat: index for index, flat in enumerate(active)}
    radius = float(cfg.flex.radius_m)
    deformable = ET.SubElement(root, "deformable")
    flex = ET.SubElement(
        deformable,
        "flex",
        {
            "name": "trampoline_flex",
            "dim": "2",
            "radius": f"{radius:.12g}",
            "body": " ".join(node_name("node", *divmod(flat, n)) for flat in active),
            # node origins are the undeformed top plane; the flex mid-surface sits one
            # radius lower so its collision top is at z = 0
            "vertex": " ".join(f"0 0 {-radius:.12g}" for _ in active),
            "element": " ".join(str(vertex_of[v]) for tri in flex_triangles(topo) for v in tri),
            "rgba": "0 0 0 0",
        },
    )
    ET.SubElement(
        flex,
        "contact",
        {
            "contype": "0",
            "conaffinity": "1",
            "condim": "3",
            "priority": str(int(cfg.flex.priority)),
            "friction": str(cfg.flex.friction),
            "solref": str(cfg.flex.solref),
            "solimp": str(cfg.flex.solimp),
            "internal": "false",
            "selfcollide": "none",
        },
    )
    ET.SubElement(flex, "edge", {"stiffness": "0", "damping": "0"})

    equality = ET.SubElement(root, "equality")
    for flat in np.flatnonzero(topo.pinned):
        ix, iy = divmod(int(flat), n)
        ET.SubElement(
            equality,
            "joint",
            {
                "name": node_name("pin", ix, iy),
                "joint1": node_name("slide", ix, iy),
                "polycoef": "0 0 0 0 0",
                "solref": str(cfg.pin.solref),
                "solimp": str(cfg.pin.solimp),
            },
        )
    return ET.tostring(root, encoding="unicode")


class MembraneForce:
    """The membrane's restoring force on its nodes, for a batch of environments.

    ``qfrc(x)`` takes ``x`` of shape ``(batch, nodes)`` (downward displacements in
    storage order) and returns the generalized forces on the same nodes; pinned
    nodes get zero.  Raises if a node moved further than the membrane radius, the
    signature of an unstable physics step.
    """

    def __init__(self, cfg: DictConfig, name: str) -> None:
        row = trampoline(cfg, name)
        self.topology = topology_of(cfg)
        self.k_lin = row["edge_stiffness_n_per_m"]
        self.k_cub = row["edge_cubic_stiffness_n_per_m3"]
        self._first = self.topology.edges[:, 0]
        self._second = self.topology.edges[:, 1]
        self._nodes = int(self.topology.active.size)
        self._pinned = self.topology.pinned
        self._index: np.ndarray | None = None
        self._rows = 0

    def _scatter(self, rows: int) -> np.ndarray:
        if self._index is None or self._rows != rows:
            offsets = (np.arange(rows, dtype=np.int64) * self._nodes)[:, None]
            self._index = np.concatenate(
                [
                    (offsets + self._first[None, :]).ravel(),
                    (offsets + self._second[None, :]).ravel(),
                ]
            )
            self._rows = rows
        return self._index

    def qfrc(self, x: np.ndarray) -> np.ndarray:
        deepest = float(np.max(np.abs(x), initial=0.0))
        if deepest > self.topology.radius_m:
            raise FloatingPointError(
                f"membrane displacement {deepest:.4f} m exceeds its radius: the step diverged"
            )
        rows = int(x.shape[0])
        delta = x[:, self._first] - x[:, self._second]
        flux = self.k_lin * delta + self.k_cub * delta**3
        weights = np.concatenate([(-flux).ravel(), flux.ravel()])
        out = np.bincount(
            self._scatter(rows), weights=weights, minlength=rows * self._nodes
        ).reshape(rows, self._nodes)
        out[:, self._pinned] = 0.0
        return out
