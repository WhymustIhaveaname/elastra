"""Penetration of the robot's collision geometry into the surface.

The depth is measured against the current, deformed surface, once per control
step:

* **contact depth**: the deepest MuJoCo contact between a robot collision geom and
  a surface collision geometry (a mattress or rigid-ground cell box, or the
  trampoline flex), ``max(0, -dist)``.  The cells and the flex move with the
  surface, so this is the overlap with the deformed surface.
* **membrane crossing depth** (trampoline only): MuJoCo's distance between a geom
  and a thin flex does not tell on which side of the membrane the geom is, so a
  geom that has passed through the membrane shows no contact depth.  Points on
  every robot collision geom (sphere bottoms, the bottom line of capsules and
  cylinders, mesh hull vertices) are therefore compared with the membrane's top
  surface, interpolated linearly on the flex triangles at the point's (x, y):
  the depth is how far the deepest point lies below it.

The reported depth is the larger of the two.  It is scored while the robot is in
the task area: over the control steps that end before the robot first leaves it
(:func:`steps_in_task_area`).  After that the trial's outcome is fixed and the robot
may be off the surface; on the trampoline, which has no floor, a robot that went
over the rim falls past the membrane's footprint.  A trial is invalid when the
depth exceeds ``limit_m`` (5 mm) for at least ``sustained_steps`` (10) consecutive
control steps (``conf/penetration/default.yaml``).
"""

from __future__ import annotations

import mujoco
import numpy as np
from omegaconf import DictConfig

from elastra.scene import Scene
from elastra.success import hull_vertices

_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
#: points per capsule / cylinder axis for the membrane crossing depth
AXIS_SAMPLES = 5


class MembraneSurface:
    """The trampoline flex as a height field over its (fixed) node grid."""

    def __init__(self, scene: Scene, topology) -> None:
        self.topo = topology
        self.n = topology.grid_size
        self.h = topology.spacing_m
        self.origin = -topology.radius_m
        qpos = np.full(self.n * self.n, -1, dtype=np.int64)
        # support joints are written in storage (flat) order
        if scene.support_qpos.size != self.n * self.n:
            raise ValueError("the trampoline scene does not carry one joint per storage node")
        qpos[:] = scene.support_qpos
        self.qpos = qpos
        self.active = topology.active

    def heights(self, data: mujoco.MjData) -> np.ndarray:
        """Top-surface height of every storage node (pinned nodes stay at 0)."""

        return -np.asarray(data.qpos[self.qpos], dtype=np.float64)

    def surface_z(self, x: np.ndarray, y: np.ndarray, z_nodes: np.ndarray) -> np.ndarray:
        """Membrane top height below the points; NaN where there is no membrane."""

        u = (x - self.origin) / self.h
        v = (y - self.origin) / self.h
        i = np.floor(u).astype(np.int64)
        j = np.floor(v).astype(np.int64)
        inside = (i >= 0) & (j >= 0) & (i < self.n - 1) & (j < self.n - 1)
        out = np.full(x.shape, np.nan)
        if not np.any(inside):
            return out
        i, j, fu, fv = i[inside], j[inside], (u - np.floor(u))[inside], (v - np.floor(v))[inside]
        c0, c1 = i * self.n + j, (i + 1) * self.n + j
        c2, c3 = (i + 1) * self.n + j + 1, i * self.n + j + 1
        a = self.active
        full = a[c0] & a[c1] & a[c2] & a[c3]
        z = np.full(i.shape, np.nan)
        # two triangles (c0, c1, c2) and (c0, c2, c3) split along the c0-c2 diagonal
        lower = full & (fu >= fv)
        upper = full & (fu < fv)
        z[lower] = (
            z_nodes[c0[lower]]
            + fu[lower] * (z_nodes[c1[lower]] - z_nodes[c0[lower]])
            + fv[lower] * (z_nodes[c2[lower]] - z_nodes[c1[lower]])
        )
        z[upper] = (
            z_nodes[c0[upper]]
            + fv[upper] * (z_nodes[c3[upper]] - z_nodes[c0[upper]])
            + fu[upper] * (z_nodes[c2[upper]] - z_nodes[c3[upper]])
        )
        # a square with three active corners carries one triangle on those corners
        partial = ~full & ((a[c0].astype(int) + a[c1] + a[c2] + a[c3]) == 3)
        if np.any(partial):
            for k in np.flatnonzero(partial):
                corners = [
                    (c0[k], 0.0, 0.0),
                    (c1[k], 1.0, 0.0),
                    (c2[k], 1.0, 1.0),
                    (c3[k], 0.0, 1.0),
                ]
                live = [c for c in corners if a[c[0]]]
                (p0, u0, v0), (p1, u1, v1), (p2, u2, v2) = live
                det = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
                w1 = ((fu[k] - u0) * (v2 - v0) - (u2 - u0) * (fv[k] - v0)) / det
                w2 = ((u1 - u0) * (fv[k] - v0) - (fu[k] - u0) * (v1 - v0)) / det
                w0 = 1.0 - w1 - w2
                if min(w0, w1, w2) >= -1.0e-12:
                    z[k] = w0 * z_nodes[p0] + w1 * z_nodes[p1] + w2 * z_nodes[p2]
        out[inside] = z
        return out


class PenetrationProbe:
    """Per-control-step penetration depth of one environment."""

    def __init__(self, scene: Scene, topology=None) -> None:
        model = scene.model
        self.robot_mask = np.zeros(model.ngeom, dtype=bool)
        self.robot_mask[scene.robot_geoms] = True
        self.support_mask = np.zeros(model.ngeom, dtype=bool)
        self.support_mask[scene.support_geoms] = True
        self.support_flex = np.asarray(scene.support_flex, dtype=np.int64)
        self.membrane = MembraneSurface(scene, topology) if topology is not None else None
        if self.membrane is not None:
            geoms = np.asarray(scene.robot_geoms, dtype=np.int64)
            self.geoms = geoms
            self.types = np.asarray(model.geom_type[geoms], dtype=np.int64)
            self.sizes = np.asarray(model.geom_size[geoms], dtype=np.float64)
            self.hulls = {
                row: hull_vertices(model, int(model.geom_dataid[int(geoms[row])]))
                for row in np.flatnonzero(self.types == _MESH).tolist()
            }

    def contact_depth(self, data: mujoco.MjData) -> float:
        ncon = int(data.ncon)
        if ncon == 0:
            return 0.0
        contact = data.contact
        geom = np.asarray(contact.geom[:ncon], dtype=np.int64)
        flex = np.asarray(contact.flex[:ncon], dtype=np.int64)
        dist = np.asarray(contact.dist[:ncon], dtype=np.float64)

        def robot(g: np.ndarray, f: np.ndarray) -> np.ndarray:
            return (g >= 0) & (f < 0) & self.robot_mask[np.where(g >= 0, g, 0)]

        def support(g: np.ndarray, f: np.ndarray) -> np.ndarray:
            by_geom = (g >= 0) & self.support_mask[np.where(g >= 0, g, 0)]
            by_flex = (f >= 0) & np.isin(f, self.support_flex)
            return by_geom | by_flex

        pair = (robot(geom[:, 0], flex[:, 0]) & support(geom[:, 1], flex[:, 1])) | (
            robot(geom[:, 1], flex[:, 1]) & support(geom[:, 0], flex[:, 0])
        )
        pair &= np.isfinite(dist)
        if not np.any(pair):
            return 0.0
        return float(max(0.0, -float(np.min(dist[pair]))))

    def _sample_points(self, data: mujoco.MjData) -> np.ndarray:
        xpos = np.asarray(data.geom_xpos[self.geoms], dtype=np.float64)
        xmat = np.asarray(data.geom_xmat[self.geoms], dtype=np.float64).reshape(-1, 3, 3)
        points = []
        down = np.array([0.0, 0.0, 1.0])
        offsets = np.linspace(-1.0, 1.0, AXIS_SAMPLES)
        for row, kind in enumerate(self.types.tolist()):
            r = self.sizes[row, 0]
            if kind == _SPHERE:
                points.append(xpos[row] - r * down)
            elif kind in (_CAPSULE, _CYLINDER):
                axis = xmat[row][:, 2]
                for s in offsets:
                    centre = xpos[row] + s * self.sizes[row, 1] * axis
                    if kind == _CAPSULE:
                        points.append(centre - r * down)
                    else:
                        rim = -down - (-axis[2]) * axis
                        norm = np.linalg.norm(rim)
                        points.append(centre + (r / norm) * rim if norm > 1.0e-12 else centre)
            elif kind == _MESH:
                points.extend(xpos[row] + self.hulls[row] @ xmat[row].T)
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def crossing_depth(self, data: mujoco.MjData) -> float:
        if self.membrane is None:
            return 0.0
        points = self._sample_points(data)
        surface = self.membrane.surface_z(points[:, 0], points[:, 1], self.membrane.heights(data))
        below = surface - points[:, 2]
        below = below[np.isfinite(below)]
        return float(max(0.0, float(below.max()))) if below.size else 0.0

    def depth(self, data: mujoco.MjData) -> tuple[float, float]:
        """``(contact depth, membrane crossing depth)``, metres."""

        return self.contact_depth(data), self.crossing_depth(data)


def steps_in_task_area(
    exit_physics_step: int | None, physics_steps_per_control_step: int, control_steps: int
) -> int:
    """Number of leading control steps that end before the robot first leaves the task
    area (all of them if it never does)."""

    if exit_physics_step is None:
        return int(control_steps)
    return min(
        int(control_steps), (int(exit_physics_step) - 1) // int(physics_steps_per_control_step)
    )


def summarize(depth: np.ndarray, cfg: DictConfig) -> dict:
    """Peak, 95th percentile and the longest run over the limit of one trace."""

    depth = np.asarray(depth, dtype=np.float64)
    over = depth > float(cfg.limit_m)
    best = run = 0
    for value in over.tolist():
        run = run + 1 if value else 0
        best = max(best, run)
    return {
        "peak_m": float(depth.max()) if depth.size else 0.0,
        "p95_m": float(np.percentile(depth, 95.0)) if depth.size else 0.0,
        "steps_over_limit": int(over.sum()),
        "longest_run_over_limit": int(best),
        "sustained": bool(best >= int(cfg.sustained_steps)),
    }
