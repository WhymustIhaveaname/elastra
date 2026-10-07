"""The get-up success criterion.

Standing at a control step (``conf/criterion/standing.yaml``):

* the pelvis height above the support surface is at least ``height_m`` (0.7 m).
  The support surface height is read below the robot's lowest collision point:
  find the lowest point over all robot collision geoms, take the support element
  (mattress cell, trampoline node) whose centre is nearest to it in xy, and read
  that element's current top.  On rigid ground the top is z = 0.
* the pelvis is upright: the world-z component of the pelvis z axis,
  ``1 - 2 (q_x^2 + q_y^2)`` of the free-joint quaternion, is at least
  ``uprightness`` (0.9).

A trial succeeds when the robot stands for ``hold_s`` (1 s, 50 control steps)
without interruption and has not left the task area by the end of the control
step that completes the hold.  The task area is the mattress footprint
``|x| <= 1.0 m, |y| <= 0.95 m``; the robot has left it as soon as the
world-axis-aligned bounding box of any robot collision geom crosses the boundary,
checked after every physics step.  Leaving the area after the hold is complete
changes nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from omegaconf import DictConfig

from elastra.scene import Scene

_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)


def hull_vertices(model: mujoco.MjModel, mesh: int) -> np.ndarray:
    """The convex-hull vertices of a mesh, in the geom frame."""

    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
    adr = int(model.mesh_graphadr[mesh])
    if adr < 0:
        return vertices.copy()
    graph = np.asarray(model.mesh_graph, dtype=np.int64)
    numvert = int(graph[adr])
    return vertices[graph[adr + 2 + numvert : adr + 2 + 2 * numvert]].copy()


def _grid_axis(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    unique = np.unique(np.round(values, 9))
    pitch = float(np.min(np.diff(unique)))
    return float(unique[0]), pitch, np.rint((values - unique[0]) / pitch).astype(np.int64)


class SurfaceHeight:
    """Reads the support surface height below the robot's lowest collision point."""

    def __init__(self, scene: Scene) -> None:
        model = scene.model
        geoms = np.asarray(scene.robot_geoms, dtype=np.int64)
        types = np.asarray(model.geom_type[geoms], dtype=np.int64)
        sizes = np.asarray(model.geom_size[geoms], dtype=np.float64)
        rows = np.arange(geoms.size)
        self.geoms = geoms
        self.sphere_rows = rows[types == _SPHERE]
        self.capsule_rows = rows[types == _CAPSULE]
        self.cylinder_rows = rows[types == _CYLINDER]
        if np.any(~np.isin(types, [_SPHERE, _CAPSULE, _CYLINDER, _MESH])):
            raise NotImplementedError(
                "robot collision geoms must be spheres, capsules, cylinders or meshes"
            )
        self.radius = sizes[:, 0].copy()
        self.half_length = sizes[:, 1].copy()
        mesh_rows, mesh_local = [], []
        for row in rows[types == _MESH].tolist():
            hull = hull_vertices(model, int(model.geom_dataid[int(geoms[row])]))
            mesh_rows.append(np.full(hull.shape[0], row, dtype=np.int64))
            mesh_local.append(hull)
        self.mesh_rows = np.concatenate(mesh_rows) if mesh_rows else np.zeros(0, np.int64)
        self.mesh_local = np.concatenate(mesh_local) if mesh_local else np.zeros((0, 3))

        # support elements: one per surface node (slide joint), or per fixed cell geom
        if scene.support_qpos.size:
            joints = np.asarray(
                [np.flatnonzero(model.jnt_qposadr == adr)[0] for adr in scene.support_qpos],
                dtype=np.int64,
            )
            bodies = model.jnt_bodyid[joints]
            self.qpos_adr = np.asarray(scene.support_qpos, dtype=np.int64)
            self.qpos0 = np.asarray(model.qpos0[self.qpos_adr], dtype=np.float64)
            self.axis_z = np.asarray(model.jnt_axis[joints][:, 2], dtype=np.float64)
            boxes: dict[int, list[int]] = {}
            for geom in np.asarray(scene.support_geoms).tolist():
                if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                    boxes.setdefault(int(model.geom_bodyid[geom]), []).append(int(geom))
            self.rest_top = np.asarray(
                [
                    float(model.body_pos[b][2])
                    + max(
                        (
                            float(model.geom_pos[g][2] + model.geom_size[g][2])
                            for g in boxes.get(int(b), [])
                        ),
                        default=0.0,
                    )
                    for b in bodies.tolist()
                ],
                dtype=np.float64,
            )
            xy = np.asarray(model.body_pos[bodies][:, :2], dtype=np.float64)
        else:
            geoms_ = np.asarray(scene.support_geoms, dtype=np.int64)
            base = np.asarray(model.body_pos[model.geom_bodyid[geoms_]], dtype=np.float64)
            centre = base + np.asarray(model.geom_pos[geoms_], dtype=np.float64)
            self.qpos_adr = np.zeros(0, dtype=np.int64)
            self.rest_top = centre[:, 2] + np.asarray(
                model.geom_size[geoms_][:, 2], dtype=np.float64
            )
            xy = centre[:, :2]
        x0, pitch_x, ix = _grid_axis(xy[:, 0])
        y0, pitch_y, iy = _grid_axis(xy[:, 1])
        self.element_xy = np.stack([x0 + ix * pitch_x, y0 + iy * pitch_y], axis=1)
        self.live = bool(scene.live_surface and self.qpos_adr.size)

    def lowest_point(self, data: mujoco.MjData) -> np.ndarray:
        """World (x, y, z) of the lowest point over the robot collision geoms."""

        xpos = np.asarray(data.geom_xpos[self.geoms], dtype=np.float64)
        xmat = np.asarray(data.geom_xmat[self.geoms], dtype=np.float64).reshape(-1, 3, 3)
        best = np.array([np.nan, np.nan, np.inf])
        if self.sphere_rows.size:
            z = xpos[self.sphere_rows, 2] - self.radius[self.sphere_rows]
            k = int(np.argmin(z))
            if z[k] < best[2]:
                row = int(self.sphere_rows[k])
                best = np.array([xpos[row, 0], xpos[row, 1], z[k]])
        for rows, capsule in ((self.capsule_rows, True), (self.cylinder_rows, False)):
            if not rows.size:
                continue
            axis = xmat[rows, :, 2]
            sign = np.where(axis[:, 2] > 0.0, -1.0, 1.0)
            cap = xpos[rows] + (sign * self.half_length[rows])[:, None] * axis
            if capsule:
                point = cap - np.outer(self.radius[rows], [0.0, 0.0, 1.0])
            else:
                rim = np.array([0.0, 0.0, -1.0]) - (-axis[:, 2])[:, None] * axis
                norm = np.linalg.norm(rim, axis=1)
                safe = np.where(norm > 1.0e-12, norm, 1.0)
                point = cap + (self.radius[rows] / safe * (norm > 1.0e-12))[:, None] * rim
            k = int(np.argmin(point[:, 2]))
            if point[k, 2] < best[2]:
                best = point[k].copy()
        if self.mesh_rows.size:
            owner = self.mesh_rows
            z = xpos[owner, 2] + np.einsum("vj,vj->v", self.mesh_local, xmat[owner, 2, :])
            k = int(np.argmin(z))
            if z[k] < best[2]:
                row = int(owner[k])
                world = xpos[row] + xmat[row] @ self.mesh_local[k]
                best = np.array([world[0], world[1], z[k]])
        return best

    def element_under(self, x: float, y: float) -> int:
        d2 = (self.element_xy[:, 0] - float(x)) ** 2 + (self.element_xy[:, 1] - float(y)) ** 2
        return int(np.argmin(d2))

    def surface_z(self, data: mujoco.MjData) -> float:
        point = self.lowest_point(data)
        element = self.element_under(point[0], point[1])
        top = float(self.rest_top[element])
        if not self.live:
            return top
        q = float(data.qpos[int(self.qpos_adr[element])])
        return top + (q - float(self.qpos0[element])) * float(self.axis_z[element])


def uprightness(data: mujoco.MjData) -> float:
    return 1.0 - 2.0 * (
        float(data.qpos[4]) * float(data.qpos[4]) + float(data.qpos[5]) * float(data.qpos[5])
    )


class TaskArea:
    """Tracks whether the robot left the task area, from its collision geoms' bounding boxes.

    ``observe`` is called after every physics step; it reads the geom placement
    MuJoCo computed for that step.
    """

    def __init__(self, scene: Scene, cfg: DictConfig) -> None:
        geoms = np.asarray(scene.robot_geoms, dtype=np.int64)
        aabb = np.asarray(scene.model.geom_aabb[geoms], dtype=np.float64).reshape(-1, 6)
        self.geoms = geoms
        self.local_centre = aabb[:, :3]
        self.local_half = np.abs(aabb[:, 3:])
        self.half = np.asarray(cfg.task_area_half_extents_m, dtype=np.float64)
        self.tolerance = float(cfg.task_area_tolerance_m)
        self.first_exit_step: int | None = None
        self.minimum_margin = math.inf

    def bounds(self, data: mujoco.MjData) -> tuple[float, float, float, float]:
        rot = np.asarray(data.geom_xmat[self.geoms], dtype=np.float64).reshape(-1, 3, 3)
        centre = np.asarray(data.geom_xpos[self.geoms], dtype=np.float64)
        centre = centre + np.einsum("nij,nj->ni", rot, self.local_centre)
        extent = np.einsum("nij,nj->ni", np.abs(rot), self.local_half)
        low = centre[:, :2] - extent[:, :2]
        high = centre[:, :2] + extent[:, :2]
        return (
            float(low[:, 0].min()),
            float(high[:, 0].max()),
            float(low[:, 1].min()),
            float(high[:, 1].max()),
        )

    def margins(self, data: mujoco.MjData) -> tuple[float, float]:
        """Signed distance of the bounding box to the x and y boundaries (negative = outside)."""

        x_min, x_max, y_min, y_max = self.bounds(data)
        return (
            min(self.half[0] - x_max, x_min + self.half[0]),
            min(self.half[1] - y_max, y_min + self.half[1]),
        )

    def observe(self, data: mujoco.MjData, physics_step: int) -> None:
        x_min, x_max, y_min, y_max = self.bounds(data)
        margin = float(
            min(
                self.half[0] - x_max,
                x_min + self.half[0],
                self.half[1] - y_max,
                y_min + self.half[1],
            )
        )
        self.minimum_margin = min(self.minimum_margin, margin)
        if self.first_exit_step is None and margin < -self.tolerance:
            self.first_exit_step = int(physics_step)


@dataclass
class Outcome:
    success: bool
    #: the robot met the standing condition at some control step
    stood: bool
    #: control-step index at which the hold was completed (None if never)
    completion_step: int | None
    #: physics step (1-based, counted from the start of the trace) of the first exit
    exit_physics_step: int | None
    longest_stand_steps: int

    @property
    def outcome_class(self) -> str:
        """``success``, ``never_stood`` or ``stood_not_completed``."""

        if self.success:
            return "success"
        return "stood_not_completed" if self.stood else "never_stood"


def standing_flags(relative_height: np.ndarray, upright: np.ndarray, cfg: DictConfig) -> np.ndarray:
    return (np.asarray(relative_height) >= float(cfg.height_m)) & (
        np.asarray(upright) >= float(cfg.uprightness)
    )


def hold_steps(cfg: DictConfig, control_dt: float) -> int:
    return int(round(float(cfg.hold_s) / float(control_dt)))


def longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for value in np.asarray(flags, dtype=bool).tolist():
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def outcome(
    stand: np.ndarray,
    exit_physics_step: int | None,
    *,
    hold: int,
    physics_steps_per_control_step: int,
) -> Outcome:
    """Score one trace of per-control-step standing flags.

    ``stand[k]`` is read after control step ``k`` (0-based), i.e. after physics
    step ``(k + 1) * physics_steps_per_control_step``.
    """

    run = 0
    completion: int | None = None
    for k, ok in enumerate(np.asarray(stand, dtype=bool).tolist()):
        run = run + 1 if ok else 0
        if run >= hold:
            completion = k
            break
    absorbed = exit_physics_step is not None and (
        completion is None
        or int(exit_physics_step) <= (completion + 1) * int(physics_steps_per_control_step)
    )
    return Outcome(
        success=bool(completion is not None and not absorbed),
        stood=bool(np.any(stand)),
        completion_step=completion,
        exit_physics_step=exit_physics_step,
        longest_stand_steps=longest_run(stand),
    )
