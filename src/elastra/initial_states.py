"""HoST's initial states: its randomised lying poses, and ``data/initial_states/host.npz``.

    elastra-make-host-initial-states

For each posture (prone, supine) and seed 0 ... 63, HoST's initial-state
randomisation (:func:`author_initial_qpos`) gives a 30-number pose: root position
(0, 0, 0.5) m, the posture's root orientation and randomised joint angles.  Seeds
0 ... 15 are the training set, seeds 16 ... 63 the 96 test states.  Each pose is
stored twice:

* ``qpos_initial``: HoST's own starting height (root at 0.5 m; used on the
  trampoline, where the robot falls onto the membrane);
* ``qpos_lowered``: lowered along z until the robot touches the unloaded mattress
  top at z = 0 (:func:`place_on_surface`; used on the mattresses and on rigid
  ground, whose cell tops are all at z = 0, so one placement serves every bed).

The ProtoMotions initial states (``data/initial_states/protomotions.npz``) are data;
``data/README.md`` describes how they were made.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import DictConfig

from elastra import config, robots
from elastra.host import POSTURES
from elastra.scene import Scene, build_scene, host_joint_order

TRAIN_SEEDS = range(0, 16)
TEST_SEEDS = range(16, 64)


def author_initial_qpos(cfg: DictConfig, assets: Path, posture: str, seed: int) -> np.ndarray:
    """HoST's own initial-state randomisation for one seed: 30 numbers (root + 23 joints).

    Root at ``initial_root_position_m`` with the posture's orientation; each joint
    at its default angle times U(0.9, 1.1) plus U(-0.1, 0.1), clipped to
    ``soft_dof_pos_limit`` times the URDF range.
    """

    host = cfg.robots.host
    names, rows = host_joint_order(robots.host_urdf(assets))
    default = np.asarray([float(host.default_joint_angles.get(n, 0.0)) for n in names])
    soft = float(host.soft_dof_pos_limit)
    lower = np.asarray([r["lower"] for r in rows]) * soft
    upper = np.asarray([r["upper"] for r in rows]) * soft
    rng = np.random.default_rng(int(seed))
    dof = default * rng.uniform(0.9, 1.1, default.shape)
    dof = dof + rng.uniform(-0.1, 0.1, default.shape)
    dof = np.clip(dof, lower, upper)
    x, y, z, w = (float(v) for v in host.initial_root_quat_xyzw[posture])
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    qpos = np.zeros(7 + len(names), dtype=np.float64)
    qpos[0:3] = list(host.initial_root_position_m)
    qpos[3:7] = [w / norm, x / norm, y / norm, z / norm]
    qpos[7:] = dof
    return qpos


def place_on_surface(
    scene: Scene,
    qpos_robot: np.ndarray,
    *,
    step: float = 0.01,
    span: float = 0.8,
    pad: float = 0.02,
    bisections: int = 48,
) -> np.ndarray:
    """Lower the robot along z until it touches the surface's collision geoms.

    Orientation and joint angles are kept.  The search starts ``pad`` above the
    robot's lowest bounding-box point over z = 0, walks down in ``step`` until a
    robot geom penetrates a surface geom (``dist < -1e-8``), then bisects.  If no
    penetration is found within ``span`` the pose is returned unchanged.
    """

    model = scene.model
    robot = {int(v) for v in scene.robot_geoms}
    cells = {int(v) for v in scene.support_geoms}
    nq = scene.robot_nq

    def data_at(dz: float) -> mujoco.MjData:
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        data.qpos[:nq] = qpos_robot
        data.qpos[2] += dz
        mujoco.mj_forward(model, data)
        return data

    def penetrates(dz: float) -> bool:
        data = data_at(dz)
        for k in range(int(data.ncon)):
            c = data.contact[k]
            pair = {int(c.geom1), int(c.geom2)}
            if pair & cells and pair & robot and float(c.dist) < -1.0e-8:
                return True
        return False

    data = data_at(0.0)
    geoms = scene.robot_geoms
    aabb = np.asarray(model.geom_aabb[geoms], dtype=np.float64).reshape(-1, 6)
    rot = np.asarray(data.geom_xmat[geoms], dtype=np.float64).reshape(-1, 3, 3)
    centre = np.asarray(data.geom_xpos[geoms]) + np.einsum("nij,nj->ni", rot, aabb[:, :3])
    extent = np.einsum("nij,nj->ni", np.abs(rot), np.abs(aabb[:, 3:]))
    high = -float((centre[:, 2] - extent[:, 2]).min()) + pad
    if penetrates(high):
        high += 0.2
        if penetrates(high):
            raise RuntimeError("the robot still penetrates the surface after the clearance offset")
    low = high
    while low > high - span and not penetrates(low):
        low -= step
    if not penetrates(low):
        return np.asarray(qpos_robot, dtype=np.float64).copy()
    for _ in range(bisections):
        middle = 0.5 * (low + high)
        if penetrates(middle):
            low = middle
        else:
            high = middle
    placed = np.asarray(qpos_robot, dtype=np.float64).copy()
    placed[2] += float(high)
    return placed


def main(argv: list[str] | None = None) -> int:
    cfg = config.load(sys.argv[1:] if argv is None else list(argv))
    assets = config.assets_dir(cfg)
    surface = build_scene(cfg, "host", "mattress/a8")
    rows = []
    for posture in POSTURES:
        for seed in [*TRAIN_SEEDS, *TEST_SEEDS]:
            initial = author_initial_qpos(cfg, assets, posture, seed)
            lowered = place_on_surface(surface, initial)
            rows.append(
                (
                    f"{posture}_{seed}",
                    posture,
                    seed,
                    "train" if seed in TRAIN_SEEDS else "test",
                    initial,
                    lowered,
                )
            )
    out = config.data_dir(cfg) / "initial_states" / "host.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        init_id=np.asarray([r[0] for r in rows]),
        posture=np.asarray([r[1] for r in rows]),
        seed=np.asarray([r[2] for r in rows], dtype=np.int64),
        split=np.asarray([r[3] for r in rows]),
        qpos_initial=np.stack([r[4] for r in rows]),
        qpos_lowered=np.stack([r[5] for r in rows]),
    )
    print(f"{out}: {len(rows)} states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
