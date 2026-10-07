"""Evaluation episodes for the two controllers, with or without a residual policy.

Each function runs one batch of environments from given initial states to the end
of the episode and returns one :class:`Trace` per environment: per control step
the pelvis height, the support surface height under the robot, the uprightness and
the two penetration depths, plus the first physics step at which the robot left
the task area.  :func:`score` turns a trace into the success outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import mujoco
import numpy as np
from omegaconf import DictConfig

from elastra import residual as residual_module
from elastra import success
from elastra.config import assets_dir
from elastra.host import HostController
from elastra.penetration import PenetrationProbe
from elastra.protomotions import ReferenceClips, Tracker
from elastra.scene import Scene
from elastra.sim import Batch
from elastra.trampoline import topology_of


@dataclass
class Trace:
    pelvis_z: list[float] = field(default_factory=list)
    surface_z: list[float] = field(default_factory=list)
    upright: list[float] = field(default_factory=list)
    contact_depth: list[float] = field(default_factory=list)
    crossing_depth: list[float] = field(default_factory=list)
    exit_physics_step: int | None = None
    physics_steps_per_control_step: int = 1
    diverged: bool = False

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            k: np.asarray(getattr(self, k), dtype=np.float64)
            for k in ("pelvis_z", "surface_z", "upright", "contact_depth", "crossing_depth")
        }


def score(trace: Trace, criterion: DictConfig, control_dt: float) -> success.Outcome:
    a = trace.arrays()
    stand = success.standing_flags(a["pelvis_z"] - a["surface_z"], a["upright"], criterion)
    if trace.diverged:
        stand[:] = False
    return success.outcome(
        stand,
        trace.exit_physics_step,
        hold=success.hold_steps(criterion, control_dt),
        physics_steps_per_control_step=trace.physics_steps_per_control_step,
    )


class _Recorder:
    def __init__(self, cfg: DictConfig, batch: Batch) -> None:
        self.batch = batch
        self.heights = [success.SurfaceHeight(scene) for scene in batch.scenes]
        self.probes = [
            PenetrationProbe(
                scene, topology_of(cfg.trampoline) if scene.surface.kind == "trampoline" else None
            )
            for scene in batch.scenes
        ]
        self.traces = [Trace(physics_steps_per_control_step=batch.decimation) for _ in batch.scenes]

    def record(self) -> None:
        for env, data in enumerate(self.batch.datas):
            trace = self.traces[env]
            trace.pelvis_z.append(float(data.qpos[2]))
            trace.surface_z.append(self.heights[env].surface_z(data))
            trace.upright.append(success.uprightness(data))
            contact, crossing = self.probes[env].depth(data)
            trace.contact_depth.append(contact)
            trace.crossing_depth.append(crossing)

    def finish(self) -> list[Trace]:
        diverged = self.batch.diverged()
        for env, trace in enumerate(self.traces):
            trace.exit_physics_step = self.batch.task_areas[env].first_exit_step
            trace.diverged = bool(diverged[env])
        return self.traces


Observer = Callable[[Batch, int], None]


def host_episodes(
    cfg: DictConfig,
    scenes: Sequence[Scene],
    qpos: Sequence[np.ndarray],
    postures: Sequence[str],
    seeds: Sequence[int],
    policy: residual_module.ResidualPolicy | None = None,
    *,
    residual_cfg: DictConfig | None = None,
    threads: int = 1,
    physics_dt: float | None = None,
    observers: Sequence[Observer] = (),
) -> list[Trace]:
    """HoST episodes: one unactuated opening step, then the controller for
    ``episode_control_steps - 1`` steps (its first 30 steps are unactuated too)."""

    host_cfg = cfg.robots.host
    control_dt = float(cfg.sim.control_dt_s)
    batch = Batch(
        scenes,
        physics_dt=float(physics_dt or cfg.sim.physics_dt_s),
        control_dt=control_dt,
        criterion=cfg.criterion,
        threads=threads,
    )
    try:
        batch.set_robot_qpos(qpos)
        controller = HostController(cfg, assets_dir(cfg), list(postures))
        rngs = [
            np.random.default_rng(int(seed) + int(host_cfg.observation_noise_seed_offset))
            for seed in seeds
        ]
        recorder = _Recorder(cfg, batch)
        n = len(scenes)
        steps = int(host_cfg.episode_control_steps)
        unactuated = int(host_cfg.unactuated_control_steps)
        horizon = steps - unactuated - 1
        surface_block = None
        if policy is not None:
            features = residual_module.surface_features(
                cfg, residual_cfg, [scene.surface for scene in scenes]
            )
        zero = np.zeros((n, int(host_cfg.num_actions)))
        batch.step(zero)
        controller.push(batch.datas, batch.scenes, zero, np.ones(n, dtype=np.int64), rngs)
        recorder.record()
        for observer in observers:
            observer(batch, 0)
        for index in range(2, steps + 1):
            step = np.full(n, index, dtype=np.int64)
            applied = controller.act(step)
            if policy is not None and index > unactuated:
                state = batch.robot_state()
                surface_block = np.concatenate(
                    [features, state["deflection"][:, None] * 10.0], axis=1
                )
                obs = residual_module.observation(
                    state,
                    applied,
                    quat=state["quat_wxyz"],
                    progress=residual_module.phase(
                        np.full(n, index - unactuated - 1), np.full(n, horizon)
                    ),
                    surface_block=surface_block,
                    position=policy.position_observation,
                )
                applied = applied + policy(obs)
            batch.step(applied)
            controller.push(batch.datas, batch.scenes, applied, step, rngs)
            recorder.record()
            for observer in observers:
                observer(batch, index - 1)
        return recorder.finish()
    finally:
        batch.close()


def protomotions_settle(
    cfg: DictConfig,
    scenes: Sequence[Scene],
    qpos: Sequence[np.ndarray],
    *,
    threads: int = 1,
    physics_dt: float | None = None,
) -> list[np.ndarray]:
    """Place each initial state on its undeformed surface (joint velocities zero) and let
    it settle under a gravity ramp; returns the full physics states."""

    batch = Batch(
        scenes,
        physics_dt=float(physics_dt or cfg.sim.physics_dt_s),
        control_dt=float(cfg.sim.control_dt_s),
        criterion=cfg.criterion,
        threads=threads,
    )
    try:
        for scene, data, q in zip(batch.scenes, batch.datas, qpos):
            mujoco.mj_resetData(scene.model, data)
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            q = np.asarray(q, dtype=np.float64)
            data.qpos[:7] = q[:7]
            data.qpos[scene.robot_qpos] = q[7:]
            mujoco.mj_forward(scene.model, data)
        settle = cfg.robots.protomotions.settle
        batch.settle(float(settle.ramp_s), float(settle.hold_s))
        return batch.states()
    finally:
        batch.close()


def protomotions_episodes(
    cfg: DictConfig,
    scenes: Sequence[Scene],
    states: Sequence[np.ndarray],
    clips: Sequence[str],
    policy: residual_module.ResidualPolicy | None = None,
    *,
    threads: int = 1,
    physics_dt: float | None = None,
    observers: Sequence[Observer] = (),
) -> list[Trace]:
    """ProtoMotions episodes from settled states; all environments share one clip length."""

    assets = assets_dir(cfg)
    reference = ReferenceClips.load(Path(assets) / "reference_clips")
    clip = np.asarray([reference.index(name) for name in clips], dtype=np.int64)
    frames = reference.frames[clip]
    if np.unique(frames).size != 1:
        raise ValueError("a ProtoMotions batch must use clips of one length")
    horizon = int(frames[0])
    batch = Batch(
        scenes,
        physics_dt=float(physics_dt or cfg.sim.physics_dt_s),
        control_dt=float(cfg.sim.control_dt_s),
        criterion=cfg.criterion,
        threads=threads,
    )
    try:
        batch.set_states(states)
        tracker = Tracker(assets, reference)
        recorder = _Recorder(cfg, batch)
        n = len(scenes)
        heading = tracker.heading(batch.robot_state()["anchor_xyzw"], clip)
        previous = np.zeros((n, len(scenes[0].robot_dof)), dtype=np.float32)
        for frame in range(horizon):
            state = batch.robot_state()
            step = np.full(n, frame, dtype=np.int64)
            target = tracker.act(state, step, clip, heading, previous)
            if policy is not None:
                obs = residual_module.observation(
                    state,
                    target,
                    quat=state["anchor_xyzw"],
                    progress=residual_module.phase(step, np.full(n, horizon)),
                    position=policy.position_observation,
                )
                target = target + policy(obs)
            batch.step(target)
            recorder.record()
            for observer in observers:
                observer(batch, frame)
            previous = target.astype(np.float32)
        return recorder.finish()
    finally:
        batch.close()
