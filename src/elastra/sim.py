"""A batch of independent robot-on-surface environments stepped on one clock.

Every environment has its own ``MjData``; environments on the same surface share
the compiled model.  One control step holds the controller's command for
``control_dt / physics_dt`` physics steps.  Before every physics step:

* the robot command is applied -- ProtoMotions: the PD target goes to the
  actuators; HoST: the joint torque ``kp * action_scale * a - kd * dq`` (live joint
  velocity, clipped to the effort limits) is written as a generalized force;
* the surface's own force on its nodes (the trampoline membrane) is added.

``mj_step`` releases the GIL, so the environments step in a thread pool.  After
the last physics step of a control step, ``mj_forward`` brings every derived
quantity up to date for the observation.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import mujoco
import numpy as np
from omegaconf import DictConfig

from elastra.scene import Scene
from elastra.success import TaskArea


def root_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _step(pair: tuple[mujoco.MjModel, mujoco.MjData]) -> None:
    mujoco.mj_step(pair[0], pair[1])


def _forward(pair: tuple[mujoco.MjModel, mujoco.MjData]) -> None:
    mujoco.mj_forward(pair[0], pair[1])


class Batch:
    def __init__(
        self,
        scenes: Sequence[Scene],
        *,
        physics_dt: float,
        control_dt: float,
        criterion: DictConfig,
        threads: int = 1,
        task_area_every_physics_step: bool = True,
    ) -> None:
        self.scenes = list(scenes)
        self.n = len(self.scenes)
        robots = {scene.robot for scene in self.scenes}
        if len(robots) != 1:
            raise ValueError("a batch holds one robot type")
        self.robot = robots.pop()
        self.physics_dt = float(physics_dt)
        self.decimation = int(round(float(control_dt) / self.physics_dt))
        if not math.isclose(self.decimation * self.physics_dt, float(control_dt), abs_tol=1e-12):
            raise ValueError(f"control step {control_dt} is not a multiple of {physics_dt}")
        self.criterion = criterion
        self.task_area_every_physics_step = bool(task_area_every_physics_step)
        # environments grouped by compiled model, in order of first appearance
        self.groups: list[tuple[Scene, np.ndarray]] = []
        seen: dict[int, int] = {}
        for env, scene in enumerate(self.scenes):
            key = id(scene)
            if key not in seen:
                seen[key] = len(self.groups)
                self.groups.append((scene, []))
            self.groups[seen[key]][1].append(env)
        self.groups = [(scene, np.asarray(envs, dtype=np.int64)) for scene, envs in self.groups]
        for scene, _ in self.groups:
            scene.model.opt.timestep = self.physics_dt
        self.datas = [mujoco.MjData(scene.model) for scene in self.scenes]
        self.pairs = [(scene.model, data) for scene, data in zip(self.scenes, self.datas)]
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(threads))) if threads > 1 else None
        # each thread steps one contiguous block of environments
        self.blocks = [
            block.tolist()
            for block in np.array_split(np.arange(self.n), max(1, min(int(threads), self.n)))
        ]
        self.command = np.zeros((self.n, len(self.scenes[0].robot_dof)), dtype=np.float64)
        self.task_areas = [TaskArea(scene, criterion) for scene in self.scenes]
        self.physics_steps = 0
        self.control_steps = 0

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)

    # -- state ------------------------------------------------------------------ #
    def set_robot_qpos(self, qpos_robot: Sequence[np.ndarray]) -> None:
        """Start every environment from a robot pose at rest on an undeformed surface."""

        for env, (scene, data) in enumerate(zip(self.scenes, self.datas)):
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.qfrc_applied[:] = 0.0
            data.ctrl[:] = 0.0
            data.qpos[: scene.robot_nq] = np.asarray(qpos_robot[env], dtype=np.float64)
            mujoco.mj_forward(scene.model, data)
        self._restart()

    def set_states(self, states: Sequence[np.ndarray]) -> None:
        """Start every environment from a full physics state (``mjSTATE_FULLPHYSICS``)."""

        for scene, data, state in zip(self.scenes, self.datas, states):
            mujoco.mj_setState(
                scene.model,
                data,
                np.asarray(state, dtype=np.float64),
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            mujoco.mj_forward(scene.model, data)
        self._restart()

    def restore(self, env_ids: Sequence[int], states: Sequence[np.ndarray]) -> None:
        """Restart some environments from full physics states; the others run on.  Their
        task-area record starts afresh (``first_exit_step`` counts on the batch clock)."""

        for env, state in zip(env_ids, states):
            env = int(env)
            scene, data = self.scenes[env], self.datas[env]
            mujoco.mj_setState(
                scene.model,
                data,
                np.asarray(state, dtype=np.float64),
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(scene.model, data)
            self.command[env] = 0.0
            self.task_areas[env] = TaskArea(scene, self.criterion)

    def _restart(self) -> None:
        self.command[:] = 0.0
        self.task_areas = [TaskArea(scene, self.criterion) for scene in self.scenes]
        self.physics_steps = 0
        self.control_steps = 0

    def states(self) -> list[np.ndarray]:
        out = []
        for scene, data in zip(self.scenes, self.datas):
            size = mujoco.mj_stateSize(scene.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            vector = np.zeros(size, dtype=np.float64)
            mujoco.mj_getState(scene.model, data, vector, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            out.append(vector)
        return out

    # -- stepping --------------------------------------------------------------- #
    def _apply_forces(self) -> None:
        for scene, envs in self.groups:
            force = None
            if scene.support_force is not None and scene.support_qpos.size:
                block = np.stack([self.datas[e].qpos[scene.support_qpos] for e in envs])
                force = scene.support_force.qfrc(block)
            if self.robot == "host":
                velocity = np.stack([self.datas[e].qvel[scene.robot_dof] for e in envs])
                torque = scene.kp * (self.command[envs] * scene.action_scale) - scene.kd * velocity
                np.clip(torque, -scene.torque_limits, scene.torque_limits, out=torque)
            for row, env in enumerate(envs):
                data = self.datas[env]
                data.qfrc_applied[:] = 0.0
                if self.robot == "host":
                    data.qfrc_applied[scene.robot_dof] = torque[row]
                if force is not None:
                    data.qfrc_applied[scene.support_dof] += force[row]

    def _map(self, function) -> None:
        if self.pool is None:
            for pair in self.pairs:
                function(pair)
        else:
            pairs = self.pairs
            list(self.pool.map(lambda block: [function(pairs[i]) for i in block], self.blocks))

    def physics_step(self) -> None:
        self._apply_forces()
        self._map(_step)
        self.physics_steps += 1
        if self.task_area_every_physics_step:
            for area, data in zip(self.task_areas, self.datas):
                area.observe(data, self.physics_steps)

    def set_command(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != self.command.shape:
            raise ValueError(f"command {command.shape} is not {self.command.shape}")
        if not np.isfinite(command).all():
            raise FloatingPointError("non-finite command")
        self.command[:] = command
        if self.robot == "protomotions":
            for scene, data, target in zip(self.scenes, self.datas, self.command):
                data.ctrl[:] = 0.0
                data.ctrl[scene.actuators] = target

    def step(self, command: np.ndarray) -> None:
        """One control step."""

        self.set_command(command)
        for _ in range(self.decimation):
            self.physics_step()
        self._map(_forward)
        self.control_steps += 1
        if not self.task_area_every_physics_step:
            for area, data in zip(self.task_areas, self.datas):
                area.observe(data, self.physics_steps)

    def settle(self, ramp_s: float, hold_s: float) -> None:
        """Load every environment quasi-statically: gravity ramps up along a half cosine
        over ``ramp_s`` and is held for ``hold_s``; the ProtoMotions actuators hold the
        initial joint angles."""

        ramp = int(round(ramp_s / self.physics_dt))
        hold = int(round(hold_s / self.physics_dt))
        gravity = {
            id(scene.model): np.asarray(scene.model.opt.gravity).copy() for scene, _ in self.groups
        }
        for scene, envs in self.groups:
            scene.model.opt.gravity[:] = 0.0
            for env in envs:
                data = self.datas[env]
                data.ctrl[:] = 0.0
                if self.robot == "protomotions":
                    data.ctrl[scene.actuators] = data.qpos[scene.robot_qpos]
        self._map(_forward)
        for step in range(1, ramp + hold + 1):
            fraction = 0.5 - 0.5 * math.cos(math.pi * (step / ramp)) if step <= ramp else 1.0
            for scene, _ in self.groups:
                scene.model.opt.gravity[:] = gravity[id(scene.model)] * fraction
            self._apply_forces()
            self._map(_step)
        for scene, _ in self.groups:
            scene.model.opt.gravity[:] = gravity[id(scene.model)]
        self._map(_forward)

    # -- read-out --------------------------------------------------------------- #
    def robot_state(self) -> dict[str, np.ndarray]:
        n = self.n
        nj = len(self.scenes[0].robot_dof)
        out = {
            "dof_pos": np.zeros((n, nj)),
            "dof_vel": np.zeros((n, nj)),
            "root_ang_vel": np.zeros((n, 3)),
            "quat_wxyz": np.zeros((n, 4)),
            "anchor_xyzw": np.zeros((n, 4)),
            "torso_z": np.zeros(n),
            "torso_up": np.zeros(n),
            "pelvis_z": np.zeros(n),
            "upright": np.zeros(n),
            "root_xy": np.zeros((n, 2)),
            "root_lin_vel_local": np.zeros((n, 3)),
            "margin_x": np.zeros(n),
            "margin_y": np.zeros(n),
            "deflection": np.zeros(n),
        }
        for env, (scene, data) in enumerate(zip(self.scenes, self.datas)):
            out["dof_pos"][env] = data.qpos[scene.robot_qpos]
            out["dof_vel"][env] = data.qvel[scene.robot_dof]
            out["root_ang_vel"][env] = data.qvel[3:6]
            out["quat_wxyz"][env] = data.qpos[3:7]
            out["anchor_xyzw"][env] = data.xquat[scene.torso_body][[1, 2, 3, 0]]
            out["torso_z"][env] = float(data.xpos[scene.torso_body][2])
            out["torso_up"][env] = float(data.xmat[scene.torso_body].reshape(3, 3)[2, 2])
            out["pelvis_z"][env] = float(data.qpos[2])
            out["upright"][env] = 1.0 - 2.0 * (
                float(data.qpos[4]) * float(data.qpos[4])
                + float(data.qpos[5]) * float(data.qpos[5])
            )
            out["root_xy"][env] = data.qpos[:2]
            out["root_lin_vel_local"][env] = root_rotation(data.qpos[3:7]).T @ data.qvel[:3]
            out["margin_x"][env], out["margin_y"][env] = self.task_areas[env].margins(data)
            if scene.support_qpos.size:
                out["deflection"][env] = float(np.max(data.qpos[scene.support_qpos], initial=0.0))
        return out

    def diverged(self) -> np.ndarray:
        """Per environment: the state is not finite, or MuJoCo found a diverged
        acceleration at some physics step (it then resets the state to the model's
        reference pose, which would otherwise pass as a standing robot)."""

        bad_acceleration = int(mujoco.mjtWarning.mjWARN_BADQACC)
        return np.asarray(
            [
                bool(
                    not np.isfinite(d.qpos).all()
                    or not np.isfinite(d.qvel).all()
                    or d.warning[bad_acceleration].number > 0
                )
                for d in self.datas
            ]
        )
