"""The HoST get-up controllers (prone and supine).

HoST [Huang et al. 2025] releases one policy per lying posture.  Each is an ONNX
network that maps the last six one-step observations (76 numbers each) to 23
joint actions; a HoST action ``a`` sets the PD target ``q + action_scale * a``.
The robot is torque-driven: every physics step the joint torque is
``kp * action_scale * a - kd * dq``, clipped to the URDF effort limits (see
:mod:`elastra.sim`).

One-step observation, in order: base angular velocity (body frame) times 0.25,
projected gravity, joint positions, joint velocities times 0.05, the action
applied at that step (original controller plus residual), and the action scale
plus uniform noise of width 0.05.  During the first 30 control steps of an
episode the action is zero and the observation row is zero, as in HoST.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig

from elastra.scene import Scene

POSTURES = ("prone", "supine")


def onnx_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def quat_rotate_inverse_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate ``vector`` by the inverse of the wxyz quaternion (Isaac Gym convention)."""

    quat = np.asarray(quat, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    w, axis = quat[0], quat[1:]
    return (
        vector * (2.0 * w * w - 1.0)
        - np.cross(axis, vector) * w * 2.0
        + axis * (axis @ vector) * 2.0
    )


class HostController:
    """Both HoST policies, run in one batch over environments of mixed posture."""

    def __init__(self, cfg: DictConfig, assets: Path, postures: list[str]) -> None:
        host = cfg.robots.host
        self.postures = list(postures)
        self.n = len(self.postures)
        self.one_step = int(host.one_step_observation_dim)
        self.history = int(host.history_length)
        self.num_actions = int(host.num_actions)
        self.clip_actions = float(host.clip_actions)
        self.clip_observations = float(host.clip_observations)
        self.unactuated = int(host.unactuated_control_steps)
        self.scales = host.obs_scales
        self.action_scale = float(host.action_scale)
        self.noise_width = float(host.action_scale_observation_noise_width)
        self.ids = {
            p: np.asarray([i for i, q in enumerate(self.postures) if q == p], dtype=np.int64)
            for p in sorted(set(self.postures))
        }
        self.sessions = {p: onnx_session(Path(assets) / str(host.policies[p])) for p in self.ids}
        self.input_names = {p: s.get_inputs()[0].name for p, s in self.sessions.items()}
        self.buffer = np.zeros((self.n, self.one_step * self.history), dtype=np.float64)

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            self.buffer[:] = 0.0
        else:
            self.buffer[np.asarray(env_ids, dtype=np.int64)] = 0.0

    def observe(
        self, data: mujoco.MjData, scene: Scene, applied: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        quat = data.qpos[3:7]
        gravity = quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0]))
        return np.concatenate(
            [
                np.asarray(data.qvel[3:6], dtype=np.float64) * float(self.scales.ang_vel),
                gravity,
                data.qpos[scene.robot_qpos] * float(self.scales.dof_pos),
                data.qvel[scene.robot_dof] * float(self.scales.dof_vel),
                applied,
                np.array([self.action_scale + (rng.random() - 0.5) * self.noise_width]),
            ]
        )

    def push(
        self,
        datas: list[mujoco.MjData],
        scenes: list[Scene],
        applied: np.ndarray,
        step: np.ndarray,
        rngs: list[np.random.Generator],
    ) -> None:
        """Append one control step's observation to every environment's history."""

        current = np.zeros((self.n, self.one_step), dtype=np.float64)
        for env in range(self.n):
            row = self.observe(datas[env], scenes[env], applied[env], rngs[env])
            if int(step[env]) <= self.unactuated:
                row = np.zeros_like(row)
            current[env] = row
        self.buffer = np.concatenate([self.buffer[:, self.one_step :], current], axis=1)

    def act(self, step: np.ndarray) -> np.ndarray:
        observation = np.clip(self.buffer, -self.clip_observations, self.clip_observations).astype(
            np.float32
        )
        actions = np.zeros((self.n, self.num_actions), dtype=np.float64)
        for posture, ids in self.ids.items():
            if ids.size:
                raw = self.sessions[posture].run(
                    None, {self.input_names[posture]: observation[ids]}
                )[0]
                actions[ids] = np.asarray(raw, dtype=np.float64)
        np.clip(actions, -self.clip_actions, self.clip_actions, out=actions)
        actions[np.asarray(step, dtype=np.int64) <= self.unactuated] = 0.0
        return actions
