"""Training a residual policy with PPO (``elastra-train``, ``conf/train/*.yaml``).

    elastra-train                                    # HoST (conf/train/host.yaml)
    elastra-train train=protomotions
    elastra-train train=protomotions_curriculum
    elastra-train train.updates=2 train.workers=8    # a short test run

The run writes to ``train.out``: ``config.yaml``, ``history.jsonl`` (one line per
update: reward, finished episodes and successes per surface, PPO statistics),
``checkpoints/update_*.pt`` every ``train.checkpoint_every`` updates and the final
``policy.pt``, which ``elastra-evaluate policy=...`` reads.

With ``train.resume=true`` (the default) a run whose ``train.out`` already holds
checkpoints continues from the latest one: networks, optimiser and random number
generators are restored, ``history.jsonl`` is cut back to that update, and every
environment restarts (the environment states are not saved, so a resumed run is not
bit-identical to an uninterrupted one).

The parallel environments are a fixed number, each on one surface.  They are split
over ``train.workers`` worker processes (environment ``e`` goes to worker
``e mod workers``); every worker steps its environments on one clock
(:class:`elastra.sim.Batch`), and the workers meet once per control step.  Every
environment draws its initial states from its own random number generator, seeded with
the training seed, its index and the update at which its curriculum stage (or the
resumed run) started, so a run does not depend on the number of workers.
An episode starts from a cached state:

* HoST: one of the training initial states of the environment's posture (even environments
  prone, odd environments supine), lowered onto the surface (dropped onto the trampoline)
  and run through HoST's 30 unactuated control steps; the episode is the next
  ``horizon_control_steps`` steps, with HoST's action history starting at zero;
* ProtoMotions: one of the training initial states, settled on the surface under a gravity
  ramp (as in the evaluation); the episode lasts as long as the reference clip of that
  initial state.

The initial state is drawn uniformly at every reset.  An episode ends when the robot
leaves the task area (a termination) or reaches its horizon (a truncation, whose
value is bootstrapped); success and falls do not end it.  Finished environments restart at
once, the others run on across updates.

A curriculum stage (``curriculum``) moves whole groups of environments to other surfaces
from a given update on; at the start of every stage all environments restart.

PPO: Gaussian actions in the pre-squash space (``delta = bound * tanh(a)``; the
log-probability is that of ``a``), GAE, a clipped policy loss, an unclipped value
loss, one Adam optimiser for actor and critic, advantages normalised per update, and
an optional stop after an epoch whose mean approximate KL divergence exceeds
``target_kl``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf

from elastra import config, success
from elastra import residual as residual_module
from elastra.config import assets_dir
from elastra.host import HostController
from elastra.protomotions import ReferenceClips, Tracker
from elastra.rollout import protomotions_settle
from elastra.scene import Scene, Surface, build_scene
from elastra.sim import Batch


# --------------------------------------------------------------------------- #
# environment surfaces and curriculum
# --------------------------------------------------------------------------- #
def environment_surfaces(train: DictConfig, seed: int) -> list[str]:
    """The surface of every environment: ``surfaces`` gives the number of environments per
    surface, and the environments are shuffled once with the training seed."""

    labels = [
        Surface.parse(str(name)).label
        for name, count in train.surfaces.items()
        for _ in range(int(count))
    ]
    if len(labels) != int(train.num_envs):
        raise ValueError(
            f"surfaces add up to {len(labels)} environments, num_envs is {train.num_envs}"
        )
    order = np.random.default_rng([seed, 1]).permutation(len(labels))
    return [labels[k] for k in order]


def stage_surfaces(train: DictConfig, surfaces: list[str], update: int) -> list[str]:
    """Environment surfaces in force at ``update``: every curriculum stage that has started
    replaces surfaces as its ``replace`` map says."""

    current = list(surfaces)
    for stage in train.get("curriculum") or []:
        if update >= int(stage.from_update):
            mapping = {
                Surface.parse(str(a)).label: Surface.parse(str(b)).label
                for a, b in stage.replace.items()
            }
            current = [mapping.get(label, label) for label in current]
    return current


def stage_starts(train: DictConfig) -> set[int]:
    return {int(stage.from_update) for stage in train.get("curriculum") or []}


# --------------------------------------------------------------------------- #
# environments
# --------------------------------------------------------------------------- #
def scenes_for(cfg: DictConfig, robot: str, labels, scenes: dict[str, Scene]) -> None:
    for label in sorted(set(labels)):
        if label not in scenes:
            scenes[label] = build_scene(cfg, robot, label)


def initial_state_rows(cfg: DictConfig, robot: str) -> np.ndarray:
    """Indices of the training initial states in ``data/initial_states/<robot>.npz``."""

    data = config.data_dir(cfg) / "initial_states"
    if robot == "host":
        states = np.load(data / "host.npz")
        return np.flatnonzero(states["split"] == str(cfg.train.initial_states))
    states = np.load(data / "protomotions.npz")
    return np.arange(len(states[f"{cfg.train.initial_states}_qpos"]))


def warm_up(
    cfg: DictConfig, robot: str, jobs: Sequence[tuple[str, int]], scenes: dict[str, Scene]
) -> list:
    """Episode start states for (surface, initial state) pairs.

    HoST: the initial state lowered onto the surface (dropped onto the trampoline) and run
    through HoST's unactuated control steps; an entry is (full physics state, torso
    height).  ProtoMotions: the initial state settled on the surface under a gravity
    ramp, as in the evaluation; an entry is the full physics state.
    """

    if not jobs:
        return []
    scenes_for(cfg, robot, [label for label, _ in jobs], scenes)
    job_scenes = [scenes[label] for label, _ in jobs]
    data = config.data_dir(cfg) / "initial_states"
    if robot == "protomotions":
        states = np.load(data / "protomotions.npz")
        poses = states[f"{cfg.train.initial_states}_qpos"]
        return protomotions_settle(cfg, job_scenes, [poses[row] for _, row in jobs])
    host = cfg.robots.host
    states = np.load(data / "host.npz")
    key = [
        "qpos_initial" if Surface.parse(label).kind == "trampoline" else "qpos_lowered"
        for label, _ in jobs
    ]
    batch = Batch(
        job_scenes,
        physics_dt=float(cfg.sim.physics_dt_s),
        control_dt=float(cfg.sim.control_dt_s),
        criterion=cfg.criterion,
    )
    try:
        batch.set_robot_qpos([states[k][row] for k, (_, row) in zip(key, jobs)])
        zero = np.zeros((len(jobs), int(host.num_actions)))
        for _ in range(int(host.unactuated_control_steps)):
            batch.step(zero)
        if batch.diverged().any():
            raise FloatingPointError("a warm-up diverged")
        torso = batch.robot_state()["torso_z"]
        return [(state, float(z)) for state, z in zip(batch.states(), torso)]
    finally:
        batch.close()


class TrainingEnvs:
    """Episode state shared by the two controllers, for the environments ``env_ids`` (indices
    into all environments of the run) on ``surfaces``; ``cache`` holds the start states
    of every (surface, initial state) pair (:func:`warm_up`)."""

    robot: str

    def __init__(
        self,
        cfg: DictConfig,
        surfaces: Sequence[str],
        env_ids: Sequence[int],
        stage: int,
        scenes: dict[str, Scene],
        cache: dict,
    ) -> None:
        self.cfg = cfg
        self.train = cfg.train
        self.n = len(surfaces)
        self.env_ids = np.asarray(env_ids, dtype=np.int64)
        self.surfaces = [Surface.parse(s) for s in surfaces]
        self.labels = [s.label for s in self.surfaces]
        self.cache = cache
        seed = int(self.train.seed)
        self.reset_rngs = [np.random.default_rng([seed, 2, int(stage), int(e)]) for e in env_ids]
        scenes_for(cfg, self.robot, self.labels, scenes)
        self.scenes = [scenes[label] for label in self.labels]
        self.batch = Batch(
            self.scenes,
            physics_dt=float(cfg.sim.physics_dt_s),
            control_dt=float(cfg.sim.control_dt_s),
            criterion=cfg.criterion,
            task_area_every_physics_step=bool(self.train.task_area_every_physics_step),
        )
        self.heights = [success.SurfaceHeight(scene) for scene in self.scenes]
        self.hold = success.hold_steps(cfg.criterion, float(cfg.sim.control_dt_s))
        self.frame = np.zeros(self.n, dtype=np.int64)
        self.horizon = np.zeros(self.n, dtype=np.int64)
        self.stand_run = np.zeros(self.n, dtype=np.int64)
        self.completed = np.zeros(self.n, dtype=bool)
        self.exited_first = np.zeros(self.n, dtype=bool)
        self.initial_torso_z = np.zeros(self.n)
        self.episode_return = np.zeros(self.n)
        self.parent = np.zeros((self.n, len(self.scenes[0].robot_dof)))

    def close(self) -> None:
        self.batch.close()

    # -- per controller ----------------------------------------------------------
    def _restart(self, env_ids: np.ndarray) -> None:
        raise NotImplementedError

    def _parent_command(self, state: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def _apply(self, command: np.ndarray) -> None:
        raise NotImplementedError

    def _observation(self, state: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    # -- common ------------------------------------------------------------------
    def reset(self, env_ids: np.ndarray | None = None) -> None:
        ids = np.arange(self.n) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
        if ids.size == 0:
            return
        self._restart(ids)
        self.frame[ids] = 0
        self.stand_run[ids] = 0
        self.completed[ids] = False
        self.exited_first[ids] = False
        self.episode_return[ids] = 0.0

    def observation(self) -> np.ndarray:
        """The residual observation of every environment (this also runs the frozen controller,
        whose command the next :meth:`step` uses)."""

        state = self.batch.robot_state()
        self.parent = self._parent_command(state)
        return self._observation(state)

    def step(self, raw_action: np.ndarray) -> dict[str, np.ndarray]:
        delta = float(self.bound) * np.tanh(np.asarray(raw_action, dtype=np.float64))
        self._apply(self.parent + delta)
        self.frame += 1
        diverged = self.batch.diverged()
        if diverged.any():
            raise FloatingPointError(f"physics diverged in environments {np.flatnonzero(diverged)}")
        state = self.batch.robot_state()
        relative = np.asarray([h.surface_z(d) for h, d in zip(self.heights, self.batch.datas)])
        relative = state["pelvis_z"] - relative
        standing = success.standing_flags(relative, state["upright"], self.cfg.criterion)
        exited = np.asarray([area.first_exit_step is not None for area in self.batch.task_areas])
        reward = residual_module.reward(
            state,
            standing,
            self.stand_run,
            exited,
            self.initial_torso_z,
            self.train.reward,
            self.hold,
        )
        self.stand_run = np.where(standing, self.stand_run + 1, 0)
        newly_complete = (self.stand_run >= self.hold) & ~self.completed & ~exited
        self.completed |= newly_complete
        self.exited_first |= exited & ~self.completed
        self.episode_return += reward
        terminated = exited
        truncated = (self.frame >= self.horizon) & ~terminated
        return {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "success": self.completed & ~self.exited_first,
        }

    def advance(self, raw_action: np.ndarray) -> dict[str, np.ndarray]:
        """One control step of the training loop: :meth:`step`, the observation after it
        (``next_obs``, which bootstraps truncated episodes), the return of every episode so
        far, and the observation after the finished episodes restart (``obs``)."""

        result = self.step(raw_action)
        result["next_obs"] = self.observation()
        result["episode_return"] = self.episode_return.copy()
        ended = np.flatnonzero(result["terminated"] | result["truncated"])
        if ended.size:
            self.reset(ended)
            result["obs"] = self.observation()
        else:
            result["obs"] = result["next_obs"]
        return result


class HostEnvs(TrainingEnvs):
    robot = "host"

    def __init__(
        self,
        cfg: DictConfig,
        surfaces: Sequence[str],
        env_ids: Sequence[int],
        stage: int,
        scenes: dict[str, Scene],
        cache: dict,
    ) -> None:
        super().__init__(cfg, surfaces, env_ids, stage, scenes, cache)
        host = cfg.robots.host
        self.residual_cfg = cfg.residuals.host
        self.bound = float(self.residual_cfg.bound_action_units)
        self.postures = ["prone" if env % 2 == 0 else "supine" for env in self.env_ids.tolist()]
        self.controller = HostController(cfg, assets_dir(cfg), self.postures)
        self.noise = [
            np.random.default_rng(
                int(self.train.seed) + int(host.observation_noise_seed_offset) + env
            )
            for env in self.env_ids.tolist()
        ]
        # every actuated step counts as "after the unactuated opening" for HoST
        self.actuated = np.full(self.n, int(host.unactuated_control_steps) + 1, dtype=np.int64)
        self.horizon[:] = int(self.train.horizon_control_steps)
        self.features = residual_module.surface_features(cfg, self.residual_cfg, self.surfaces)
        states = np.load(config.data_dir(cfg) / "initial_states" / "host.npz")
        chosen = states["split"] == str(self.train.initial_states)
        self.by_posture = {
            p: np.flatnonzero(chosen & (states["posture"] == p)) for p in ("prone", "supine")
        }

    def _restart(self, env_ids: np.ndarray) -> None:
        picked = []
        for env in env_ids.tolist():
            pool = self.by_posture[self.postures[env]]
            picked.append(int(pool[self.reset_rngs[env].integers(0, pool.size)]))
        entries = [self.cache[self.labels[env]][row] for env, row in zip(env_ids, picked)]
        self.batch.restore(env_ids, [state for state, _ in entries])
        self.initial_torso_z[env_ids] = [z for _, z in entries]
        self.controller.reset(env_ids)

    def _parent_command(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return self.controller.act(self.actuated)

    def _apply(self, command: np.ndarray) -> None:
        self.batch.step(command)
        self.controller.push(
            self.batch.datas, self.batch.scenes, command, self.actuated, self.noise
        )

    def _observation(self, state: dict[str, np.ndarray]) -> np.ndarray:
        surface_block = None
        if bool(self.residual_cfg.surface_observation):
            surface_block = np.concatenate(
                [self.features, state["deflection"][:, None] * 10.0], axis=1
            )
        return residual_module.observation(
            state,
            self.parent,
            quat=state["quat_wxyz"],
            progress=residual_module.phase(self.frame, self.horizon),
            surface_block=surface_block,
            position=bool(self.residual_cfg.position_observation),
        )


class ProtomotionsEnvs(TrainingEnvs):
    robot = "protomotions"

    def __init__(
        self,
        cfg: DictConfig,
        surfaces: Sequence[str],
        env_ids: Sequence[int],
        stage: int,
        scenes: dict[str, Scene],
        cache: dict,
    ) -> None:
        super().__init__(cfg, surfaces, env_ids, stage, scenes, cache)
        self.residual_cfg = cfg.residuals.protomotions
        self.bound = float(self.residual_cfg.bound_rad)
        assets = assets_dir(cfg)
        self.clips = ReferenceClips.load(Path(assets) / "reference_clips")
        self.tracker = Tracker(assets, self.clips)
        states = np.load(config.data_dir(cfg) / "initial_states" / "protomotions.npz")
        prefix = str(self.train.initial_states)
        self.poses = states[f"{prefix}_qpos"]
        self.pose_clip = np.asarray(
            [self.clips.index(str(c)) for c in states[f"{prefix}_clip"]], dtype=np.int64
        )
        self.clip = np.zeros(self.n, dtype=np.int64)
        self.heading = np.zeros((self.n, 4))
        self.previous = np.zeros((self.n, len(self.scenes[0].robot_dof)), dtype=np.float32)

    def _restart(self, env_ids: np.ndarray) -> None:
        poses = np.asarray(
            [self.reset_rngs[env].integers(0, len(self.poses)) for env in env_ids.tolist()],
            dtype=np.int64,
        )
        self.batch.restore(
            env_ids, [self.cache[self.labels[env]][int(p)] for env, p in zip(env_ids, poses)]
        )
        self.clip[env_ids] = self.pose_clip[poses]
        self.horizon[env_ids] = self.clips.frames[self.clip[env_ids]]
        state = self.batch.robot_state()
        self.heading[env_ids] = self.tracker.heading(
            state["anchor_xyzw"][env_ids], self.clip[env_ids]
        )
        self.previous[env_ids] = 0.0
        self.initial_torso_z[env_ids] = state["torso_z"][env_ids]

    def _parent_command(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return self.tracker.act(state, self.frame, self.clip, self.heading, self.previous)

    def _apply(self, command: np.ndarray) -> None:
        self.batch.step(command)
        self.previous = command.astype(np.float32)

    def _observation(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return residual_module.observation(
            state,
            self.parent,
            quat=state["anchor_xyzw"],
            progress=residual_module.phase(self.frame, self.horizon),
            position=bool(self.residual_cfg.position_observation),
        )


ENVS = {"host": HostEnvs, "protomotions": ProtomotionsEnvs}


# --------------------------------------------------------------------------- #
# worker processes
# --------------------------------------------------------------------------- #
def _serve(conn, cfg_container: dict) -> None:
    """A worker process: answers ``(command, payload)`` messages until ``close``."""

    cfg = OmegaConf.create(cfg_container)
    robot = str(cfg.train.controller)
    scenes: dict[str, Scene] = {}
    envs: TrainingEnvs | None = None
    while True:
        command, payload = conn.recv()
        try:
            if command == "close":
                if envs is not None:
                    envs.close()
                conn.send(("ok", None))
                return
            if command == "warm_up":
                result = warm_up(cfg, robot, payload, scenes)
            elif command == "start":
                if envs is not None:
                    envs.close()
                envs = ENVS[robot](cfg, scenes=scenes, **payload)
                envs.reset()
                result = envs.observation()
            elif command == "advance":
                result = envs.advance(payload)
            else:
                raise ValueError(f"unknown command {command}")
            conn.send(("ok", result))
        except Exception:
            import traceback

            conn.send(("error", traceback.format_exc()))


class ShardedEnvs:
    """All environments of a run, split over worker processes (environment ``e`` on worker
    ``e mod workers``).  The start states are computed once, spread over the workers, and
    kept for later curriculum stages."""

    def __init__(self, cfg: DictConfig, workers: int) -> None:
        import multiprocessing as mp

        self.cfg = cfg
        self.robot = str(cfg.train.controller)
        self.num_envs = int(cfg.train.num_envs)
        self.workers = max(1, min(int(workers), self.num_envs))
        self.shards = [np.arange(w, self.num_envs, self.workers) for w in range(self.workers)]
        self.rows = initial_state_rows(cfg, self.robot)
        self.cache: dict = {}
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        context = mp.get_context("spawn")
        container = OmegaConf.to_container(cfg, resolve=True)
        self.conns, self.processes = [], []
        for _ in range(self.workers):
            parent, child = context.Pipe()
            process = context.Process(target=_serve, args=(child, container), daemon=True)
            process.start()
            child.close()
            self.conns.append(parent)
            self.processes.append(process)

    def _call(self, messages: list[tuple[str, object]]) -> list:
        for conn, message in zip(self.conns, messages):
            conn.send(message)
        replies = [conn.recv() for conn in self.conns]
        errors = [payload for status, payload in replies if status == "error"]
        if errors:
            raise RuntimeError("a training worker failed:\n" + errors[0])
        return [payload for _, payload in replies]

    def start(self, surfaces: Sequence[str], stage: int) -> np.ndarray:
        """Put every environment on its surface and start its first episode; returns the
        observations."""

        missing = [
            (label, int(row))
            for label in sorted(set(surfaces))
            if label not in self.cache
            for row in self.rows
        ]
        parts = [missing[w :: self.workers] for w in range(self.workers)]
        for part, states in zip(parts, self._call([("warm_up", part) for part in parts])):
            for (label, row), state in zip(part, states):
                self.cache.setdefault(label, {})[row] = state
        messages = []
        for ids in self.shards:
            labels = [surfaces[e] for e in ids.tolist()]
            payload = {
                "surfaces": labels,
                "env_ids": ids,
                "stage": int(stage),
                "cache": {label: self.cache[label] for label in set(labels)},
            }
            messages.append(("start", payload))
        return self._gather(self._call(messages))

    def advance(self, raw_action: np.ndarray) -> dict[str, np.ndarray]:
        replies = self._call([("advance", raw_action[ids]) for ids in self.shards])
        return {key: self._gather([reply[key] for reply in replies]) for key in replies[0]}

    def _gather(self, parts: list[np.ndarray]) -> np.ndarray:
        first = np.asarray(parts[0])
        out = np.zeros((self.num_envs,) + first.shape[1:], dtype=first.dtype)
        for ids, part in zip(self.shards, parts):
            out[ids] = part
        return out

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
                conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Generalised advantage estimates for a ``(steps, envs)`` rollout."""

    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in range(steps - 1, -1, -1):
        next_values = last_values if t == steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        running = delta + gamma * lam * not_done * running
        advantages[t] = running
    return advantages


def ppo_update(
    actor, critic, optimizer, rollout: dict[str, np.ndarray], ppo: DictConfig, generator
) -> dict[str, float]:
    import torch
    from torch.distributions import Normal

    obs = torch.as_tensor(rollout["obs"], dtype=torch.float32)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.float32)
    old_logp = torch.as_tensor(rollout["logp"], dtype=torch.float32)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    size = obs.shape[0]
    minibatch = size // int(ppo.minibatches)
    parameters = list(actor.parameters()) + list(critic.parameters())
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    count = 0
    epochs_run = 0
    for _ in range(int(ppo.epochs)):
        order = torch.randperm(size, generator=generator)
        epoch_kl = []
        for start in range(0, minibatch * int(ppo.minibatches), minibatch):
            idx = order[start : start + minibatch]
            mean, log_std = actor(obs[idx])
            dist = Normal(mean, log_std.exp())
            logp = dist.log_prob(actions[idx]).sum(-1)
            ratio = (logp - old_logp[idx]).exp()
            clipped = ratio.clamp(1.0 - float(ppo.clip_ratio), 1.0 + float(ppo.clip_ratio))
            policy_loss = -torch.min(ratio * advantages[idx], clipped * advantages[idx]).mean()
            value_loss = torch.nn.functional.mse_loss(critic(obs[idx]), returns[idx])
            entropy = dist.entropy().sum(-1).mean()
            loss = (
                policy_loss + float(ppo.value_coef) * value_loss - float(ppo.entropy_coef) * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(ppo.max_grad_norm))
            optimizer.step()
            kl = float((old_logp[idx] - logp).mean().detach())
            epoch_kl.append(kl)
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["approx_kl"] += kl
            count += 1
        epochs_run += 1
        if ppo.target_kl is not None and float(np.mean(epoch_kl)) > float(ppo.target_kl):
            break
    out = {k: v / max(count, 1) for k, v in stats.items()}
    out["epochs_run"] = epochs_run
    return out


# --------------------------------------------------------------------------- #
# the training loop
# --------------------------------------------------------------------------- #
def _policy_blob(cfg: DictConfig, actor, critic, obs_dim: int, update: int) -> dict:
    robot = str(cfg.train.controller)
    res = cfg.residuals[robot]
    bound = float(res.bound_action_units if robot == "host" else res.bound_rad)
    return {
        "robot": robot,
        "obs_dim": int(obs_dim),
        "hidden": [int(h) for h in res.hidden],
        "bound": bound,
        "position_observation": bool(res.position_observation),
        "surface_observation": bool(res.surface_observation),
        "updates": int(update),
        "description": str(cfg.train.description),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }


def latest_checkpoint(out: Path) -> Path | None:
    paths = sorted((out / "checkpoints").glob("update_*.pt"))
    return paths[-1] if paths else None


def _cut_history(path: Path, update: int) -> None:
    """Keep the history lines up to ``update`` (the lines a resumed run will write again
    are dropped)."""

    if not path.is_file():
        return
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    kept = [line for line in lines if int(json.loads(line)["update"]) <= update]
    path.write_text("".join(line + "\n" for line in kept))


def train(cfg: DictConfig) -> Path:
    import torch

    tcfg = cfg.train
    seed = int(tcfg.seed)
    out = config.repo_path(tcfg.out)
    checkpoint = latest_checkpoint(out)
    if checkpoint is not None and not bool(tcfg.resume):
        raise FileExistsError(
            f"{out} already holds checkpoints: set train.resume=true to continue that run "
            "or choose another train.out"
        )
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    torch.set_num_threads(int(tcfg.torch_threads))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    blob = None
    update = 0
    if checkpoint is not None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        update = int(blob["updates"])
        _cut_history(out / "history.jsonl", update)
        print(f"resuming from {checkpoint} (update {update})", flush=True)
    else:
        (out / "history.jsonl").write_text("")
    envs = ShardedEnvs(cfg, int(tcfg.workers))
    try:
        return _train_loop(cfg, envs, update, blob, generator, out)
    finally:
        envs.close()


def _train_loop(
    cfg: DictConfig, envs: ShardedEnvs, update: int, blob: dict | None, generator, out: Path
) -> Path:
    """PPO from ``update`` on (from the checkpoint ``blob`` if there is one)."""

    import torch
    from torch.distributions import Normal

    tcfg = cfg.train
    robot = str(tcfg.controller)
    seed = int(tcfg.seed)
    env_surfaces = environment_surfaces(tcfg, seed)
    res = cfg.residuals[robot]
    action_dim = int(cfg.robots.host.num_actions) if robot == "host" else 29
    first_update = update
    labels = stage_surfaces(tcfg, env_surfaces, update)
    obs = envs.start(labels, update)
    actor, critic = residual_module.build_actor_critic(
        obs.shape[1],
        action_dim,
        hidden=tuple(int(h) for h in res.hidden),
        initial_log_std=float(res.initial_log_std),
        seed=seed,
    )
    ppo = tcfg.ppo
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=float(ppo.learning_rate)
    )
    if blob is not None:
        actor.load_state_dict(blob["actor"])
        critic.load_state_dict(blob["critic"])
        optimizer.load_state_dict(blob["optimizer"])
        torch.set_rng_state(blob["rng"]["torch"])
        generator.set_state(blob["rng"]["generator"])
    history = (out / "history.jsonl").open("a")
    steps = int(tcfg.rollout_steps)
    gamma = float(ppo.gamma)
    try:
        while update < int(tcfg.updates):
            if update in stage_starts(tcfg) and update > first_update:
                labels = stage_surfaces(tcfg, env_surfaces, update)
                obs = envs.start(labels, update)
            started = time.time()
            buffer = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones")}
            finished = []
            for _ in range(steps):
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.float32)
                    mean, log_std = actor(x)
                    dist = Normal(mean, log_std.exp())
                    raw = dist.sample()
                    logp = dist.log_prob(raw).sum(-1)
                    value = critic(x)
                result = envs.advance(raw.numpy().astype(np.float64))
                reward = result["reward"].copy()
                cut = np.flatnonzero(result["truncated"])
                if cut.size:
                    with torch.no_grad():
                        boot = critic(torch.as_tensor(result["next_obs"][cut], dtype=torch.float32))
                    reward[cut] += gamma * boot.numpy().astype(np.float64)
                done = result["terminated"] | result["truncated"]
                buffer["obs"].append(obs)
                buffer["actions"].append(raw.numpy())
                buffer["logp"].append(logp.numpy())
                buffer["values"].append(value.numpy().astype(np.float64))
                buffer["rewards"].append(reward)
                buffer["dones"].append(done.astype(np.float64))
                ended = np.flatnonzero(done)
                for env in ended.tolist():
                    finished.append(
                        {
                            "surface": labels[env],
                            "success": bool(result["success"][env]),
                            "exited": bool(result["terminated"][env]),
                            "return": float(result["episode_return"][env]),
                        }
                    )
                obs = result["obs"]
            with torch.no_grad():
                last_values = critic(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            rewards = np.asarray(buffer["rewards"])
            values = np.asarray(buffer["values"])
            dones = np.asarray(buffer["dones"])
            advantages = gae(
                rewards, values, dones, last_values.astype(np.float64), gamma, float(ppo.gae_lambda)
            )
            rollout = {
                "obs": np.concatenate(buffer["obs"]),
                "actions": np.concatenate(buffer["actions"]),
                "logp": np.concatenate(buffer["logp"]),
                "advantages": advantages.reshape(-1),
                "returns": (advantages + values).reshape(-1),
            }
            stats = ppo_update(actor, critic, optimizer, rollout, ppo, generator)
            update += 1
            by_surface = {}
            for row in finished:
                entry = by_surface.setdefault(row["surface"], [0, 0, 0])
                entry[0] += 1
                entry[1] += row["success"]
                entry[2] += row["exited"]
            record = {
                "update": update,
                "wall_s": time.time() - started,
                "mean_step_reward": float(rewards.mean()),
                "episodes": len(finished),
                "successes": sum(r["success"] for r in finished),
                "by_surface": {
                    k: {"episodes": v[0], "successes": v[1], "exits": v[2]}
                    for k, v in sorted(by_surface.items())
                },
                **stats,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(
                f"update {update}: reward/step {record['mean_step_reward']:.3f}, "
                f"episodes {record['episodes']}, successes {record['successes']}, "
                f"kl {stats['approx_kl']:.4f}, {record['wall_s']:.1f} s",
                flush=True,
            )
            if update % int(tcfg.checkpoint_every) == 0 or update == int(tcfg.updates):
                blob = _policy_blob(cfg, actor, critic, obs.shape[1], update)
                blob["optimizer"] = optimizer.state_dict()
                blob["rng"] = {
                    "torch": torch.get_rng_state(),
                    "generator": generator.get_state(),
                }
                path = out / "checkpoints" / f"update_{update:06d}.pt"
                torch.save(blob, path.with_suffix(".part"))
                os.replace(path.with_suffix(".part"), path)
        final = out / "policy.pt"
        torch.save(_policy_blob(cfg, actor, critic, obs.shape[1], update), final)
        return final
    finally:
        history.close()


def main(argv: list[str] | None = None) -> int:
    cfg = config.load(sys.argv[1:] if argv is None else list(argv), config_name="training")
    path = train(cfg)
    print(f"policy written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
