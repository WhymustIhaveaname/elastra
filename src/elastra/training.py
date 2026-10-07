"""Training a residual policy with PPO (``elastra-train``, ``conf/train/*.yaml``).

    elastra-train                                    # HoST (conf/train/host.yaml)
    elastra-train train=protomotions
    elastra-train train=protomotions_curriculum
    elastra-train train.updates=2 train.threads=8    # a short test run

The run writes to ``train.out``: ``config.yaml``, ``history.jsonl`` (one line per
update: reward, finished episodes and successes per surface, PPO statistics),
``checkpoints/update_*.pt`` every ``train.checkpoint_every`` updates and the final
``policy.pt``, which ``elastra-evaluate policy=...`` reads.

The parallel environments are a fixed number, each on one surface, stepped together
on one clock (:class:`elastra.sim.Batch`).  An episode starts from a cached state:

* HoST: one of the training initial states of the environment's posture (even environments prone,
  odd environments supine), lowered onto the surface (dropped onto the trampoline) and run
  through HoST's 30 unactuated control steps; the episode is the next
  ``horizon_control_steps`` steps, with HoST's action history starting at zero;
* ProtoMotions: one of the training initial states, settled on the surface under a gravity
  ramp (as in the evaluation); the episode lasts as long as the reference clip of that initial state.

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
        raise ValueError(f"surfaces add up to {len(labels)} environments, num_envs is {train.num_envs}")
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
class TrainingEnvs:
    """Episode state shared by the two controllers."""

    robot: str

    def __init__(
        self,
        cfg: DictConfig,
        surfaces: Sequence[str],
        rng: np.random.Generator,
        scenes: dict[str, Scene],
    ) -> None:
        self.cfg = cfg
        self.train = cfg.train
        self.n = len(surfaces)
        self.surfaces = [Surface.parse(s) for s in surfaces]
        self.labels = [s.label for s in self.surfaces]
        self.rng = rng
        for label in sorted(set(self.labels)):
            if label not in scenes:
                scenes[label] = build_scene(cfg, self.robot, label)
        self.scenes = [scenes[label] for label in self.labels]
        self.batch = Batch(
            self.scenes,
            physics_dt=float(cfg.sim.physics_dt_s),
            control_dt=float(cfg.sim.control_dt_s),
            criterion=cfg.criterion,
            threads=int(self.train.threads),
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


class HostEnvs(TrainingEnvs):
    robot = "host"

    def __init__(
        self,
        cfg: DictConfig,
        surfaces: Sequence[str],
        rng: np.random.Generator,
        scenes: dict[str, Scene],
        cache: dict,
        seed: int,
    ) -> None:
        super().__init__(cfg, surfaces, rng, scenes)
        host = cfg.robots.host
        self.residual_cfg = cfg.residuals.host
        self.bound = float(self.residual_cfg.bound_action_units)
        self.postures = ["prone" if env % 2 == 0 else "supine" for env in range(self.n)]
        self.controller = HostController(cfg, assets_dir(cfg), self.postures)
        self.noise = [
            np.random.default_rng(seed + int(host.observation_noise_seed_offset) + env)
            for env in range(self.n)
        ]
        # every actuated step counts as "after the unactuated opening" for HoST
        self.actuated = np.full(self.n, int(host.unactuated_control_steps) + 1, dtype=np.int64)
        self.horizon[:] = int(self.train.horizon_control_steps)
        self.features = residual_module.surface_features(cfg, self.residual_cfg, self.surfaces)
        self.cache = cache
        self._fill_cache()
        states = np.load(config.data_dir(cfg) / "initial_states" / "host.npz")
        chosen = states["split"] == str(self.train.initial_states)
        self.by_posture = {
            p: np.flatnonzero(chosen & (states["posture"] == p)) for p in ("prone", "supine")
        }

    def _fill_cache(self) -> None:
        """Warm-up states (after HoST's unactuated steps) for every (surface, initial state)."""

        cfg = self.cfg
        host = cfg.robots.host
        states = np.load(config.data_dir(cfg) / "initial_states" / "host.npz")
        rows = np.flatnonzero(states["split"] == str(self.train.initial_states))
        missing = [label for label in sorted(set(self.labels)) if label not in self.cache]
        if not missing:
            return
        jobs = [(label, int(row)) for label in missing for row in rows]
        scenes = [self.scenes[self.labels.index(label)] for label, _ in jobs]
        key = [
            "qpos_initial" if Surface.parse(label).kind == "trampoline" else "qpos_lowered"
            for label, _ in jobs
        ]
        batch = Batch(
            scenes,
            physics_dt=float(cfg.sim.physics_dt_s),
            control_dt=float(cfg.sim.control_dt_s),
            criterion=cfg.criterion,
            threads=int(self.train.threads),
        )
        try:
            batch.set_robot_qpos([states[k][row] for k, (_, row) in zip(key, jobs)])
            zero = np.zeros((len(jobs), int(host.num_actions)))
            for _ in range(int(host.unactuated_control_steps)):
                batch.step(zero)
            if batch.diverged().any():
                raise FloatingPointError("a warm-up diverged")
            torso = batch.robot_state()["torso_z"]
            for (label, row), state, z in zip(jobs, batch.states(), torso):
                self.cache.setdefault(label, {})[row] = (state, float(z))
        finally:
            batch.close()

    def _restart(self, env_ids: np.ndarray) -> None:
        picked = []
        for env in env_ids.tolist():
            pool = self.by_posture[self.postures[env]]
            picked.append(int(pool[self.rng.integers(0, pool.size)]))
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
        rng: np.random.Generator,
        scenes: dict[str, Scene],
        cache: dict,
        seed: int,
    ) -> None:
        super().__init__(cfg, surfaces, rng, scenes)
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
        self.cache = cache
        self._fill_cache()

    def _fill_cache(self) -> None:
        """Settled states for every (surface, training initial state)."""

        missing = [label for label in sorted(set(self.labels)) if label not in self.cache]
        if not missing:
            return
        jobs = [(label, pose) for label in missing for pose in range(len(self.poses))]
        scenes = [self.scenes[self.labels.index(label)] for label, _ in jobs]
        settled = protomotions_settle(
            self.cfg, scenes, [self.poses[p] for _, p in jobs], threads=int(self.train.threads)
        )
        for (label, pose), state in zip(jobs, settled):
            self.cache.setdefault(label, {})[pose] = state

    def _restart(self, env_ids: np.ndarray) -> None:
        poses = self.rng.integers(0, len(self.poses), size=env_ids.size)
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


def train(cfg: DictConfig) -> Path:
    import torch
    from torch.distributions import Normal

    tcfg = cfg.train
    robot = str(tcfg.controller)
    seed = int(tcfg.seed)
    out = config.repo_path(tcfg.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    torch.set_num_threads(int(tcfg.torch_threads))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    rng = np.random.default_rng(seed)
    env_surfaces = environment_surfaces(tcfg, seed)
    res = cfg.residuals[robot]
    action_dim = int(cfg.robots.host.num_actions) if robot == "host" else 29

    scenes: dict[str, Scene] = {}
    cache: dict = {}
    update = 0
    envs = ENVS[robot](cfg, stage_surfaces(tcfg, env_surfaces, update), rng, scenes, cache, seed)
    envs.reset()
    obs = envs.observation()
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
    history = (out / "history.jsonl").open("a")
    steps = int(tcfg.rollout_steps)
    gamma = float(ppo.gamma)
    try:
        while update < int(tcfg.updates):
            if update in stage_starts(tcfg) and update > 0:
                envs.close()
                envs = ENVS[robot](
                    cfg, stage_surfaces(tcfg, env_surfaces, update), rng, scenes, cache, seed
                )
                envs.reset()
                obs = envs.observation()
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
                result = envs.step(raw.numpy().astype(np.float64))
                reward = result["reward"].copy()
                next_obs = envs.observation()
                cut = np.flatnonzero(result["truncated"])
                if cut.size:
                    with torch.no_grad():
                        boot = critic(torch.as_tensor(next_obs[cut], dtype=torch.float32))
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
                            "surface": envs.labels[env],
                            "success": bool(result["success"][env]),
                            "exited": bool(result["terminated"][env]),
                            "return": float(envs.episode_return[env]),
                        }
                    )
                if ended.size:
                    envs.reset(ended)
                    obs = envs.observation()
                else:
                    obs = next_obs
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
                torch.save(blob, out / "checkpoints" / f"update_{update:06d}.pt")
        final = out / "policy.pt"
        torch.save(_policy_blob(cfg, actor, critic, obs.shape[1], update), final)
        return final
    finally:
        history.close()
        envs.close()


def main(argv: list[str] | None = None) -> int:
    cfg = config.load(sys.argv[1:] if argv is None else list(argv), config_name="training")
    path = train(cfg)
    print(f"policy written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
