"""The residual policy: network, observation and reward.

The residual adds a bounded correction to the original controller's command:
``delta = bound * tanh(a)`` with ``a`` the network output (sampled from a diagonal
Gaussian during training, the mean at evaluation).  For HoST the correction is in
HoST action units (one unit moves the PD target by HoST's action scale), for
ProtoMotions it is in radians added to the tracker's PD target.

Observation (fixed scales, no running normalisation), in order:

    joint positions, joint velocities x 0.1, base orientation quaternion,
    base angular velocity x 0.25, the original controller's output,
    torso height, torso uprightness, pelvis height,
    [HoST only] surface type one-hot (mattress, trampoline, rigid), stiffness input,
                maximum surface deflection x 10,
    [optional]  root xy, root linear velocity in the root frame x 0.1,
                signed distances of the robot to the x and y task-area boundaries,
    progress of the trial (step / steps planned, clipped to [0, 1]).

The stiffness input is ``log2(s / s_ref) / 6`` with ``s`` the mattress stiffness per
unit area (cell stiffness / pitch^2) or the trampoline edge stiffness and ``s_ref``
the reference value of ``conf/residual/host.yaml``; on rigid ground it is 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

SURFACE_KINDS = ("mattress", "trampoline", "rigid")


def stiffness_input(cfg: DictConfig, residual_cfg: DictConfig, surface) -> float:
    spec = residual_cfg.stiffness_input
    if surface.kind == "rigid":
        return float(spec.rigid_value)
    if surface.kind == "mattress":
        # k / pitch^2 is the bed's stiffness per unit area: the input does not
        # change when the same bed is discretised more finely
        k = float(cfg.mattress.beds[surface.name].k_n_per_m)
        k_ref = float(spec.mattress_reference_k_n_per_m)
        grid = 2.0 * np.log2(float(spec.mattress_reference_pitch_m) / float(cfg.mattress.pitch_m))
        return float((np.log2(k / k_ref) + grid) / float(spec.log2_range))
    k = float(cfg.trampoline.trampolines[surface.name].edge_stiffness_n_per_m)
    k_ref = float(spec.trampoline_reference_k_n_per_m)
    return float(np.log2(k / k_ref) / float(spec.log2_range))


def surface_features(cfg: DictConfig, residual_cfg: DictConfig, surfaces: list) -> np.ndarray:
    rows = []
    for surface in surfaces:
        onehot = [1.0 if surface.kind == kind else 0.0 for kind in SURFACE_KINDS]
        rows.append([*onehot, stiffness_input(cfg, residual_cfg, surface)])
    return np.asarray(rows, dtype=np.float64)


def phase(step: np.ndarray, horizon: np.ndarray) -> np.ndarray:
    step = np.asarray(step, dtype=np.float64)
    return np.clip(step / np.maximum(np.asarray(horizon, dtype=np.float64), 1.0), 0.0, 1.0)[:, None]


def observation(
    state: dict[str, np.ndarray],
    parent: np.ndarray,
    *,
    quat: np.ndarray,
    progress: np.ndarray,
    surface_block: np.ndarray | None = None,
    position: bool = False,
) -> np.ndarray:
    """One batch of residual observations (see the module docstring for the layout)."""

    blocks = [
        state["dof_pos"],
        state["dof_vel"] * 0.1,
        quat,
        state["root_ang_vel"] * 0.25,
        np.asarray(parent, dtype=np.float64),
        state["torso_z"][:, None],
        state["torso_up"][:, None],
        state["pelvis_z"][:, None],
    ]
    if surface_block is not None:
        blocks.append(surface_block)
    if position:
        blocks.append(
            np.concatenate(
                [
                    state["root_xy"],
                    state["root_lin_vel_local"] * 0.1,
                    state["margin_x"][:, None],
                    state["margin_y"][:, None],
                ],
                axis=1,
            )
        )
    blocks.append(progress)
    return np.concatenate(blocks, axis=1).astype(np.float64)


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
def build_actor_critic(
    obs_dim: int,
    action_dim: int,
    *,
    hidden: tuple[int, ...] = (256, 128),
    initial_log_std: float = -1.0,
    seed: int = 0,
):
    """Actor and critic MLPs (tanh, orthogonal init), seeded.

    The actor outputs the mean of the pre-squash action and carries a
    state-independent log standard deviation; its output layer is initialised
    with gain 0.01, so a new policy starts as the original controller plus noise.
    """

    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    def mlp(out_dim: int, final_gain: float):
        layers, last = [], int(obs_dim)
        for size in hidden:
            linear = nn.Linear(last, int(size))
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.Tanh()]
            last = int(size)
        head = nn.Linear(last, out_dim)
        nn.init.orthogonal_(head.weight, gain=final_gain)
        nn.init.zeros_(head.bias)
        layers.append(head)
        return nn.Sequential(*layers)

    class Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = mlp(action_dim, 0.01)
            self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

        def forward(self, obs):
            mean = self.body(obs)
            return mean, self.log_std.expand_as(mean)

    class Critic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = mlp(1, 1.0)

        def forward(self, obs):
            return self.body(obs).squeeze(-1)

    return Actor(), Critic()


@dataclass
class ResidualPolicy:
    """A trained residual actor used greedily: ``delta = bound * tanh(mean(obs))``."""

    actor: object
    bound: float
    obs_dim: int
    position_observation: bool

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            mean, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32))
            return (self.bound * torch.tanh(mean)).numpy().astype(np.float64)


def load_policy(path, action_dim: int) -> ResidualPolicy:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=True)
    actor, _ = build_actor_critic(int(blob["obs_dim"]), action_dim, hidden=tuple(blob["hidden"]))
    actor.load_state_dict(blob["actor"])
    actor.eval()
    return ResidualPolicy(
        actor, float(blob["bound"]), int(blob["obs_dim"]), bool(blob["position_observation"])
    )


# --------------------------------------------------------------------------- #
# reward
# --------------------------------------------------------------------------- #
def reward(
    state: dict[str, np.ndarray],
    standing: np.ndarray,
    stand_run: np.ndarray,
    exited: np.ndarray,
    initial_torso_z: np.ndarray,
    cfg: DictConfig,
    hold: int,
) -> np.ndarray:
    """Per-control-step training reward (``reward`` in ``conf/train/*.yaml``)::

        h + w_u u + w_J [the hold completes now] + w_s [standing] - w_O [left the area]

    * ``h``: torso-height progress, ``(z_torso - z_torso,0) / (z_stand - z_torso,0)``
      clipped to [0, 1], with ``z_torso,0`` the torso height at the episode start and
      ``z_stand`` the torso height of the standing robot (``standing_torso_z_m``);
    * ``u``: the torso's uprightness (world z component of its z axis), clipped to [0, 1];
    * ``[standing]``: the standing condition of the success criterion
      (:func:`elastra.success.standing_flags`);
    * ``[the hold completes now]``: this step is the ``hold``-th consecutive standing
      step (``stand_run`` counts the standing steps before this one); it is paid again
      if the robot stands for another full hold after an interruption;
    * ``[left the area]``: the robot left the task area during this step (the episode
      ends there).
    """

    span = np.maximum(float(cfg.standing_torso_z_m) - initial_torso_z, 1.0e-3)
    progress = np.clip((state["torso_z"] - initial_torso_z) / span, 0.0, 1.0)
    upright = np.clip(state["torso_up"], 0.0, 1.0)
    completes = standing & (stand_run + 1 == int(hold))
    return (
        float(cfg.height_weight) * progress
        + float(cfg.upright_weight) * upright
        + float(cfg.completion_bonus) * completes.astype(np.float64)
        + float(cfg.standing_bonus) * standing.astype(np.float64)
        - float(cfg.exit_penalty) * exited.astype(np.float64)
    )
