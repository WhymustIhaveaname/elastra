# Elastra

MuJoCo models of a mattress and a trampoline, and residual policies that let two humanoid get-up controllers (HoST and the ProtoMotions motion tracker, both on the Unitree G1) stand up on them.
The repository contains the surface models, the success criterion, the evaluation and training code, the initial states, and the two trained residual policies.

## Install

Python 3.13, CPU only.
The policies and the initial states are stored with [Git LFS](https://git-lfs.com) (`git lfs install` before cloning).
With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                           # creates .venv with the package and its commands
uv run pytest                     # unit tests (the scene tests need the assets)
```

The commands below are installed into the environment (`[project.scripts]` in `pyproject.toml`); run them with `uv run <command>` or after activating `.venv`.

## Third-party assets

The robot models, the original controllers and the reference motions are not stored here (sources and licenses: [THIRD_PARTY.md](THIRD_PARTY.md)).

```bash
export HF_TOKEN=...               # a Hugging Face account that accepted the BONES-SEED license
uv run elastra-download-assets
```

Every file is checked against the sha256 in `conf/assets/default.yaml`.
`--skip-bones` skips the BONES-SEED clips; HoST then runs, ProtoMotions does not.

## Surfaces

A surface is named `mattress/<bed>`, `trampoline/<name>` or `rigid`.

* **Mattress** (`src/elastra/mattress.py`, `conf/mattress/default.yaml`): 20 x 19 rigid box cells, 0.1 m apart (2.0 m x 1.9 m).
  Each cell slides vertically on a joint with stiffness `k`, damping `c` and mass `m`; every interior cell has a five-point Laplacian (curvature) tendon with stiffness `b` and damping `d`.
  A bed is the five numbers `k, b, c, d, m`; the beds `a1` ... `a64` are the seven beds of the paper.
  Every number can be overridden, e.g. `mattress.beds.a4.c_n_s_per_m=8.0`.
* **Trampoline** (`src/elastra/trampoline.py`, `conf/trampoline/default.yaml`): a circular membrane of radius 1.2 m on a 25 x 25 grid of nodes 0.1 m apart (377 free nodes, pinned rim) with edge force `k_lin d + k_cub d^3`; the robot touches a flex surface on the nodes.
  Trampolines `a1` and `a2`.
* **Rigid ground**: the mattress cell boxes fixed to the world.

The grid spacing of both surfaces is 0.1 m (cell pitch, node spacing).
`mattress=refined` and `trampoline=refined` halve it (0.05 m) with the parameters rescaled so that the continuum they discretise stays the same; the refined grids need a physics step of at most 0.15625 ms (a quarter of the default 0.625 ms) on the stiffest beds.

## Success and penetration

A trial succeeds when the robot stands for 1 s without interruption before it leaves the task area (the 2.0 m x 1.9 m mattress footprint, for every surface).
Standing: the pelvis is at least 0.7 m above the support surface under the robot's lowest collision point, and the pelvis z axis has a world-z component of at least 0.9 (`src/elastra/success.py`, `conf/criterion/standing.yaml`).

Penetration is the depth of the robot's collision geometry in the deformed surface, once per control step: the deepest MuJoCo contact, and on the trampoline also how far any robot point lies below the membrane.
A trial with more than 5 mm for 10 consecutive control steps while the robot is in the task area is invalid (`src/elastra/penetration.py`, `conf/penetration/default.yaml`).

## Evaluate

```bash
uv run elastra-evaluate controller=host policy=checkpoints/host_residual.pt
uv run elastra-evaluate controller=host policy=null          # HoST alone
uv run elastra-evaluate controller=protomotions policy=checkpoints/protomotions_residual.pt
uv run elastra-evaluate controller=protomotions policy=null
```

HoST runs its 96 test initial states (48 prone, 48 supine) on every surface, ProtoMotions its 40 test initial states (`data/README.md`).
Each run writes `summary.md` (per surface: successes, successes without sustained penetration, never stood / stood but not completed, penetration depths), `summary.json`, `cells.json` (one row per trial) and `traces.npz` to `out` (`outputs/evaluation/<controller>` by default).
`workers` sets the number of worker processes.
Any configuration value can be overridden on the command line, for example the physics step `sim.physics_dt_s=0.0003125` or the grid `mattress=refined trampoline=refined sim.physics_dt_s=0.00015625`.

## Train

```bash
uv run elastra-train                                   # HoST, conf/train/host.yaml
uv run elastra-train train=protomotions                # ProtoMotions
uv run elastra-train train=protomotions_curriculum     # ProtoMotions, with soft beds after update 500
```

PPO on 256 environments; the training configurations (surfaces of the parallel environments, curriculum, PPO and reward settings) are in `conf/train/`.
One update steps 256 environments for 64 control steps (524288 physics steps); `train.threads` sets the number of threads that step them.
The run writes `history.jsonl`, checkpoints and the final `policy.pt` to `train.out`; evaluate it with `policy=<path>/policy.pt`.

## Load response

```bash
uv run elastra-load-response                           # physics steps 0.625, 0.3125, 0.15625 ms
uv run elastra-load-response mattress=refined trampoline=refined out=outputs/load_response/refined
```

A 10 kg rigid ball of radius 0.125 m is loaded onto each surface quasi-statically (static deflection) and dropped from rest (deflection over time).

## Mattress calibration

```bash
uv run elastra-indentation                             # EN 1957 loading pad, bed a1
```

The loading pad of EN 1957 is pressed into the mattress, and the stiffness scale s, a factor on `k` and `b` of bed `a1`, is fitted to the loads that Vlaović et al. (2024) measured on a polyurethane foam mattress at 10 to 50 mm deflection (`data/load_deflection/`).
The fitted bed (s = 4.10, between beds `a4` and `a8`) over-predicts the held-out loads at 60 to 100 mm by 5.9 % to 10.8 %, and by up to 22.9 % with the pad on a cell centre or on the 0.05 m grid ([docs/mattress_calibration.md](docs/mattress_calibration.md)).
The mattress has no nonlinear compression term.

## Validation

[docs/validation.md](docs/validation.md): success counts of the released policies, penetration depth on every surface, and the dependence of the load response and of the success counts on the physics step and the grid spacing.
[docs/mattress_calibration.md](docs/mattress_calibration.md): the mattress against a measured load-deflection curve.

## Repository layout

```
conf/            Hydra configuration: every physical and experimental parameter
src/elastra/     the package
  mattress.py, trampoline.py, scene.py    surfaces and robot-on-surface scenes
  robots.py, host.py, protomotions.py     robot models and the two original controllers
  sim.py, rollout.py                      batched simulation and evaluation episodes
  success.py, penetration.py              success criterion and penetration depth
  residual.py, training.py                residual policy and PPO training
  evaluation.py, load_response.py         the evaluation and load-response commands
  indentation.py                          the EN 1957 loading pad and the mattress calibration
  assets.py, initial_states.py            asset download, HoST initial states
data/            initial states, a measured mattress load-deflection curve (data/README.md)
checkpoints/     the trained residual policies (Git LFS)
docs/            validation and calibration results
tests/           unit tests
```

## License

The code in this repository is under the MIT License ([LICENSE](LICENSE)).
Third-party assets keep their own licenses ([THIRD_PARTY.md](THIRD_PARTY.md)).
