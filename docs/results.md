# Results: the residual policies in `checkpoints/`

This page reports how the two residual policies in `checkpoints/` were trained and how they perform, all computed with the code and configuration of this repository.
They are compared with the original controllers and with the earlier residual policies that [validation.md](validation.md) evaluates (released in commit `303f0c0`).

## Training

* HoST residual policy (`checkpoints/host_residual.pt`): `elastra-train train.workers=48`, configuration `conf/train/host.yaml`.
* ProtoMotions residual policy (`checkpoints/protomotions_residual.pt`): `elastra-train train=protomotions_curriculum train.workers=48`, configuration `conf/train/protomotions_curriculum.yaml`.
* Both: 1000 PPO updates of 256 environments, 64 control steps per update, success criterion relative to the support surface (`conf/criterion/standing.yaml`), trained from scratch in one run.
* Surfaces of the environments: mattresses `a8`, `a16`, `a32`, `a64` (32 environments each), rigid ground (51) and trampolines `a1` (39) and `a2` (38); from update 500 on, the environments of mattresses `a16`, `a32`, `a64` are on mattresses `a1`, `a2`, `a4`.
* The HoST residual policy observes the robot's position (root xy, root velocity, distances to the task-area boundary) and the surface from the first update; the ProtoMotions residual policy observes neither.
* Initial states: the 32 HoST training states and the 39 ProtoMotions training states (`data/README.md`), none of which is a test initial state.
* Each run used 48 cores of an Intel Xeon 6728P, the machine of the evaluation runs.
  HoST took 5.8 h (20.8 s per update), ProtoMotions 4.5 h (16.0 s per update); neither run was interrupted or resumed.
* The number of worker processes does not change a run: two runs of the HoST configuration with 32 and with 48 workers wrote identical training histories over their first two updates.

![Training curves](figures/training.png)

The figure shows, per 20 updates, the fraction of the finished training episodes that succeeded, grouped by surface (`elastra-figures`, from `history.jsonl`).
Training episodes use sampled actions; the evaluation below uses the mean action.

Success fraction of the finished training episodes, per surface, over four blocks of 100 updates:

| run, updates | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST, 1-100 | | | | 0.25 | 0.54 | 0.73 | 0.84 | 0.31 | 0.48 | 0.96 |
| HoST, 401-500 | | | | 0.68 | 0.90 | 0.96 | 0.96 | 0.77 | 0.82 | 0.96 |
| HoST, 501-600 | 0.01 | 0.06 | 0.26 | 0.73 | | | | 0.80 | 0.80 | 0.97 |
| HoST, 901-1000 | 0.03 | 0.13 | 0.44 | 0.87 | | | | 0.86 | 0.85 | 0.98 |
| ProtoMotions, 1-100 | | | | 0.06 | 0.16 | 0.30 | 0.39 | 0.11 | 0.23 | 0.48 |
| ProtoMotions, 401-500 | | | | 0.40 | 0.50 | 0.66 | 0.66 | 0.59 | 0.61 | 0.64 |
| ProtoMotions, 501-600 | 0.00 | 0.01 | 0.14 | 0.41 | | | | 0.60 | 0.61 | 0.62 |
| ProtoMotions, 901-1000 | 0.00 | 0.02 | 0.17 | 0.43 | | | | 0.58 | 0.59 | 0.59 |

In the first 500 updates the success fraction rises on every training surface for both controllers.
After mattresses `a1`, `a2`, `a4` enter at update 500, HoST raises it on mattresses `a1`, `a2`, `a4` from 0.01, 0.06, 0.26 to 0.03, 0.13, 0.44 and keeps improving on mattress `a8` and the trampolines.
ProtoMotions stays at 0.00 to 0.02 on mattresses `a1` and `a2` and rises from 0.14 to 0.17 on `a4`; on the other surfaces it stays within 0.05 of its level over updates 401 to 500.
The approximate KL divergence of an update stayed below 0.018 (HoST) and 0.039 (ProtoMotions; its largest values are in the two updates after mattresses `a1`, `a2`, `a4` enter, when every environment restarts), and every update ran all four PPO epochs.

## Evaluation

```bash
elastra-evaluate controller=host policy=null out=outputs/evaluation/host_original
elastra-evaluate controller=host policy=checkpoints/host_residual.pt out=outputs/evaluation/host_residual
elastra-evaluate controller=protomotions policy=null out=outputs/evaluation/protomotions_original
elastra-evaluate controller=protomotions policy=checkpoints/protomotions_residual.pt out=outputs/evaluation/protomotions_residual
elastra-figures                    # docs/figures/host.png, protomotions.png, training.png
```

The setup is that of [validation.md](validation.md): the 96 HoST test initial states and the 40 ProtoMotions test initial states on every surface, the same success criterion and penetration rule, grid spacing 0.1 m, physics step 0.625 ms, and the same machine (Intel Xeon 6728P).
The original controllers give the same counts as in validation.md, trial for trial.
The rows "+ earlier residual" are copied from validation.md; "+ residual" is the policy in `checkpoints/`.
No trial diverged.

### HoST

Success (success without sustained penetration), 96 trials per surface:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0 (0) | 3 (3) | 6 (6) | 24 (24) | 51 (51) | 51 (51) | 91 (91) | 53 (52) | 48 (42) | 96 (96) | 423 (416) |
| HoST + earlier residual | 0 (0) | 11 (11) | 51 (51) | 88 (88) | 95 (95) | 96 (96) | 96 (96) | 94 (87) | 93 (80) | 96 (96) | 720 (700) |
| HoST + residual | 4 (4) | 17 (17) | 46 (46) | 86 (86) | 88 (88) | 82 (82) | 85 (85) | 91 (87) | 83 (73) | 96 (96) | 678 (664) |

Outcome classes, success / stood but not completed / never stood:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0 / 31 / 65 | 3 / 54 / 39 | 6 / 52 / 38 | 24 / 52 / 20 | 51 / 45 / 0 | 51 / 45 / 0 | 91 / 5 / 0 | 53 / 41 / 2 | 48 / 38 / 10 | 96 / 0 / 0 | 423 / 363 / 174 |
| HoST + earlier residual | 0 / 20 / 76 | 11 / 80 / 5 | 51 / 44 / 1 | 88 / 8 / 0 | 95 / 1 / 0 | 96 / 0 / 0 | 96 / 0 / 0 | 94 / 2 / 0 | 93 / 3 / 0 | 96 / 0 / 0 | 720 / 158 / 82 |
| HoST + residual | 4 / 32 / 60 | 17 / 59 / 20 | 46 / 49 / 1 | 86 / 10 / 0 | 88 / 8 / 0 | 82 / 14 / 0 | 85 / 11 / 0 | 91 / 5 / 0 | 83 / 13 / 0 | 96 / 0 / 0 | 678 / 201 / 81 |

![HoST results](figures/host.png)

With the residual policy HoST succeeds in 678 of 960 trials, against 423 for the original controller and 720 with the earlier residual policy.
On mattresses `a1` and `a2` it succeeds in 4 and 17 trials, more than the original controller (0, 3) and than the earlier residual policy (0, 11); on `a4` in 46 (earlier residual policy: 51).
On the softest mattress, `a1`, it never stands in 60 trials, stands without completing the 1 s hold in 32 and succeeds in 4; the original controller has 65, 31 and 0, the earlier residual policy 76, 20 and 0.
On the mattresses that leave the training surfaces at update 500, `a16`, `a32` and `a64`, it succeeds in 88, 82 and 85 trials, fewer than the earlier residual policy (95, 96, 96); on `a64` it succeeds less often than the original controller (85 against 91).
Every one of its failures on these three mattresses stood but did not complete the hold.
A second run of the second training half, from the same update-500 checkpoint with another seed, keeps 96, 94 and 93 on these mattresses ([comparison with the earlier training](#comparison-with-the-earlier-training)).
On the trampolines it succeeds in 91 and 83 trials (87 and 73 without sustained penetration) and on rigid ground in all 96.

### ProtoMotions

Success (success without sustained penetration), 40 trials per surface:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProtoMotions | 0 (0) | 1 (1) | 1 (1) | 8 (8) | 4 (4) | 13 (13) | 13 (13) | 3 (3) | 3 (3) | 16 (16) | 62 (62) |
| ProtoMotions + earlier residual | 0 (0) | 0 (0) | 1 (1) | 14 (14) | 20 (20) | 18 (18) | 22 (22) | 22 (22) | 24 (24) | 22 (22) | 143 (143) |
| ProtoMotions + residual | 2 (2) | 0 (0) | 6 (6) | 8 (8) | 7 (7) | 14 (14) | 13 (13) | 12 (11) | 9 (9) | 11 (11) | 82 (81) |

Outcome classes, success / stood but not completed / never stood:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProtoMotions | 0 / 0 / 40 | 1 / 7 / 32 | 1 / 6 / 33 | 8 / 6 / 26 | 4 / 7 / 29 | 13 / 4 / 23 | 13 / 4 / 23 | 3 / 5 / 32 | 3 / 6 / 31 | 16 / 4 / 20 | 62 / 49 / 289 |
| ProtoMotions + earlier residual | 0 / 2 / 38 | 0 / 15 / 25 | 1 / 16 / 23 | 14 / 12 / 14 | 20 / 10 / 10 | 18 / 14 / 8 | 22 / 11 / 7 | 22 / 7 / 11 | 24 / 4 / 12 | 22 / 11 / 7 | 143 / 102 / 155 |
| ProtoMotions + residual | 2 / 2 / 36 | 0 / 14 / 26 | 6 / 19 / 15 | 8 / 18 / 14 | 7 / 23 / 10 | 14 / 12 / 14 | 13 / 16 / 11 | 12 / 9 / 19 | 9 / 11 / 20 | 11 / 13 / 16 | 82 / 137 / 181 |

![ProtoMotions results](figures/protomotions.png)

With the residual policy the ProtoMotions tracker succeeds in 82 of 400 trials, against 62 for the original controller and 143 with the earlier residual policy.
It succeeds more often than the original controller on mattresses `a1`, `a4`, `a16`, `a32` and on both trampolines, as often on `a8` and `a64`, and less often on mattress `a2` (0 against 1) and on rigid ground (11 against 16).
It reaches the standing condition in 219 trials (82 successes and 137 trials that stood without completing the hold), against 111 for the original controller and 245 for the earlier residual policy.
Runs of 500 updates with different seeds and the earlier evaluation rules are compared in the [comparison with the earlier training](#comparison-with-the-earlier-training).

#### Training and test initial states

The ProtoMotions test initial states are fallen poses that are not among the training initial states.
To see whether the result reflects how the policy fits its training poses or how it generalises to the test poses, the policy was also evaluated from the 39 training initial states (`initial_states=train`, 390 trials on the ten surfaces), together with its checkpoint at update 500 and with the earlier residual policy.
These runs only diagnose the result: the policy in `checkpoints/` is the one at update 1000, as the training configuration defines.

Successes over the ten surfaces:

| policy | training initial states (390 trials) | test initial states (400 trials) |
| --- | ---: | ---: |
| ProtoMotions + residual, update 1000 (`checkpoints/`) | 149 | 82 |
| ProtoMotions + residual, update 500 | 206 | 115 |
| ProtoMotions + earlier residual | 119 | 143 |

From its training initial states the residual policy succeeds more often than the earlier residual policy (149 against 119), from the test initial states less often (82 against 143).
On the seven surfaces of its second training half, the policy succeeds in 95 of 273 trials from the training initial states (0.35), close to the success fraction of its training episodes over updates 901 to 1000 (0.36): training and evaluation agree.
The second half of training, with mattresses `a1`, `a2`, `a4` in place of `a16`, `a32`, `a64`, lowered the successes from both sets of initial states (206 to 149 and 115 to 82).

### Penetration

Trials with a sustained penetration; per-trial peak depth, median / maximum over the trials (mm):

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0; 1.0 / 4.0 | 0; 1.3 / 2.6 | 0; 1.5 / 3.3 | 0; 1.5 / 3.9 | 0; 2.0 / 5.4 | 0; 2.1 / 5.9 | 0; 2.2 / 5.9 | 10; 3.4 / 68.3 | 24; 4.6 / 65.3 | 0; 0.4 / 1.7 |
| HoST + residual | 0; 1.2 / 3.1 | 0; 1.5 / 3.7 | 0; 1.9 / 3.4 | 0; 1.9 / 4.7 | 0; 1.9 / 5.0 | 0; 2.1 / 5.3 | 0; 2.7 / 7.6 | 6; 3.1 / 31.6 | 16; 3.9 / 38.5 | 0; 0.6 / 2.0 |
| ProtoMotions | 0; 0.9 / 2.4 | 0; 1.0 / 2.5 | 0; 1.3 / 3.5 | 0; 1.6 / 4.3 | 0; 1.8 / 3.7 | 0; 2.2 / 5.5 | 0; 2.7 / 7.8 | 0; 2.5 / 37.2 | 0; 2.7 / 16.6 | 0; 0.4 / 1.5 |
| ProtoMotions + residual | 0; 1.0 / 2.2 | 0; 1.7 / 2.4 | 0; 2.0 / 4.8 | 0; 1.8 / 9.0 | 0; 2.0 / 4.7 | 0; 2.1 / 4.5 | 0; 3.1 / 10.5 | 1; 3.3 / 85.6 | 1; 4.0 / 93.7 | 0; 0.4 / 1.5 |

Largest contact depth / largest membrane crossing depth over all trials (mm):

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST + residual | 3.1 / 0.0 | 3.7 / 0.0 | 3.4 / 0.0 | 4.7 / 0.0 | 5.0 / 0.0 | 5.3 / 0.0 | 7.6 / 0.0 | 8.8 / 31.6 | 10.7 / 38.5 | 2.0 / 0.0 |
| ProtoMotions + residual | 2.2 / 0.0 | 2.4 / 0.0 | 4.8 / 0.0 | 9.0 / 0.0 | 4.7 / 0.0 | 4.5 / 0.0 | 10.5 / 0.0 | 28.7 / 85.6 | 22.4 / 93.7 | 1.5 / 0.0 |

The rows of the original controllers are those of validation.md.
On the mattresses and on rigid ground no trial of either residual policy has a sustained penetration; the per-trial peak depth has a median of 0.4 to 3.1 mm and a maximum of 10.5 mm.
On the trampolines HoST with the residual policy has a sustained penetration in 6 and 16 trials (trampolines `a1` and `a2`), against 10 and 24 for the original controller; the contact depth stays below 11 mm, and in every one of these trials the membrane crossing depth alone exceeds 5 mm for at least 10 consecutive control steps (validation.md, section 1, traces such crossings to HoST's foot spheres).
The ProtoMotions residual policy has one trial with a sustained penetration on each trampoline: test initial state `fall_23` on trampoline `a1` (a success, crossing depth up to 85.6 mm) and `fall_18` on trampoline `a2` (stood but not completed, 93.7 mm); in both the robot leaves the task area afterwards, 0.03 s and 0.9 s after the longest run of control steps deeper than 5 mm ends.

## Comparison with the earlier training

The earlier residual policies of [validation.md](validation.md) (commit `303f0c0`) were trained with an earlier implementation of the training code.
With the evaluation of this repository the released HoST policy succeeds in 678 of 960 trials and the earlier one in 720, the released ProtoMotions policy in 82 of 400 and the earlier one in 143.
This section lists how the earlier training differs from the training of this repository and measures what the differences do.
No error in the training code was found; the gaps lie within the spread between training runs that differ only in their random numbers.

### What differs

The two trainings were compared item by item: physics and control clocks, solver and contact settings, robot models and actuation, the original controllers and their inputs, the residual observation and action, reward, success criterion, initial states and their placement, episodes and resets, surface counts, PPO settings and initialisation.
Stepped side by side from the same state with the same residual actions, the earlier and the new training environments give bit-identical robot states, residual observations, rewards and termination flags once the first three items of the table are set as in the earlier training (60 control steps for HoST and 50 for ProtoMotions, each on a mattress, rigid ground and trampoline `a1`).
With the same three items, the HoST start states after the 30 unactuated steps (32 per surface on mattresses `a16`, `a64` and rigid ground) and the settled ProtoMotions training states (39 per surface on mattress `a8` and rigid ground) are bit-identical to the ones the earlier code computed.

| item | earlier training | this repository |
| --- | --- | --- |
| mattress | the cell spring `k` plus two terms on every cell: a densification force for cell deflections beyond 0.138 m, taken from the load table of an earlier foam model (on bed `a1` 10 N at 0.16 m, 72 N at 0.18 m and 248 N at 0.20 m per cell, scaled with the bed multiplier), and an extra downward load of 0.143 N | the linear cell spring only (`conf/mattress/default.yaml`, [mattress_calibration.md](mattress_calibration.md)) |
| rigid ground | the mattress cells on slide joints, held at zero deflection by equality constraints (they move by at most 13 µm under the robot) | the cell boxes fixed to the world |
| task area, ProtoMotions | the robot has left it when the centre of a robot collision geom crosses the boundary, checked once per control step in training and in evaluation | when the bounding box of a robot collision geom crosses it, checked once per control step in training and after every physics step in evaluation (`conf/criterion/standing.yaml`); HoST used bounding boxes in both |
| random numbers | one generator for the initial-state draws of all environments; another assignment of environments to surfaces (same counts per surface) | one generator per environment (`src/elastra/training.py`) |
| residual squashing | `bound * tanh` in single precision | in double precision (differences below 1e-7 rad) |

Everything else is the same.
The position observation of the HoST residual policy is used from the first update here; the earlier HoST run added it at update 500, with the weights of the new inputs starting at zero.

To measure the first three items, the training and the evaluation of this repository were run with them set as in the earlier training; "earlier items" below means these three items and the single-precision squashing.
All runs below used the machine of the evaluation runs (Intel Xeon 6728P).

### HoST

At update 500, before mattresses `a1`, `a2`, `a4` enter, the HoST run of this repository and the earlier one both succeed in 94, 96 and 96 of 96 trials on mattresses `a16`, `a32`, `a64`.
The released policy loses part of this in the second half of training (88, 82, 85 at update 1000).
The second half (updates 501 to 1000) was therefore run again from the same update-500 checkpoint, with and without the earlier mattress terms and with two seeds.
A rerun with the released configuration and seed reproduces the released run bit for bit (checked over its first two updates), so the runs with seed 283402 differ from it only in the mattress physics of training; seed 283403 gives a second draw of the random numbers of the second half.

Successes out of 96 per surface, HoST with the residual policy, evaluation of this repository:

| updates 501 to 1000 trained with | seed | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear mattress (`checkpoints/`) | 283402 | 4 | 17 | 46 | 86 | 88 | 82 | 85 | 91 | 83 | 96 | 678 |
| linear mattress | 283403 | 2 | 17 | 67 | 95 | 96 | 94 | 93 | 85 | 90 | 96 | 735 |
| densification term | 283402 | 3 | 20 | 62 | 94 | 96 | 96 | 94 | 89 | 88 | 96 | 738 |
| densification term | 283403 | 2 | 12 | 64 | 91 | 94 | 94 | 94 | 94 | 86 | 96 | 727 |
| densification, extra load, rigid ground as earlier | 283402 | 4 | 23 | 71 | 93 | 96 | 96 | 95 | 93 | 92 | 96 | 759 |
| earlier training | | 0 | 11 | 51 | 88 | 95 | 96 | 96 | 94 | 93 | 96 | 720 |

With seed 283403 the second half on the linear mattress keeps the stiff mattresses (96, 94, 93), and the densification term changes the total by -8 trials instead of +60.
The loss on the stiff mattresses is therefore a property of the released run, not of the linear mattress.
In that run all 33 failures on `a16` to `a64` stand up (32 start prone), and then either the pelvis stays below 0.7 m above the surface with the torso upright (24 trials) or the robot falls (9) before the 1 s hold is complete.

The densification force acts only where a cell is deflected beyond 0.138 m.
With the released policy (8 training start states) this happens in 77 %, 37 % and 5 % of the control steps on mattresses `a1`, `a2`, `a4` and never on `a8`, `a16` and `a64`.
On mattress `a1` the deepest cell under the robot stays at or below 0.19 m in 95 % of the control steps with the term and 0.23 m without it, where it reaches 0.26 m, past the 0.247 m cell travel.

With the earlier mattress terms and rigid ground also in the evaluation, the totals are 672 and 746 (linear mattress, seeds 283402 and 283403), 752 and 746 (densification term), 766 (all three) and 717 (earlier policy).

### ProtoMotions

The earlier ProtoMotions training had 500 updates and no curriculum, so the runs here have 500 updates without the curriculum (`train=protomotions`; the first 500 updates of the released curriculum run are the run with seed 287121).
Successes out of 400 trials on the test initial states, with the evaluation of this repository, with the same evaluation and the earlier exit rule (geom centres, once per control step), and with the earlier evaluation (earlier exit rule, mattress terms and rigid ground):

| training | seed | this repository | earlier exit rule | earlier evaluation |
| --- | --- | ---: | ---: | ---: |
| this repository (`train=protomotions`) | 287121 | 115 | 154 | 154 |
| this repository | 287122 | 128 | 149 | 162 |
| this repository | 287123 | 104 | 143 | 151 |
| this repository | 287124 | 95 | 128 | 132 |
| earlier items | 287121 | 91 | 158 | 163 |
| earlier items | 287122 | 89 | 124 | 123 |
| earlier items | 287123 | 70 | 103 | 97 |
| earlier exit rule only | 287121 | 73 | 108 | 105 |
| earlier training (earlier policy) | | 143 | 157 | 158 |
| original controller | | 62 | 76 | 70 |

The earlier code's own evaluation of the earlier policy gave 160.
Runs with the earlier items succeed in 70 to 91 trials with the evaluation of this repository and in 97 to 163 with the earlier one; runs of this repository in 95 to 128 and 132 to 162.
The earlier policy succeeds more often than every run with the evaluation of this repository and is among the best runs with the earlier evaluation; on these seeds the training of this repository does at least as well as the earlier training.
In an evaluation the exit rule changes only the scoring: a trajectory is the same under both rules, and every trial that succeeds under the earlier rule but not under the rule of this repository is one in which the bounding box of a robot collision geom, but not its centre, left the task area before the hold was complete (14 trials for the earlier policy and for the original controller, 21 to 67 for the runs above).

### Choices left open

The three items are deliberate choices of this repository; the measurements above do not single out any of them as the cause of a gap.

* The mattress densification term (with or without the extra load).
  The mattress of this repository is linear, as calibrated in [mattress_calibration.md](mattress_calibration.md).
  With the term, the second HoST training half gives 738 and 727 of 960 (seeds 283402, 283403) against 678 and 735 without it.
  For ProtoMotions with the curriculum, the second half with the term gives 71 of 400 against 82 without it (108 against 112 under the earlier exit rule).
* The exit rule for ProtoMotions (bounding boxes or centres), in training and in evaluation: the table above.
* Rigid ground as earlier and the extra load: with them and the densification term, the second HoST half gives 759 of 960 (seed 283402) against 738 with the densification term alone.
