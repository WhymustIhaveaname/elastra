# Results: the residual policies in `checkpoints/`

This page reports how the two residual policies in `checkpoints/` were trained and how they perform, all computed with the code and configuration of this repository.

## Training

* HoST residual policy (`checkpoints/host_residual.pt`): `elastra-train train.workers=48`, configuration `conf/train/host.yaml`.
* ProtoMotions residual policy (`checkpoints/protomotions_residual.pt`): `elastra-train train=protomotions train.workers=32`, configuration `conf/train/protomotions.yaml`.
* Both: 256 environments, 64 control steps per update, success criterion relative to the support surface (`conf/criterion/standing.yaml`), trained from scratch in one run without interruption.
* Surfaces of the environments: mattresses `a8`, `a16`, `a32`, `a64` (32 environments each), rigid ground (51) and trampolines `a1` (39) and `a2` (38).
* HoST trains for 1000 updates in two stages.
  In the first 500 updates the residual policy observes the surface (type, stiffness, largest deflection) but not the robot's position; from update 500 on it also observes the position (root xy, root velocity, distances to the task-area boundary), whose input weights start at zero, and the environments of mattresses `a16`, `a32`, `a64` move to mattresses `a1`, `a2`, `a4`.
* ProtoMotions trains for 500 updates on the surfaces above; its residual policy observes neither the robot's position nor the surface.
* The HoST policy is the last update. The ProtoMotions policy is the saved checkpoint (one every 10 updates) with the highest success rate of the training episodes that ended in the 50 updates up to it (`policy_checkpoint` in `conf/train/protomotions.yaml`), which is update 260.
* Initial states: the 32 HoST training states and the 39 ProtoMotions training states (`data/README.md`), none of which is a test initial state.
* The HoST run used 48 cores of an Intel Xeon Gold 6258R (20.5 s per update, 5.7 h); the ProtoMotions run used 32 cores of an Intel Xeon 6728P that other jobs shared (24.4 s per update, 3.4 h).

![Training curves](figures/training.png)

The figure shows, per 20 updates, the fraction of the finished training episodes that succeeded, grouped by surface (`elastra-figures`, from `history.jsonl`).
Training episodes use sampled actions; the evaluation below uses the mean action.

Success fraction of the finished training episodes, per surface, over blocks of 100 updates:

| run, updates | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST, 1-100 | | | | 0.23 | 0.49 | 0.63 | 0.77 | 0.33 | 0.44 | 0.97 |
| HoST, 401-500 | | | | 0.59 | 0.86 | 0.92 | 0.96 | 0.65 | 0.75 | 0.96 |
| HoST, 501-600 | 0.00 | 0.09 | 0.25 | 0.66 | | | | 0.69 | 0.77 | 0.97 |
| HoST, 901-1000 | 0.01 | 0.13 | 0.32 | 0.83 | | | | 0.83 | 0.86 | 0.99 |
| ProtoMotions, 1-100 | | | | 0.07 | 0.21 | 0.36 | 0.40 | 0.17 | 0.21 | 0.47 |
| ProtoMotions, 201-300 | | | | 0.42 | 0.55 | 0.67 | 0.74 | 0.60 | 0.67 | 0.76 |
| ProtoMotions, 401-500 | | | | 0.34 | 0.50 | 0.58 | 0.55 | 0.50 | 0.52 | 0.60 |

In the first 500 updates of HoST the success fraction rises on every training surface.
After mattresses `a1`, `a2`, `a4` enter at update 500, HoST raises it on them from 0.00, 0.09, 0.25 to 0.01, 0.13, 0.32 and keeps improving on mattress `a8`, the trampolines and rigid ground.
ProtoMotions reaches its highest success fractions over updates 201 to 300 and ends 0.05 to 0.19 below them.
Its success rate over all surfaces is 0.65 in updates 211 to 260, the highest of any 50 updates that end at a saved checkpoint, and 0.55 in the last 50 updates; the policy of update 260 is the one in `checkpoints/`.
The last update (500) succeeds in 127 of the 400 test trials of the evaluation below, update 260 in 180.
The approximate KL divergence of an update stayed below 0.024 (HoST) and 0.019 (ProtoMotions), and every update ran all four PPO epochs.

## Evaluation

```bash
elastra-evaluate controller=host policy=null out=outputs/evaluation/host_original
elastra-evaluate controller=host policy=checkpoints/host_residual.pt out=outputs/evaluation/host_residual
elastra-evaluate controller=protomotions policy=null out=outputs/evaluation/protomotions_original
elastra-evaluate controller=protomotions policy=checkpoints/protomotions_residual.pt out=outputs/evaluation/protomotions_residual
elastra-figures                    # docs/figures/host.png, protomotions.png, training.png
```

The setup is that of [validation.md](validation.md): the 96 HoST test initial states and the 40 ProtoMotions test initial states on every surface, the success criterion and penetration rule of `conf/criterion/standing.yaml` and `conf/penetration/default.yaml`, grid spacing 0.1 m and physics step 0.625 ms.
The ProtoMotions evaluation steps the environments of one reference clip over a block of surfaces as one batch (`protomotions_blocks` in `conf/evaluate.yaml`).
No trial diverged.
The evaluations of HoST, HoST + residual and ProtoMotions ran on an Intel Xeon Gold 6258R, that of ProtoMotions + residual on an Intel Xeon 6728P.
A run is reproduced bit for bit on the same machine; on another CPU model a few outcomes can change (on an AMD Ryzen 9 9950X3D HoST + residual succeeds in 95 instead of 96 trials on rigid ground).

### HoST

Success (success without sustained penetration), 96 trials per surface:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0 (0) | 3 (3) | 6 (6) | 24 (24) | 51 (51) | 51 (51) | 91 (91) | 53 (52) | 48 (42) | 96 (96) | 423 (416) |
| HoST + residual | 1 (1) | 10 (10) | 43 (43) | 90 (90) | 96 (96) | 96 (96) | 95 (95) | 85 (75) | 84 (69) | 96 (96) | 696 (671) |

Outcome classes, success / stood but not completed / never stood:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0 / 31 / 65 | 3 / 54 / 39 | 6 / 52 / 38 | 24 / 52 / 20 | 51 / 45 / 0 | 51 / 45 / 0 | 91 / 5 / 0 | 53 / 41 / 2 | 48 / 38 / 10 | 96 / 0 / 0 | 423 / 363 / 174 |
| HoST + residual | 1 / 39 / 56 | 10 / 76 / 10 | 43 / 52 / 1 | 90 / 6 / 0 | 96 / 0 / 0 | 96 / 0 / 0 | 95 / 1 / 0 | 85 / 11 / 0 | 84 / 12 / 0 | 96 / 0 / 0 | 696 / 197 / 67 |

![HoST results](figures/host.png)

With the residual policy HoST succeeds in 696 of 960 trials, against 423 for the original controller.
It succeeds more often on every mattress, by 37 to 66 trials on mattresses `a4` to `a32`, and in 95 or 96 of 96 trials on mattresses `a16` to `a64`.
On the softest mattress, `a1`, it never stands in 56 trials, stands without completing the 1 s hold in 39 and succeeds in 1; the original controller never stands in 65 and stands without completing the hold in 31.
On the trampolines it succeeds in 85 and 84 trials (75 and 69 without sustained penetration), against 53 and 48 for the original controller, and on rigid ground in all 96.

### ProtoMotions

Success (success without sustained penetration), 40 trials per surface:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProtoMotions | 0 (0) | 1 (1) | 1 (1) | 9 (9) | 7 (7) | 15 (15) | 16 (16) | 4 (4) | 6 (6) | 17 (17) | 76 (76) |
| ProtoMotions + residual | 1 (1) | 1 (1) | 7 (7) | 22 (22) | 24 (24) | 25 (25) | 26 (26) | 21 (21) | 24 (24) | 29 (29) | 180 (180) |

Outcome classes, success / stood but not completed / never stood:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProtoMotions | 0 / 0 / 40 | 1 / 7 / 32 | 1 / 6 / 33 | 9 / 5 / 26 | 7 / 4 / 29 | 15 / 2 / 23 | 16 / 1 / 23 | 4 / 4 / 32 | 6 / 3 / 31 | 17 / 3 / 20 | 76 / 35 / 289 |
| ProtoMotions + residual | 1 / 5 / 34 | 1 / 15 / 24 | 7 / 14 / 19 | 22 / 7 / 11 | 24 / 5 / 11 | 25 / 4 / 11 | 26 / 7 / 7 | 21 / 5 / 14 | 24 / 2 / 14 | 29 / 6 / 5 | 180 / 70 / 150 |

![ProtoMotions results](figures/protomotions.png)

With the residual policy the ProtoMotions tracker succeeds in 180 of 400 trials, none of them with a sustained penetration, against 76 for the original controller.
It succeeds at least as often as the original controller on every surface, most of all on the trampolines (21 and 24 against 4 and 6) and on mattress `a16` (24 against 7).
It reaches the standing condition in 250 trials (180 successes and 70 trials that stood without completing the hold), against 111 for the original controller.
On mattress `a1` the original controller never stands; with the residual policy the robot succeeds in 1 trial and stands without completing the hold in 5.

### Penetration

Trials with a sustained penetration; per-trial peak depth, median / maximum over the trials (mm):

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0; 1.0 / 4.0 | 0; 1.3 / 2.6 | 0; 1.5 / 3.3 | 0; 1.5 / 3.9 | 0; 2.0 / 5.4 | 0; 2.1 / 5.9 | 0; 2.2 / 5.9 | 10; 3.4 / 68.3 | 24; 4.6 / 65.3 | 0; 0.4 / 1.7 |
| HoST + residual | 0; 1.2 / 3.6 | 0; 1.6 / 6.3 | 0; 1.8 / 4.4 | 0; 1.9 / 4.7 | 0; 1.6 / 4.0 | 0; 1.9 / 4.0 | 0; 2.7 / 5.5 | 14; 3.5 / 37.9 | 18; 4.2 / 40.4 | 0; 0.6 / 1.5 |
| ProtoMotions | 0; 0.9 / 2.4 | 0; 1.1 / 2.5 | 0; 1.4 / 3.5 | 0; 1.8 / 4.9 | 0; 1.8 / 5.1 | 0; 2.2 / 5.5 | 0; 2.7 / 7.8 | 0; 2.5 / 94.1 | 0; 2.8 / 83.8 | 0; 0.4 / 1.5 |
| ProtoMotions + residual | 0; 1.0 / 2.4 | 0; 1.7 / 3.4 | 0; 2.0 / 3.8 | 0; 2.3 / 4.1 | 0; 2.3 / 4.6 | 0; 2.9 / 6.5 | 0; 3.7 / 6.1 | 1; 3.3 / 62.5 | 1; 3.7 / 66.9 | 0; 0.5 / 2.0 |

Largest contact depth / largest membrane crossing depth over all trials (mm):

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST + residual | 3.6 / 0.0 | 6.3 / 0.0 | 4.4 / 0.0 | 4.7 / 0.0 | 4.0 / 0.0 | 4.0 / 0.0 | 5.5 / 0.0 | 8.8 / 37.9 | 10.7 / 40.4 | 1.5 / 0.0 |
| ProtoMotions + residual | 2.4 / 0.0 | 3.4 / 0.0 | 3.8 / 0.0 | 4.1 / 0.0 | 4.6 / 0.0 | 6.5 / 0.0 | 6.1 / 0.0 | 51.4 / 62.5 | 43.7 / 66.9 | 2.0 / 0.0 |

On the mattresses and on rigid ground no trial of either controller has a sustained penetration; the per-trial peak depth has a median of 0.4 to 3.7 mm and a maximum of 7.8 mm.
On the trampolines HoST with the residual policy has a sustained penetration in 14 and 18 trials (trampolines `a1` and `a2`), against 10 and 24 for the original controller; its contact depth stays below 11 mm, and in every one of these trials the membrane crossing depth alone exceeds 5 mm for at least 10 consecutive control steps ([validation.md](validation.md), section 1, traces such crossings to HoST's foot spheres).
The ProtoMotions residual policy has a sustained penetration in 1 trial on each trampoline, the original controller in none: test initial state `fall_36`, which never stands on either trampoline; in both trials the contact depth and the crossing depth each exceed 5 mm for at least 10 consecutive control steps.

### ProtoMotions with other training seeds

The test result of the ProtoMotions residual policy varies with the training seed.
The policy in `checkpoints/` is the run with the seed of `conf/train/protomotions.yaml`, 287121.
Two more runs of the same command, with `train.seed=287122` and `train.seed=287123`, were trained (48 workers) and evaluated as above on an Intel Xeon Gold 6258R; their policies are not in the repository.

Success (success without sustained penetration) of the `policy.pt` of each run, 40 trials per surface:

| training seed, update of `policy.pt` | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 287121, update 260 (`checkpoints/`) | 1 (1) | 1 (1) | 7 (7) | 22 (22) | 24 (24) | 25 (25) | 26 (26) | 21 (21) | 24 (24) | 29 (29) | 180 (180) |
| 287122, update 430 | 0 (0) | 0 (0) | 2 (2) | 9 (9) | 14 (14) | 20 (20) | 19 (19) | 22 (20) | 28 (27) | 24 (24) | 138 (135) |
| 287123, update 320 | 0 (0) | 2 (2) | 3 (3) | 14 (14) | 22 (22) | 21 (21) | 23 (23) | 19 (19) | 22 (22) | 28 (28) | 154 (154) |

The three policies succeed in 180, 138 and 154 of the 400 test trials, against 76 for the original controller; the last updates of the three runs succeed in 127, 114 and 117.
On the training episodes the three policies are close: their success rates over the 50 updates up to the update of `policy.pt` are 0.65, 0.65 and 0.64 (0.55, 0.63 and 0.55 over the last 50 updates).
The training success rate thus does not show the differences between the runs on the test initial states.
