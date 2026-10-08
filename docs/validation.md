# Validation: penetration depth, physics step and grid spacing

This page reports three checks of the surface models and of the evaluation, all computed with the code and configuration of this repository:

1. the success counts of the earlier residual policies (below) and of the original controllers on every surface, with the penetration depth measured in every trial;
2. the load response of the surfaces and the success counts when the physics step is halved and quartered;
3. the load response and the success counts when the grid spacing of the mattress and of the trampoline is halved.

No policy was retrained for any of these runs.

The residual policies on this page are the earlier ones, released in commit `303f0c0` as `checkpoints/host_residual.pt` and `checkpoints/protomotions_residual.pt`.
They were trained with an earlier implementation of the training code, with the surface counts of the training configurations of this repository:
the HoST residual for 500 PPO updates without the position observation, then for 500 more updates with the position observation added (the weights of the new inputs starting at zero) and mattresses `a16`, `a32`, `a64` replaced by `a1`, `a2`, `a4`;
the ProtoMotions residual for 500 updates without that replacement.
That training differs from the one of this repository in the mattress (a densification term and an extra load per cell), in the rigid ground (cells held by equality constraints) and, for ProtoMotions, in the task-area rule (geom centres); [results.md](results.md#comparison-with-the-earlier-training) lists the differences and measures their effect.
In the tables, "+ residual" means these earlier policies.
The files in `checkpoints/` are now the policies trained with the code of this repository; their results are in [results.md](results.md).
`git checkout 303f0c0 -- checkpoints/` restores the earlier policies (`git checkout HEAD -- checkpoints/` returns to the current ones).

The mattress is compared with a measured load-deflection curve in [mattress_calibration.md](mattress_calibration.md).

## Setup

* Controllers: the original HoST controller alone and with the earlier HoST residual policy; the original ProtoMotions tracker alone and with the earlier ProtoMotions residual policy.
* Initial states (`data/README.md`): HoST runs its 96 test states (48 prone, 48 supine) on every surface, ProtoMotions its 40 test initial states.
* Surfaces: mattresses `a1` ... `a64`, trampolines `a1`, `a2`, rigid ground.
* Success: the pelvis is at least 0.7 m above the support surface under the robot's lowest collision point and the pelvis z axis has a world-z component of at least 0.9, without interruption for 1 s, before the robot leaves the task area (`conf/criterion/standing.yaml`).
* Penetration (`src/elastra/penetration.py`): once per control step, the larger of the *contact depth* (the deepest MuJoCo contact between a robot collision geom and the surface) and, on the trampoline, the *membrane crossing depth* (how far the deepest robot point lies below the membrane's top surface).
  A trial has a *sustained penetration* when the depth exceeds 5 mm for at least 10 consecutive control steps (0.2 s) while the robot is in the task area; such a trial is invalid, and "success without sustained penetration" counts the valid successes.
* Default discretisation: grid spacing 0.1 m (cell pitch of the mattress, node spacing of the trampoline), physics step 0.625 ms (1600 Hz), control step 20 ms.
* The evaluation runs used one machine (Intel Xeon 6728P); the load-response runs and the diagnostic reruns used an AMD Ryzen 9 9950X3D.
  A run is reproduced bit for bit on the same machine; on another CPU model the last bits of the trajectories differ and a few outcomes can change.
  Of five evaluation runs repeated on the AMD machine, four gave the same counts (HoST on mattress `a2` and on rigid ground, HoST + residual on trampoline `a1`, ProtoMotions + residual on rigid ground, ProtoMotions on trampoline `a2` at 0.3125 ms) and one gave 12 instead of 14 successes (ProtoMotions + residual on mattress `a8`).

Commands (any `out=` directory; with the earlier policies restored in `checkpoints/`, see above):

```bash
# success and penetration, default discretisation
elastra-evaluate controller=host policy=null
elastra-evaluate controller=host policy=checkpoints/host_residual.pt
elastra-evaluate controller=protomotions policy=null
elastra-evaluate controller=protomotions policy=checkpoints/protomotions_residual.pt
# the same with a smaller physics step, or on the refined grids
elastra-evaluate ... sim.physics_dt_s=0.0003125
elastra-evaluate ... sim.physics_dt_s=0.00015625
elastra-evaluate ... sim.physics_dt_s=0.00015625 mattress=refined trampoline=refined \
    'surfaces=[mattress/a1,mattress/a2,mattress/a4,mattress/a8,mattress/a16,mattress/a32,mattress/a64,trampoline/a1,trampoline/a2]'
# load response
elastra-load-response
elastra-load-response mattress=refined trampoline=refined out=outputs/load_response/refined
elastra-load-response 'load.position_xy_m=[0.05,0.0]' ...
```

## 1. Success and penetration at the default discretisation

Success (success without sustained penetration), per surface:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0 (0) | 3 (3) | 6 (6) | 24 (24) | 51 (51) | 51 (51) | 91 (91) | 53 (52) | 48 (42) | 96 (96) |
| HoST + residual | 0 (0) | 11 (11) | 51 (51) | 88 (88) | 95 (95) | 96 (96) | 96 (96) | 94 (87) | 93 (80) | 96 (96) |
| ProtoMotions | 0 (0) | 1 (1) | 1 (1) | 8 (8) | 4 (4) | 13 (13) | 13 (13) | 3 (3) | 3 (3) | 16 (16) |
| ProtoMotions + residual | 0 (0) | 0 (0) | 1 (1) | 14 (14) | 20 (20) | 18 (18) | 22 (22) | 22 (22) | 24 (24) | 22 (22) |

Outcome classes, success / stood but not completed / never stood.
"Stood" means the standing condition held at some control step:

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0 / 31 / 65 | 3 / 54 / 39 | 6 / 52 / 38 | 24 / 52 / 20 | 51 / 45 / 0 | 51 / 45 / 0 | 91 / 5 / 0 | 53 / 41 / 2 | 48 / 38 / 10 | 96 / 0 / 0 |
| HoST + residual | 0 / 20 / 76 | 11 / 80 / 5 | 51 / 44 / 1 | 88 / 8 / 0 | 95 / 1 / 0 | 96 / 0 / 0 | 96 / 0 / 0 | 94 / 2 / 0 | 93 / 3 / 0 | 96 / 0 / 0 |
| ProtoMotions | 0 / 0 / 40 | 1 / 7 / 32 | 1 / 6 / 33 | 8 / 6 / 26 | 4 / 7 / 29 | 13 / 4 / 23 | 13 / 4 / 23 | 3 / 5 / 32 | 3 / 6 / 31 | 16 / 4 / 20 |
| ProtoMotions + residual | 0 / 2 / 38 | 0 / 15 / 25 | 1 / 16 / 23 | 14 / 12 / 14 | 20 / 10 / 10 | 18 / 14 / 8 | 22 / 11 / 7 | 22 / 7 / 11 | 24 / 4 / 12 | 22 / 11 / 7 |

Penetration: number of trials with a sustained penetration; per-trial peak depth, median / maximum over the trials (mm):

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 0; 1.0 / 4.0 | 0; 1.3 / 2.6 | 0; 1.5 / 3.3 | 0; 1.5 / 3.9 | 0; 2.0 / 5.4 | 0; 2.1 / 5.9 | 0; 2.2 / 5.9 | 10; 3.4 / 68.3 | 24; 4.6 / 65.3 | 0; 0.4 / 1.7 |
| HoST + residual | 0; 1.2 / 4.7 | 0; 1.5 / 6.1 | 0; 1.7 / 4.2 | 0; 1.9 / 3.6 | 0; 1.7 / 4.1 | 0; 1.8 / 4.1 | 0; 2.1 / 5.9 | 7; 3.3 / 42.2 | 14; 3.9 / 46.6 | 0; 0.6 / 2.4 |
| ProtoMotions | 0; 0.9 / 2.4 | 0; 1.0 / 2.5 | 0; 1.3 / 3.5 | 0; 1.6 / 4.3 | 0; 1.8 / 3.7 | 0; 2.2 / 5.5 | 0; 2.7 / 7.8 | 0; 2.5 / 37.2 | 0; 2.7 / 16.6 | 0; 0.4 / 1.5 |
| ProtoMotions + residual | 0; 1.2 / 4.0 | 0; 1.5 / 2.5 | 0; 2.2 / 4.2 | 0; 2.0 / 3.8 | 0; 2.3 / 5.5 | 0; 2.6 / 7.2 | 0; 3.1 / 11.8 | 0; 3.2 / 18.4 | 2; 4.3 / 63.5 | 0; 0.5 / 1.9 |

Largest contact depth / largest membrane crossing depth over all trials (mm):

| controller | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HoST | 4.0 / 0.0 | 2.6 / 0.0 | 3.3 / 0.0 | 3.9 / 0.0 | 5.4 / 0.0 | 5.9 / 0.0 | 5.9 / 0.0 | 48.4 / 68.3 | 46.0 / 65.3 | 1.7 / 0.0 |
| HoST + residual | 4.7 / 0.0 | 6.1 / 0.0 | 4.2 / 0.0 | 3.6 / 0.0 | 4.1 / 0.0 | 4.1 / 0.0 | 5.9 / 0.0 | 8.8 / 42.2 | 10.7 / 46.6 | 2.4 / 0.0 |
| ProtoMotions | 2.4 / 0.0 | 2.5 / 0.0 | 3.5 / 0.0 | 4.3 / 0.0 | 3.7 / 0.0 | 5.5 / 0.0 | 7.8 / 0.0 | 25.7 / 37.2 | 16.0 / 16.6 | 1.5 / 0.0 |
| ProtoMotions + residual | 4.0 / 0.0 | 2.5 / 0.0 | 4.2 / 0.0 | 3.8 / 0.0 | 5.5 / 0.0 | 7.2 / 0.0 | 11.8 / 0.0 | 18.4 / 14.7 | 33.3 / 63.5 | 1.9 / 0.0 |

On the mattresses and on rigid ground no trial has a sustained penetration; the per-trial peak depth has a median of 0.4 to 3.1 mm and a maximum of 11.8 mm, and the deeper readings last less than 10 control steps.

On the trampolines HoST has sustained penetrations in 10 and 24 trials (original controller, trampolines `a1` and `a2`) and in 7 and 14 trials (with the residual).
In these trials the contact depth is at most 7.5 mm; the penetration is a membrane crossing.
HoST's feet collide through nine spheres of radius 5 mm each, and the membrane's collision surface is a flex of radius 2 mm.
When the centre of a foot sphere passes below the membrane's mid-surface, MuJoCo's two-sided flex contact pushes the sphere further down, and the foot stays below the membrane.
A diagnostic rerun of HoST on both trampolines (on the AMD machine; 10, 24, 6 and 13 trials with a sustained penetration) recorded the robot body of the deepest point: of the 5995 control steps with more than 5 mm of crossing depth in those trials, 5985 belong to a foot sphere, and in 5980 the centre of that sphere was below the membrane's mid-surface.
The ProtoMotions G1 has capsule feet of radius 8 to 10 mm; it has two trials with a sustained penetration (with the residual, trampoline `a2`), in which the contact depth itself reaches 12.8 and 33.3 mm.

One ProtoMotions test initial state (`fall_03`) slides 0.13 m on rigid ground while it is settled under the gravity ramp and starts 1 cm outside the task area; it is a failure for both ProtoMotions rows on rigid ground.

The HoST outcome classes on the soft mattresses are the three classes of the paper's HoST figure: on mattress `a1` the original controller never stands in 65 trials and stands without completing the hold in 31; with the earlier residual policy the counts are 76 and 20.

## 2. Physics step

### Load response

A rigid ball of radius 0.125 m and mass 10 kg is placed with its bottom on the unloaded surface at the surface centre (on the 0.1 m mattress grid the centre is the middle of the edge shared by two cells).
Static deflection: gravity ramps up along a half cosine over 3 s and is held for 2 s; the deflection at the end.
Drop: the ball is released at rest under full gravity; the peak deflection over 2 s.
Contact and constraint time constants are kept at their values for the 0.625 ms step (`sim.contact_reference_dt_s`), so a smaller step changes only the integration.

Static deflection / drop peak deflection (mm), 0.1 m grid:

| surface | 0.625 ms | 0.3125 ms | 0.15625 ms | largest change |
| --- | ---: | ---: | ---: | ---: |
| mattress/a1 | 93.371 / 163.108 | 93.371 / 163.110 | 93.370 / 163.122 | 0.01 % |
| mattress/a2 | 51.596 / 96.441 | 51.596 / 96.442 | 51.596 / 96.445 | 0.00 % |
| mattress/a4 | 30.317 / 54.930 | 30.317 / 54.933 | 30.317 / 54.937 | 0.01 % |
| mattress/a8 | 18.784 / 32.599 | 18.784 / 32.611 | 18.784 / 32.610 | 0.04 % |
| mattress/a16 | 9.424 / 18.273 | 9.424 / 18.273 | 9.424 / 18.272 | 0.00 % |
| mattress/a32 | 4.743 / 9.188 | 4.743 / 9.186 | 4.743 / 9.186 | 0.02 % |
| mattress/a64 | 2.403 / 4.642 | 2.403 / 4.641 | 2.403 / 4.641 | 0.03 % |
| trampoline/a1 | 21.644 / 41.062 | 21.649 / 41.008 | 21.651 / 41.011 | 0.13 % |
| trampoline/a2 | 11.556 / 23.332 | 11.553 / 23.299 | 11.551 / 23.307 | 0.14 % |

Halving or quartering the physics step changes the static deflection and the drop peak deflection by at most 0.14 %.

### Success counts

Success (success without sustained penetration) on the 0.1 m grid at the three physics steps, and on the 0.05 m grid at 0.15625 ms (section 3):

HoST:

| grid, physics step | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 m, 0.625 ms | 0 (0) | 3 (3) | 6 (6) | 24 (24) | 51 (51) | 51 (51) | 91 (91) | 53 (52) | 48 (42) | 96 (96) |
| 0.1 m, 0.3125 ms | 0 (0) | 4 (4) | 2 (2) | 26 (26) | 54 (54) | 54 (54) | 93 (93) | 55 (54) | 58 (58) | 96 (96) |
| 0.1 m, 0.15625 ms | 0 (0) | 1 (1) | 3 (3) | 24 (24) | 56 (56) | 52 (52) | 93 (93) | 56 (56) | 54 (53) | 96 (96) |
| 0.05 m, 0.15625 ms | 0 (0) | 2 (2) | 3 (3) | 28 (28) | 37 (37) | 45 (45) | 95 (95) | 20 (15) | 27 (15) |  |

HoST + residual:

| grid, physics step | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 m, 0.625 ms | 0 (0) | 11 (11) | 51 (51) | 88 (88) | 95 (95) | 96 (96) | 96 (96) | 94 (87) | 93 (80) | 96 (96) |
| 0.1 m, 0.3125 ms | 0 (0) | 12 (12) | 45 (45) | 87 (87) | 94 (94) | 96 (96) | 96 (96) | 93 (93) | 94 (92) | 96 (96) |
| 0.1 m, 0.15625 ms | 2 (2) | 11 (11) | 50 (50) | 87 (87) | 96 (96) | 96 (96) | 96 (96) | 94 (94) | 94 (94) | 96 (96) |
| 0.05 m, 0.15625 ms | 3 (3) | 17 (17) | 40 (40) | 84 (84) | 96 (96) | 96 (96) | 96 (96) | 82 (80) | 45 (24) |  |

ProtoMotions:

| grid, physics step | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 m, 0.625 ms | 0 (0) | 1 (1) | 1 (1) | 8 (8) | 4 (4) | 13 (13) | 13 (13) | 3 (3) | 3 (3) | 16 (16) |
| 0.1 m, 0.3125 ms | 0 (0) | 0 (0) | 0 (0) | 6 (6) | 8 (8) | 12 (12) | 13 (13) | 3 (2) | 5 (5) | 17 (17) |
| 0.1 m, 0.15625 ms | 1 (1) | 0 (0) | 0 (0) | 5 (5) | 9 (9) | 11 (11) | 13 (13) | 2 (2) | 4 (4) | 16 (16) |
| 0.05 m, 0.15625 ms | 0 (0) | 1 (1) | 1 (1) | 3 (3) | 6 (6) | 9 (9) | 13 (13) | 1 (1) | 5 (5) |  |

ProtoMotions + residual:

| grid, physics step | mattress a1 | a2 | a4 | a8 | a16 | a32 | a64 | trampoline a1 | a2 | rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 m, 0.625 ms | 0 (0) | 0 (0) | 1 (1) | 14 (14) | 20 (20) | 18 (18) | 22 (22) | 22 (22) | 24 (24) | 22 (22) |
| 0.1 m, 0.3125 ms | 0 (0) | 0 (0) | 1 (1) | 11 (11) | 22 (22) | 23 (23) | 24 (24) | 22 (22) | 22 (22) | 23 (23) |
| 0.1 m, 0.15625 ms | 0 (0) | 0 (0) | 0 (0) | 13 (13) | 19 (19) | 25 (25) | 23 (23) | 23 (23) | 23 (23) | 19 (19) |
| 0.05 m, 0.15625 ms | 0 (0) | 0 (0) | 4 (4) | 13 (13) | 22 (22) | 22 (22) | 23 (23) | 15 (15) | 22 (22) |  |

Trials with a sustained penetration, summed over the ten surfaces (nine on the 0.05 m grid, which has no rigid ground):

| controller | 0.1 m, 0.625 ms | 0.1 m, 0.3125 ms | 0.1 m, 0.15625 ms | 0.05 m, 0.15625 ms |
| --- | ---: | ---: | ---: | ---: |
| HoST | 34 | 1 | 2 | 94 |
| HoST + residual | 21 | 2 | 0 | 83 |
| ProtoMotions | 0 | 2 | 0 | 0 |
| ProtoMotions + residual | 2 | 1 | 1 | 0 |

On the 0.1 m grid, halving or quartering the physics step changes the success count of a surface by at most 10 of 96 trials for HoST (trampoline `a2`, 48 to 58), 6 of 96 with the residual (mattress `a4`, 51 to 45), 5 of 40 for ProtoMotions (mattress `a16`, 4 to 9) and 7 of 40 with the residual (mattress `a32`, 18 to 25).
Summed over the ten surfaces the counts are 423, 442 and 435 of 960 (HoST), 720, 713 and 722 (HoST + residual), 62, 64 and 61 of 400 (ProtoMotions) and 143, 148 and 145 (ProtoMotions + residual) at 0.625, 0.3125 and 0.15625 ms.
At every physics step and on every surface the residual policy has at least as many successes as its original controller, except ProtoMotions on the softest mattresses, where the original controller has one success and the residual none (mattress `a2` at 0.625 ms, `a1` at 0.15625 ms).

The sustained penetrations of HoST on the trampolines (the foot spheres crossing the membrane, section 1) depend on the physics step: 34 and 21 trials at 0.625 ms, at most 2 at the smaller steps.
On the trampolines the counts of successes without a sustained penetration therefore rise with the smaller steps (HoST + residual on trampoline `a2`: 80, 92, 94).

## 3. Grid spacing

### Definition

The refined grids halve the spacing (0.05 m) and keep the continuum that the grid discretises (`conf/mattress/refined.yaml`, `conf/trampoline/refined.yaml`):

* mattress, 40 x 38 cells on the same 2.0 m x 1.9 m footprint: the bed is a foundation of stiffness k / pitch^2 per unit area plus a plate of bending stiffness b pitch^2, so k, c and m scale with pitch^2 (1/4) and b with 1/pitch^2 (x 4);
* trampoline, 49 x 49 node grid clipped to the circle (1661 free nodes instead of 377) on the same 1.2 m circle: the edge stiffness is a membrane tension and stays the same, the 3.0 kg membrane mass and the total node damping are spread over the free nodes.

Contact, cell travel and thickness, and the floor are unchanged.
Halving the spacing raises the highest frequency of the discretised bed four-fold and that of the membrane two-fold.
With MuJoCo's explicit integration of the tendon springs and of the membrane force, the refined mattress is stable in the load-response test at 0.625 ms on `a1` ... `a4`, needs 0.3125 ms on `a8` and `a16` and 0.15625 ms on `a32` and `a64`; the refined trampoline needs 0.3125 ms.
All refined runs therefore use 0.15625 ms and are compared with the 0.1 m grid at 0.15625 ms.

### Load response

Static deflection / drop peak deflection (mm), 0.05 m grid:

| surface | 0.625 ms | 0.3125 ms | 0.15625 ms | change from 0.1 m grid, 0.15625 ms |
| --- | ---: | ---: | ---: | ---: |
| mattress/a1 | 102.678 / 189.648 | 102.677 / 189.654 | 102.677 / 189.656 | +10.0 % / +16.3 % |
| mattress/a2 | 61.529 / 110.921 | 61.529 / 110.931 | 61.529 / 110.929 | +19.2 % / +15.0 % |
| mattress/a4 | 37.368 / 65.513 | 37.368 / 65.524 | 37.368 / 65.526 | +23.3 % / +19.3 % |
| mattress/a8 | diverged | 18.701 / 36.308 | 18.701 / 36.308 | -0.4 % / +11.3 % |
| mattress/a16 | diverged | 9.390 / 18.221 | 9.390 / 18.221 | -0.4 % / -0.3 % |
| mattress/a32 | diverged | diverged | 4.734 / 9.176 | -0.2 % / -0.1 % |
| mattress/a64 | diverged | diverged | 2.405 / 4.651 | +0.1 % / +0.2 % |
| trampoline/a1 | diverged | 22.224 / 44.231 | 22.226 / 44.222 | +2.7 % / +7.8 % |
| trampoline/a2 | diverged | 12.711 / 25.561 | 12.711 / 25.554 | +10.0 % / +9.6 % |

Static deflection (mm) at three ball positions on the two grids, 0.15625 ms.
On the 0.1 m grid (0, 0) is the middle of the edge shared by two cells and (0.05, 0) is a cell centre; on the 0.05 m grid (0, 0) and (0.05, 0) are corners shared by four cells and (0.025, 0.025) is a cell centre:

| surface | 0.1 m: (0, 0) | 0.1 m: (0.05, 0) | 0.1 m: (0.025, 0.025) | 0.05 m: (0, 0) | 0.05 m: (0.05, 0) | 0.05 m: (0.025, 0.025) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mattress/a1 | 93.37 | 83.78 | 98.24 | 102.68 | 102.68 | 105.38 |
| mattress/a2 | 51.60 | 51.17 | 51.27 | 61.53 | 61.53 | 61.58 |
| mattress/a4 | 30.32 | 32.60 | 32.49 | 37.37 | 37.37 | 33.09 |
| mattress/a8 | 18.78 | 22.18 | 22.26 | 18.70 | 18.70 | 19.35 |
| mattress/a16 | 9.42 | 12.77 | 13.03 | 9.39 | 9.39 | 11.46 |
| mattress/a32 | 4.74 | 6.42 | 6.42 | 4.73 | 4.73 | 6.04 |
| mattress/a64 | 2.40 | 3.25 | 3.25 | 2.41 | 2.41 | 3.12 |

On the stiffer mattresses (`a8` ... `a64`) the static deflection at the surface centre changes by less than 0.5 % when the spacing is halved, on `a1`, `a2` and `a4` it grows by 10 %, 19 % and 23 %, and on the trampolines by 3 % and 10 %.
On both mattress grids the deflection also depends on where the ball touches the grid: on the 0.1 m grid bed `a16` deflects 9.42 mm on the cell edge at (0, 0) and 12.77 mm at the cell centre (0.05, 0), 36 % more; on the 0.05 m grid bed `a64` deflects 2.41 mm on the cell corner at (0, 0) and 3.12 mm at the cell centre (0.025, 0.025), 30 % more.
Seen as a plate on an elastic foundation, with bending stiffness D and foundation stiffness K per unit area, the bed has the characteristic length (D / K)^(1/4) = pitch (b / k)^(1/4) = 0.083 m.
It is the same for all seven beds (their b / k is equal) and shorter than the 0.1 m cell pitch, so a load the size of the ball is not resolved by the 0.1 m grid.

### Success counts

See the table of section 2: rows "0.05 m, 0.15625 ms", compared with "0.1 m, 0.15625 ms".

Compared with the 0.1 m grid at the same physics step (0.15625 ms), the ProtoMotions counts on the 0.05 m grid change by at most 3 trials per surface (original controller) and by at most 4 per mattress (with the residual); with the residual the count on trampoline `a1` falls from 23 to 15.
HoST changes by at most 4 trials on mattresses `a1`, `a2`, `a4`, `a8` and `a64` and falls from 56 to 37 on `a16` and from 52 to 45 on `a32`; with the residual it changes by at most 6 trials on the mattresses except `a4` (50 to 40).

On the trampolines the HoST counts fall from 56 to 20 and from 54 to 27 (original controller, trampolines `a1` and `a2`) and from 94 to 82 and from 94 to 45 (with the residual), and the trials with a sustained penetration rise from 2 and 0 to 94 and 83 of the 192 trampoline trials (the largest contact depth in these trials is 12.3 and 19.0 mm).
ProtoMotions, whose feet are capsules, has no sustained penetration on the 0.05 m grid.
For HoST the trampoline results are therefore not converged in the grid spacing.
