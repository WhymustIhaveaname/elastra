# Data

## `initial_states/host.npz`: HoST initial states

128 lying poses of the 23-DoF G1, written by `elastra-make-host-initial-states` with HoST's initial-state randomisation: root at (0, 0, 0.5) m, the posture's root orientation (prone or supine) and randomised joint angles, one pose per posture and seed 0 ... 63.

| array | shape | meaning |
|---|---|---|
| `init_id` | (128,) | `prone_16`, `supine_63`, ... (posture and seed) |
| `posture` | (128,) | `prone` or `supine` |
| `seed` | (128,) | seed of the randomisation |
| `split` | (128,) | `train` (seeds 0 ... 15, 32 states) or `test` (seeds 16 ... 63, 96 states) |
| `qpos_initial` | (128, 30) | root position, root quaternion (w, x, y, z), 23 joint angles; HoST's own starting height, used on the trampolines (the robot falls onto the membrane) |
| `qpos_lowered` | (128, 30) | the same pose lowered along z until the robot touches z = 0, the top of the unloaded mattress and of rigid ground |

The evaluation runs the 96 `test` states on every surface.

## `initial_states/protomotions.npz`: ProtoMotions initial states

Poses of the 29-DoF G1 (`qpos`: root position, root quaternion (w, x, y, z), 29 joint angles).
Before an episode each pose is settled on the surface under a gravity ramp (`robots.protomotions.settle` in `conf/robot/protomotions.yaml`).
`*_clip` is the reference clip the tracker follows (`prone`, `side` or `supine`).

* `test_*`, 40 states, the evaluation set.
  Each is a passive fall of the robot with its motors off on rigid ground: from the standing joint angles plus uniform offsets in [-0.35, 0.35] rad, a root tilt of 0.3 ... 1.2 rad about a random horizontal axis and small random root and joint velocities, simulated for 3 s; the final pose, with the root's horizontal position and yaw removed, is the state.
  Out of 2400 candidates, those that ended standing, outside the joint ranges, or with a sustained ground penetration were dropped, and near duplicates (joint RMS distance below 0.15 rad, also against the training poses) were merged; the 40 states were fixed before any policy was run on them.
  The clip is the one whose starting root orientation is closest.
  `test_source_id` is `fall_00` ... `fall_39`.
* `train_*`, 39 states, the residual's training set: lying poses from the HumanUP pose pool (https://github.com/RunpeiDong/HumanUP, Apache-2.0), mapped to the 29-DoF G1 and placed on rigid ground with a horizontal offset, a yaw rotation and small joint offsets.
  `train_source_id` names the HumanUP pose and the placement.
