"""The ProtoMotions G1 motion tracker and its three get-up reference clips.

The tracker (NVlabs ProtoMotions, ``g1-bones-deploy``) is the released ONNX
deployment pipeline.  Each control step it takes the torso orientation, joint
positions and velocities, the base angular velocity, the previous PD target and
the reference clip 1, 2, 4 and 8 steps ahead, and returns 29 PD targets.  The
reference orientation is turned about the vertical once per episode so the
clip's initial heading matches the robot's (the yaw-only offset of the official
MuJoCo deployment).

Each initial state is paired with one of three BONES-SEED clips of a person
getting up from the back (``supine``), the side (``side``) or the front
(``prone``), resampled to 50 Hz (:func:`build_reference_clips`).  An episode lasts
as many control steps as its clip has frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

INPUT_NAMES = (
    "current_anchor_rot",
    "current_dof_pos",
    "current_dof_vel",
    "current_root_local_ang_vel",
    "historical_processed_actions",
    "mimic_future_anchor_rot",
    "mimic_future_dof_pos",
    "mimic_future_dof_vel",
)
OUTPUT_NAMES = ("actions", "joint_pos_targets", "stiffness_targets", "damping_targets")
FUTURE_STEPS = (1, 2, 4, 8)
CLIPS = ("supine", "side", "prone")


# --------------------------------------------------------------------------- #
# quaternions (xyzw storage)
# --------------------------------------------------------------------------- #
def _normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = (np.asarray(a, dtype=np.float64)[..., i] for i in range(4))
    bx, by, bz, bw = (np.asarray(b, dtype=np.float64)[..., i] for i in range(4))
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _yaw_xyzw(q: np.ndarray) -> np.ndarray:
    q = _normalize(q)
    x, y, z, w = (q[..., i] for i in range(4))
    half = 0.5 * np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zeros = np.zeros_like(half)
    return np.stack((zeros, zeros, np.sin(half), np.cos(half)), axis=-1)


def heading_offset_xyzw(robot_anchor: np.ndarray, reference_anchor: np.ndarray) -> np.ndarray:
    """The yaw-only rotation taking the reference's heading to the robot's."""

    conjugate = _yaw_xyzw(reference_anchor).copy()
    conjugate[..., :3] *= -1.0
    return _normalize(quat_multiply_xyzw(_yaw_xyzw(robot_anchor), conjugate))


# --------------------------------------------------------------------------- #
# reference clips
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReferenceClips:
    """The three clips padded to the longest by repeating their last frame."""

    names: tuple[str, ...]
    frames: np.ndarray  # (3,)
    anchor_rotation_xyzw: np.ndarray  # (3, T, 4)
    joint_position: np.ndarray  # (3, T, 29)
    joint_velocity: np.ndarray  # (3, T, 29)

    @classmethod
    def load(cls, directory: Path) -> ReferenceClips:
        clips = [np.load(Path(directory) / f"{name}.npz") for name in CLIPS]
        frames = np.asarray([c["joint_position"].shape[0] for c in clips], dtype=np.int64)
        longest = int(frames.max())

        def pad(field: str) -> np.ndarray:
            out = []
            for clip in clips:
                array = np.asarray(clip[field], dtype=np.float64)
                out.append(
                    np.concatenate([array, np.repeat(array[-1:], longest - array.shape[0], 0)])
                )
            return np.stack(out)

        return cls(
            CLIPS, frames, pad("anchor_rotation_xyzw"), pad("joint_position"), pad("joint_velocity")
        )

    def index(self, name: str) -> int:
        return self.names.index(name)


def build_reference_clips(assets: Path) -> list[Path]:
    """Resample the three BONES-SEED CSVs to 50 Hz reference clips.

    Each clip keeps 2 s of lying before the lowest pelvis height (the start of the
    get-up) and everything after it; positions and joint angles are interpolated
    linearly, the root orientation by slerp; joint velocities are the central
    difference smoothed by a Gaussian of sigma 2 frames; the torso orientation of
    every frame comes from forward kinematics of the G1 model.
    """

    import xml.etree.ElementTree as ET

    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation, Slerp

    from elastra.config import CONF_DIR
    from elastra.robots import protomotions_scene_root

    manifest = OmegaConf.load(CONF_DIR / "assets" / "default.yaml").bones_seed
    robot_cfg = OmegaConf.load(CONF_DIR / "robot" / "protomotions.yaml")
    root = protomotions_scene_root(Path(assets), robot_cfg)
    contact = root.find("contact")
    root.remove(contact)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, 30)]
    qpos_adr = np.asarray([model.joint(n).qposadr[0] for n in joint_names], dtype=np.int64)
    torso = model.body("torso_link").id
    source_fps, target_fps, lead_in_s = 120.0, 50.0, 2.0
    out_dir = Path(assets) / "reference_clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, item in manifest.clips.items():
        values = np.loadtxt(
            Path(assets) / "bones_seed" / Path(item.member).name,
            delimiter=",",
            skiprows=1,
            dtype=np.float64,
        )
        root_position = values[:, 1:4] / 100.0
        quat_wxyz = Rotation.from_euler("xyz", values[:, 4:7], degrees=True).as_quat()[
            :, [3, 0, 1, 2]
        ]
        joints = np.deg2rad(values[:, 7:])
        lowest = int(np.argmin(root_position[:, 2]))
        start = max(0, lowest - int(round(lead_in_s * source_fps)))
        root_position = root_position[start:].copy()
        root_position[:, :2] -= root_position[0, :2]
        quat_wxyz, joints = quat_wxyz[start:], joints[start:]
        source_t = np.arange(root_position.shape[0], dtype=np.float64) / source_fps
        duration = float(source_t[-1])
        target_t = np.arange(int(round(duration * target_fps)) + 1, dtype=np.float64) / target_fps
        target_t[-1] = duration

        def resample(array: np.ndarray) -> np.ndarray:
            out = np.empty((target_t.size, array.shape[1]))
            for column in range(array.shape[1]):
                out[:, column] = np.interp(target_t, source_t, array[:, column])
            return out

        root_r = resample(root_position)
        joints_r = resample(joints)
        quat_r = Slerp(source_t, Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]))(target_t).as_quat()
        quat_r_wxyz = quat_r[:, [3, 0, 1, 2]]
        velocity = gaussian_filter1d(
            np.gradient(joints_r, 1.0 / target_fps, axis=0), sigma=2.0, axis=0, mode="nearest"
        )
        anchor = np.zeros((target_t.size, 4))
        for frame in range(target_t.size):
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[:3] = root_r[frame]
            data.qpos[3:7] = quat_r_wxyz[frame]
            data.qpos[qpos_adr] = joints_r[frame]
            mujoco.mj_forward(model, data)
            anchor[frame] = data.xquat[torso][[1, 2, 3, 0]]
        path = out_dir / f"{name}.npz"
        np.savez(
            path,
            joint_position=joints_r,
            joint_velocity=velocity,
            anchor_rotation_xyzw=anchor,
            root_position=root_r,
            root_rotation_xyzw=quat_r,
        )
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# the tracker
# --------------------------------------------------------------------------- #
class Tracker:
    """The frozen ProtoMotions tracker, batched over environments with different clips."""

    def __init__(self, assets: Path, clips: ReferenceClips) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(Path(assets) / "protomotions" / "unified_pipeline.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        names = tuple(i.name for i in self.session.get_inputs())
        if names != INPUT_NAMES:
            raise RuntimeError(f"unexpected tracker inputs {names}")
        self.clips = clips
        self.offsets = np.asarray(FUTURE_STEPS, dtype=np.int64)

    def heading(self, anchor_xyzw: np.ndarray, clip: np.ndarray) -> np.ndarray:
        return heading_offset_xyzw(anchor_xyzw, self.clips.anchor_rotation_xyzw[clip, 0])

    def act(
        self,
        state: dict[str, np.ndarray],
        step: np.ndarray,
        clip: np.ndarray,
        heading: np.ndarray,
        previous_target: np.ndarray,
    ) -> np.ndarray:
        """PD targets (n, 29) for one control step."""

        last = self.clips.frames[clip] - 1
        frames = np.minimum(np.asarray(step, dtype=np.int64), last)
        future = np.minimum(frames[:, None] + self.offsets[None, :], last[:, None])
        rows = clip[:, None]
        future_anchor = _normalize(
            quat_multiply_xyzw(heading[:, None, :], self.clips.anchor_rotation_xyzw[rows, future])
        ).astype(np.float32)
        feed = {
            "current_anchor_rot": state["anchor_xyzw"].astype(np.float32),
            "current_dof_pos": state["dof_pos"].astype(np.float32),
            "current_dof_vel": state["dof_vel"].astype(np.float32),
            "current_root_local_ang_vel": state["root_ang_vel"].astype(np.float32),
            "historical_processed_actions": np.asarray(previous_target, dtype=np.float32)[
                :, None, :
            ],
            "mimic_future_anchor_rot": future_anchor,
            "mimic_future_dof_pos": self.clips.joint_position[rows, future].astype(np.float32),
            "mimic_future_dof_vel": self.clips.joint_velocity[rows, future].astype(np.float32),
        }
        outputs = self.session.run(list(OUTPUT_NAMES), feed)
        return np.asarray(outputs[1], dtype=np.float64)
