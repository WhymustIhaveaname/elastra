"""The EN 1957 loading pad pressed into the mattress, fitted to a measured load-deflection curve.

    elastra-indentation                                     # bed a1, pad at the surface centre
    elastra-indentation 'pad.position_xy_m=[0.05,0.0]'      # pad centred on a cell
    elastra-indentation mattress=refined sim.physics_dt_s=0.00015625 \
        out=outputs/indentation/refined

The loading pad of EN 1957 (``conf/indentation.yaml``: a rigid disc with a convex spherical
face and a rounded front edge) is a mocap body above the mattress without the robot.  The
bed first settles under its own weight; the pad is then placed with its apex on the bed top
and pushed down step by step: it moves at ``loading.speed_m_s`` to the next deflection of the
measured curve and is held there for ``loading.hold_s``.  The load is the vertical contact
force of the cells on the pad at the end of the hold; the deflection is the depth of the pad
apex below the unloaded bed top.

The stiffness scale ``s`` multiplies ``k`` and ``b`` of the bed ``bed``.  It is fitted by least
squares to the measured loads at ``fit_deflections_mm``: with the loads ``F`` simulated at the
current ``s`` and the measured loads ``F*``, ``s`` is replaced by ``s (F . F*) / (F . F)`` and
the bed is simulated again, until that factor is within ``fit.tolerance`` of 1.  (The load is
proportional to ``s`` up to the contact compliance and the joint range, which stops a cell at
``z = 0``, ``m g / k`` above its unloaded position, so this converges in two or three runs.)
The loads at the other deflections of the curve are predictions.

Writes ``indentation.json`` (configuration, every run, and per deflection the measured load, the
simulated load and the relative error) and ``summary.md`` to ``out``.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf

from elastra import config
from elastra.mattress import BED_FIELDS
from elastra.scene import Surface, freeze_contact_time_constants, strip_robot, surface_xml

#: the bed name under which the scaled bed is added to a copy of the configuration
SCALED_BED = "indentation"


@dataclass(frozen=True)
class Pad:
    """The pad's face in its meridional plane, apex at the origin, ``z`` up.

    The face is a sphere of radius ``R`` out to ``r_d``, then the edge, a torus of radius
    ``f`` tangent to the sphere and to the cylinder ``r = a``.
    """

    radius_m: float  # a
    face_radius_m: float  # R
    edge_radius_m: float  # f

    @classmethod
    def from_cfg(cls, pad: DictConfig) -> Pad:
        return cls(
            radius_m=0.5 * float(pad.diameter_m),
            face_radius_m=float(pad.face_radius_m),
            edge_radius_m=float(pad.edge_radius_m),
        )

    @property
    def edge_centre(self) -> tuple[float, float]:
        """``(r, z)`` of the centre of the edge circle."""

        a, big_r, f = self.radius_m, self.face_radius_m, self.edge_radius_m
        r = a - f
        return r, big_r - math.sqrt((big_r - f) ** 2 - r * r)

    @property
    def face_edge_radius_m(self) -> float:
        """``r_d``, where the spherical face meets the edge."""

        r, _ = self.edge_centre
        return self.face_radius_m * r / (self.face_radius_m - self.edge_radius_m)

    def height(self, radial_m: np.ndarray | float) -> np.ndarray:
        """Height of the face above the apex at radius ``radial_m`` (``inf`` beyond the pad)."""

        r = np.abs(np.asarray(radial_m, dtype=np.float64))
        big_r, f = self.face_radius_m, self.edge_radius_m
        centre_r, centre_z = self.edge_centre
        sphere = big_r - np.sqrt(np.maximum(big_r * big_r - r * r, 0.0))
        edge = centre_z - np.sqrt(np.maximum(f * f - (r - centre_r) ** 2, 0.0))
        face = np.where(r <= self.face_edge_radius_m, sphere, edge)
        return np.where(r <= self.radius_m, face, np.inf)

    def vertices(self, rings: int, segments: int, cylinder_height_m: float) -> np.ndarray:
        """Vertices of the pad's convex hull: the apex, ``rings`` rings on the spherical face
        and on the edge, and the top of the cylinder."""

        angle = 2.0 * math.pi * np.arange(segments) / segments
        circle = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        radii = [self.face_edge_radius_m * i / rings for i in range(1, rings + 1)]
        profile = [(r, float(self.height(r))) for r in radii]
        centre_r, centre_z = self.edge_centre
        start = math.asin(self.face_edge_radius_m / self.face_radius_m)
        for i in range(1, rings + 1):
            theta = start + (0.5 * math.pi - start) * i / rings
            profile.append(
                (
                    centre_r + self.edge_radius_m * math.sin(theta),
                    centre_z - self.edge_radius_m * math.cos(theta),
                )
            )
        profile.append((self.radius_m, centre_z + float(cylinder_height_m)))
        rows = [np.zeros((1, 3))]
        for r, z in profile:
            rows.append(np.column_stack([r * circle, np.full(segments, z)]))
        return np.concatenate(rows)


def scaled_cfg(cfg: DictConfig, scale: float) -> DictConfig:
    """A copy of ``cfg`` with bed :data:`SCALED_BED`: bed ``cfg.bed`` with ``k`` and ``b``
    multiplied by ``scale``."""

    out = copy.deepcopy(cfg)
    row = {field: float(cfg.mattress.beds[cfg.bed][field]) for field in BED_FIELDS}
    row["k_n_per_m"] *= scale
    row["b_n_per_m"] *= scale
    OmegaConf.update(out, f"mattress.beds.{SCALED_BED}", row, force_add=True)
    return out


def indentation_model(cfg: DictConfig, scale: float) -> mujoco.MjModel:
    """The bed, scaled by ``scale``, without the robot, with the pad as a mocap body."""

    cfg = scaled_cfg(cfg, scale)
    surface = Surface("mattress", SCALED_BED)
    root = ET.fromstring(strip_robot(surface_xml(cfg, surface), "indentation"))
    pad_cfg = cfg.pad
    vertices = Pad.from_cfg(pad_cfg).vertices(
        int(pad_cfg.rings), int(pad_cfg.segments), float(pad_cfg.cylinder_height_m)
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "mesh",
        {"name": "pad", "vertex": " ".join(f"{v:.12g}" for v in vertices.ravel())},
    )
    x, y = (float(v) for v in pad_cfg.position_xy_m)
    body = ET.SubElement(
        root.find("worldbody"), "body", {"name": "pad", "mocap": "true", "pos": f"{x} {y} 0.5"}
    )
    contact = cfg.mattress.contact
    ET.SubElement(
        body,
        "geom",
        {
            "name": "pad",
            "type": "mesh",
            "mesh": "pad",
            "contype": "1",
            "conaffinity": "0",
            # above the cells' priority, so the pad's condim applies
            "priority": str(int(contact.priority) + 1),
            "condim": str(int(cfg.pad_contact.condim)),
            "solref": str(contact.solref),
            "solimp": str(contact.solimp),
        },
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = float(cfg.sim.physics_dt_s)
    model.opt.gravity[:] = list(cfg.sim.gravity_m_s2)
    freeze_contact_time_constants(model, float(cfg.sim.contact_reference_dt_s))
    return model


def pad_load(model: mujoco.MjModel, data: mujoco.MjData, pad_geom: int) -> tuple[float, int]:
    """Upward contact force of the cells on the pad (N), and the number of contacts."""

    force = np.zeros(6)
    total = 0.0
    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        if pad_geom not in (int(contact.geom1), int(contact.geom2)):
            continue
        mujoco.mj_contactForce(model, data, index, force)
        # force[0] is the normal force; the normal frame[0:3] points from geom1 to geom2
        on_geom2 = force[0] * np.asarray(contact.frame[:3])
        total += float(on_geom2[2] if int(contact.geom2) == pad_geom else -on_geom2[2])
        count += 1
    return total, count


def indent(cfg: DictConfig, scale: float, deflections_mm: list[float]) -> dict:
    """Press the pad into the bed scaled by ``scale``; the load at every deflection."""

    if np.any(np.diff([0.0, *deflections_mm]) <= 0.0):
        raise ValueError(f"the deflections {deflections_mm} must be positive and increasing")
    model = indentation_model(cfg, scale)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    pad_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad")
    mocap = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pad")])
    joints = [
        j
        for j in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(
            "mattress_slide_"
        )
    ]
    qpos = model.jnt_qposadr[joints]
    dof = model.jnt_dofadr[joints]
    loading = cfg.loading
    for _ in range(int(round(float(loading.settle_s) / dt))):
        mujoco.mj_step(model, data)
    # the top of the unloaded bed: the highest cell top (the cells sag uniformly)
    top = -float(np.min(data.qpos[qpos]))
    data.mocap_pos[mocap, 2] = top
    mujoco.mj_forward(model, data)
    unloaded_load, _ = pad_load(model, data, pad_geom)
    speed = float(loading.speed_m_s)
    hold = int(round(float(loading.hold_s) / dt))
    depth = 0.0
    rows = []
    for deflection_mm in deflections_mm:
        target = float(deflection_mm) / 1000.0
        steps = max(1, int(math.ceil((target - depth) / speed / dt)))
        for step in range(1, steps + 1):
            data.mocap_pos[mocap, 2] = top - (depth + (target - depth) * step / steps)
            mujoco.mj_step(model, data)
        depth = target
        for _ in range(hold):
            mujoco.mj_step(model, data)
        if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            raise FloatingPointError(f"the indentation diverged at {deflection_mm} mm")
        mujoco.mj_forward(model, data)
        load, contacts = pad_load(model, data, pad_geom)
        cell_speed = float(np.max(np.abs(data.qvel[dof])))
        rows.append(
            {
                "deflection_mm": float(deflection_mm),
                "load_n": load,
                "contacts": contacts,
                "max_cell_deflection_mm": 1000.0 * float(np.max(data.qpos[qpos])),
                "max_cell_speed_m_s": cell_speed,
                "settled": cell_speed <= float(loading.settled_cell_speed_m_s),
            }
        )
    bed = scaled_cfg(cfg, scale).mattress.beds[SCALED_BED]
    return {
        "scale": float(scale),
        "bed": {field: float(bed[field]) for field in BED_FIELDS},
        "unloaded_top_z_m": top,
        "load_at_first_contact_n": unloaded_load,
        "rows": rows,
    }


def read_curve(path: Path) -> list[tuple[float, float]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [(float(row["deflection_mm"]), float(row["load_n"])) for row in rows]


def fit(cfg: DictConfig) -> dict:
    curve = read_curve(config.repo_path(cfg.curve))
    deflections = [d for d, _ in curve]
    measured = np.asarray([f for _, f in curve])
    fit_set = {float(d) for d in cfg.fit_deflections_mm}
    missing = fit_set - set(deflections)
    if missing:
        raise ValueError(f"fit deflections {sorted(missing)} are not on the curve")
    used = np.asarray([d in fit_set for d in deflections])
    scale = 1.0
    runs = []
    for _ in range(int(cfg.fit.max_iterations)):
        run = indent(cfg, scale, deflections)
        simulated = np.asarray([row["load_n"] for row in run["rows"]])
        factor = float(simulated[used] @ measured[used] / (simulated[used] @ simulated[used]))
        run["least_squares_factor"] = factor
        runs.append(run)
        print(f"s = {scale:.6f}: least-squares factor {factor:.6f}", flush=True)
        if abs(factor - 1.0) <= float(cfg.fit.tolerance):
            break
        scale *= factor
    else:
        raise RuntimeError(f"the stiffness scale did not converge in {len(runs)} runs")
    final = runs[-1]
    rows = []
    for (_, load), row, in_fit in zip(curve, final["rows"], used):
        rows.append(
            {
                **row,
                "measured_load_n": load,
                "relative_error": (row["load_n"] - load) / load,
                "used_in_fit": bool(in_fit),
            }
        )
    return {"scale": final["scale"], "bed": final["bed"], "rows": rows, "runs": runs}


def summary_markdown(cfg: DictConfig, result: dict) -> str:
    bed = result["bed"]
    lines = [
        f"Stiffness scale s = {result['scale']:.4f} (bed {cfg.bed}: k and b times s): "
        f"k = {bed['k_n_per_m']:.2f} N/m, b = {bed['b_n_per_m']:.2f} N/m.",
        f"Pad at ({', '.join(f'{float(v):g}' for v in cfg.pad.position_xy_m)}) m, "
        f"grid pitch {float(cfg.mattress.pitch_m):g} m, "
        f"physics step {float(cfg.sim.physics_dt_s):g} s.",
        "",
        "| deflection (mm) | measured (N) | simulated (N) | relative error | used |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        used = "fit" if row["used_in_fit"] else "held out"
        lines.append(
            f"| {row['deflection_mm']:g} | {row['measured_load_n']:.1f} | {row['load_n']:.1f} | "
            f"{100.0 * row['relative_error']:+.1f} % | {used} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    cfg = config.load(argv, config_name="indentation")
    out = config.repo_path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    result = fit(cfg)
    unsettled = [row["deflection_mm"] for row in result["rows"] if not row["settled"]]
    if unsettled:
        print(f"warning: not settled at {unsettled} mm", flush=True)
    result["config"] = OmegaConf.to_container(cfg, resolve=True)
    result["overrides"] = list(argv)
    (out / "indentation.json").write_text(json.dumps(result, indent=1))
    text = summary_markdown(cfg, result)
    (out / "summary.md").write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
