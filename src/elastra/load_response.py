"""Load response of the surfaces to a rigid ball, for the physics-step and grid-spacing studies.

    elastra-load-response                                   # 0.625, 0.3125, 0.15625 ms
    elastra-load-response mattress=refined trampoline=refined \
        out=outputs/load_response/refined

A rigid ball (``conf/load_response.yaml``) is placed with its bottom on the unloaded
surface top at ``load.position_xy_m`` (by default the centre of the surface; on the
20 x 19 mattress that is the middle of the shared edge of two cells) and loaded in
two ways:

* static: gravity ramps up along a half cosine over ``static.ramp_s`` and is held for
  ``static.hold_s``; the static deflection is the ball's downward displacement at the end;
* drop: the ball is released at rest under full gravity; the deflection is recorded every
  ``record_dt_s`` for ``drop.duration_s``.

Writes ``load_response.json`` with, per surface and physics step, the static deflection, the
peak deflection of the drop, its time, and the deflection curve.  A run is marked unstable (and
its deflections are not reported) when the integration diverges: MuJoCo reports a bad
acceleration (it then resets the state), or a membrane node moves beyond the
membrane radius.
"""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from elastra import config
from elastra.scene import Surface, freeze_contact_time_constants, strip_robot, surface_xml
from elastra.trampoline import MembraneForce


def ball_scene(cfg, surface: Surface, physics_dt: float):
    root = ET.fromstring(strip_robot(surface_xml(cfg, surface), "load_response"))
    load = cfg.load
    radius = float(load.radius_m)
    x, y = (float(v) for v in load.position_xy_m)
    body = ET.SubElement(
        root.find("worldbody"),
        "body",
        {"name": "load", "pos": f"{x:.12g} {y:.12g} {radius:.12g}"},
    )
    ET.SubElement(body, "freejoint", {"name": "load"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "load",
            "type": "sphere",
            "size": f"{radius:.12g}",
            "mass": f"{float(load.mass_kg):.12g}",
            "contype": "1",
            "conaffinity": "1",
            "friction": str(load.friction),
        },
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = float(physics_dt)
    model.opt.gravity[:] = list(cfg.sim.gravity_m_s2)
    freeze_contact_time_constants(model, float(cfg.sim.contact_reference_dt_s))
    prefix = "trampoline_slide_" if surface.kind == "trampoline" else "mattress_slide_"
    joints = [
        j
        for j in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(prefix)
    ]
    qpos = model.jnt_qposadr[joints].astype(np.int64)
    dof = model.jnt_dofadr[joints].astype(np.int64)
    force = MembraneForce(cfg.trampoline, surface.name) if surface.kind == "trampoline" else None
    return model, qpos, dof, force


def run(
    model, qpos, dof, force, radius: float, *, steps: int, ramp_steps: int, record_every: int
) -> tuple[np.ndarray, bool]:
    """The ball's deflection every ``record_every`` steps and whether the integration diverged
    (MuJoCo's bad-acceleration warning, or a membrane node beyond its radius)."""

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gravity = np.asarray(model.opt.gravity).copy()
    z = ball_z_index(model)
    deflection = []
    diverged = False
    try:
        for step in range(1, steps + 1):
            fraction = (
                0.5 - 0.5 * math.cos(math.pi * step / ramp_steps) if step <= ramp_steps else 1.0
            )
            model.opt.gravity[:] = gravity * fraction
            data.qfrc_applied[:] = 0.0
            if force is not None:
                data.qfrc_applied[dof] += force.qfrc(data.qpos[qpos][None, :])[0]
            mujoco.mj_step(model, data)
            if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
                diverged = True
                break
            if step % record_every == 0:
                deflection.append(radius - float(data.qpos[z]))
    except FloatingPointError:
        diverged = True
    finally:
        model.opt.gravity[:] = gravity
    return np.asarray(deflection), diverged


def ball_z_index(model) -> int:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "load")
    return int(model.jnt_qposadr[joint]) + 2


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    cfg = config.load(argv, config_name="load_response")
    out = config.repo_path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    radius = float(cfg.load.radius_m)
    record_dt = float(cfg.record_dt_s)
    results = []
    for name in cfg.surfaces:
        surface = Surface.parse(str(name))
        for dt in cfg.physics_dts:
            dt = float(dt)
            every = int(round(record_dt / dt))
            if not math.isclose(every * dt, record_dt, rel_tol=1e-9):
                raise ValueError(f"record step {record_dt} is not a multiple of {dt}")
            model, qpos, dof, force = ball_scene(cfg, surface, dt)
            static_steps = int(round((float(cfg.static.ramp_s) + float(cfg.static.hold_s)) / dt))
            static, static_diverged = run(
                model,
                qpos,
                dof,
                force,
                radius,
                steps=static_steps,
                ramp_steps=int(round(float(cfg.static.ramp_s) / dt)),
                record_every=every,
            )
            drop, drop_diverged = run(
                model,
                qpos,
                dof,
                force,
                radius,
                steps=int(round(float(cfg.drop.duration_s) / dt)),
                ramp_steps=0,
                record_every=every,
            )
            stable = not (static_diverged or drop_diverged) and bool(
                np.isfinite(static).all() and np.isfinite(drop).all()
            )
            row = {
                "surface": surface.label,
                "physics_dt_s": dt,
                "stable": stable,
                "static_deflection_m": None,
                "drop_peak_deflection_m": None,
                "drop_peak_time_s": None,
                "drop_final_deflection_m": None,
                "drop_deflection_m": drop.tolist(),
            }
            if stable:
                peak = int(np.argmax(drop))
                row.update(
                    {
                        "static_deflection_m": float(static[-1]),
                        "drop_peak_deflection_m": float(drop[peak]),
                        "drop_peak_time_s": float((peak + 1) * record_dt),
                        "drop_final_deflection_m": float(drop[-1]),
                    }
                )
            results.append(row)
            if stable:
                print(
                    f"{surface.label} dt={dt:.8f}: static {1000 * row['static_deflection_m']:.3f} mm, "
                    f"drop peak {1000 * row['drop_peak_deflection_m']:.3f} mm at "
                    f"{row['drop_peak_time_s']:.3f} s",
                    flush=True,
                )
            else:
                print(f"{surface.label} dt={dt:.8f}: unstable", flush=True)
    meta = {
        "load": {"radius_m": radius, "mass_kg": float(cfg.load.mass_kg)},
        "record_dt_s": record_dt,
        "overrides": list(argv),
    }
    (out / "load_response.json").write_text(json.dumps({"meta": meta, "rows": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
