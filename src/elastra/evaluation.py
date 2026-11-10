"""Evaluate a get-up controller, with or without a residual policy, on a set of surfaces.

    elastra-evaluate controller=host policy=checkpoints/host_residual.pt
    elastra-evaluate controller=host policy=null           # HoST alone
    elastra-evaluate controller=protomotions policy=checkpoints/protomotions_residual.pt

Every surface is run from the fixed test initial states (``data/initial_states``):
HoST's 96 test states (48 prone, 48 supine) or the 40 ProtoMotions test initial states.
Writes to ``out`` (``conf/evaluate.yaml``):

* ``cells.json``    one row per (surface, initial state): outcome, penetration summary
                    (over the control steps before the robot first leaves the task
                    area, see :mod:`elastra.penetration`)
* ``traces.npz``    per control step of the whole episode: pelvis height, surface
                    height, uprightness, contact and membrane-crossing penetration depth
* ``summary.json`` / ``summary.md``   per surface: successes, successes without a
                    sustained penetration, outcome classes, penetration statistics

Any configuration value can be overridden, e.g. ``sim.physics_dt_s=0.0003125`` or
``mattress=refined``.

One job is one (surface, posture) group for HoST and one (surface, reference clip)
group for ProtoMotions; its environments are stepped together and the residual
policy evaluates them as one batch.  A run is reproduced bit for bit by the same
command on the same machine.  The last bits of a trajectory can change with the CPU
model and with the batch size of the residual's matrix products, and a get-up is
sensitive enough to such differences that a few outcomes can change.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time

import numpy as np

from elastra import config, penetration, rollout
from elastra.scene import Surface, build_scene


def jobs_for(cfg) -> list[dict]:
    """HoST: one job per (surface, posture).  ProtoMotions: one job per (block, clip), a block
    being a group of surfaces (``protomotions_blocks``) whose environments of one clip are
    stepped as one batch, ordered surface by surface."""

    data = config.data_dir(cfg) / "initial_states"
    jobs = []
    if cfg.controller == "host":
        states = np.load(data / "host.npz")
        chosen = np.flatnonzero(states["split"] == str(cfg.initial_states))
        for surface in cfg.surfaces:
            for posture in ("prone", "supine"):
                rows = [int(i) for i in chosen if str(states["posture"][i]) == posture]
                jobs.append({"surfaces": [str(surface)], "group": posture, "rows": rows})
    else:
        states = np.load(data / "protomotions.npz")
        prefix = str(cfg.initial_states)
        clips = states[f"{prefix}_clip"]
        blocks = [[str(s) for s in block] for block in cfg.protomotions_blocks]
        in_blocks = [Surface.parse(s).label for block in blocks for s in block]
        wanted = [Surface.parse(str(s)).label for s in cfg.surfaces]
        if sorted(in_blocks) != sorted(wanted):
            raise ValueError(
                f"protomotions_blocks must hold every evaluated surface exactly once: "
                f"blocks {sorted(in_blocks)}, surfaces {sorted(wanted)}"
            )
        for block in blocks:
            for clip in sorted({str(c) for c in clips}):
                rows = [int(i) for i in np.flatnonzero(clips == clip)]
                if rows:
                    jobs.append({"surfaces": block, "group": clip, "rows": rows})
    return jobs


def run_job(args: tuple) -> dict:
    overrides, job = args
    import torch

    cfg = config.load(overrides, config_name="evaluate")
    robot = str(cfg.controller)
    torch.set_num_threads(1)
    from elastra import residual

    surfaces = [Surface.parse(s) for s in job["surfaces"]]
    data = config.data_dir(cfg) / "initial_states"
    built = {s.label: build_scene(cfg, robot, s) for s in surfaces}
    pairs = [(s, row) for s in surfaces for row in job["rows"]]
    scenes = [built[s.label] for s, _ in pairs]
    n = len(pairs)
    policy = None
    if cfg.policy:
        policy = residual.load_policy(
            config.repo_path(cfg.policy),
            int(cfg.robots.host.num_actions) if robot == "host" else 29,
        )
    started = time.time()
    if robot == "host":
        states = np.load(data / "host.npz")
        keys = ["qpos_initial" if s.kind == "trampoline" else "qpos_lowered" for s, _ in pairs]
        traces = rollout.host_episodes(
            cfg,
            scenes,
            [states[k][i] for k, (_, i) in zip(keys, pairs)],
            [str(states["posture"][i]) for _, i in pairs],
            [int(states["seed"][i]) for _, i in pairs],
            policy,
            residual_cfg=cfg.residuals.host,
            threads=int(cfg.threads),
        )
        ids = [str(states["init_id"][i]) for _, i in pairs]
    else:
        states = np.load(data / "protomotions.npz")
        prefix = str(cfg.initial_states)
        qpos = [states[f"{prefix}_qpos"][i] for _, i in pairs]
        settled = rollout.protomotions_settle(cfg, scenes, qpos, threads=int(cfg.threads))
        traces = rollout.protomotions_episodes(
            cfg, scenes, settled, [job["group"]] * n, policy, threads=int(cfg.threads)
        )
        ids = [str(states[f"{prefix}_source_id"][i]) for _, i in pairs]
    rows = []
    for (surface, _), init_id, trace in zip(pairs, ids, traces):
        outcome = rollout.score(trace, cfg.criterion, float(cfg.sim.control_dt_s))
        arrays = trace.arrays()
        inside = penetration.steps_in_task_area(
            trace.exit_physics_step, trace.physics_steps_per_control_step, arrays["pelvis_z"].size
        )
        contact = arrays["contact_depth"][:inside]
        crossing = arrays["crossing_depth"][:inside]
        pen = penetration.summarize(np.maximum(contact, crossing), cfg.penetration)
        rows.append(
            {
                "surface": surface.label,
                "initial_state": init_id,
                "group": job["group"],
                "block": "+".join(s.label for s in surfaces),
                "success": bool(outcome.success),
                "outcome": outcome.outcome_class,
                "completion_step": outcome.completion_step,
                "exit_physics_step": outcome.exit_physics_step,
                "longest_stand_steps": int(outcome.longest_stand_steps),
                "max_relative_height_m": float(np.max(arrays["pelvis_z"] - arrays["surface_z"])),
                "diverged": bool(trace.diverged),
                "penetration_scored_steps": int(inside),
                "penetration": pen,
                "peak_contact_depth_m": float(contact.max(initial=0.0)),
                "peak_crossing_depth_m": float(crossing.max(initial=0.0)),
                "valid": not pen["sustained"],
                "trace": {k: v.tolist() for k, v in arrays.items()},
            }
        )
    return {"job": job, "rows": rows, "wall_s": time.time() - started}


def summarize(rows: list[dict], surfaces: list[str]) -> list[dict]:
    out = []
    for surface in surfaces:
        label = Surface.parse(surface).label
        group = [r for r in rows if r["surface"] == label]
        peaks = np.asarray([r["penetration"]["peak_m"] for r in group])
        out.append(
            {
                "surface": label,
                "trials": len(group),
                "successes": sum(r["success"] for r in group),
                "successes_valid": sum(r["success"] and r["valid"] for r in group),
                "invalid": sum(not r["valid"] for r in group),
                "never_stood": sum(r["outcome"] == "never_stood" for r in group),
                "stood_not_completed": sum(r["outcome"] == "stood_not_completed" for r in group),
                "peak_penetration_max_m": float(peaks.max()) if group else None,
                "peak_penetration_median_m": float(np.median(peaks)) if group else None,
                "peak_contact_depth_max_m": float(max(r["peak_contact_depth_m"] for r in group))
                if group
                else None,
                "peak_crossing_depth_max_m": float(max(r["peak_crossing_depth_m"] for r in group))
                if group
                else None,
                "diverged": sum(r["diverged"] for r in group),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    cfg = config.load(argv, config_name="evaluate")
    out = config.repo_path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(cfg)
    workers = max(1, int(cfg.workers))
    started = time.time()
    results = []
    if workers == 1:
        for job in jobs:
            results.append(run_job((argv, job)))
            print(f"{'+'.join(job['surfaces'])} {job['group']}: done", flush=True)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap_unordered(run_job, [(argv, job) for job in jobs]):
                job = result["job"]
                ok = sum(r["success"] for r in result["rows"])
                print(
                    f"{'+'.join(job['surfaces'])} {job['group']}: {ok}/{len(result['rows'])} "
                    f"({result['wall_s']:.0f} s)",
                    flush=True,
                )
                results.append(result)
    rows = [row for result in results for row in result["rows"]]
    order = {Surface.parse(s).label: k for k, s in enumerate(cfg.surfaces)}
    rows.sort(key=lambda r: (order[r["surface"]], r["initial_state"]))
    np.savez_compressed(
        out / "traces.npz",
        **{
            f"{r['surface']}|{r['initial_state']}|{k}": np.asarray(v)
            for r in rows
            for k, v in r.pop("trace").items()
        },
    )
    (out / "cells.json").write_text(json.dumps(rows, indent=1))
    summary = summarize(rows, list(cfg.surfaces))
    meta = {
        "controller": str(cfg.controller),
        "policy": cfg.policy,
        "physics_dt_s": float(cfg.sim.physics_dt_s),
        "overrides": list(argv),
        "wall_s": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, "surfaces": summary}, indent=1))
    title = f"+ residual {cfg.policy}" if cfg.policy else "(original controller)"
    lines = [
        f"# {cfg.controller} {title}",
        "",
        f"physics step {float(cfg.sim.physics_dt_s)} s; overrides: {' '.join(argv) or 'none'}",
        "",
        "| surface | trials | success | success, no sustained penetration | "
        "sustained penetration | never stood | stood, not completed | diverged | "
        "peak depth median / max (mm) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['surface']} | {s['trials']} | {s['successes']} | "
            f"{s['successes_valid']} | {s['invalid']} | {s['never_stood']} | "
            f"{s['stood_not_completed']} | {s['diverged']} | "
            f"{1000 * s['peak_penetration_median_m']:.1f} / "
            f"{1000 * s['peak_penetration_max_m']:.1f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
