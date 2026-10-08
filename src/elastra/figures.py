"""Figures of the evaluation and training results (``elastra-figures``, ``conf/figures.yaml``).

    elastra-figures                          # the result paths in conf/figures.yaml
    elastra-figures out=outputs/figures

Reads ``summary.json`` of four ``elastra-evaluate`` runs (each original controller without
and with its residual policy) and ``history.jsonl`` of the two ``elastra-train`` runs, and writes
to ``out``:

* ``host.png``: HoST, (a) successes per surface, (b) outcome classes on the softest
  mattress;
* ``protomotions.png``: ProtoMotions, (a) successes per surface, (b) outcome classes
  over all trials;
* ``training.png``: the fraction of finished training episodes that succeeded, per
  update, for both training runs.

Colours are taken from the Okabe-Ito palette.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from elastra import config
from elastra.scene import Surface

# Okabe-Ito colours
ORANGE = "#E69F00"
SKY_BLUE = "#56B4E9"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
REDDISH_PURPLE = "#CC79A7"
INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"

CONTROLLER_COLOURS = {"original": ORANGE, "residual": BLUE}
OUTCOME_CLASSES = [
    ("never_stood", "never stood", VERMILLION),
    ("stood_not_completed", "stood but not completed", REDDISH_PURPLE),
    ("successes", "success", SKY_BLUE),
]
NAMES = {"host": "HoST", "protomotions": "ProtoMotions"}


def _summary(path) -> dict[str, dict]:
    data = json.loads((config.repo_path(path) / "summary.json").read_text())
    return {row["surface"]: row for row in data["surfaces"]}


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=8, width=0.6, color=MUTED)


def _surface_axis(ax, surfaces: list[str]) -> None:
    """Tick labels a1 ... a64 with the surface kind written once below each group."""

    parsed = [Surface.parse(s) for s in surfaces]
    ax.set_xticks(range(len(surfaces)))
    ax.set_xticklabels(["rigid\nground" if s.kind == "rigid" else s.name for s in parsed])
    for kind, title in (("mattress", "mattress"), ("trampoline", "trampoline")):
        positions = [k for k, s in enumerate(parsed) if s.kind == kind]
        if positions:
            ax.annotate(
                title,
                xy=(float(np.mean(positions)), 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -26),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color=MUTED,
            )
    for k in range(1, len(parsed)):
        if parsed[k].kind != parsed[k - 1].kind:
            ax.axvline(k - 0.5, color=GRID, linewidth=0.8, zorder=0)


def _successes_panel(ax, original: dict, residual: dict, surfaces: list[str], name: str) -> None:
    labels = [Surface.parse(s).label for s in surfaces]
    trials = int(original[labels[0]]["trials"])
    width = 0.38
    x = np.arange(len(labels))
    for offset, key, rows, label in (
        (-width / 2 - 0.01, "original", original, f"{name}, original controller"),
        (width / 2 + 0.01, "residual", residual, f"{name} with the residual policy"),
    ):
        colour = CONTROLLER_COLOURS[key]
        valid = np.asarray([rows[s]["successes_valid"] for s in labels], dtype=float)
        total = np.asarray([rows[s]["successes"] for s in labels], dtype=float)
        ax.bar(x + offset, valid, width, color=colour, label=label, zorder=2)
        ax.bar(
            x + offset,
            total - valid,
            width,
            bottom=valid,
            color="white",
            edgecolor=colour,
            hatch="////",
            linewidth=0.0,
            zorder=2,
        )
    ax.bar(
        [np.nan],
        [0],
        color="white",
        edgecolor=MUTED,
        hatch="////",
        linewidth=0.0,
        label="success with sustained penetration (invalid)",
    )
    ax.set_ylim(0, trials * 1.04)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(f"successes (of {trials} trials)", fontsize=8, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    _surface_axis(ax, surfaces)
    _style(ax)


def _outcomes_panel(ax, rows: list[tuple[str, dict]]) -> None:
    """Horizontal stacked bars of the three outcome classes, one bar per controller."""

    for y, (label, counts) in enumerate(rows):
        left = 0.0
        total = float(sum(counts[key] for key, _, _ in OUTCOME_CLASSES))
        for key, _, colour in OUTCOME_CLASSES:
            value = float(counts[key])
            if value > 0:
                ax.barh(
                    y, value, left=left, height=0.6, color=colour, edgecolor="white", linewidth=1.5
                )
                if value / total >= 0.06:
                    ax.text(
                        left + value / 2,
                        y,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=INK,
                    )
            left += value
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])
    ax.invert_yaxis()
    total = float(sum(rows[0][1][key] for key, _, _ in OUTCOME_CLASSES))
    ax.set_xlim(0, total)
    ax.set_xlabel("trials", fontsize=8, color=INK)
    _style(ax)
    from matplotlib.patches import Patch

    handles = [Patch(color=colour, label=text) for _, text, colour in OUTCOME_CLASSES]
    ax.legend(
        handles=handles,
        ncol=3,
        fontsize=8,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.2,
    )


def controller_figure(cfg, robot: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = _summary(cfg.evaluation[f"{robot}_original"])
    residual = _summary(cfg.evaluation[f"{robot}_residual"])
    surfaces = list(original)
    name = NAMES[robot]
    trials = int(original[surfaces[0]]["trials"])
    fig = plt.figure(figsize=(7.4, 5.6), dpi=200)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.95)
    top = fig.add_subplot(grid[0])
    _successes_panel(top, original, residual, surfaces, name)
    top.legend(
        ncol=2,
        fontsize=8,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.2,
    )
    top.set_title(
        f"(a) {name}: successes per surface, {trials} trials each",
        loc="left",
        fontsize=9,
        color=INK,
        pad=34,
    )
    bottom = fig.add_subplot(grid[1])
    if robot == "host":
        softest = Surface.parse(str(cfg.softest_mattress)).label
        rows = [
            (f"{name}, original controller", original[softest]),
            (f"{name} with the residual policy", residual[softest]),
        ]
        title = (
            f"(b) {name}: outcomes on the softest mattress, {Surface.parse(softest).name}, "
            f"{trials} trials"
        )
    else:
        keys = [key for key, _, _ in OUTCOME_CLASSES]
        rows = [
            (
                f"{name}, original controller",
                {k: sum(r[k] for r in original.values()) for k in keys},
            ),
            (
                f"{name} with the residual policy",
                {k: sum(r[k] for r in residual.values()) for k in keys},
            ),
        ]
        total = sum(r["trials"] for r in original.values())
        title = f"(b) {name}: outcomes over all {total} trials"
    _outcomes_panel(bottom, rows)
    bottom.set_title(title, loc="left", fontsize=9, color=INK, pad=22)
    path = out / f"{robot}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _history(path) -> list[dict]:
    lines = (config.repo_path(path) / "history.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def training_figure(cfg, out: Path) -> Path:
    """Success fraction of the finished episodes over a window of updates, per surface group."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = int(cfg.training_window)
    groups = [
        ("rigid ground", lambda s: s.kind == "rigid", BLUE),
        ("trampolines a1, a2", lambda s: s.kind == "trampoline", ORANGE),
        ("mattress a8", lambda s: s.kind == "mattress" and s.name == "a8", SKY_BLUE),
        (
            "mattresses a16, a32, a64, then a1, a2, a4",
            lambda s: s.kind == "mattress" and s.name != "a8",
            REDDISH_PURPLE,
        ),
    ]
    runs = list(cfg.training.items())
    fig, axes = plt.subplots(len(runs), 1, figsize=(7.4, 2.5 * len(runs)), dpi=200, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (robot, path) in zip(axes, runs):
        history = _history(path)
        updates = np.asarray([r["update"] for r in history])
        for label, member, colour in groups:
            episodes = np.zeros(len(history))
            successes = np.zeros(len(history))
            for k, record in enumerate(history):
                for surface, row in record["by_surface"].items():
                    if member(Surface.parse(surface)):
                        episodes[k] += row["episodes"]
                        successes[k] += row["successes"]
            ends = np.arange(window, len(history) + 1, window)
            fraction = [
                successes[e - window : e].sum() / max(episodes[e - window : e].sum(), 1.0)
                for e in ends
            ]
            ax.plot(updates[ends - 1], fraction, color=colour, linewidth=1.6, label=label)
        run = OmegaConf.load(config.repo_path(path) / "config.yaml")
        for stage in run.train.curriculum or []:
            update = int(stage.from_update)
            if update > updates.max():
                continue
            replaced = ", ".join(Surface.parse(str(a)).name for a in stage.replace)
            added = ", ".join(Surface.parse(str(b)).name for b in stage.replace.values())
            ax.axvline(update, color=MUTED, linewidth=0.8, zorder=0)
            ax.annotate(
                f"from update {update}: mattresses {replaced} replaced by {added}",
                xy=(update, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(4, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=7,
                color=MUTED,
            )
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, float(updates.max()))
        ax.set_ylabel("successful episodes\n(fraction)", fontsize=8, color=INK)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_title(f"{NAMES[robot]} residual policy training", loc="left", fontsize=9, color=INK)
        _style(ax)
    axes[-1].set_xlabel(
        f"PPO update (each point: the {window} updates up to it)", fontsize=8, color=INK
    )
    axes[0].legend(
        ncol=2,
        fontsize=8,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.12),
        borderaxespad=0.2,
    )
    fig.tight_layout()
    path = out / "training.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    cfg = config.load(sys.argv[1:] if argv is None else list(argv), config_name="figures")
    out = config.repo_path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    for robot in ("host", "protomotions"):
        print(controller_figure(cfg, robot, out))
    print(training_figure(cfg, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
