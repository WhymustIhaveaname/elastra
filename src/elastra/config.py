"""Load the Hydra configuration tree in ``conf/``.

Every command takes ``key=value`` overrides on the command line, for example

    elastra-evaluate controller=host mattress.beds.a4.c_n_s_per_m=8.0

Relative paths in the configuration (``paths.*``, ``policy``, ``out``) are relative
to the repository root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "conf"


def load(overrides: Sequence[str] = (), *, config_name: str = "config") -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.set_readonly(cfg, False)
    return cfg


def repo_path(value: str | Path) -> Path:
    """``value`` as an absolute path; a relative path is taken from the repository root."""

    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def assets_dir(cfg: DictConfig) -> Path:
    return repo_path(cfg.paths.assets)


def data_dir(cfg: DictConfig) -> Path:
    return repo_path(cfg.paths.data)


def outputs_dir(cfg: DictConfig) -> Path:
    return repo_path(cfg.paths.outputs)
