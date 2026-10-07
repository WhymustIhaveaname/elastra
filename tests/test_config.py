import re

import pytest

from elastra import config
from elastra.mattress import BED_FIELDS

BEDS = ("a1", "a2", "a4", "a8", "a16", "a32", "a64")


def test_every_bed_is_five_numbers(cfg):
    assert tuple(cfg.mattress.beds) == BEDS
    for name in BEDS:
        assert set(cfg.mattress.beds[name]) == set(BED_FIELDS)


def test_no_alpha_in_configuration():
    for path in config.CONF_DIR.rglob("*.yaml"):
        assert not re.search(r"alpha", path.read_text(), flags=re.IGNORECASE), path


@pytest.mark.parametrize("name", BEDS)
def test_refined_mattress_keeps_the_continuum(cfg, name):
    """Half the pitch: k, c, m scale with pitch^2 and b with 1 / pitch^2."""

    fine = config.load(["mattress=refined"]).mattress
    ratio = (float(fine.pitch_m) / float(cfg.mattress.pitch_m)) ** 2
    coarse_bed, fine_bed = cfg.mattress.beds[name], fine.beds[name]
    for field in ("k_n_per_m", "c_n_s_per_m", "cell_mass_kg"):
        assert float(fine_bed[field]) == pytest.approx(ratio * float(coarse_bed[field]), rel=1e-12)
    assert float(fine_bed.b_n_per_m) == pytest.approx(
        float(coarse_bed.b_n_per_m) / ratio, rel=1e-12
    )
    assert float(fine_bed.d_n_s_per_m) == float(coarse_bed.d_n_s_per_m) == 0.0
