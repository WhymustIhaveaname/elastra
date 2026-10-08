import numpy as np
import pytest

from elastra import config, indentation, mattress


@pytest.fixture(scope="module")
def indentation_cfg():
    return config.load([], config_name="indentation")


def test_pad_face(indentation_cfg):
    pad = indentation.Pad.from_cfg(indentation_cfg.pad)
    assert pad.height(0.0) == 0.0
    # the face and the edge meet without a step, and the edge ends on the cylinder r = a
    r_d = pad.face_edge_radius_m
    np.testing.assert_allclose(pad.height(r_d - 1e-9), pad.height(r_d + 1e-9), atol=1e-8)
    np.testing.assert_allclose(pad.height(pad.radius_m), pad.edge_centre[1], atol=1e-12)
    assert np.isinf(pad.height(pad.radius_m + 1e-6))
    # every mesh vertex below the cylinder top lies on the face
    vertices = pad.vertices(24, 96, 0.15)
    face = vertices[vertices[:, 2] <= pad.edge_centre[1] + 1e-12]
    # (the last edge ring is on r = a up to rounding)
    radial = np.minimum(np.hypot(face[:, 0], face[:, 1]), pad.radius_m)
    np.testing.assert_allclose(pad.height(radial), face[:, 2], atol=1e-12)


def test_load_without_curvature_term(indentation_cfg, needs_assets):
    """With b = 0 every cell under the pad is pressed down to the pad independently: the
    load is k times the sum over the cells of the pad depth below their tops' nearest point
    to the pad axis."""

    cfg = config.load(
        ["mattress.beds.a1.b_n_per_m=0.0", "loading.hold_s=0.5"], config_name="indentation"
    )
    run = indentation.indent(cfg, 1.0, [20.0, 40.0])
    pad = indentation.Pad.from_cfg(cfg.pad)
    half = 0.5 * float(cfg.mattress.pitch_m)
    offset = np.abs(mattress.cell_centres(cfg.mattress) - np.asarray(cfg.pad.position_xy_m))
    nearest = np.hypot(*np.moveaxis(np.maximum(offset - half, 0.0), -1, 0)).ravel()
    k = float(cfg.mattress.beds.a1.k_n_per_m)
    for row in run["rows"]:
        depth = row["deflection_mm"] / 1000.0
        pressed = depth - pad.height(nearest)
        expected = k * np.sum(np.maximum(pressed, 0.0))
        assert row["settled"]
        assert row["contacts"] == np.count_nonzero(pressed > 0.0)
        np.testing.assert_allclose(row["load_n"], expected, rtol=2e-3)
