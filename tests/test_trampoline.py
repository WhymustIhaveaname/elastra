import numpy as np
import pytest

from elastra import config, trampoline


@pytest.mark.parametrize("variant, free", [("default", 377), ("refined", 1661)])
def test_free_nodes(variant, free):
    topo = trampoline.topology_of(config.load([f"trampoline={variant}"]).trampoline)
    assert int(np.count_nonzero(topo.active & ~topo.pinned)) == free


@pytest.mark.parametrize("name", ["a1", "a2"])
def test_membrane_force_is_linear_for_small_displacements(cfg, name):
    force = trampoline.MembraneForce(cfg.trampoline, name)
    nodes = force.topology.active.size
    assert np.all(force.qfrc(np.zeros((1, nodes))) == 0.0)
    rng = np.random.default_rng(0)
    x = 1.0e-3 * rng.standard_normal((2, nodes))
    x[:, force.topology.pinned] = 0.0
    out = force.qfrc(x)
    # graph Laplacian of the edges: the cubic term is negligible at 1 mm
    edges = force.topology.edges
    expected = np.zeros_like(x)
    delta = x[:, edges[:, 0]] - x[:, edges[:, 1]]
    np.add.at(expected.T, edges[:, 0], (-force.k_lin * delta).T)
    np.add.at(expected.T, edges[:, 1], (force.k_lin * delta).T)
    expected[:, force.topology.pinned] = 0.0
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(force.qfrc(-x), -out, rtol=1e-12, atol=1e-15)


def test_diverged_step_is_reported(cfg):
    force = trampoline.MembraneForce(cfg.trampoline, "a1")
    x = np.zeros((1, force.topology.active.size))
    x[0, 0] = 2.0 * float(cfg.trampoline.radius_m)
    with pytest.raises(FloatingPointError):
        force.qfrc(x)
