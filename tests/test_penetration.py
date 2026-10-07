import numpy as np
import pytest

from elastra import penetration


def test_sustained_needs_consecutive_steps(cfg):
    limit = float(cfg.penetration.limit_m)
    steps = int(cfg.penetration.sustained_steps)
    over = 2.0 * limit
    short = np.r_[np.zeros(5), np.full(steps - 1, over), np.zeros(3), np.full(steps - 1, over)]
    out = penetration.summarize(short, cfg.penetration)
    assert not out["sustained"]
    assert out["longest_run_over_limit"] == steps - 1
    assert out["steps_over_limit"] == 2 * (steps - 1)
    long = np.r_[np.zeros(5), np.full(steps, over)]
    assert penetration.summarize(long, cfg.penetration)["sustained"]
    assert penetration.summarize(long, cfg.penetration)["peak_m"] == over


def test_membrane_surface_matches_the_flex(cfg, needs_assets):
    """The crossing depth's membrane top is MuJoCo's flex surface: the triangles through
    the flex vertices, raised by the flex radius."""

    import mujoco

    from elastra.scene import build_scene
    from elastra.trampoline import flex_triangles, topology_of

    scene = build_scene(cfg, "protomotions", "trampoline/a1")
    model = scene.model
    data = mujoco.MjData(model)
    topo = topology_of(cfg.trampoline)
    rng = np.random.default_rng(0)
    deflection = 0.05 * rng.random(topo.active.size)
    deflection[topo.pinned] = 0.0
    data.qpos[scene.support_qpos] = deflection
    mujoco.mj_forward(model, data)
    surface = penetration.MembraneSurface(scene, topo)
    heights = surface.heights(data)

    start = int(model.flex_vertadr[0])
    vertices = data.flexvert_xpos[start : start + int(model.flex_vertnum[0])]
    vertex_of = {int(flat): k for k, flat in enumerate(np.flatnonzero(topo.active))}
    radius = float(cfg.trampoline.flex.radius_m)
    checked = 0
    for triangle in flex_triangles(topo)[::7]:
        corners = vertices[[vertex_of[v] for v in triangle]]
        weights = rng.dirichlet(np.ones(3))
        point = weights @ corners
        expected = point[2] + radius
        got = surface.surface_z(np.array([point[0]]), np.array([point[1]]), heights)[0]
        assert got == pytest.approx(expected, abs=1e-12)
        checked += 1
    assert checked > 100
