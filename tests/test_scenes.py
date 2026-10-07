import mujoco
import numpy as np

from elastra.scene import Surface, build_scene, strip_robot, surface_xml


def _joints(model, prefix):
    return [
        j
        for j in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(prefix)
    ]


def test_mattress_structure(cfg, needs_assets):
    scene = build_scene(cfg, "protomotions", "mattress/a1")
    model = scene.model
    nx, ny = int(cfg.mattress.grid.nx), int(cfg.mattress.grid.ny)
    assert len(_joints(model, "mattress_slide_")) == nx * ny
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, t) or "" for t in range(model.ntendon)
    ]
    assert sum(n.startswith("mattress_curvature_") for n in names) == (nx - 2) * (ny - 2)
    assert scene.robot_qpos.size == 29


def test_rigid_ground_has_no_moving_cells(cfg, needs_assets):
    scene = build_scene(cfg, "host", "rigid")
    assert not _joints(scene.model, "mattress_slide_")
    assert scene.robot_qpos.size == 23


def test_unloaded_mattress_sags_uniformly(cfg, needs_assets):
    """Every stencil row sums to zero, so the cells' weight sags the bed by m g / k."""

    model = mujoco.MjModel.from_xml_string(
        strip_robot(surface_xml(cfg, Surface.parse("mattress/a1")), "bed")
    )
    data = mujoco.MjData(model)
    for _ in range(int(round(4.0 / model.opt.timestep))):
        mujoco.mj_step(model, data)
    x = data.qpos[[model.jnt_qposadr[j] for j in _joints(model, "mattress_slide_")]]
    bed = cfg.mattress.beds.a1
    expected = float(bed.cell_mass_kg) * 9.81 / float(bed.k_n_per_m)
    np.testing.assert_allclose(x, expected, rtol=1e-4)
