"""MuJoCo grasp evaluation (graspgen.sim) on synthetic objects, without the dataset.

Grasps are built by hand: the gripper approaches along -z (from above) with its fingertip pads
centred at a given point and its closing axis, the x axis of an ACRONYM grasp pose, at a given yaw.
"""

import mujoco
import numpy as np
import pytest
import trimesh

from graspgen.sim.decompose import decompose
from graspgen.sim.gripper import ASSET_DIR, GripperConfig, gripper_spec
from graspgen.sim.lift import GraspSim, SimConfig

PAD_DEPTH = 0.103  # distance of the fingertip pad centre from the hand base along the approach axis


def top_grasp(pad_centre, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, s, 0], [s, -c, 0], [0, 0, -1]]  # x: closing axis, z: approach
    T[:3, 3] = np.add(pad_centre, [0, 0, PAD_DEPTH])
    return T


@pytest.fixture(scope="module")
def plate():
    """Upright plate, 2 cm thick along x, 12 cm wide along y, 10 cm high."""
    return GraspSim([trimesh.creation.box((0.02, 0.12, 0.10)).vertices], mass=0.2)


def evaluate(sim, T):
    return {key: value[0] for key, value in sim.evaluate(T[None]).items()}


def test_object_mass(plate):
    assert plate.model.body("object").mass[0] == pytest.approx(0.2, rel=1e-4)


def test_grasp_across_plate_succeeds(plate):
    out = evaluate(plate, top_grasp((0, 0, 0.03)))
    assert not out["collision"] and out["contact"] and out["success"]


def test_closing_axis_is_grasp_x(plate):
    # Turned by 90 deg the fingers would have to straddle the 12 cm width: they hit the plate.
    out = evaluate(plate, top_grasp((0, 0, 0.03), yaw=np.pi / 2))
    assert out["collision"] and not out["success"]


def test_grasp_beside_plate_fails(plate):
    out = evaluate(plate, top_grasp((0.15, 0, 0.03)))
    assert not out["collision"] and not out["contact"] and not out["success"]


def test_palm_in_object_collides(plate):
    out = evaluate(plate, top_grasp((0, 0, -0.03)))
    assert out["collision"] and out["penetration"] > 0.01 and not out["success"]


def test_heavy_object_slips():
    # 2 x 40 N grip force with friction 1 cannot carry 20 kg.
    heavy = GraspSim([trimesh.creation.box((0.02, 0.12, 0.10)).vertices], mass=20.0)
    out = evaluate(heavy, top_grasp((0, 0, 0.03)))
    assert out["contact"] and not out["success"]


def test_results_do_not_depend_on_order(plate):
    Ts = np.stack([top_grasp((0, 0, 0.03)), top_grasp((0.15, 0, 0.03)), top_grasp((0, 0.03, 0.03))])
    forward, backward = plate.evaluate(Ts), plate.evaluate(Ts[::-1])
    for key in forward:
        np.testing.assert_array_equal(forward[key], backward[key][::-1])


def test_decomposition_keeps_cavity(tmp_path):
    """A grasp on the wall of a ring puts one finger inside it, which only works with convex parts."""
    ring = trimesh.creation.annulus(r_min=0.035, r_max=0.04, height=0.08)
    wall_grasp = top_grasp((0.0375, 0, 0.03))

    hull_only = GraspSim([ring.vertices], mass=0.15)
    assert evaluate(hull_only, wall_grasp)["collision"]

    parts = decompose(ring.vertices, ring.faces, cache_dir=tmp_path)
    assert len(parts) > 1
    out = evaluate(GraspSim(parts, mass=0.15), wall_grasp)
    assert not out["collision"] and out["success"]

    cached = decompose(ring.vertices, ring.faces, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    for part, part_cached in zip(parts, cached, strict=True):
        np.testing.assert_array_equal(part, part_cached)


def test_simulate_colliding_is_optional(plate):
    colliding = top_grasp((0, 0, -0.03))
    sim = GraspSim([trimesh.creation.box((0.02, 0.12, 0.10)).vertices], 0.2, cfg=SimConfig(collision_tol=1.0))
    assert not evaluate(sim, colliding)["collision"]  # tolerance above the penetration depth
    assert evaluate(plate, colliding)["collision"]


def test_gripper_config_reaches_model():
    cfg = SimConfig(timestep=0.001, gripper=GripperConfig(grip_force=10.0, close_speed=0.05, friction=0.4))
    model = GraspSim([trimesh.creation.box((0.02, 0.12, 0.10)).vertices], 0.2, cfg=cfg).model

    assert model.opt.timestep == 0.001
    for name in ("finger1", "finger2"):
        np.testing.assert_allclose(model.actuator(name).forcerange, [-10.0, 10.0])
    assert model.joint("finger_joint1").damping[0] == pytest.approx(10.0 / 0.05)
    assert model.geom("hand").friction[0] == 0.4


def test_hand_xml_holds_the_defaults():
    """hand.xml can be opened on its own; it must then behave like the default configuration."""
    from_file = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "hand.xml"))
    configured = gripper_spec(GripperConfig(), SimConfig().timestep).compile()

    assert from_file.opt.timestep == configured.opt.timestep
    for field in ("dof_damping", "geom_friction", "actuator_gainprm", "actuator_forcerange"):
        np.testing.assert_allclose(getattr(from_file, field), getattr(configured, field))
    np.testing.assert_allclose(from_file.actuator_biasprm, configured.actuator_biasprm, rtol=1e-4)
