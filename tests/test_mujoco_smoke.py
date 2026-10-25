"""Minimal MuJoCo lift-and-hold test with a synthetic parallel-jaw gripper.

Stand-in for the M2 setup (Menagerie Panda hand + ACRONYM meshes), which needs
downloaded assets. Checks that the pipeline (mocap-driven gripper, position-controlled
fingers, free object, contacts) can hold an object, and that an open gripper cannot.

The hand is a dynamic body welded to a mocap target. Making the hand itself the
mocap body does not work: mocap bodies are teleported, so their contacts have zero
velocity and friction cannot carry the object.
"""

import mujoco
import numpy as np

# Fingers (group 2) only collide with the object (group 1 | 2), not with the floor.
GRIPPER_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" contype="1" conaffinity="1"/>
    <body name="target" mocap="true" pos="0 0 0.02"/>
    <body name="hand" pos="0 0 0.02">
      <freejoint/>
      <geom type="box" size="0.02 0.05 0.005" pos="0 0 0.05" mass="0.5" contype="0" conaffinity="0"/>
      <body name="left" pos="0 0.04 0">
        <joint name="left" type="slide" axis="0 -1 0" range="0 0.04" damping="5"/>
        <geom type="box" size="0.01 0.005 0.015" mass="0.05" friction="1.5" contype="2" conaffinity="2"/>
      </body>
      <body name="right" pos="0 -0.04 0">
        <joint name="right" type="slide" axis="0 1 0" range="0 0.04" damping="5"/>
        <geom type="box" size="0.01 0.005 0.015" mass="0.05" friction="1.5" contype="2" conaffinity="2"/>
      </body>
    </body>
    <body name="object" pos="0 0 0.02">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1" friction="1.5" contype="3" conaffinity="3"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="target" body2="hand"/>
  </equality>
  <actuator>
    <position joint="left" kp="500" ctrlrange="0 0.04" forcerange="-20 20"/>
    <position joint="right" kp="500" ctrlrange="0 0.04" forcerange="-20 20"/>
  </actuator>
</mujoco>
"""


def lift_and_hold(close: bool, lift_height: float = 0.1) -> float:
    """Close (or not), lift the hand, hold; return the object's height gain."""
    model = mujoco.MjModel.from_xml_string(GRIPPER_XML)
    data = mujoco.MjData(model)
    obj = model.body("object").id
    dt = model.opt.timestep

    def run(seconds):
        for _ in range(int(seconds / dt)):
            mujoco.mj_step(model, data)

    run(0.3)  # settle
    z_start = data.xpos[obj, 2]

    data.ctrl[:] = 0.04 if close else 0.0
    run(0.5)

    z_hand = data.mocap_pos[0, 2]
    for z in np.linspace(z_hand, z_hand + lift_height, int(1.0 / dt)):
        data.mocap_pos[0, 2] = z
        mujoco.mj_step(model, data)
    run(0.5)  # hold

    return data.xpos[obj, 2] - z_start


def test_closed_gripper_lifts_object():
    assert lift_and_hold(close=True) > 0.08


def test_open_gripper_does_not_lift_object():
    assert abs(lift_and_hold(close=False)) < 0.01
