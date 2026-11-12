"""Floating Franka Panda hand for the MuJoCo grasp evaluation.

The model is assets/panda_hand/hand.xml. Geometry, inertias and the finger coupling follow
`franka_emika_panda/hand.xml` of MuJoCo Menagerie (collision geoms only, see the README next to
it). What differs is how the hand is held and driven, because here it is not mounted on an arm:

- The hand is rigidly attached to a `lift` body that has one vertical slide joint. Its pose on that
  body is written into the model for every grasp, so it sits exactly at the grasp pose and lifting
  moves it with a real velocity. (A mocap body would be teleported, and contacts of a teleported
  body carry no friction load; a weld to a mocap target is soft and adds six degrees of freedom.)
- Each finger has a position actuator with a force limit, which makes the grip force an explicit
  parameter. Menagerie's actuator saturates at about 2 N per finger, too little for the heavier
  ACRONYM objects.

Frames: ACRONYM grasp poses are poses of the hand base with z the approach direction and the fingers
opening along x. The Panda hand opens along y, hence `T_HAND_FROM_GRASP`.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).parent / "assets" / "panda_hand"

FINGER_OPEN = 0.04  # joint position of one finger when fully open [m]; the opening width is twice that

# Rotation by +90 deg about z: maps the hand's opening axis (y) onto ACRONYM's (x).
T_HAND_FROM_GRASP = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@dataclass(frozen=True)
class GripperConfig:
    grip_force: float = 40.0  # force limit of each finger [N]
    close_speed: float = 0.1  # speed of an unobstructed finger [m/s]
    friction: float = 1.0  # sliding friction of the finger geoms
    lift_kp: float = 20000.0  # stiffness of the lift axis [N/m]


def gripper_spec(cfg: GripperConfig, timestep: float) -> mujoco.MjSpec:
    """The hand on its lift axis, without an object, as an editable model specification.

    hand.xml carries the defaults of `GripperConfig`; everything that depends on the configuration
    is overwritten here, so the values in the file never take effect in an evaluation.
    """
    spec = mujoco.MjSpec.from_file(str(ASSET_DIR / "hand.xml"))
    spec.option.timestep = timestep

    for geom in spec.geoms:
        geom.friction[0] = cfg.friction

    # A force-limited finger accelerates until joint damping balances the actuator, which sets the
    # closing speed. The gain saturates the actuator from 5 mm of position error on.
    finger_kp = cfg.grip_force / 0.005
    for joint, actuator in [("finger_joint1", "finger1"), ("finger_joint2", "finger2")]:
        spec.joint(joint).damping[0] = cfg.grip_force / cfg.close_speed
        _set_position_gains(spec.actuator(actuator), kp=finger_kp)
        spec.actuator(actuator).forcerange = [-cfg.grip_force, cfg.grip_force]

    # Critical damping for about 1 kg of hand and object
    _set_position_gains(spec.actuator("lift"), kp=cfg.lift_kp, kv=2.0 * np.sqrt(cfg.lift_kp))
    return spec


def _set_position_gains(actuator, kp: float, kv: float = 0.0) -> None:
    """What `<position kp= kv=>` expands to: force = kp * (ctrl - q) - kv * qvel."""
    actuator.gainprm[0] = kp
    actuator.biasprm[1] = -kp
    actuator.biasprm[2] = -kv
