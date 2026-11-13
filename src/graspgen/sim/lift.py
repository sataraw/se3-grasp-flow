"""Grasp evaluation in MuJoCo: collision and contact checks and a lift-and-hold test.

One `GraspSim` holds one object and is reused for all grasps on it. A grasp is a pose of the
gripper in the object's frame (ACRONYM convention, metres). For each grasp:

1. collision  The open hand is placed at the grasp pose. The grasp collides if the hand penetrates
              the object by more than `collision_tol`.
2. contact    The grasp has contact if the object reaches into the volume swept by the fingertip
              pads, i.e. there is something to close on.
3. success    Lift-and-hold. Without gravity (the object floats, as in the ACRONYM and Isaac Gym
              protocols) the fingers close. Then gravity is switched on, the hand is raised by
              `lift_height` and held. The grasp succeeds if afterwards both fingers still touch
              the object and the object has followed the hand.

Colliding grasps are failures and are not simulated unless `simulate_colliding` is set: starting
a simulation in penetration measures the contact solver's push-out, not the grasp.

Gravity acts along -z of the frame the grasps are given in, so with randomly rotated test objects
the load direction relative to the grasp is random as well.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from graspgen.sim.decompose import convex_hull
from graspgen.sim.gripper import FINGER_OPEN, T_HAND_FROM_GRASP, GripperConfig, gripper_spec

GRAVITY = 9.81
CHECK_INTERVAL = 25  # steps between checks for a lost object


@dataclass(frozen=True)
class SimConfig:
    timestep: float = 0.002
    gripper: GripperConfig = field(default_factory=GripperConfig)
    collision_tol: float = 0.001  # tolerated penetration of the open hand [m]
    simulate_colliding: bool = False
    close_time: float = 0.8  # fingers close, no gravity [s]
    settle_time: float = 0.2  # gravity on, hand at rest [s]
    lift_height: float = 0.15  # [m]
    lift_time: float = 1.0  # [s]
    hold_time: float = 1.0  # [s]
    min_lift_fraction: float = 0.5  # the object must rise by this fraction of lift_height


class GraspSim:
    """A free-floating object, given as convex parts, and the Panda hand."""

    def __init__(self, parts: list[np.ndarray], mass: float, friction: float = 1.0, cfg: SimConfig = None):
        self.cfg = cfg or SimConfig()

        hulls = [hull for part in parts if (hull := convex_hull(np.asarray(part, dtype=np.float64)))]
        if not hulls:
            raise ValueError("object has no part with a volume")
        density = mass / sum(volume for _, volume in hulls)

        spec = gripper_spec(self.cfg.gripper, self.cfg.timestep)
        hand = spec.geom("hand")  # the object gets the gripper's (stiff) contact parameters
        body = spec.worldbody.add_body(name="object")
        body.add_freejoint()
        for i, (vertices, _) in enumerate(hulls):
            # Vertices only: MuJoCo builds the convex hull, which is also what it collides with.
            spec.add_mesh(name=f"part{i}", uservert=vertices.ravel())
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"part{i}",
                density=density,
                friction=[friction, 0.005, 0.0001],
                condim=4,
                solref=hand.solref,
                solimp=hand.solimp,
                contype=0,
                conaffinity=1,
            )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        m = self.model
        self._hand = m.body("hand").id
        self._object = m.body("object").id
        self._fingers = (m.body("left_finger").id, m.body("right_finger").id)
        self._closing_region = m.geom("closing_region").id
        self._object_geoms = np.flatnonzero(m.geom_bodyid == self._object)
        self._finger_qpos = [m.joint(name).qposadr[0] for name in ("finger_joint1", "finger_joint2")]
        self._lift_ctrl = m.actuator("lift").id
        self._finger_ctrl = [m.actuator("finger1").id, m.actuator("finger2").id]

    # ------------------------------------------------------------------ state

    def reset(self, T_grasp: np.ndarray) -> None:
        """Open hand at the grasp pose, object at rest at the origin of its frame, no gravity."""
        m, d = self.model, self.data
        T_hand = np.asarray(T_grasp, dtype=np.float64) @ T_HAND_FROM_GRASP
        m.body_pos[self._hand] = T_hand[:3, 3]
        mujoco.mju_mat2Quat(m.body_quat[self._hand], T_hand[:3, :3].ravel())
        m.opt.gravity[:] = 0.0

        mujoco.mj_resetData(m, d)
        d.qpos[self._finger_qpos] = FINGER_OPEN
        d.ctrl[self._finger_ctrl] = FINGER_OPEN
        mujoco.mj_forward(m, d)

    def _object_contacts(self):
        """(other body, penetration depth) for every contact that involves the object."""
        m, d = self.model, self.data
        for contact in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
            if b1 == self._object:
                yield b2, -contact.dist
            elif b2 == self._object:
                yield b1, -contact.dist

    def penetration(self) -> float:
        """Deepest penetration of the hand into the object in the current state [m]."""
        return max((depth for _, depth in self._object_contacts()), default=0.0)

    def fingers_touching(self) -> int:
        """Number of fingers (0, 1 or 2) in contact with the object."""
        touching = {body for body, _ in self._object_contacts()}
        return sum(finger in touching for finger in self._fingers)

    def object_between_fingers(self) -> bool:
        m, d = self.model, self.data
        return any(
            mujoco.mj_geomDistance(m, d, self._closing_region, geom, 0.0, None) < 0.0
            for geom in self._object_geoms
        )

    def _unstable(self) -> bool:
        return self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0

    # ------------------------------------------------------------- evaluation

    def _step(self, duration: float, lift: tuple[float, float] | None = None) -> bool:
        """Advance by `duration`; `lift=(start, end)` ramps the lift target smoothly.

        Returns False as soon as the object is lost (no finger touches it) or the simulation
        diverges.
        """
        m, d = self.model, self.data
        steps = round(duration / m.opt.timestep)
        for i in range(steps):
            if lift is not None:
                s = (i + 1) / steps
                d.ctrl[self._lift_ctrl] = lift[0] + (lift[1] - lift[0]) * s * s * (3.0 - 2.0 * s)
            mujoco.mj_step(m, d)
            if (i + 1) % CHECK_INTERVAL == 0 and (self._unstable() or self.fingers_touching() == 0):
                return False
        return not self._unstable()

    def lift_and_hold(self) -> bool:
        """Close, lift, hold from the state set by `reset`; True if the object is still grasped."""
        cfg, m, d = self.cfg, self.model, self.data
        d.ctrl[self._finger_ctrl] = m.actuator_ctrlrange[self._finger_ctrl, 0]
        mujoco.mj_step(m, d, round(cfg.close_time / m.opt.timestep))
        if self._unstable() or self.fingers_touching() < 2:
            return False

        m.opt.gravity[2] = -GRAVITY
        if not self._step(cfg.settle_time):
            return False
        z_start = d.xipos[self._object, 2]
        if not self._step(cfg.lift_time, lift=(0.0, cfg.lift_height)):
            return False
        if not self._step(cfg.hold_time):
            return False

        lifted = d.xipos[self._object, 2] - z_start > cfg.min_lift_fraction * cfg.lift_height
        return bool(lifted) and self.fingers_touching() == 2

    def evaluate(self, Ts_grasp: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate grasps (N, 4, 4); returns per-grasp `collision`, `contact`, `success`, `penetration`."""
        n = len(Ts_grasp)
        out = {
            "collision": np.zeros(n, dtype=bool),
            "contact": np.zeros(n, dtype=bool),
            "success": np.zeros(n, dtype=bool),
            "penetration": np.zeros(n),
        }
        for i, T_grasp in enumerate(Ts_grasp):
            self.reset(T_grasp)
            out["penetration"][i] = self.penetration()
            out["collision"][i] = out["penetration"][i] > self.cfg.collision_tol
            out["contact"][i] = self.object_between_fingers()
            if out["contact"][i] and (self.cfg.simulate_colliding or not out["collision"][i]):
                out["success"][i] = self.lift_and_hold()
        return out


def evaluate_object(
    parts: list[np.ndarray], mass: float, friction: float, Ts_grasp: np.ndarray, cfg: SimConfig = None
) -> dict[str, np.ndarray]:
    """Build the simulation of one object and evaluate grasps on it (picklable, for process pools)."""
    return GraspSim(parts, mass, friction, cfg).evaluate(Ts_grasp)
