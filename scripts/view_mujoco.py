"""Replay grasps of a MuJoCo evaluation (scripts/eval_mujoco.py) in the interactive MuJoCo viewer.

Takes the output directory of an evaluation, rebuilds the simulation of one test case with the
settings in its summary.json and replays the stored grasps in real time with `GraspSim.lift_and_hold`:
close without gravity, gravity on, lift, shake if the evaluation did, hold. A replay ends early when
the object is lost, as in the evaluation. The stored outcome of each grasp is printed next to the
outcome of the replay.

    uv run python scripts/view_mujoco.py results/mujoco/calibration/gt                 # first case
    uv run python scripts/view_mujoco.py results/mujoco/calibration/gt --list
    uv run python scripts/view_mujoco.py results/mujoco/equigraspflow_full/SE3_Euler-5 --case 15 --show fail
    uv run python scripts/view_mujoco.py results/mujoco/calibration/random --case 3 --grasps 0 7 --slow 4

Viewer: drag to rotate, right-drag to pan, scroll to zoom; closing the window ends the replay.
Keys:  Enter        replay the current grasp
       Right/Left   next / previous grasp
       3, 4         toggle the gripper / the closing region between the fingertips (contact check)
The grasps play one after the other; after the last one the viewer waits for a key.
Grasps that collide or have no contact are not simulated by the evaluation; they are shown for
a moment at the grasp pose. The object pose can differ from the evaluation by about a millimetre,
because the reference loader centres every case on a freshly sampled point cloud.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from graspgen import egf
from graspgen.sim.acronym import load_test_cases
from graspgen.sim.gripper import GripperConfig
from graspgen.sim.lift import GraspSim, SimConfig

KEY_STEPS = {257: 0, 262: 1, 263: -1}  # GLFW key codes: Enter, Right, Left -> change of grasp


class Jump(Exception):
    """A key was pressed: go to another grasp (or replay the current one)."""


def poll(viewer, pressed: list[int]) -> None:
    """Refresh the viewer; leave the running replay on a key press or when the window is closed."""
    if not viewer.is_running():
        raise KeyboardInterrupt
    if pressed:
        raise Jump(pressed.pop())
    viewer.sync()


def wait(viewer, pressed: list[int], seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        poll(viewer, pressed)
        time.sleep(0.02)


def replay(sim: GraspSim, viewer, pressed: list[int], slow: float) -> bool:
    """`GraspSim.lift_and_hold` from the state set by `reset`, stepped in real time."""
    last = [time.perf_counter()]

    def step():
        poll(viewer, pressed)
        time.sleep(max(0.0, sim.model.opt.timestep * slow - (time.perf_counter() - last[0])))
        last[0] = time.perf_counter()

    sim.on_step = step
    return sim.lift_and_hold()


def sim_config(summary: dict) -> SimConfig:
    """The simulation settings of an evaluation; settings it did not record keep their defaults.

    Except friction: evaluations from before it was a setting (no "friction", or null) ran with the larger of
    the gripper's friction (1.0) and the object's (1.0 in every ACRONYM file), so they replay at 1.0.
    """
    stored = dict(summary.get("sim", {}))
    gripper = stored.pop("gripper", {})
    if stored.get("friction") is None:
        stored["friction"] = max(gripper.get("friction", 1.0), 1.0)
    fields = {f.name for f in dataclasses.fields(SimConfig)} - {"gripper"}
    gripper_fields = {f.name for f in dataclasses.fields(GripperConfig)}
    return SimConfig(
        gripper=GripperConfig(**{k: v for k, v in gripper.items() if k in gripper_fields}),
        **{k: v for k, v in stored.items() if k in fields},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("eval_dir", type=Path, help="output directory of eval_mujoco.py")
    parser.add_argument("--list", action="store_true", help="list the cases with their rates and exit")
    parser.add_argument("--case", type=int, default=0, help="case index, see --list (default: 0)")
    parser.add_argument("--grasps", type=int, nargs="+", help="grasp indices (default: the first --num)")
    parser.add_argument("--show", choices=["all", "success", "fail"], default="all")
    parser.add_argument("--num", type=int, default=5, help="number of grasps to replay (default: 5)")
    parser.add_argument("--slow", type=float, default=1.0, help="slow-motion factor (default: 1)")
    args = parser.parse_args()

    per_case = pd.read_csv(args.eval_dir / "per_case.csv")
    if args.list:
        print(per_case[["category", "obj_idx", "rot_idx", "success_rate", "collision_rate"]].to_string())
        return
    stored = np.load(args.eval_dir / "grasps.npz")
    row = per_case.iloc[args.case]

    idxs = np.arange(stored["success"].shape[1])
    if args.grasps:
        idxs = np.array(args.grasps)
    elif args.show != "all":
        idxs = idxs[stored["success"][args.case] == (args.show == "success")]
    idxs = idxs if args.grasps else idxs[: args.num]
    if len(idxs) == 0:
        raise SystemExit(f"case {args.case} has no grasp with --show {args.show}")

    # The evaluation keeps the first objects of every category; load just as many
    max_objects = int(per_case.groupby("category").obj_idx.nunique().max())
    summary = json.loads((args.eval_dir / "summary.json").read_text())
    coacd_threshold = summary.get("sim", {}).get("coacd_threshold", 0.05)
    cases, _ = load_test_cases(
        egf.load_config("equigraspflow_full").data.test.dataset, max_objects, coacd_threshold
    )
    case = next(
        c for c in cases if (c.category, c.obj_idx, c.rot_idx) == (row.category, row.obj_idx, row.rot_idx)
    )
    sim = GraspSim(case.parts, case.mass, sim_config(summary))
    print(f"case {args.case}: {row.category} {row.obj_idx}, rotation {row.rot_idx}, {case.mass * 1e3:.0f} g")
    if sim.cfg.shake:
        print(f"shake: {sim.cfg.shake_amplitude * 1e3:g} mm at {sim.cfg.shake_frequency:g} Hz")
    print(f"friction: {sim.cfg.friction:g}")

    def show(i: int) -> None:
        sim.reset(stored["Ts_grasp"][args.case, i])
        collision, contact = stored["collision"][args.case, i], stored["contact"][args.case, i]
        if collision or not contact:
            print(f"grasp {i}: {'collision' if collision else 'no contact'}, not simulated")
            wait(viewer, pressed, 1.5 * args.slow)
            return
        wait(viewer, pressed, 0.5)
        result = replay(sim, viewer, pressed, args.slow)
        print(f"grasp {i}: stored success={bool(stored['success'][args.case, i])}, replay={result}")
        wait(viewer, pressed, 0.5)

    pressed = []  # filled by the viewer thread, read by `poll`
    sim.reset(stored["Ts_grasp"][args.case, idxs[0]])
    with mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=lambda key: key in KEY_STEPS and pressed.append(KEY_STEPS[key])
    ) as viewer:
        viewer.opt.geomgroup[3] = 1  # the gripper's collision geoms; the viewer hides group 3 by default
        viewer.cam.lookat[:] = (0.0, 0.0, 0.07)
        viewer.cam.distance = 0.8
        pos = 0
        try:
            while True:
                try:
                    show(idxs[pos])
                    if pos == len(idxs) - 1:
                        print("last grasp; Enter replays, Left/Right changes the grasp")
                        wait(viewer, pressed, float("inf"))
                    pos += 1
                except Jump as jump:
                    pos = int(np.clip(pos + jump.args[0], 0, len(idxs) - 1))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
