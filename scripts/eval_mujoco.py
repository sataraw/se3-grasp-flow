"""Evaluate grasps on the test split in MuJoCo: collision, contact and lift-and-hold success.

The test cases are those of the EMD evaluation (scripts/eval_egf.py): every test object in every
test rotation, with the same point cloud. For each case `--num-grasps` grasps are evaluated with
the Panda hand (see graspgen/sim/lift.py for the protocol). Grasp sources:

    gt        successful ACRONYM grasps, drawn at random     -> should succeed (calibration)
    random    samples of the model prior (uniform rotation,
              Gaussian position around the centroid)         -> should fail (calibration)
    egf       grasps sampled from an EquiGraspFlow run

    uv run python scripts/eval_mujoco.py --source gt random
    uv run python scripts/eval_mujoco.py                                    # released checkpoint, RK4-MK 20
    uv run python scripts/eval_mujoco.py --solver SE3_Euler --steps 1 2 5 10 20   # NFE sweep
    uv run python scripts/eval_mujoco.py --source gt --max-objects 2 --num-grasps 20   # quick look

Outputs: results/mujoco/calibration/<source>/ and results/mujoco/<run-dir name>/<solver>-<steps>/ with

    summary.json     rates (mean over all grasps and per category), latency, settings
    per_case.csv     rates of every (object, rotation)
    grasps.npz       evaluated poses [m] and per-grasp outcomes, (cases, num_grasps, ...)

Metrics, as fractions of all evaluated grasps:

    collision_rate   open hand penetrates the object
    contact_rate     object reaches between the fingers
    success_rate     object still held after closing, lifting and holding (collisions count as failures)

Latency (egf only), wall clock on `--device`:

    single_grasp_ms     one point cloud, one grasp, median over --latency-repeats calls
    batch_ms            median over the evaluation's sample calls (all rotations of an object at once)
    batch_per_grasp_ms  batch_ms divided by the number of grasps in the batch

The first run decomposes the test meshes into convex parts (cached in data/cache/coacd).
"""

import argparse
import copy
import dataclasses
import json
import multiprocessing
import os
import random
import time
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from graspgen import egf
from graspgen.paths import CHECKPOINT_DIR, RESULTS_DIR
from graspgen.sim.acronym import GraspCase, load_test_cases
from graspgen.sim.gripper import GripperConfig
from graspgen.sim.lift import SimConfig, evaluate_object

STAGES_PER_STEP = {"SE3_Euler": 1, "SE3_RK_mk": 4}
METRICS = ["success", "collision", "contact"]


def load_run_config(run_dir: Path):
    ymls = sorted(run_dir.glob("*.yml"))
    if len(ymls) != 1:
        raise FileNotFoundError(f"expected exactly one .yml in {run_dir}, found {[p.name for p in ymls]}")
    return OmegaConf.load(ymls[0])


def synchronize(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def timed_sample(model, pc, nums_grasps, device):
    synchronize(device)
    start = time.perf_counter()
    Ts = model.sample(pc, nums_grasps)
    synchronize(device)
    return Ts, time.perf_counter() - start


def gt_grasps(cases: list[GraspCase], num_grasps: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [
        case.Ts_gt[rng.choice(len(case.Ts_gt), size=num_grasps, replace=len(case.Ts_gt) < num_grasps)]
        for case in cases
    ]


def random_grasps(cases: list[GraspCase], num_grasps: int, scale: float) -> list[np.ndarray]:
    grasps = []
    for _ in cases:
        Ts = egf.SO3_uniform_R3_normal(num_grasps, "cpu").double().numpy()
        Ts[:, :3, 3] /= scale
        grasps.append(Ts)
    return grasps


def egf_grasps(model, cases: list[GraspCase], num_grasps: int, scale: float, device: str, repeats: int):
    """Sample grasps for all cases, object by object as in the reference test; returns grasps and latency."""
    model.eval()

    # Single-grasp latency, after a warm-up
    pc = torch.from_numpy(cases[0].pc).unsqueeze(0).to(device)
    one = torch.tensor([1], device=device)
    for _ in range(3):
        model.sample(pc, one)
    single = [timed_sample(model, pc, one, device)[1] for _ in range(repeats)]

    grasps, batch = [], []
    by_object = groupby(cases, key=lambda case: (case.category, case.obj_idx))
    for _, group in tqdm(list((key, list(group)) for key, group in by_object), desc="Sampling", leave=False):
        pc = torch.from_numpy(np.stack([case.pc for case in group])).to(device)
        nums_grasps = torch.full((len(group),), num_grasps, device=device)
        Ts_rots, seconds = timed_sample(model, pc, nums_grasps, device)
        batch.append(seconds)
        for Ts in Ts_rots:
            Ts = Ts.double().cpu().numpy()
            Ts[:, :3, 3] /= scale
            grasps.append(Ts)

    batch_ms = 1e3 * float(np.median(batch))
    latency = {
        "device": device,
        "single_grasp_ms": 1e3 * float(np.median(single)),
        "batch_ms": batch_ms,
        "batch_per_grasp_ms": batch_ms / (num_grasps * len(cases) / len(batch)),
    }
    return grasps, latency


def simulate(pool, cases: list[GraspCase], grasps: list[np.ndarray], cfg: SimConfig) -> list[dict]:
    tasks = [(case.parts, case.mass, case.friction, Ts, cfg) for case, Ts in zip(cases, grasps, strict=True)]
    jobs = [pool.apply_async(evaluate_object, task) for task in tasks]
    return [job.get() for job in tqdm(jobs, desc="Simulating", leave=False)]


def summarize(cases: list[GraspCase], results: list[dict]) -> tuple[dict, pd.DataFrame]:
    per_case = pd.DataFrame(
        [
            {
                "category": case.category,
                "obj_idx": case.obj_idx,
                "rot_idx": case.rot_idx,
                "name": case.name,
                "num_parts": len(case.parts),
                "mass": case.mass,
                **{f"{key}_rate": result[key].mean() for key in METRICS},
            }
            for case, result in zip(cases, results, strict=True)
        ]
    )
    # Every case has the same number of grasps, so the mean over cases is the mean over grasps.
    metrics = {
        f"{key}_rate": {
            "mean": float(per_case[f"{key}_rate"].mean()),
            "per_category": per_case.groupby("category", sort=False)[f"{key}_rate"].mean().to_dict(),
        }
        for key in METRICS
    }
    return metrics, per_case


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source", nargs="+", choices=["gt", "random", "egf"], default=["egf"])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=CHECKPOINT_DIR / "equigraspflow_full",
        help="directory with the training config (.yml) and checkpoint",
    )
    parser.add_argument("--checkpoint", default="model_best_val_loss.pkl")
    parser.add_argument("--solver", choices=sorted(STAGES_PER_STEP), help="default: from the run config")
    parser.add_argument("--steps", type=int, nargs="+", help="ODE steps, several for a sweep")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist")
    parser.add_argument("--device", help="GPU index, 'cpu' or 'mps' (default: GPU 0 if available)")
    parser.add_argument("--num-grasps", type=int, default=100, help="grasps per (object, rotation)")
    parser.add_argument("--max-objects", type=int, help="only the first objects of each category")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--logdir", type=Path, default=RESULTS_DIR / "mujoco")
    sim = parser.add_argument_group("simulation")
    sim.add_argument("--grip-force", type=float, default=GripperConfig.grip_force, help="per finger [N]")
    sim.add_argument("--lift-height", type=float, default=SimConfig.lift_height, help="[m]")
    sim.add_argument("--collision-tol", type=float, default=SimConfig.collision_tol, help="[m]")
    sim.add_argument("--simulate-colliding", action="store_true", help="lift colliding grasps too")
    sim.add_argument("--coacd-threshold", type=float, default=0.05, help="concavity tolerance")
    args = parser.parse_args()

    # The calibration sources need the test set only, which is the same in every full-point-cloud run
    base = load_run_config(args.run_dir) if "egf" in args.source else egf.load_config("equigraspflow_full")
    cfg = OmegaConf.merge(base, OmegaConf.from_dotlist(args.set))
    cfg.model.checkpoint = str(args.run_dir / args.checkpoint)
    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    device = args.device if args.device in ("cpu", "mps") else f"cuda:{args.device}"
    if args.solver:
        cfg.model.ode_solver.name = args.solver
    solver = cfg.model.ode_solver.name
    sim_cfg = SimConfig(
        gripper=GripperConfig(grip_force=args.grip_force),
        lift_height=args.lift_height,
        collision_tol=args.collision_tol,
        simulate_colliding=args.simulate_colliding,
    )

    # Same seeding as the reference test scripts, so the test point clouds match scripts/eval_egf.py
    seed = cfg.get("seed", 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)

    cases, scale = load_test_cases(cfg.data.test.dataset, args.max_objects, args.coacd_threshold)
    print(f"{len(cases)} test cases x {args.num_grasps} grasps, {args.workers} workers")

    # (source, steps): one evaluation each
    settings = [(source, None) for source in args.source if source != "egf"]
    if "egf" in args.source:
        settings += [("egf", steps) for steps in args.steps or [cfg.model.ode_solver.num_steps]]

    rows = []
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for source, steps in settings:
            summary = {"source": source, "num_cases": len(cases), "num_grasps": args.num_grasps}
            if source == "gt":
                name, logdir = "gt", args.logdir / "calibration" / "gt"
                grasps = gt_grasps(cases, args.num_grasps, rng)
            elif source == "random":
                name, logdir = "random", args.logdir / "calibration" / "random"
                grasps = random_grasps(cases, args.num_grasps, scale)
            else:
                name = f"{solver}-{steps}"
                logdir = args.logdir / args.run_dir.name / name
                cfg.model.ode_solver.num_steps = steps
                model = egf.build_model(cfg.model).to(device)
                grasps, latency = egf_grasps(
                    model, cases, args.num_grasps, scale, device, args.latency_repeats
                )
                summary |= {
                    "run_dir": str(args.run_dir),
                    "checkpoint": args.checkpoint,
                    "solver": solver,
                    "steps": steps,
                    "nfe": steps * STAGES_PER_STEP[solver],
                    "guidance": cfg.model.guidance,
                    "latency": latency,
                }

            start = time.perf_counter()
            results = simulate(pool, cases, grasps, sim_cfg)
            summary["sim_time_s"] = time.perf_counter() - start
            summary["metrics"], per_case = summarize(cases, results)
            summary["sim"] = dataclasses.asdict(sim_cfg) | {"coacd_threshold": args.coacd_threshold}

            logdir.mkdir(parents=True, exist_ok=True)
            (logdir / "summary.json").write_text(json.dumps(summary, indent=2))
            per_case.to_csv(logdir / "per_case.csv", index=False)
            np.savez_compressed(
                logdir / "grasps.npz",
                Ts_grasp=np.stack(grasps),
                **{key: np.stack([result[key] for result in results]) for key in [*METRICS, "penetration"]},
            )
            if source == "egf":
                OmegaConf.save(copy.deepcopy(cfg), logdir / "config.yml")

            print(f"\n{name}: {logdir}")
            print(json.dumps(summary["metrics"], indent=2))
            rows.append(
                {
                    "name": name,
                    "nfe": summary.get("nfe"),
                    **{key: summary["metrics"][f"{key}_rate"]["mean"] for key in METRICS},
                    "ms/grasp": summary.get("latency", {}).get("single_grasp_ms"),
                }
            )

    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
