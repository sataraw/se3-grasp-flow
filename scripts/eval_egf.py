"""Evaluate an EquiGraspFlow run on the test split with the reference test procedure.

Reuses test()/print_results()/log_results()/save_figure() from the unmodified
test_full.py / test_partial.py; only the sampler can be swapped, so the same script covers the
paper setting (RK4-MK, 20 steps) and the NFE sweeps of M2/M3.

    uv run python scripts/eval_egf.py                                  # released full checkpoint
    uv run python scripts/eval_egf.py --run-dir data/checkpoints/equigraspflow_partial
    uv run python scripts/eval_egf.py --solver SE3_Euler --steps 5

NFE counts evaluations of the guided vector field. With classifier-free guidance each one runs the
vector-field network twice (conditional + unconditional); the encoder runs once per sample call.

Outputs: results/eval/<run-dir name>/<name>/ with logging.log, emd.xlsx (full only),
visualizations.json (view with scripts/show_vis.py) and summary.json:

    nfe                      guided vector-field evaluations per sample call (steps x 1 Euler, x 4 RK4-MK)
    wall_time_s              whole test loop incl. EMD and plot building, not sampling latency
    emd.per_category         mean EMD over the (object, rotation[, view]) test cases of a category
    emd.mean                 mean over all test cases, i.e. weighted by object count
    emd.num_evaluations      number of test cases (full: 83 objects x 3 rotations = 249)

One EMD value: as many grasps are generated as the object has ground-truth grasps, an optimal
one-to-one assignment pairs them, and the EMD is the mean pair distance sqrt(theta^2 + |dp|^2), with
theta the relative rotation angle in rad and dp the translation difference in m.
"""

import argparse
import copy
import json
import logging
import random
import runpy
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from graspgen import egf
from graspgen.paths import CHECKPOINT_DIR, EGF_ROOT, RESULTS_DIR

STAGES_PER_STEP = {"SE3_Euler": 1, "SE3_RK_mk": 4}


def load_run_config(run_dir: Path):
    ymls = sorted(run_dir.glob("*.yml"))
    if len(ymls) != 1:
        raise FileNotFoundError(f"expected exactly one .yml in {run_dir}, found {[p.name for p in ymls]}")
    return OmegaConf.load(ymls[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=CHECKPOINT_DIR / "equigraspflow_full",
        help="directory with the training config (.yml) and checkpoint",
    )
    parser.add_argument("--checkpoint", default="model_best_val_loss.pkl")
    parser.add_argument("--solver", choices=sorted(STAGES_PER_STEP), help="default: from the run config")
    parser.add_argument("--steps", type=int, help="number of ODE steps (default: from the run config)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist")
    parser.add_argument("--device", default="0", help="GPU index or 'cpu'")
    parser.add_argument("--name", help="output subdirectory (default: <solver>-<steps>)")
    parser.add_argument("--logdir", type=Path, default=RESULTS_DIR / "eval")
    args = parser.parse_args()

    cfg = OmegaConf.merge(load_run_config(args.run_dir), OmegaConf.from_dotlist(args.set))
    cfg.model.checkpoint = str(args.run_dir / args.checkpoint)
    cfg.device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    if args.solver:
        cfg.model.ode_solver.name = args.solver
    if args.steps:
        cfg.model.ode_solver.num_steps = args.steps
    solver, steps = cfg.model.ode_solver.name, cfg.model.ode_solver.num_steps
    partial = cfg.data.test.dataset.name != "full"

    args.logdir = args.logdir / args.run_dir.name / (args.name or f"{solver}-{steps}")
    args.logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.logdir / "logging.log",
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %I:%M:%S %p",
        level=logging.DEBUG,
        force=True,
    )
    OmegaConf.save(cfg, args.logdir / "config.yml")
    print(f"Result directory: {args.logdir}")

    # Same seeding as the reference test scripts
    seed = cfg.get("seed", 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

    ref_script = EGF_ROOT / ("test_partial.py" if partial else "test_full.py")
    ref = runpy.run_path(str(ref_script), run_name="egf_test")

    test_loader = egf.get_dataloader("test", copy.deepcopy(cfg.data.test))
    model = egf.build_model(cfg.model).to(cfg.device)
    metrics = egf.build_metrics(cfg.metrics)
    dataset = test_loader.dataset
    if partial:
        visualizer = ref["PlotlySubplotsVisualizer"](rows=dataset.num_rots, cols=dataset.num_views)
        visualizer.fig.update_layout(height=2700)
    else:
        visualizer = ref["PlotlySubplotsVisualizer"](rows=1, cols=dataset.num_rots)

    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    if partial:
        results = ref["test"](args, model, test_loader, metrics, cfg.device, visualizer)
    else:
        results = ref["test"](model, test_loader, metrics, cfg.device, visualizer)
    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()
    wall_time = time.perf_counter() - start

    ref["print_results"](test_loader, results)
    if not partial:
        ref["log_results"](args.logdir, test_loader, results)
    ref["save_figure"](args.logdir, visualizer)

    summary = {
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "point_cloud": "partial" if partial else "full",
        "solver": solver,
        "steps": steps,
        "nfe": steps * STAGES_PER_STEP[solver],
        "guidance": cfg.model.guidance,
        "wall_time_s": wall_time,
        "metrics": {
            key: {
                "mean": float(np.nanmean(result)),
                "per_category": {
                    t: float(np.nanmean(r)) for t, r in zip(dataset.obj_types, result, strict=True)
                },
                "num_evaluations": int(np.sum(~np.isnan(result))),
            }
            for key, result in results.items()
        },
    }
    (args.logdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["metrics"], indent=2))
    print(f"NFE {summary['nfe']}, test wall time {wall_time:.1f} s")


if __name__ == "__main__":
    main()
