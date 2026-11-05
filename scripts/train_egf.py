"""Train EquiGraspFlow with the unmodified reference train.py.

The config is one of third_party/EquiGraspFlow/configs (by name) or a path to a .yml,
optionally changed with OmegaConf dotlist overrides. The resolved config is written next to
the run outputs, so a run can always be traced back to exactly what it trained with.

    uv run python scripts/train_egf.py --config equigraspflow_full
    uv run python scripts/train_egf.py --config equigraspflow_full --run smoke \
        --set trainer.num_epochs=2 trainer.val_interval=50 data.train.num_workers=0

Outputs: results/train/<config name>/<run>/ (tensorboard logs, logging.log, config, checkpoints).
"""

import argparse
import runpy
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from graspgen import egf  # noqa: F401  (patches DATASET_DIR before train.py imports the loaders)
from graspgen.paths import EGF_CONFIG_DIR, EGF_ROOT, RESULTS_DIR


def resolve_config(config: str) -> Path:
    path = Path(config)
    return path if path.suffix in (".yml", ".yaml") else EGF_CONFIG_DIR / f"{config}.yml"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="equigraspflow_full", help="config name or path to a .yml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist")
    parser.add_argument("--device", default="0", help="GPU index or 'cpu'")
    parser.add_argument("--run", default=datetime.now().strftime("%Y%m%d-%H%M"))
    parser.add_argument("--logdir", default=str(RESULTS_DIR / "train"))
    args = parser.parse_args()

    config_path = resolve_config(args.config)
    # Windows spawns DataLoader workers by pickling the dataset, which holds open3d meshes that
    # cannot be pickled. Loading is in-memory anyway, so workers buy nothing; --set can override.
    platform_defaults = (
        ["data.train.num_workers=0", "data.val.num_workers=0"] if sys.platform == "win32" else []
    )
    cfg = OmegaConf.merge(
        OmegaConf.load(config_path), OmegaConf.from_dotlist([*platform_defaults, *args.set])
    )

    # train.py names the run directory after the config file, so keep the file name
    resolved = Path(args.logdir) / "_configs" / args.run / config_path.name
    resolved.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, resolved)

    sys.argv = [
        str(EGF_ROOT / "train.py"),
        *("--config", str(resolved)),
        *("--device", args.device),
        *("--logdir", args.logdir),
        *("--run", args.run),
    ]
    runpy.run_path(str(EGF_ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
