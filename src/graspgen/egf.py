"""Adapter for the EquiGraspFlow reference implementation.

EquiGraspFlow (third_party/EquiGraspFlow, pinned git submodule) is not an installable
package: it imports its own modules as top-level packages (`utils`, `models`, `loaders`, ...)
and reads the dataset from a hardcoded relative `DATASET_DIR = 'dataset'`. This module
makes it importable from anywhere without modifying it:

1. puts the submodule on sys.path, so its absolute imports resolve,
2. points both `DATASET_DIR` constants at `graspgen.paths.DATASET_DIR`,
3. re-exports the pieces we build on.

Our own code imports EquiGraspFlow only through this module, e.g.
`from graspgen.egf import VNLinear, exp_so3`. Never name a module of ours `utils`,
`models`, `loaders`, `losses`, `metrics` or `trainers` at the top level.

Note: their factory functions (`get_model`, `get_dataloader`, ...) `pop` keys from the
config they receive, so pass a copy (`build_model` does this).
"""

import copy
import sys

from omegaconf import DictConfig, OmegaConf

from graspgen.paths import DATASET_DIR, EGF_CONFIG_DIR, EGF_ROOT

if not (EGF_ROOT / "models" / "equi_grasp_flow.py").exists():
    raise ImportError(f"EquiGraspFlow submodule not found at {EGF_ROOT}. Run `git submodule update --init`.")

if str(EGF_ROOT) not in sys.path:
    sys.path.insert(0, str(EGF_ROOT))

import loaders.acronym
import loaders.utils

loaders.acronym.DATASET_DIR = str(DATASET_DIR)
loaders.utils.DATASET_DIR = str(DATASET_DIR)

# Re-exports (grouped by the reference module they come from)
from loaders import get_dataloader, get_dataset  # noqa: E402
from loaders.acronym import AcronymFullPointCloud, AcronymPartialPointCloud  # noqa: E402
from losses import get_losses  # noqa: E402
from losses.mse_loss import MSELoss  # noqa: E402
from metrics import get_metrics  # noqa: E402
from metrics.emd import EMDCalculator  # noqa: E402
from models import get_model  # noqa: E402
from models.equi_grasp_flow import EquiGraspFlow, get_traj  # noqa: E402
from models.vn_dgcnn import VNDGCNNEncoder  # noqa: E402
from models.vn_layers import VNLinear, VNLinearLeakyReLU  # noqa: E402
from models.vn_vector_fields import VNVectorFields  # noqa: E402
from utils import Lie  # noqa: E402
from utils.distributions import SO3_uniform_R3_normal, get_dist  # noqa: E402
from utils.Lie import (  # noqa: E402
    Lie_bracket,
    SE3_geodesic_dist,
    bracket_so3,
    exp_so3,
    inv_SO3,
    is_SE3,
    log_SO3,
    super_fibonacci_spiral,
)
from utils.ode_solvers import SE3_RK4_MK, SE3_Euler, get_ode_solver  # noqa: E402


def load_config(name: str = "equigraspflow_full") -> DictConfig:
    """Load one of the reference configs from third_party/EquiGraspFlow/configs."""
    return OmegaConf.load(EGF_CONFIG_DIR / f"{name}.yml")


def build_model(cfg_model: DictConfig) -> EquiGraspFlow:
    """`get_model` on a copy of the config, so the caller's config stays intact."""
    return get_model(copy.deepcopy(cfg_model))


def build_losses(cfg_losses) -> dict:
    """`get_losses` on a copy of the config."""
    return get_losses(copy.deepcopy(cfg_losses))


def build_metrics(cfg_metrics) -> dict:
    """`get_metrics` on a copy of the config."""
    return get_metrics(copy.deepcopy(cfg_metrics))


__all__ = [
    "SE3_RK4_MK",
    "AcronymFullPointCloud",
    "AcronymPartialPointCloud",
    "EMDCalculator",
    "EquiGraspFlow",
    "Lie",
    "Lie_bracket",
    "MSELoss",
    "SE3_Euler",
    "SE3_geodesic_dist",
    "SO3_uniform_R3_normal",
    "VNDGCNNEncoder",
    "VNLinear",
    "VNLinearLeakyReLU",
    "VNVectorFields",
    "bracket_so3",
    "build_losses",
    "build_metrics",
    "build_model",
    "exp_so3",
    "get_dataloader",
    "get_dataset",
    "get_dist",
    "get_losses",
    "get_metrics",
    "get_model",
    "get_ode_solver",
    "get_traj",
    "inv_SO3",
    "is_SE3",
    "load_config",
    "log_SO3",
    "super_fibonacci_spiral",
]
