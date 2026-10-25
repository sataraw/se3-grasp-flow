"""Filesystem locations used across the project.

The data root can be moved (e.g. to a larger disk or a mounted bucket) with the
GRASPGEN_DATA environment variable; everything else is relative to the repo.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Untouched reference implementation (git submodule)
EGF_ROOT = REPO_ROOT / "third_party" / "EquiGraspFlow"
EGF_CONFIG_DIR = EGF_ROOT / "configs"

DATA_DIR = Path(os.environ.get("GRASPGEN_DATA", REPO_ROOT / "data"))
# EquiGraspFlow's preprocessed ACRONYM subset: grasps/, meshes/, splits/
DATASET_DIR = DATA_DIR / "dataset"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

RESULTS_DIR = REPO_ROOT / "results"
