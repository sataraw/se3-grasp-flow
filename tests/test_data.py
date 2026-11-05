"""The downloaded ACRONYM subset and checkpoints are complete and consistent.

Skipped when data/ is absent (e.g. CI or a fresh clone); see docs/setup.md for the download.
Checks the raw files only, without building the datasets, so it stays fast.
"""

import h5py
import numpy as np
import pytest
import torch

from graspgen import egf, paths

CATEGORIES = ["Bowl", "Laptop", "Mug", "Pencil"]
SPLITS = ["train", "val", "test"]

pytestmark = pytest.mark.skipif(not paths.DATASET_DIR.exists(), reason=f"no dataset at {paths.DATASET_DIR}")


@pytest.mark.parametrize("category", CATEGORIES)
def test_splits_index_disjoint_existing_files(category):
    num_files = len(list((paths.DATASET_DIR / "grasps" / category).glob("*.h5")))
    splits = {s: np.load(paths.DATASET_DIR / "splits" / category / f"idxs_{s}.npy") for s in SPLITS}

    for idxs in splits.values():
        assert len(idxs) > 0
        assert idxs.min() >= 0 and idxs.max() < num_files
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            assert not set(splits[a]) & set(splits[b]), f"{a}/{b} overlap"


@pytest.mark.parametrize("category", CATEGORIES)
def test_grasp_files_reference_existing_meshes(category):
    for path in sorted((paths.DATASET_DIR / "grasps" / category).glob("*.h5")):
        with h5py.File(path) as data:
            mesh = paths.DATASET_DIR / data["object/file"][()].decode("utf-8")
            assert mesh.exists(), f"{path.name}: missing {mesh}"
            assert data["grasps/transforms"].shape[1:] == (4, 4)


@pytest.mark.parametrize("name", ["equigraspflow_full", "equigraspflow_partial"])
def test_checkpoint_loads_into_model(name):
    run_dir = paths.CHECKPOINT_DIR / name
    if not run_dir.exists():
        pytest.skip(f"no checkpoint at {run_dir}")
    state = torch.load(run_dir / "model_best_val_loss.pkl", map_location="cpu")["model_state"]
    model = egf.build_model(egf.load_config(name).model)
    model.load_state_dict(state)  # strict: no missing or unexpected keys
