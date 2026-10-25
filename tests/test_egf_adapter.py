"""The EquiGraspFlow reference code is importable and runs end to end on random data."""

import pytest
import torch

from graspgen import egf, paths


@pytest.fixture(scope="module")
def cfg():
    return egf.load_config("equigraspflow_full")


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return egf.build_model(cfg.model)


def random_batch(num_objs=2, num_pts=1024, num_grasps=8):
    pc = torch.randn(num_objs, 3, num_pts)
    pc -= pc.mean(dim=2, keepdim=True)
    Ts_grasp = [egf.SO3_uniform_R3_normal(num_grasps, "cpu") for _ in range(num_objs)]
    return {"pc": pc, "Ts_grasp": Ts_grasp}


def test_dataset_dir_is_patched():
    import loaders.acronym
    import loaders.utils

    assert loaders.acronym.DATASET_DIR == str(paths.DATASET_DIR)
    assert loaders.utils.DATASET_DIR == str(paths.DATASET_DIR)


def test_build_model_keeps_config(cfg):
    egf.build_model(cfg.model)
    assert cfg.model.name == "equigraspflow"


def test_train_step(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    losses = egf.build_losses(cfg.losses)
    results = model.step(random_batch(), losses, "train", optimizer)
    assert torch.isfinite(torch.tensor(results["scalar/train/loss"]))


@pytest.mark.parametrize("nfe", [1, 5])
def test_sample_euler(model, nfe):
    model.ode_solver = egf.SE3_Euler(num_steps=nfe)
    batch = random_batch(num_grasps=16)
    nums_grasps = torch.tensor([16, 16])

    model.eval()
    grasps = model.sample(batch["pc"], nums_grasps)

    assert len(grasps) == 2
    for T in grasps:
        assert T.shape == (16, 4, 4)
        assert egf.is_SE3(T)
