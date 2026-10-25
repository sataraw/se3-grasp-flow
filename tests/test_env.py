"""Environment checks: pinned versions and GPU availability."""

import pytest
import torch


def test_pinned_versions():
    import numpy
    import open3d

    assert numpy.__version__.startswith("1.26")
    assert open3d.__version__.startswith("0.16")  # required by the partial point cloud code


def test_mujoco_import():
    import mujoco

    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    assert model.nbody == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matmul():
    x = torch.randn(256, 256, device="cuda")
    torch.testing.assert_close((x @ torch.eye(256, device="cuda")).cpu(), x.cpu())
