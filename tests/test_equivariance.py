"""SO(3)-equivariance of the EquiGraspFlow networks, checked in float64.

This is the check M3 requires for GMF-SG ("equivariance within a specified numerical tolerance"),
run here on the reference encoder and vector field.
"""

import pytest
import torch
from scipy.spatial.transform import Rotation

from graspgen import egf

ATOL = 1e-9


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return egf.build_model(egf.load_config("equigraspflow_full").model).double().eval()


@pytest.fixture(scope="module")
def R():
    return torch.from_numpy(Rotation.random(random_state=0).as_matrix())


def test_encoder_equivariant(model, R):
    pc = torch.randn(2, 3, 512, dtype=torch.float64)

    z = model.encoder(pc)  # (B, C, 3): C vectors per object
    z_rot = model.encoder(R @ pc)

    torch.testing.assert_close(z_rot, z @ R.T, atol=ATOL, rtol=0)


def test_vector_field_equivariant(model, R):
    torch.manual_seed(1)
    num = 16
    z = torch.randn(num, model.encoder.dims[-1], 3, dtype=torch.float64)
    t = torch.rand(num, 1, dtype=torch.float64)
    x_t = egf.SO3_uniform_R3_normal(num, "cpu").double()

    T_rot = torch.eye(4, dtype=torch.float64)
    T_rot[:3, :3] = R

    v = model.vector_field(z, t, x_t).view(num, 2, 3)  # (omega_s, v)
    v_rot = model.vector_field(z @ R.T, t, T_rot @ x_t).view(num, 2, 3)

    torch.testing.assert_close(v_rot, v @ R.T, atol=ATOL, rtol=0)
