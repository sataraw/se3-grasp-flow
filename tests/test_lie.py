"""EquiGraspFlow's SO(3) exp/log against scipy."""

import torch
from scipy.spatial.transform import Rotation

from graspgen import egf


def random_rotvecs(num, max_angle=3.0, seed=0):
    rotvecs = Rotation.random(num, random_state=seed).as_rotvec()
    angles = (rotvecs**2).sum(axis=1, keepdims=True) ** 0.5
    return torch.from_numpy(rotvecs / angles * angles.clip(max=max_angle))


def test_exp_matches_scipy():
    w = random_rotvecs(100)
    expected = torch.from_numpy(Rotation.from_rotvec(w.numpy()).as_matrix())
    torch.testing.assert_close(egf.exp_so3(w), expected)


# log_SO3 regularizes with EPS = 1e-4 (theta / (2 sin(theta) + EPS)), so it is only
# approximately the inverse of exp: ~1e-4 error up to 2.5 rad, ~1e-3 at 3.0 rad,
# ~4e-3 at 3.1 rad (float64). Keep this in mind for losses built on log (GMF-SG).
LOG_ATOL = {2.5: 3e-4, 3.0: 1.5e-3}


def test_log_inverts_exp():
    for max_angle, atol in LOG_ATOL.items():
        w = random_rotvecs(100, max_angle=max_angle)
        w_back = egf.bracket_so3(egf.log_SO3(egf.exp_so3(w)))
        torch.testing.assert_close(w_back, w, atol=atol, rtol=0)


def test_geodesic_dist_rotation_only():
    w = random_rotvecs(10, max_angle=2.5)
    T = torch.eye(4, dtype=w.dtype).repeat(10, 1, 1)
    T[:, :3, :3] = egf.exp_so3(w)
    identity = torch.eye(4, dtype=w.dtype).repeat(10, 1, 1)

    torch.testing.assert_close(egf.SE3_geodesic_dist(identity, T), w.norm(dim=1), atol=LOG_ATOL[2.5], rtol=0)
