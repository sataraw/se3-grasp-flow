"""Test cases for the MuJoCo evaluation, taken from EquiGraspFlow's ACRONYM test set.

A test case is one (object, rotation) pair of the reference test set, i.e. exactly what the EMD
evaluation runs on: the same point cloud goes into the model, and grasps live in the same frame
(origin at the point-cloud centroid, object rotated). The only difference is the unit: the
reference dataset is scaled by `scale` (8), here everything is in metres, since masses and forces
have to be physical.
"""

import copy
import os
from dataclasses import dataclass

import h5py
import numpy as np
from tqdm import tqdm

from graspgen import egf
from graspgen.paths import CACHE_DIR, DATASET_DIR
from graspgen.sim.decompose import decompose


@dataclass
class GraspCase:
    category: str
    obj_idx: int  # index into the sorted grasp files of the category, as in the reference splits
    rot_idx: int
    name: str  # grasp file name without extension
    pc: np.ndarray  # (3, num_pts) model input, in dataset scale
    Ts_gt: np.ndarray  # (N, 4, 4) successful ACRONYM grasps [m]
    parts: list[np.ndarray]  # convex parts of the object, vertices (K_i, 3) [m]
    mass: float  # [kg]
    friction: float


def load_test_cases(
    cfg_dataset,
    max_objects: int | None = None,
    coacd_threshold: float = 0.05,
    split: str = "test",
) -> tuple[list[GraspCase], float]:
    """Build the reference dataset and return its test cases and the dataset scale.

    `cfg_dataset` is a `data.<split>.dataset` node of a reference config (full point clouds).
    `max_objects` keeps only the first objects of every category, for quick runs.
    """
    cfg_dataset = copy.deepcopy(cfg_dataset)
    if cfg_dataset.name != "full":
        raise NotImplementedError("the MuJoCo evaluation supports full point clouds only")
    cfg_dataset.split = split
    dataset = egf.get_dataset(cfg_dataset)
    scale = dataset.scale

    cases = []
    for i, category in enumerate(tqdm(dataset.obj_types, desc="Convex decomposition", leave=False)):
        filenames = sorted(os.listdir(DATASET_DIR / "grasps" / category))
        for j, obj_idx in enumerate(dataset.obj_idxs_types[i][:max_objects]):
            filename = filenames[obj_idx]
            with h5py.File(DATASET_DIR / "grasps" / category / filename) as data:
                mesh = egf.load_mesh(data)  # metres, in the frame of the mesh file
                mass = float(data["object/mass"][()])
                friction = float(data["object/friction"][()])
            vertices = np.asarray(mesh.vertices)
            parts = decompose(
                vertices,
                np.asarray(mesh.triangles),
                threshold=coacd_threshold,
                cache_dir=CACHE_DIR / "coacd",
                name=f"{category}/{filename}",
            )

            for k, R in enumerate(dataset.Rs):
                # The dataset mesh is R @ (scale * vertices - centre); recover the centre so that
                # the parts, decomposed once in the mesh file's frame, can follow.
                vertices_case = np.asarray(dataset.mesh_list_types[i][j][k].vertices) / scale
                offsets = vertices - vertices_case @ R
                centre = offsets.mean(axis=0)
                if not np.allclose(offsets, centre, atol=1e-5):
                    raise RuntimeError(f"{filename}: dataset mesh does not match the mesh file")

                Ts_gt = np.array(dataset.Ts_grasp_list_types[i][j][k], dtype=np.float64)
                Ts_gt[:, :3, 3] /= scale
                cases.append(
                    GraspCase(
                        category=category,
                        obj_idx=int(obj_idx),
                        rot_idx=k,
                        name=os.path.splitext(filename)[0],
                        pc=np.asarray(dataset.pc_list_types[i][j][k], dtype=np.float32),
                        Ts_gt=Ts_gt,
                        parts=[(part - centre) @ R.T for part in parts],
                        mass=mass,
                        friction=friction,
                    )
                )
    return cases, scale
