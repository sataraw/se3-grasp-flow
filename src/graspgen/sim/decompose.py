"""Approximate convex decomposition of object meshes with CoACD.

MuJoCo collides a mesh geom as its convex hull, which would fill the inside of a mug or a bowl. An
object is therefore split into convex parts, each of which becomes one geom.
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

MIN_PART_VOLUME = 1e-12  # [m^3]; smaller hulls are numerical debris and are dropped


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Hull vertices and volume of a point set, or None if it is degenerate (flat or tiny)."""
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return None
    if hull.volume < MIN_PART_VOLUME:
        return None
    return points[hull.vertices], hull.volume


def decompose(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold: float = 0.05,
    cache_dir: Path | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
    """Split a mesh into convex parts; returns the hull vertices (K_i, 3) of each part.

    `threshold` is CoACD's concavity tolerance (0.01 fine ... 1 coarse). Results are cached in
    `cache_dir` under a hash of the mesh and the parameters.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha1(vertices.tobytes() + faces.tobytes())
        digest.update(f"coacd|{threshold}|{seed}".encode())
        cache_file = Path(cache_dir) / f"{digest.hexdigest()}.npz"
        if cache_file.exists():
            with np.load(cache_file) as data:
                return [data[key] for key in data.files]

    import coacd  # imported late: only needed on a cache miss

    coacd.set_log_level("error")
    result = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=threshold, seed=seed)
    parts = [hull[0] for part, _ in result if (hull := convex_hull(np.asarray(part))) is not None]
    if not parts:
        raise ValueError("convex decomposition produced no valid part")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, *parts)
    return parts
