"""Approximate convex decomposition of object meshes with CoACD.

MuJoCo collides a mesh geom as its convex hull, which would fill the inside of a mug or a bowl. An
object is therefore split into convex parts, each of which becomes one geom.

CoACD runs in a child process (`python -m graspgen.sim.decompose`): it can crash with a segmentation
fault on a mesh it cannot handle, which must not take the evaluation down with it, and it brings
its own OpenMP runtime, which is best kept apart from those of torch and open3d.
"""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

MIN_PART_VOLUME = 1e-12  # [m^3]; smaller hulls are numerical debris and are dropped
# CoACD's manifold preprocessing: "auto" skips it for meshes that already look manifold. Tried in
# this order; forcing it ("on") is the fallback for meshes on which "auto" crashes.
PREPROCESS_MODES = ("auto", "on")


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
    name: str = "mesh",
) -> list[np.ndarray]:
    """Split a mesh into convex parts; returns the hull vertices (K_i, 3) of each part.

    `threshold` is CoACD's concavity tolerance (0.01 fine ... 1 coarse). Results are cached in
    `cache_dir` under a hash of the mesh and the parameters. `name` only labels error messages.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha1(vertices.tobytes() + faces.tobytes())
        digest.update(f"coacd|{threshold}|{seed}".encode())
        cache_file = Path(cache_dir) / f"{digest.hexdigest()}.npz"
        if cache_file.exists():
            return _load_parts(cache_file)

    with tempfile.TemporaryDirectory() as tmp:
        mesh_file, parts_file = Path(tmp) / "mesh.npz", Path(tmp) / "parts.npz"
        np.savez(mesh_file, vertices=vertices, faces=faces)
        failures = []
        for mode in PREPROCESS_MODES:
            command = [sys.executable, "-m", "graspgen.sim.decompose"]
            command += [str(mesh_file), str(parts_file), str(threshold), str(seed), mode]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                break
            failures.append(
                f"preprocess={mode}: exit code {result.returncode} {result.stderr.strip()[-300:]}"
            )
            print(f"CoACD failed on {name} ({failures[-1]})", file=sys.stderr)
        else:
            raise RuntimeError(f"convex decomposition of {name} failed: " + "; ".join(failures))
        parts = _load_parts(parts_file)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, *parts)
    return parts


def _load_parts(path: Path) -> list[np.ndarray]:
    with np.load(path) as data:
        return [data[key] for key in data.files]


def _run_coacd(mesh_file: str, parts_file: str, threshold: str, seed: str, preprocess_mode: str) -> None:
    """Child process: decompose the mesh in `mesh_file` and write the parts to `parts_file`."""
    import coacd

    coacd.set_log_level("error")
    with np.load(mesh_file) as data:
        mesh = coacd.Mesh(data["vertices"], data["faces"])
    result = coacd.run_coacd(
        mesh, threshold=float(threshold), seed=int(seed), preprocess_mode=preprocess_mode
    )
    parts = [hull[0] for part, _ in result if (hull := convex_hull(np.asarray(part))) is not None]
    if not parts:
        sys.exit("convex decomposition produced no valid part")
    np.savez(parts_file, *parts)


if __name__ == "__main__":
    _run_coacd(*sys.argv[1:])
