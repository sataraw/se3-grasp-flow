# Setup

## 1. Tools

- **uv** ([install](https://docs.astral.sh/uv/getting-started/installation/)). It manages Python 3.10 itself.
- **Git** with submodule support.
- **LaTeX** with `latexmk` (MiKTeX or TeX Live) and Perl (comes with Git for Windows / TeX Live).
- **NVIDIA GPU (optional)**: torch is installed from the CUDA 12.8 index, which supports RTX 50xx (Blackwell)
  and older cards. Without a GPU everything runs on CPU.

> **Windows PATH pitfall.** If `latexmk` or `kpsewhich` fail with
> `MiKTeX cannot retrieve attributes for the directory '...\claude.exe\'`, your PATH contains a *file*
> (`C:\Users\<you>\.local\bin\claude.exe`) instead of the folder. Replace that entry with
> `C:\Users\<you>\.local\bin` in the user environment variables.

## 2. Environment

```bash
git clone --recurse-submodules <repo-url>   # or: git submodule update --init
uv sync                                     # creates .venv with Python 3.10 + all dependencies
uv run pytest                               # data-free smoke tests, ~20 s
```

Pinned on purpose:

| Package | Pin | Why |
|---|---|---|
| Python | 3.10 | open3d 0.16 has no wheels for newer Python |
| open3d | 0.16.0 | EquiGraspFlow's partial point clouds assert exactly 0.16 |
| numpy | 1.26 | matches the reference environment |
| torch | ≥ 2.7, cu128 index | Blackwell support; same wheels on Linux VMs |

Lint/format: `uv run ruff check src tests` and `uv run ruff format src tests`.

## 3. Data (not in git)

EquiGraspFlow publishes the preprocessed ACRONYM subset (Bowl, Laptop, Mug, Pencil: grasps, watertight
meshes, splits) and pretrained checkpoints on Google Drive (links in
`third_party/EquiGraspFlow/README.md`). Download them into:

```
data/
  dataset/       grasps/  meshes/  splits/
  checkpoints/   equigraspflow_full/  equigraspflow_partial/
```

Set `GRASPGEN_DATA=/some/other/path` to keep `data/` elsewhere (e.g. a larger disk).

## 4. Using EquiGraspFlow from our code

The reference code is a pinned git submodule and is **never edited**. It is not a package (its modules are
top-level `utils`, `models`, `loaders`, ...), so `graspgen.egf` puts it on `sys.path`, points its hardcoded
`DATASET_DIR` at `data/dataset`, and re-exports what we use:

```python
from graspgen import egf

cfg = egf.load_config("equigraspflow_full")    # third_party/EquiGraspFlow/configs/*.yml
model = egf.build_model(cfg.model)             # copies cfg; their factories pop keys
model.ode_solver = egf.SE3_Euler(num_steps=5)  # NFE = Euler steps, as in GraspMeanFlow
```

Rules:
- Import EquiGraspFlow only through `graspgen.egf`.
- Never create our own top-level modules named `utils`, `models`, `loaders`, `losses`, `metrics`, `trainers`.
- New methods (GMF-SG, MuJoCo evaluation, ...) are new modules in `graspgen` that build on the re-exports.
  If a reference file ever has to change, copy that one file into `graspgen` and modify the copy.

Known numerical limit: their `log_SO3` regularizes with `EPS = 1e-4` and is accurate to ~1e-4 rad up to
2.5 rad and ~1e-3 rad at 3 rad (see `tests/test_lie.py`). Relevant for anything built on log, such as the
GMF-SG semigroup loss.

## 5. LaTeX

```bash
cd latex
latexmk          # builds proposal/build/proposal.pdf
latexmk -pvc     # rebuild on every save
latexmk -c       # remove build files
```

`IEEEtran.cls`, `IEEEtran.bst` and the shared bibliography `adlr.bib` live in `latex/` so both of us
build with identical versions. New documents go in their own folder (e.g. `latex/report/report.tex`) and are
added to `@default_files` in `latex/latexmkrc`. Submission names follow the course: `proposal-<id>.pdf`.
