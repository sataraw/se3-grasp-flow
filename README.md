# Generative Networks for Robot Grasping

ADLR WS 2026/27 project (idea 4) by Georgii Fatakhov and Awar Satar.

Few-step, SE(3)-equivariant grasp generation on ACRONYM: from EquiGraspFlow (flow matching)
to a reimplementation of GraspMeanFlow-SG, evaluated by EMD, geometric checks, and a MuJoCo
lift-and-hold test across sampling budgets (NFE). See [docs/proposal.md](docs/proposal.md).

## Quick start

```bash
git clone --recurse-submodules <repo-url>
uv sync
uv run pytest
```

Details (installation, data, LaTeX build, conventions): [docs/setup.md](docs/setup.md).

## Layout

```
src/graspgen/            our package
  paths.py               data / checkpoint / results locations (GRASPGEN_DATA overrides data/)
  egf.py                 adapter that makes the EquiGraspFlow reference code importable
third_party/EquiGraspFlow/  reference implementation, git submodule, never edited
tests/                   data-free smoke tests (env, adapter, equivariance, Lie, MuJoCo)
latex/                   IEEEtran documents built with latexmk (proposal/, later report/)
docs/                    proposal and project documentation
data/                    datasets and checkpoints (not in git)
```
