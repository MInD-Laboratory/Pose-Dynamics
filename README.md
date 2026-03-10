# Pose Dynamics

This repository contains the code to reproduce all results and figures for the three
case studies in the accompanying methods paper on nonlinear approaches to analysing
pose dynamics in behavioural research.

The paper demonstrates how recurrence quantification analysis (RQA) and related
nonlinear time-series methods can be applied to body-pose data to capture movement
structure that is invisible to conventional linear (mean/variance) summaries.

## Repository Structure

```
Pose_Dynamics/
├── src/pose_dynamics/          # Shared Python library (install with pip)
│   ├── config.py               # Global analysis parameters
│   ├── preprocessing/          # Signal cleaning, interpolation, Procrustes alignment
│   ├── features/               # Linear kinematics, PCA (principal movements)
│   ├── nonlinear/              # AMI/FNN parameter estimation, RQA wrappers
│   ├── rqa/                    # C++-backed recurrence engine (auto, cross, multivariate)
│   └── projects/               # Dataset-specific pipelines (Mosaic, mirror_game, MATB)
├── projects/
│   ├── MATB/                   # Case Study 1 — figure notebook + submodule
│   ├── Mosaic/                 # Case Study 2 — notebooks + config
│   └── mirror_game/            # Case Study 3 — notebooks + data + results
└── pyproject.toml              # Package metadata and build configuration
```

## Case Studies

### Case Study 1: Multi-Attribute Task Battery (MATB)

Participants performed a simulated multi-tasking workload battery while body pose was
recorded.  The analysis examines how postural dynamics — quantified with RQA — reflect
cognitive workload.

The full analysis lives in the submodule at
[projects/MATB/Measuring_Workload_Dynamics_in_OpenMATB/](projects/MATB/Measuring_Workload_Dynamics_in_OpenMATB/).
The notebook [projects/MATB/MATB.ipynb](projects/MATB/MATB.ipynb) reads pre-computed
outputs from the submodule and reproduces the paper figures.

See the submodule README for its own setup and usage instructions.

### Case Study 2: MOSAIC

Face-to-face dyadic interaction study.  OpenPose 2-D keypoints are preprocessed,
projected onto principal movements via PCA, and analysed with Cross-RQA to quantify
interpersonal movement coupling.

- Notebooks: [projects/Mosaic/notebooks/](projects/Mosaic/notebooks/)
- Full details: [projects/Mosaic/README.md](projects/Mosaic/README.md)

### Case Study 3: The Mirror Game

Joint-improvisation paradigm with three visual-feedback conditions (back-to-back,
unidirectional, face-to-face).  ZED 3-D pose is preprocessed, projected onto
principal movements, and analysed with Cross-RQA and multivariate CRQA.

- Notebooks: [projects/mirror_game/notebooks/](projects/mirror_game/notebooks/)
- Full details: [projects/mirror_game/README.md](projects/mirror_game/README.md)

## Data Availability

Data for the Mirror Game and MOSAIC case studies are available at:
**https://osf.io/bauqs**

Download and place each dataset in the corresponding `data/raw/` directory before
running the notebooks (see each case study README for the expected layout).

---

## Setup

### Prerequisites

- Python 3.10 or later

### 1. Clone the Repository

Clone the repository and initialise the MATB submodule in one step:

```bash
git clone --recurse-submodules https://github.com/MInD-Laboratory/Pose-Dynamics.git
cd Pose_Dynamics
```

If you have already cloned without `--recurse-submodules`, initialise the submodule
manually:

```bash
git submodule update --init --recursive
```

### 2. Create a Virtual Environment

Using the standard library `venv`:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

Or with conda:

```bash
conda create -n pose_dynamics python=3.11
conda activate pose_dynamics
```

### 3. Install the Package

Install the `pose_dynamics` library from `pyproject.toml` in editable mode so that
changes to the source are reflected immediately without reinstalling:

```bash
pip install -e .
```

### 4. Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm
```

If you intend to run the mixed-effects model notebooks also install:

```bash
pip install statsmodels
```

### 5. (Optional) Jupyter

To run the notebooks interactively:

```bash
pip install jupyter
jupyter lab
```

### 6. Verify the Installation

```python
from pose_dynamics.config import get_cfg
cfg = get_cfg()
print(cfg.RQA.eDim, cfg.SAMPLING.pose_fps)  # expected output: 4  30.0
```

---

## Quickstart

After setup, a typical workflow for the Mirror Game case study is:

```bash
# 1. Place raw data CSVs in projects/mirror_game/data/raw/
# 2. Open JupyterLab and run notebooks in order (1 → 5)
jupyter lab projects/mirror_game/notebooks/
```

For the MOSAIC case study:

```bash
# 1. Set data_path in projects/Mosaic/config.py to point to your data directory
# 2. Run notebooks in order (1 → 3)
jupyter lab projects/Mosaic/notebooks/
```

---

## Package Overview

The `pose_dynamics` library (`src/pose_dynamics/`) provides reusable modules shared
across all case studies:

| Module | Purpose |
|--------|---------|
| `config.py` | Centralised RQA, windowing, sampling, and interpolation parameters |
| `preprocessing/pose_preprocessing.py` | DataFrame ↔ pose-array conversion, interpolation, filtering, centring |
| `preprocessing/signal_cleaning.py` | Resampling, Butterworth filtering, normalisation, sliding windows |
| `preprocessing/geometry.py` | Centroid computation, Procrustes alignment (rigid and similarity) |
| `features/linear_features.py` | Per-frame displacement, speed, and acceleration magnitudes |
| `features/dimred.py` | PCA fitting and projection onto principal movements |
| `nonlinear/state_space_recon.py` | AMI and FNN for embedding parameter selection |
| `nonlinear/rqa_utils.py` | Auto-RQA, Cross-RQA, and multivariate CRQA wrappers |
| `rqa/` | C++-backed recurrence engine with Python bindings |
