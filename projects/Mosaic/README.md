# Case Study 2: MOSAIC

## Overview

MOSAIC is a dyadic interaction study in which pairs of participants sat face-to-face
and engaged in structured social tasks.  Pose was estimated from video at 60 Hz using
OpenPose, yielding 2-D keypoint positions for each participant.  The goal of this case
study is to characterise the temporal structure of individual and coupled body movement
using linear kinematics and Cross-Recurrence Quantification Analysis (CRQA).

## Data

The data from case study 2 can be found here: **https://osf.io/3a59k** under `raw data/study2`. Please note that there are 6 files that were not included in the full publication, but are included in the analyses for this repository. These can be found at **https://osf.io/bauq** under `Case_Study_2_Mosaic`.

Place the downloaded data in the project root at `data\Mosaic`. Alteratively, update the `DATA_PATH` in the [analysis notebook](notebooks/1_analysis_pipeline.ipynb). Each CSV contains per-frame
OpenPose keypoint coordinates with columns of the form
`<KeypointName>_x_offset`, `<KeypointName>_y_offset`, `<KeypointName>_confidence`.

## Analysis Pipeline

The analysis is implemented as a sequence of Jupyter notebooks in `notebooks/`.
Run them in the order listed below.

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `analysis_pipeline.ipynb` | Preprocessing, Region of Interest extraction, Procrustes alignment, linear feature extraction and windowed CRQA |
| 2 | `mixed_effects_models.ipynb` | Linear mixed-effects models testing the effect of experimental condition on recurrence and kinematic measures |
| 3 | `visualisation.ipynb` | Visualisations for for the recurrence and kinematic results |

### Step 1 — Preprocessing and Feature Extraction

Raw keypoint data are preprocessed using the following steps:

1. **Confidence masking** — keypoints with detection confidence below 0.3 are set to
   NaN.  This removes frames in which the pose estimator was uncertain about the
   keypoint location.
2. **Resolution normalisation** — pixel coordinates are divided by the frame dimensions
   (720 × 720) so all keypoints lie in [0, 1], making the representation
   resolution-independent.
3. **Interpolation** — NaN gaps shorter than 60 frames (1 s at 60 Hz) are filled by
   linear interpolation.  Longer gaps are left as NaN.
4. **Low-pass filtering** — a 4th-order zero-phase Butterworth filter with a 10 Hz
   cutoff suppresses high-frequency noise while preserving biologically meaningful
   movement dynamics.
5. **Procrustes alignment** — each trial's mean pose is aligned to a global template (on a window-by-window basis)
   via rigid Procrustes (translation + rotation) so that between-participant
   differences in position and orientation do not confound movement measures.
7. **Windowed feature extraction** — for each 60-second overlapping window (50 % overlap):
   - *Linear kinematics*: RMS position, mean and SD of velocity magnitude.
   - *CRQA*: Metrics including %REC, %DET, entropy, and laminarity computed on each ROI.

### Step 2 — Statistical Modelling

Linear mixed-effects models (LMMs) are fitted to test whether experimental condition
predicts the CRQA and linear measures, with participant pair included as a random effect.

### Step 3 - Visualisation

Produces visualisations of the CRQA and linear measures results.

## Configuration

All dataset-level parameters are centralised in `config.py` and the shared library
at `src/pose_dynamics/projects/Mosaic/`.  Key analysis values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sampling rate | 60 Hz | OpenPose frame rate |
| Window size (RQA) | 60 s (3600 frames) | Analysis window for CRQA and linear measures |
| Window overlap | 50 % | Step = window / 2 |
| Embedding dimension (eDim) | 4 | Chosen via False Nearest Neighbours (FNN) |
| Time delay (tLag) | 10 frames | Chosen via Auto Mutual Information (AMI) |
| Recurrence radius | 0.2 | Applied after z-score normalisation |
| Theiler window (CRQA) | 0 | No autocorrelation correction in cross-RQA |
| Min. line length | 2 | Minimum diagonal/vertical line length counted |

## Source Modules

The shared Python library code for this case study lives in
`src/pose_dynamics/projects/Mosaic/`:

| Module | Description |
|--------|-------------|
| `pipeline.py` | Data loading, preprocessing pipeline, keypoint extraction, and windowing utilities |
| `alignment.py` | Procrustes alignment, symmetric template construction, and limb-length constraints |
| `features.py` | Velocity computation and linear kinematic metrics |
| `visualization.py` | Alignment diagnostics, step-by-step Procrustes visualisation |
