# Case Study 2: MOSAIC

## Overview

MOSAIC is a dyadic interaction study in which pairs of participants sat face-to-face
and engaged in structured social tasks.  Pose was estimated from video at 60 Hz using
OpenPose, yielding 2-D keypoint positions for each participant.  The goal of this case
study is to characterise the temporal structure of individual and coupled body movement
using linear kinematics and Cross-Recurrence Quantification Analysis (CRQA).

## Data

Raw data are available at the joint Open Science Framework (OSF) repository:
**https://osf.io/bauqs**

Place the downloaded data such that the path set in `config.py` (`data_path`) points
to the directory containing per-session CSV files.  Each CSV contains per-frame
OpenPose keypoint coordinates with columns of the form
`<KeypointName>_x_offset`, `<KeypointName>_y_offset`, `<KeypointName>_confidence`.

## Analysis Pipeline

The analysis is implemented as a sequence of Jupyter notebooks in `notebooks/`.
Run them in the order listed below.

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `preprocessing_crqa_linear_metrics.ipynb` | Preprocessing, Procrustes alignment, PCA, windowed CRQA and linear feature extraction |
| 2 | `MOSAIC_PCA.ipynb` | Diagnostic PCA visualisations — scree plot, cumulative variance, and principal-movement animations |
| 3 | `mixed_effects_models.ipynb` | Linear mixed-effects models testing the effect of experimental condition on recurrence and kinematic measures |

### Step 1 — Preprocessing and Feature Extraction

Raw keypoint data are preprocessed in the following order:

1. **Confidence masking** — keypoints with detection confidence below 0.4 are set to
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
5. **Procrustes alignment** — each trial's mean pose is aligned to a global template
   via rigid Procrustes (translation + rotation) so that between-participant
   differences in position and orientation do not confound movement measures.
6. **PCA** — the aligned keypoint time series is projected onto the leading principal
   components to produce a compact set of *principal movements* (PMs), each capturing
   an independent mode of body motion.
7. **Windowed feature extraction** — for each 60-second overlapping window (50 % overlap):
   - *Linear kinematics*: RMS position, mean and SD of velocity and acceleration for
     each PM.
   - *CRQA*: %REC, %DET, entropy, laminarity, trapping time, and divergence computed
     on each PM pair across the two participants in the dyad.

### Step 2 — Diagnostic PCA

Visualises how much variance each PC captures (scree plot) and animates the spatial
pattern of each retained principal movement to aid interpretation.

### Step 3 — Statistical Modelling

Linear mixed-effects models (LMMs) are fitted to test whether experimental condition
predicts the CRQA and linear measures, with participant pair included as a random effect.

## Configuration

All dataset-level parameters are centralised in `config.py` and the shared library
at `src/pose_dynamics/projects/Mosaic/`.  Key analysis values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sampling rate | 60 Hz | OpenPose frame rate |
| Window size (RQA) | 60 s (3 600 frames) | Analysis window for CRQA |
| Window size (linear) | 5 s (300 frames) | Analysis window for kinematic features |
| Window overlap | 50 % | Step = window / 2 |
| Embedding dimension (eDim) | 4 | Chosen via False Nearest Neighbours (FNN) |
| Time delay (tLag) | 15 frames | Chosen via Auto Mutual Information (AMI) |
| Recurrence radius | 0.15 | Applied after z-score normalisation |
| Theiler window (CRQA) | 0 | No autocorrelation correction in cross-RQA |
| Min. line length | 2 | Minimum diagonal/vertical line length counted |

## Source Modules

The shared Python library code for this case study lives in
`src/pose_dynamics/projects/Mosaic/`:

| Module | Description |
|--------|-------------|
| `pipeline.py` | Data loading, preprocessing pipeline, keypoint extraction, and windowing utilities |
| `alignment.py` | Procrustes alignment, symmetric template construction, and limb-length constraints |
| `features.py` | Velocity computation, custom facial features (blink and lip distance), and linear kinematic metrics |
| `visualization.py` | Alignment diagnostics, step-by-step Procrustes visualisation, and PCA movement animations |
