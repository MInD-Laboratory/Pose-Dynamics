# Case Study 3: The Mirror Game

## Overview

The Mirror Game is a joint-improvisation paradigm in which two participants move
together simultaneously.  Visual feedback between partners was manipulated across
three conditions that systematically vary the coupling information available:

| Condition | Code | Description |
|-----------|------|-------------|
| Back-to-back | `b2b` | No visual feedback between partners |
| Unidirectional | `uni` | One designated leader can see the follower; follower cannot see the leader |
| Face-to-face | `f2f` | Mutual visual feedback — both participants can see each other |

Within each pair, one participant was assigned as the *leader* (whose movement the
other was asked to follow) and the other as the *follower*.  Leader/follower roles
were counter-balanced across the two experimental blocks of six trials each.

Pose was captured at 30 Hz using a ZED stereo camera with the ZED body-tracking SDK,
yielding 3-D positions for 38 anatomical keypoints per participant per frame.

## Data

Raw data are available at the joint Open Science Framework (OSF) repository:
**https://osf.io/bauqs**

Place the downloaded CSV files in `data/raw/`.  Each file corresponds to one
participant in one trial and follows the naming convention:

```
P###_T##_P#_pose_3d.csv
```

where `P###` is the pair ID, `T##` is the trial number (T1–T12), and `P#` is
the person within the pair (P1 or P2).

## Analysis Pipeline

The analysis is implemented as a numbered sequence of Jupyter notebooks in
`notebooks/`.  Run them in the order shown below.

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `1_preprocessing.ipynb` | Load raw CSVs, resample to 30 Hz, temporally align pairs, interpolate, filter, centre on pelvis, and Procrustes-align all trials to a shared canonical template |
| 2 | `2_diagnostic_pca.ipynb` | Fit PCA on the aligned pose data; inspect scree plot and cumulative explained variance to select the number of principal movements |
| 3 | `3_linear_kinematics.ipynb` | Extract windowed linear kinematic features (speed, acceleration) and test condition effects with LMMs |
| 4 | `4_rqa_parameter_estimation.ipynb` | Estimate RQA parameters — time delay τ via AMI and embedding dimension m via FNN — from representative trials |
| 5 | `5_recurrence_analysis.ipynb` | Compute windowed CRQA and multivariate CRQA on the principal-movement trajectories; fit LMMs on recurrence measures |

### Step 1 — Preprocessing

Each raw trial file is preprocessed as follows:

1. **Resampling** — raw data are resampled to a uniform 30 Hz grid using
   cumulative millisecond timestamps.
2. **Temporal alignment** — the P1 and P2 files for each trial are truncated to
   their common temporal overlap so both time series are the same length.
3. **Interpolation** — NaN gaps up to 60 frames (2 s at 30 Hz) are filled by
   linear interpolation.  Longer gaps remain as NaN.
4. **Low-pass filtering** — a 4th-order zero-phase Butterworth filter with a
   10 Hz cutoff removes measurement noise without distorting movement dynamics.
5. **Centring** — all keypoints are expressed relative to the pelvis (keypoint 0)
   so that locomotion does not confound postural dynamics.
6. **Procrustes alignment** — a canonical body-frame template is constructed
   from the mean poses of all trials.  Each trial's mean pose is then aligned to
   this template via rigid Procrustes (translation + rotation) so that orientation
   differences between participants and recording sessions are removed.
7. **PCA** — the aligned, centred keypoint sequences are projected onto the
   leading principal components to yield a set of *principal movements* (PMs)
   that serve as the input to subsequent recurrence analyses.

Pre-processed data are saved to `data/preprocessed/` as a single long-format CSV.

### Step 2 — Diagnostic PCA

Explores how much variance each principal component (PM) explains and animates
the spatial pattern of each retained PM to aid physical interpretation (e.g.,
arm swing, torso sway).

### Step 3 — Linear Kinematics

Extracts windowed speed and acceleration features from the PM time series.
Linear mixed-effects models test whether condition (b2b / uni / f2f) and role
(leader / follower) predict kinematic variability.

### Step 4 — RQA Parameter Estimation

Selects the delay-embedding parameters for recurrence analysis:

- **Time delay (τ)** — chosen as the first local minimum of the Auto Mutual
  Information (AMI) function, computed over lags 1–140 frames.
- **Embedding dimension (m)** — chosen as the smallest dimension at which the
  False Nearest Neighbours (FNN) proportion falls to near zero.

Estimated parameters for this dataset: **τ = 15 frames**, **m = 4**.

### Step 5 — Recurrence Analysis

Two recurrence analyses are carried out on each windowed PM pair:

- **Cross-RQA (CRQA)** — applied pairwise to leader and follower PM scores to
  quantify interpersonal movement coupling.
- **Multivariate CRQA** — all retained PMs are analysed simultaneously in a
  joint phase space, providing a single set of recurrence measures that captures
  full-body coupling.

Recurrence measures (%REC, %DET, entropy, laminarity, trapping time, divergence)
are then modelled with linear mixed-effects models to test condition and role effects.

Results are saved to `data/rqa/` and supplementary tables are generated in
`results/tables/`.

## Configuration

Key analysis parameters (set in `src/pose_dynamics/projects/mirror_game/config.py`
and `src/pose_dynamics/config.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sampling rate | 30 Hz | ZED body-tracking output rate |
| Keypoints | 38 | Full-body 3-D skeleton from ZED SDK |
| Window size | 60 s (1 800 frames) | Analysis window for CRQA |
| Window overlap | 50 % | Step = window / 2 |
| Embedding dimension (eDim) | 4 | Chosen via FNN analysis |
| Time delay (tLag) | 15 frames | Chosen via AMI analysis |
| Recurrence radius | 0.15 | Applied after z-score normalisation |
| Theiler window (CRQA) | 0 | Cross-RQA: no autocorrelation exclusion |
| Min. line length | 2 | Minimum diagonal/vertical line counted |

## Directory Structure

```
mirror_game/
├── notebooks/                  # Analysis notebooks (run in numerical order)
│   ├── 1_preprocessing.ipynb
│   ├── 2_diagnostic_pca.ipynb
│   ├── 3_linear_kinematics.ipynb
│   ├── 4_rqa_parameter_estimation.ipynb
│   └── 5_recurrence_analysis.ipynb
├── data/
│   ├── raw/                    # Raw per-trial CSVs (download from OSF)
│   ├── preprocessed/           # Output of notebook 1
│   ├── features/               # Linear kinematic features (notebook 3)
│   └── rqa/                    # CRQA and multivariate CRQA results (notebook 5)
├── results/
│   ├── figures/                # Publication figures and animations
│   └── tables/                 # Supplementary LaTeX/CSV tables
└── Mirror_Game_Conditions.csv  # Condition and leader assignment per pair/trial
```

## Source Modules

The shared library code for this case study lives in
`src/pose_dynamics/projects/mirror_game/`:

| Module | Description |
|--------|-------------|
| `config.py` | Keypoint mapping (38-point ZED skeleton), body-region groupings, condition labels, and RQA measure names |
| `pipeline.py` | File indexing, resampling, temporal alignment, condition merging, and leader/follower role assignment |
| `plots.py` | Visualisation utilities specific to the Mirror Game results figures |
