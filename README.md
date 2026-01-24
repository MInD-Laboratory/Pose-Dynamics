'pose-dynamics' is a Python package for processing pose-estimation data and computing time-series and coordination metrics used in movement science and cognitive science. The pipeline is informed by "Nonlinear Methods for Analyzing Pose in Behavioral Research" (see docs/paper reference).

The package provides a standardized pipeline for:

- loading and coercing pose-estimation outputs
- windowing time series
- preprocessing (interpolation, filtering, alignment)
- linear kinematic analysis including PCA
- estimating embedding parameters (AMI, FNN)
- computing recurrence-based measures (RQA, CRQA, MdRQA)
- extracting kinematic and geometric features

The repository also includes three **self-contained case studies** under `examples/` that demonstrate how the library is used in practice. 

## Installation

This package is intended to be installed from source for development and research use.

**Setup**
```
git clone https://github.com/cartersale/Pose-Dynamics.git # Clone the repo
cd pose-dynamics
python -m venv .venv # Create virtual environment
source .venv/bin/activate 
pip install -e . # Install the package 
pytest tests/test_imports.py # Verify the installation worked
```

## Data Requirements & Formats
Each input file is assumed to correspond to a single trial or session. Metadata such as subject or trial identifiers are supplied externally (e.g., as function arguments or via configuration files), not embedded in the raw pose data.

**Supported input formats**: Pose data are expected to be provided as wide-format CSV files with one row
per frame.

The following formats are supported:

#### 2D pose with confidence

Columns are expected to follow the pattern:

```
x0, y0, prob0, x1, y1, prob1, ...
```

where the numeric suffix indicates the keypoint index.

#### 3D pose

Columns are expected to follow the pattern:

```
x0, y0, z0, x1, y1, z1, ...
```

Confidence values are optional for 3D data.

### Frame rate and time

Pose files are not required to include timestamps.
Instead, the user is expected to provide the frame rate (`fps`) when loading the data.

Each file is treated as an ordered sequence of frames. A frame index is assigned
internally starting at 0.

If timestamps are present (e.g., `timestamp_ns` or `dt_ms`), they are used preferentially.
Otherwise, time in seconds is derived from the frame index and the provided frame rate.

### Canonical representation

All pose data are coerced into a **canonical long-form representation** with the following
columns.

**Required:**

* `frame` : integer frame index
* `kp`    : integer keypoint index
* `x`, `y`: spatial coordinates

**Optional:**

* `z`     : depth coordinate
* `conf`  : confidence value
* `t`     : time in seconds (derived)
* `subject`, `trial` : metadata identifiers

All downstream modules in `pose_dynamics` assume this canonical representation.

## IO: Ingest (wide CSV -> canonical parquet)

Run ingest on a directory of wide pose CSVs (one CSV per trial):

```
pose-dynamics ingest-csv \
  --in data/raw_pose_csv/ \
  --out artifacts/ingest/run_001/ \
  --fps 30
```

Behavior:
- If a CSV has a `time` column, `--fps` is not required for that file.
- Unrecognized columns are ignored and logged in `recording.json`.

Outputs:
- `pose.parquet`
- `recording.json`
- `qc_ingest.json`

## Preprocessing (selection → timebase → confidence → missing → alignment → normalization → filtering → windowing)

Preprocess uses a YAML config (stored in examples/ or projects/), e.g.
`examples/case_study_1_matb/configs/preprocess.yaml`.

Run preprocessing:

```
pose-dynamics preprocess \
  --in artifacts/ingest/run_001/pose.parquet \
  --recording artifacts/ingest/run_001/recording.json \
  --config examples/case_study_1_matb/configs/preprocess.yaml \
  --out artifacts/preprocess/run_001/
```

Outputs:
- `pose_clean.parquet`
- `windows.parquet`
- `qc_preprocess.json` (includes filtering metadata per trial if filtering is enabled)
- `provenance.json`
- `alignment_transforms.json` (only if alignment is enabled)

## Feature extraction (kinematics + geometry)

Feature extraction is a separate step that consumes preprocessed outputs and
computes per-window features. You can restrict feature computation to a subset
of keypoints that were already preprocessed.

Run feature extraction:

```
pose-dynamics feature-extract \
  --pose artifacts/preprocess/run_001/pose_clean.parquet \
  --windows artifacts/preprocess/run_001/windows.parquet \
  --config examples/case_study_1_matb/configs/features.yaml \
  --out artifacts/features/run_001/
```

Outputs:
- `features.parquet`
- `qc_features.json`
- `provenance_features.json`

## PCA (global or per-trial)

PCA can be computed on both a pose summary and the extracted features. The PCA
config controls whether PCA is global or per trial.

Run PCA:

```
pose-dynamics pca \
  --pose artifacts/preprocess/run_001/pose_clean.parquet \
  --windows artifacts/preprocess/run_001/windows.parquet \
  --features artifacts/features/run_001/features.parquet \
  --config examples/case_study_1_matb/configs/pca.yaml \
  --out artifacts/pca/run_001/
```

Outputs:
- `pca_scores.parquet`
- `pca_components.json`
- `qc_pca.json`
- `provenance_pca.json`

## RQA parameter estimation (AMI/FNN/epsilon sensitivity)

Use this step to select embedding delay $\tau$, embedding dimension $m$, and an
epsilon radius based on AMI/FNN curves and recurrence-rate sensitivity. This
always uses the preprocessing windows.

List available keypoints:

```
pose-dynamics list-keypoints \
  --pose artifacts/preprocess/run_001/pose_clean.parquet
```

Estimate parameters:

```
pose-dynamics rqa-params \
  --pose artifacts/preprocess/run_001/pose_clean.parquet \
  --windows artifacts/preprocess/run_001/windows.parquet \
  --config examples/case_study_1_matb/configs/rqa_params.yaml \
  --out artifacts/rqa_params/run_001/
```

Outputs:
- `rqa_params.json`
- `qc_rqa_params.json`
- `provenance_rqa_params.json`
- `plots/ami_<kp>.png`, `plots/fnn_<kp>.png`, `plots/epsilon_<kp>.png`

## RQA / CRQA

RQA uses the preprocessing windows and selected keypoints to compute recurrence
metrics per window. CRQA uses `--pose-y` for the paired stream.

Run RQA:

```
pose-dynamics rqa \
  --pose artifacts/preprocess/run_001/pose_clean.parquet \
  --windows artifacts/preprocess/run_001/windows.parquet \
  --config examples/case_study_1_matb/configs/rqa.yaml \
  --out artifacts/rqa/run_001/
```

Outputs:
- `rqa_stats.parquet`
- `qc_rqa.json`
- `provenance_rqa.json`
- `plots/rp_<trial>_<window>.png`, `plots/drp_<trial>_<window>.png`

