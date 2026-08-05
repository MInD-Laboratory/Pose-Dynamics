# Getting started

You have a folder of pose recordings and you want features and metrics out of them.
This guide takes you there in three commands:

```bash
pose-dynamics inspect  trial.csv     # 1. see your keypoint numbering
pose-dynamics new-config study.yaml  # 2. write a config, fill in the details
pose-dynamics run      study.yaml    # 3. get features + metrics over the whole folder
```

Along the way you'll make two real choices — which keypoints/features you care about,
and the two embedding parameters `(τ, m)` — and this guide is mostly about how to
make them well. (Everything here also works from a notebook; see `run_dataset.ipynb`.)

## Before you start: the one file format

The package reads **one format**: a wide CSV, one file per person per trial, columns
`x0,y0[,z0][,c0], x1,y1,...`. Converting your estimator's output to it is the only
setup step — see the [canonical format](canonical_format.md) (there are example
converters in `examples/`). Everything below assumes a folder of such files.

## Step 1 — Inspect: see your keypoints

Different estimators number keypoints differently. Before you can say "the arms" you
need to know which indices *are* the arms:

```bash
pose-dynamics inspect trial.csv
```

This prints the shape (how many keypoints, 2D or 3D, confidence or not) and saves a
plot of the skeleton with **every point labelled by its index**. Look at it: the
head cluster, the wrists, the ankles will be obvious. Write down the indices you
care about — you'll use them in the config.

## Step 2 — Choose (τ, m)

Recurrence analysis reconstructs movement in a *state space* built from two numbers:
a delay **τ** and a dimension **m**. The package presents the evidence rather than
picking for you, because automated minimum-picking is unreliable on empirical data.
Open **`quickstart.ipynb`**, run it on one of your files, and read the two curves:

- **AMI (for τ):** the curve drops then flattens. **Choose τ where it stops
  dropping** — the onset of the plateau. A clear dip (first minimum) is the easy
  case. A dependable choice is a turn that shows up across many of your signals,
  rather than a one-sample wiggle in a single noisy curve or simply the lowest
  point. If the curve never flattens, your data is probably oversampled —
  downsample and look again.
- **FNN (for m):** the curve falls steeply then levels off. **Choose m at the elbow.**
  It usually levels at a small non-zero floor (measurement noise) — don't chase it to
  zero. When in doubt, round *up*: over-embedding is safer than under-embedding.

Commit **one** `(τ, m)` and apply it to your whole dataset — not a different value per
trial. Note the numbers; they go straight into the config.

## Step 3 — Write the config

```bash
pose-dynamics new-config study.yaml
```

This writes a commented template. Open it and fill it in. Here is every field that
matters and how to set it:

```yaml
data: ./data                 # your folder of canonical CSVs
frame_rate: 60               # Hz — must match how you recorded (it is not guessed)

tau: 20                      # from Step 2
m: 4                         # from Step 2
```

**Preprocessing** — clean the trajectories:

```yaml
conf_threshold: 0.30         # mark keypoints below this confidence as missing
filter_cutoff: 10.0          # Hz low-pass. Set it to smooth jitter WITHOUT flattening
filter_order: 4              #   the real movement (10 Hz for 2D webcam, ~5 for 3D body)
```

**The feature pipeline** — *what* you measure. This is a **list of steps** (see
[feature steps](feature_steps.md)). Delete the block entirely to just analyse each
keypoint's speed; keep/adapt it to build regional or derived features. The indices
come from Step 1:

```yaml
features:
  - step: coordinate_normalization
    params: {width: 720, height: 720}     # your video resolution
  - step: roi_centroid
    params:
      rois:
        arms:       [2, 3, 4, 5, 6, 7]     # <- the indices you read off the plot
        upper_body: [1, 2, 5, 8]
  - step: velocity_magnitude
    params: {method: diff}
```

**What to compute, and the one judgement call in the analysis:**

```yaml
compute_linear: true         # magnitude summaries (mean / std / rms / max)
compute_recurrence: true     # recurrence metrics

radius_mode: fixed_rrec      # <- read this carefully:
target_rec: 5.0              #    "fixed_rrec" pins recurrence rate to this target and
                             #    SOLVES the radius. %REC is then just a check that it
                             #    converged; the *radius* carries the density signal.
                             #    Use "fixed_radius" + `radius:` instead if you want
                             #    %REC itself to be an outcome you compare.

window_s: 30                 # analysis window in seconds (remove for whole-trial)
```

**Where the output goes:**

```yaml
features_dir: ./features     # one CSV of feature time series per input file
output_csv: ./metrics.csv    # one tidy table of metrics (file × feature × window)
```

## Step 4 — Run

```bash
pose-dynamics run study.yaml
```

You get:

- `features/<trial>_features.csv` — the extracted feature time series, per file. If
  all you wanted was *features over a bunch of CSVs*, you're done here.
- `metrics.csv` — a tidy table, one row per **file × feature × window**, with the
  magnitude columns (`mean, std, rms, max`) and the recurrence columns
  (`perc_recur, perc_determ, laminarity, mean_line_length, maxl_found, entropy`,
  plus `radius_used`).
- a printed **data-quality summary** — any flagged or excluded trials are listed.

Take `metrics.csv` into your own statistics — the package computes no inferential
statistics itself.

## Step 5 — Which metrics to report

You now have many columns. Don't test all of them and keep whatever is significant.

- A sensible **core set** is **RR, DET, and Lmax** (recurrence density,
  predictability, longest sustained structured episode). Add **laminarity / trapping
  time** for persistence questions, **entropy** for timescale-diversity questions.
- **Decide in advance** which metrics your manipulation should move, and why; report
  anything else as exploratory. The metrics are mathematically coupled, so testing
  all of them inflates false positives.
- **Absolute values are not meaningful on their own** — they shift with `(τ, m)` and
  the radius. What should be stable is the *comparative* pattern across your
  conditions. And if you chose `fixed_rrec`, remember %REC is pinned by design —
  compare the **radius** and the line-length family instead.

## Where to go from here

- **[Feature steps](feature_steps.md)** — the full list of steps and how to compose
  them (including writing your own).
- **[Configuration reference](configuration.md)** — every parameter, default, and the
  paper's guidance.
- **Reproduce a case study** — `notebooks/case1_matb_reproduction.ipynb` and the
  others show real, published analyses end to end, including dyadic (two-person)
  cross-recurrence, which the standard single-person pipeline does not cover.
