# Configuration reference

Every parameter, its default, its units, and — where the paper gives guidance —
what the paper recommends. Parameters are passed explicitly to the relevant
function or dataclass; there is no global config singleton.

Where the paper offers a *range* rather than a value, that is noted: these are the
knobs you are expected to set for your data, not accept blindly. The case-study
configs (`pose_dynamics.case_studies.*.config`) show committed values for real data.

## The standard pipeline (`StudyConfig` + `run_study`)

The one-stop way to run a whole dataset. A `StudyConfig` bundles the parameters
below into one editable object (or a YAML/JSON file) and runs from a notebook
(`run_study(config)`) or the command line:

```bash
pose-dynamics new-config study.yaml   # write a template config
pose-dynamics run study.yaml          # run it over the folder
```

It writes per-file **feature time series** (`features_dir`) and/or a tidy **metrics
table** (`output_csv`) — linear + recurrence metrics per file × feature × window —
plus a per-trial quality report. Load a file with `StudyConfig.from_file(path)`.

`pose-dynamics new-config` writes a template that **lists every field with its
default** (embedding, preprocessing, the feature pipeline, all RQA settings —
`min_line`/l_min, `rescale`, `norm`, `theiler` — windowing, quality, output), so you
can see and change any of them.

| Field | Default | Notes |
|-------|---------|-------|
| `data` | — | folder or glob of canonical CSVs (required). |
| `frame_rate` | — | Hz (required). |
| `tau`, `m` | — | the committed embedding (required). |
| `features` | `None` | a feature pipeline (list of steps `{step, params}`, see [feature steps](feature_steps.md)). `None` = per-keypoint speed. |
| `default_signal` | `"speed"` | per-keypoint signal when no `features` given (`"speed"`/`"displacement"`). |
| `compute_linear`, `compute_recurrence` | `True`, `True` | which metric families to compute. |
| `features_dir` | `None` | write per-file feature time series here. |
| `window_s` | `None` | analysis window (s); `None` = whole trial. |
| `interp_cap` | `None` | `None` resolves to the principled `(m−1)·τ`. |
| `radius_mode`, `target_rec`, `radius` | `"fixed_rrec"`, `5.0`, `None` | recurrence density (see below). |
| `output_csv` | `None` | write the results table if set. |

All other preprocessing/RQA/quality fields mirror the parameters documented below,
with the same defaults.

## Preprocessing

| Parameter | Default | Units | Paper guidance |
|-----------|---------|-------|----------------|
| `mask_low_confidence(threshold=)` | — | 0–1 | Mask low-confidence keypoints as missing. Case studies used **0.30**. |
| `interpolate_gaps(max_gap=)` | — | frames | Fill only gaps ≤ cap; longer gaps left missing. Conservative limit is **`(m−1)·τ`** (derived from delay embedding); a provisional fixed cap (e.g. 60 frames ≈ 1 s at 60 Hz) is used before `(m, τ)` are known. |
| `butterworth_filter(cutoff_hz=)` | — | Hz | Low-pass to remove jitter without flattening movement. Cases used **10 Hz** (2D webcam), **5 Hz** (3D full body). Must be < Nyquist (`frame_rate/2`). |
| `butterworth_filter(order=)` | `4` | — | Zero-phase (applied forward+backward). |

## Linear kinematic metrics (`pose_dynamics.linear`)

A standalone family — `load → preprocess → linear metrics` is a complete analysis;
you need not run recurrence. Reductions (pose → scalars), no config required:

| Function | Returns |
|----------|---------|
| `per_frame_kinematics(seq)` | per-frame displacement / speed / acceleration magnitudes (2D or 3D). |
| `kinematic_summary(seq, stats=)` | tidy per-keypoint DataFrame of `{quantity}_{stat}` (e.g. `speed_rms`). |
| `region_kinematic_summary(seq, regions, stats=)` | per-region (centroid) summary. |
| `summarise_signal(x, stats=)` | summarize any 1-D signal. |

`stats` defaults to `("mean", "std", "min", "max", "rms")` (also `"median"`). The
paper: RMS weights larger excursions and gives a compact "energetic" index of
movement; keep physically meaningful units (don't z-score) when amplitude is the
question, and detrend rather than per-window z-score.

## Gap policy / data quality (`assess_quality`)

| Parameter | Default | Units | Guidance |
|-----------|---------|-------|----------|
| `max_missing_frac` | `0.30` | fraction | Trial-level threshold; exceeding it sets the trial's status to `on_exceed`. |
| `per_keypoint_max_missing_frac` | `0.30` | fraction | A keypoint missing more than this is flagged as a candidate to drop (paper: drop a landmark missing ≳ 20–30% rather than reconstruct it). |
| `on_exceed` | `"flag"` | — | `"retain"`, `"flag"`, or `"exclude"`. Nothing is dropped silently — flagged/excluded trials appear in the report. |

## Embedding selection (`select_embedding`)

The framework **presents evidence and proposes**; you commit `(τ, m)` with
`evidence.commit(tau, m)`. Applied fixed across a study.

| Parameter | Default | Units | Guidance |
|-----------|---------|-------|----------|
| `tau_grid` | `(10, 25)` | frames | Presented/plausible delay band. Paper: `τ ≈ 10–30` at 30–60 Hz; cases ranged 10–20. |
| `m_grid` | `(3, 6)` | — | Presented dimension band. Paper: `m` of 3–5 common; all three cases converged on **4**. |
| `ami_max_lag` | `140` | frames | Range over which AMI is computed. |
| `fnn_max_dim` | `10` | — | Range over which FNN is computed. |
| `rel_frac` | `1/e` | — | Relative-AMI level for the τ heuristic (proposal only). |
| `fnn_tol` | `10.0` | % | FNN noise-floor tolerance for the m heuristic (proposal only). |
| `subset`, `seed` | `None`, `0` | — | Optionally compute on a random subset of signals (size + seed logged). |

**Committed value** (`EmbeddingParams`): `tau`, `m`, and `multivariate` (set `True`
to skip embedding for multivariate RQA). Carries `max_interp_gap = (m−1)·τ` and a
default `theiler_window = τ`.

## RQA / CRQA (`RqaParams`)

Fixed across a study by default and written into every result's provenance.

| Parameter | Default | Units | Guidance |
|-----------|---------|-------|----------|
| `eDim`, `tLag` | — | —, frames | The committed `(m, τ)`. |
| `radius_mode` | `"fixed_rrec"` | — | `"fixed_rrec"`: you give a target %REC, the framework solves the radius (bisection); the **achieved radius** becomes the informative output. `"fixed_radius"`: you give the radius, %REC is the outcome (the paper's case-study mode). |
| `radius` | `None` | fraction of rescaled distance | For `fixed_radius`. Cases used **0.2** (auto), **0.3** (cross). Paper: a constant ε of ~15–25% of mean pairwise distance. |
| `target_rec` | `None` | % | For `fixed_rrec`. Paper target %REC ≈ **2–5%**. |
| `rescale` | `"mean"` | — | Rescale pairwise distances by their **mean** (paper default) / `"max"` / `"none"`. |
| `norm` | `"zscore"` | — | The single normalization applied before embedding (`"zscore"`, `"minmax"`, `"center"`, `"none"`). Passed once — the framework never double-normalizes. |
| `theiler` | `None → τ` | frames | Auto-RQA excludes a band of width τ around the diagonal; cross-RQA forces **0**. |
| `min_line` | `2` | frames | Minimum diagonal/vertical line length (l_min). Paper: start at 2; raise only if determinism ceilings (95–100%). Case 1 used 4. |
| `bisect_tol`, `bisect_max_iter`, `radius_hi` | `0.05`, `50`, `2.0` | %, —, — | Radius-search tolerance, iteration cap, initial upper bound. Non-convergence is reported, never hidden. |

## Windowing (`make_windows`)

| Parameter | Default | Units | Guidance |
|-----------|---------|-------|----------|
| `window_s` | — | seconds | Paper: ≳ 1000 samples per window; 30–120 s for 30–100 Hz movement data. |
| `overlap` | `0.5` | fraction | 50% overlap is the standard compromise (smooth temporal profile). |
| `max_missing` | `0.5` | fraction | Windows above this fraction of missing data are flagged. |

## Metrics

Auto/cross/multivariate RQA return the same metric family (see `RqaResult.metrics`):
`perc_recur` (RR), `perc_determ` (DET), `laminarity`, `mean_line_length`,
`std_line_length`, `maxl_found` (Lmax), `entropy`, `trapping_time`, `vmax`,
`divergence`, plus `radius_used` and convergence info. The paper's practical core
set is RR, DET, and Lmax (or mean line length); interpret them together, not in
isolation.
