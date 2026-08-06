<p align="center">
  <img src="https://raw.githubusercontent.com/MInD-Laboratory/Pose-Dynamics/main/docs/images/logo.svg" alt="pose-dynamics" width="520">
</p>

# Pose-Dynamics

A reproducible pipeline for **analyzing movement from pose data** — from any 2D or
3D pose estimator. It takes frame-by-frame landmark trajectories and turns them into
interpretable measures of behaviour at three levels, which you can use together or
on their own:

- **Clean pose data** — confidence masking, principled interpolation, zero-phase
  filtering, and an explicit, reported gap policy. Stop here if all you need is
  conditioned trajectories.
- **Linear kinematics** — displacement, velocity, acceleration and their summaries
  (mean, SD, max, RMS): *how much* and *how fast* someone moves.
- **Recurrence analysis** — *how movement is organised
  in time* — its stability, coordination, and transitions, which amplitude measures
  cannot see.

This package accompanies the methods paper [*Nonlinear methods for analyzing pose in behavioral
research*](https://arxiv.org/abs/2604.01453) and implements the main stages the paper describes, including the three case studies.

Every stage is **inspectable**: it returns an object you can plot and sanity-check.
The primary interface is a notebook, and the key parameter choices are made by the
researcher, from evidence the framework presents.

<p align="center">
  <img src="https://raw.githubusercontent.com/MInD-Laboratory/Pose-Dynamics/main/docs/images/checkpoint_and_rp.png" alt="A filter checkpoint and a recurrence plot" width="820">
</p>

<p align="center"><em>Left: a preprocessing checkpoint (raw vs. filtered — smoothed, not flattened).
Right: the recurrence plot the analysis produces.</em></p>

## Quickstart (no Python required)

Install [Docker](https://docs.docker.com/get-docker/), then from this folder:

```bash
docker compose up          # first run builds the image (a few minutes)
```

Open the `http://127.0.0.1:8888/…` link it prints — it opens **straight to the
quickstart notebook**. Choose **Run ▸ Run All Cells** and read top to bottom: it
runs on bundled example data first, so you see it work before your own data is
involved, and it teaches you to read each checkpoint and choose the two parameters
the analysis needs.

**To use your own data:** drop your [canonical CSV files](https://github.com/MInD-Laboratory/Pose-Dynamics/blob/main/docs/canonical_format.md)
into the `data/` folder next to this command (they appear in the notebook at
`/work/data/`), then change the one marked cell.

*Prefer Python?* `pip install pose-dynamics` (3.12+; needs a C/C++ compiler for the
`rqa-analysis` recurrence core), then open `notebooks/quickstart.ipynb`.

## Get features over a folder of CSVs — three commands

New here? **Start with the [Getting started guide](https://github.com/MInD-Laboratory/Pose-Dynamics/blob/main/docs/getting_started.md).** It
walks the whole workflow: no notebook, no code — inspect your data, write a config,
run it over the folder.

```bash
pose-dynamics inspect  trial.csv     # 1. see your keypoints, labelled by index
pose-dynamics new-config study.yaml  # 2. write a config (declare the feature pipeline)
pose-dynamics run      study.yaml    # 3. features + metrics for the whole folder
```

This writes per-file **feature time series** and a tidy **metrics table** (magnitude
+ recurrence). Ready-to-edit config templates are in [`configs/`](https://github.com/MInD-Laboratory/Pose-Dynamics/tree/main/configs/).

## Documentation

| Document | For |
|----------|-----|
| [Pose analysis quick guide](https://github.com/MInD-Laboratory/Pose-Dynamics/blob/main/docs/guide.html) | A decision-tree walkthrough of the whole field — acquisition, feature selection, pre-processing, and choosing between linear and recurrence analysis — for orienting before you touch a config. |
| [Getting started](https://github.com/MInD-Laboratory/Pose-Dynamics/blob/main/docs/getting_started.md) | The three-command workflow, writing a config, and choosing (τ, m). **Start here.** |
| [Canonical format](https://github.com/MInD-Laboratory/Pose-Dynamics/blob/main/docs/canonical_format.md) | The one input format. Read this to bring your own data. |
| `notebooks/quickstart.ipynb` | Interactive: read each checkpoint and commit (τ, m) on one file. |
| `notebooks/run_dataset.ipynb` | The same run from a notebook (edit a config cell). |
| [Configuration reference](https://github.com/MInD-Laboratory/Pose-Dynamics/blob/main/docs/configuration.md) | Every parameter, its default, units, and the paper's guidance. |
| [Feature steps](https://github.com/MInD-Laboratory/Pose-Dynamics/blob/main/docs/feature_steps.md) | The steps a feature pipeline is built from, and how to add your own. |

## Citation

If you use this software, please cite the accompanying paper:

```bibtex
@article{sale2026nonlinear,
  title={Nonlinear methods for analyzing pose in behavioral research},
  author={Sale, Carter and Macpherson, Margaret C and Patil, Gaurav and Miles, Kelly and Kallen, Rachel W and Wallot, Sebastian and Richardson, Michael J},
  journal={arXiv preprint arXiv:2604.01453},
  year={2026}
}
```

## Scope

**Supported.** One input format (wide CSV, one file per person per trial; 2D or 3D
and confidence inferred from the header) and one in-memory model — `PoseSequence`,
with a `Dyad` container for two-person analysis. On top of those: preprocessing
(confidence masking, run-limited interpolation, zero-phase filtering, explicit gap
policy), linear kinematic metrics, a composable feature-step library, embedding
selection (AMI/FNN), and a wrapper over
[`rqa-analysis`](https://pypi.org/project/rqa-analysis/) for auto-, cross-, and
multivariate cross-recurrence. Each level is a valid entry point; recurrence is
optional. The three published case studies reproduce as worked examples.

**Out of scope.** Pose estimation, real-time analysis, and inferential statistics —
mixed-effects models, effect sizes, and corrections live in the case-study notebooks
rather than in the library. Estimator outputs are converted to the canonical format
once, using the example converters in `examples/`. Embedding parameters are not
auto-selected: the framework presents AMI/FNN evidence and proposes values, and the
researcher commits them.

**Reproducibility.** Every result carries the resolved configuration that produced
it, and every `PoseSequence` carries an ordered provenance log of the stages applied.
Regression tests pin each case study's outputs. Releases follow semantic versioning,
with the canonical format and the `PoseSequence`/`Dyad` API as the stable public
surface.

License: MIT.
