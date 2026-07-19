"""
Case 3 (Mirror Game) reproduction entry point.

Reproduces the Case-3 finding: how visual coupling (back-to-back / unidirectional
/ face-to-face) shapes whole-body movement magnitude (acceleration RMS) and
leader-follower coordination (cross-recurrence rate and maximum diagonal line
length), from a five-keypoint subset.

The ZED stream is variable-rate, so each trial is resampled to a uniform 30 Hz
grid from its timestamps. Per-keypoint magnitude (the CRQA signal) is invariant to
the rigid per-trial Procrustes rotation, so alignment is not needed for this
figure (it matters only for the PCA diagnostic). Notebooks call these functions.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.pose_sequence import PoseSequence
from ...features import FeaturePipeline
from ...preprocessing import butterworth_filter
from ...rqa import RqaParams, run_cross_rqa
from ...embedding import EmbeddingParams
from . import config as C

_FILE_RE = re.compile(r"^P(\d+)_T(\d+)_P(\d+)_pose_3d$")
COLORS = ["#c9d8e8", "#7fa8cf", "#2f6fae"]


# ----------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------
def default_conditions_csv() -> Path:
    """Path to the bundled Mirror_Game conditions table."""
    return Path(__file__).parent / "data" / "Mirror_Game_Conditions.csv"


def load_condition_map(conditions_csv: str | Path | None = None) -> dict[tuple[int, int], str]:
    """Map ``(pair, trial 1..12)`` -> condition from the Mirror_Game conditions CSV.

    Trials 1-6 are block 1 (columns ``block1_1..block1_6``), 7-12 are block 2.
    Defaults to the bundled conditions table.
    """
    df = pd.read_csv(conditions_csv or default_conditions_csv())
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    out: dict[tuple[int, int], str] = {}
    for _, row in df.iterrows():
        pair = int(row["Pair"])
        for t in range(1, 7):
            out[(pair, t)] = str(row[f"block1_{t}"]).strip()
            out[(pair, t + 6)] = str(row[f"block2_{t}"]).strip()
    return out


def parse_file(path: str | Path) -> tuple[int, int, int]:
    """Return ``(pair, trial, person)`` from a ``P###_T##_P#_pose_3d.csv`` name."""
    m = _FILE_RE.match(Path(path).stem)
    if not m:
        raise ValueError(f"Unexpected mirror-game filename: {path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ----------------------------------------------------------------------
# Loading + resampling
# ----------------------------------------------------------------------
def load_and_resample(path: str | Path) -> PoseSequence:
    """Load a ZED 3-D export and resample to a uniform 30 Hz grid from timestamps."""
    df = pd.read_csv(path)
    coord_cols = [c for c in df.columns if c[0] in "xyz" and c[1:].isdigit()]
    n_kp = len({c[1:] for c in coord_cols})
    T = len(df)

    # source time (seconds), from absolute timestamps (fallback: cumulative dt_ms)
    if "timestamp_ns" in df.columns:
        ts = df["timestamp_ns"].to_numpy(float)
        t_src = (ts - ts[0]) / 1e9
    else:
        t_src = np.cumsum(df["dt_ms"].to_numpy(float)) / 1000.0
        t_src = t_src - t_src[0]

    # uniform target grid
    duration = t_src[-1]
    t_dst = np.arange(0.0, duration, 1.0 / C.TARGET_RATE)

    coords = np.empty((len(t_dst), n_kp, 3))
    for k in range(n_kp):
        for i, ax in enumerate("xyz"):
            coords[:, k, i] = np.interp(t_dst, t_src, df[f"{ax}{k}"].to_numpy(float))

    pair, trial, person = parse_file(path)
    return PoseSequence(
        coords=coords, keypoint_names=[f"kp{i}" for i in range(n_kp)],
        frame_rate=C.TARGET_RATE, source_file=str(path),
        meta={"pair": pair, "trial": trial, "person": person},
    )


# ----------------------------------------------------------------------
# Feature pipeline (centre on pelvis -> filter -> select 5 kp -> magnitudes)
# ----------------------------------------------------------------------
def feature_pipeline_config() -> list[dict]:
    return [
        {"step": "center", "params": {"reference": C.PELVIS}},
        {"step": "select_keypoints",
         "params": {"indices": C.SUBSET_INDICES, "names": C.SUBSET_NAMES}},
    ]


def cross_params() -> RqaParams:
    return RqaParams.from_embedding(
        EmbeddingParams(tau=C.TAU, m=C.M), radius_mode="fixed_rrec",
        target_rec=C.TARGET_REC, rescale=C.RESCALE, min_line=C.MIN_LINE, norm=C.NORM,
    )


def _prep_subset(seq: PoseSequence) -> PoseSequence:
    """Centre on pelvis, low-pass filter, and keep the five-keypoint subset."""
    seq = butterworth_filter(
        FeaturePipeline.from_config([{"step": "center", "params": {"reference": C.PELVIS}}])
        .run(seq).pose,
        cutoff_hz=C.FILTER_CUTOFF, order=C.FILTER_ORDER,
    )
    return FeaturePipeline.from_config(
        [{"step": "select_keypoints",
          "params": {"indices": C.SUBSET_INDICES, "names": C.SUBSET_NAMES}}]
    ).run(seq).pose


# ----------------------------------------------------------------------
# Per-dyad processing
# ----------------------------------------------------------------------
def process_dyad(p1: PoseSequence, p2: PoseSequence, condition: str) -> dict:
    """Kinematics (acceleration RMS) and per-keypoint CRQA (%REC, Lmax), averaged."""
    a = _prep_subset(p1)
    b = _prep_subset(p2)
    n = min(a.n_frames, b.n_frames)               # shared-clock: trim to overlap
    ac, bc = a.coords[:n], b.coords[:n]
    fps = C.TARGET_RATE

    # kinematics: acceleration magnitude RMS, averaged over the five keypoints
    accel_rms = []
    for coords in (ac, bc):
        vel = np.gradient(coords, axis=0) * fps
        acc = np.gradient(vel, axis=0) * fps
        acc_mag = np.linalg.norm(acc, axis=2)      # (n, 5)
        accel_rms.append(np.sqrt(np.mean(acc_mag ** 2, axis=0)))  # per keypoint
    accel_rms = np.mean(accel_rms, axis=0).mean()  # avg over persons + keypoints

    # per-keypoint CRQA on magnitude time series, averaged over the five keypoints.
    # Under fixed-rec mode %REC is pinned to the target (a convergence check); the
    # achieved radius is the informative density measure.
    cp = cross_params()
    recs, lmaxs, radii = [], [], []
    for k in range(len(C.SUBSET_INDICES)):
        mag_a = np.linalg.norm(ac[:, k, :], axis=1)
        mag_b = np.linalg.norm(bc[:, k, :], axis=1)
        res = run_cross_rqa(mag_a, mag_b, cp)
        recs.append(res.rec_rate)
        lmaxs.append(res.metrics["maxl_found"])
        radii.append(res.radius_used)

    return {
        "pair": p1.meta["pair"], "trial": p1.meta["trial"], "condition": condition,
        "accel_rms": float(accel_rms),
        "cross_perc_recur": float(np.mean(recs)),   # pinned ~2.5% (convergence check)
        "cross_radius": float(np.mean(radii)),       # informative density measure
        "cross_lmax": float(np.mean(lmaxs)),
    }


def run_reproduction(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    pairs: list[int] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Process all (or selected) dyad-trials into a tidy results table.

    ``conditions_csv`` defaults to the bundled Mirror_Game conditions table.
    """
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)

    # discover pair/trial with both persons present (ignore macOS ._ files)
    files = [p for p in data_dir.glob("P*_T*_P*_pose_3d.csv") if not p.name.startswith("._")]
    by_key: dict[tuple[int, int], dict[int, Path]] = {}
    for f in files:
        pr, tr, pe = parse_file(f)
        by_key.setdefault((pr, tr), {})[pe] = f

    keys = sorted(k for k, v in by_key.items() if {1, 2} <= set(v))
    if pairs is not None:
        keys = [k for k in keys if k[0] in pairs]

    rows = []
    for i, (pr, tr) in enumerate(keys):
        cond = cond_map.get((pr, tr))
        if cond not in C.CONDITION_ORDER:
            continue
        if progress:
            print(f"[{i + 1}/{len(keys)}] pair {pr} trial {tr} ({cond})")
        p1 = load_and_resample(by_key[(pr, tr)][1])
        p2 = load_and_resample(by_key[(pr, tr)][2])
        # need enough frames for delay embedding + a meaningful recurrence plot
        min_frames = (C.M + 1) * C.TAU + 50
        if min(p1.n_frames, p2.n_frames) < min_frames:
            if progress:
                print(f"    skipped (too short: {min(p1.n_frames, p2.n_frames)} frames)")
            continue
        rows.append(process_dyad(p1, p2, cond))
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
_PANELS = [
    ("accel_rms", "Acceleration RMS"),
    ("cross_radius", "Cross radius (@2.5% REC)"),
    ("cross_lmax", "Cross Lmax"),
]


def plot_case3_figure(df: pd.DataFrame, axes=None):
    """Group-averaged (mean +/- SEM) kinematics and CRQA by visual-coupling condition."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = np.asarray(axes).flatten()
    for (metric, ylab), ax in zip(_PANELS, axes):
        stats = (
            df[["condition", metric]].dropna()
            .groupby("condition", observed=True)[metric]
            .agg(["mean", "sem"]).reindex(C.CONDITION_ORDER)
        )
        ax.bar(range(len(C.CONDITION_ORDER)), stats["mean"], yerr=stats["sem"],
               color=COLORS, edgecolor="black", linewidth=2, capsize=5)
        ax.set_xticks(range(len(C.CONDITION_ORDER)))
        ax.set_xticklabels([c.upper() for c in C.CONDITION_ORDER])
        ax.set_ylabel(ylab, fontsize=12)
    axes[0].figure.tight_layout()
    return axes
