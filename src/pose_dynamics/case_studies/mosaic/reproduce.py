"""
Case 2 (MOSAIC) reproduction entry point.

Case 2 quantifies *interpersonal* coordination during conversation across four
background-noise levels, via cross-recurrence between the two partners' ROI
velocity-magnitude signals. That dyadic analysis needs **both** partners (the
left- and right-camera files) and many pairs; :func:`run_reproduction` implements
it but requires the full dataset.

The individual-level pieces — ROI velocity-magnitude linear metrics and auto-RQA —
need only one participant and are exercised by :func:`run_individual`.

Notebooks call these functions; they hold no analysis logic themselves.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.pose_sequence import PoseSequence
from ...features import FeaturePipeline
from ...preprocessing import butterworth_filter, interpolate_gaps, mask_low_confidence
from ...rqa import RqaParams, run_auto_rqa, run_cross_rqa
from ...embedding import EmbeddingParams
from ...windowing import make_windows
from . import config as C

_FILE_RE = re.compile(r"^S(\d+)_T(\d+)_(left|right)$")
COLORS = ["#cfe0ef", "#93b8db", "#4f8cc0", "#20517e"]


# ----------------------------------------------------------------------
# Conditions / filenames
# ----------------------------------------------------------------------
def parse_file(path: str | Path) -> tuple[int, int, str]:
    """Return ``(session, trial, camera)`` from a ``S###_T#_right.csv`` name."""
    m = _FILE_RE.match(Path(path).stem)
    if not m:
        raise ValueError(f"Unexpected MOSAIC filename: {path}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def default_conditions_csv() -> Path:
    """Path to the bundled MOSAIC conditions table."""
    return Path(__file__).parent / "data" / "Mosaic_Conditions.csv"


def load_condition_map(conditions_csv: str | Path | None = None) -> dict[tuple[int, int], str]:
    """Map ``(session, trial)`` -> condition from the MOSAIC conditions CSV.

    Defaults to the bundled conditions table.
    """
    df = pd.read_csv(conditions_csv or default_conditions_csv())
    return {(int(r["session"]), int(r["trial"])): str(r["condition"]).strip()
            for _, r in df.iterrows()}


# ----------------------------------------------------------------------
# ROI resolution + loading
# ----------------------------------------------------------------------
def resolve_rois(columns: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    """Resolve ROI keypoint names present in a header.

    Returns ``(keypoint_names, roi_index_map)`` where the index map gives, per ROI,
    the positions of its keypoints within ``keypoint_names`` (their union).
    """
    base = sorted({c.replace("_x_offset", "").replace("_y_offset", "").replace("_confidence", "")
                   for c in columns if c.endswith(("_x_offset", "_y_offset", "_confidence"))})

    roi_names: dict[str, list[str]] = {}
    for roi, exact in C.ROI_EXACT.items():
        names = [n for n in exact if n in base]
        if roi == "centre_face":
            names += [n for n in base if any(s in n for s in C.CENTRE_FACE_SUBSTRINGS)
                      and n not in names]
        roi_names[roi] = names

    keypoint_names = sorted({n for names in roi_names.values() for n in names})
    pos = {n: i for i, n in enumerate(keypoint_names)}
    roi_index_map = {roi: [pos[n] for n in names] for roi, names in roi_names.items() if names}
    return keypoint_names, roi_index_map


def load_mosaic_file(path: str | Path) -> tuple[PoseSequence, dict[str, list[int]]]:
    """Load a MOSAIC OpenPose file, keeping only ROI keypoints."""
    df = pd.read_csv(path)
    keypoint_names, roi_index_map = resolve_rois(list(df.columns))
    T = len(df)
    coords = np.empty((T, len(keypoint_names), 2))
    conf = np.empty((T, len(keypoint_names)))
    for i, name in enumerate(keypoint_names):
        coords[:, i, 0] = df[f"{name}_x_offset"].to_numpy(float)
        coords[:, i, 1] = df[f"{name}_y_offset"].to_numpy(float)
        conf[:, i] = df[f"{name}_confidence"].to_numpy(float)

    session, trial, camera = parse_file(path)
    seq = PoseSequence(
        coords=coords, keypoint_names=keypoint_names, frame_rate=C.FRAME_RATE,
        confidence=conf, source_file=str(path),
        meta={"session": session, "trial": trial, "camera": camera},
    )
    return seq, roi_index_map


# ----------------------------------------------------------------------
# Preprocessing + ROI velocity-magnitude signals
# ----------------------------------------------------------------------
def _downsample(seq: PoseSequence, factor: int) -> PoseSequence:
    """Decimate by an integer factor after low-pass filtering (60 -> 30 Hz)."""
    return PoseSequence(
        coords=seq.coords[::factor], keypoint_names=seq.keypoint_names,
        frame_rate=seq.frame_rate / factor,
        confidence=None if seq.confidence is None else seq.confidence[::factor],
        mask=seq.mask[::factor], source_file=seq.source_file,
        provenance=seq.provenance.appended("downsample", {"factor": factor}),
        meta=dict(seq.meta),
    )


def roi_velocity_signals(seq: PoseSequence, roi_index_map: dict[str, list[int]]):
    """Preprocess and reduce to one ROI velocity-magnitude signal per ROI.

    mask -> interpolate -> filter -> normalize to [0,1] -> downsample to 30 Hz,
    then ROI centroid -> velocity magnitude.
    """
    seq = mask_low_confidence(seq, C.CONF_THRESHOLD)
    seq = interpolate_gaps(seq, C.INTERP_CAP)
    seq = butterworth_filter(seq, C.FILTER_CUTOFF, C.FILTER_ORDER)
    seq = _downsample(seq, int(round(C.FRAME_RATE / C.TARGET_RATE)))

    pipe = FeaturePipeline.from_config([
        {"step": "coordinate_normalization",
         "params": {"width": C.VIDEO_WIDTH, "height": C.VIDEO_HEIGHT, "mode": "unit"}},
        {"step": "roi_centroid", "params": {"rois": roi_index_map}},
        {"step": "velocity_magnitude", "params": {"method": "diff"}},
    ])
    return pipe.run(seq).features  # columns: {roi}_speed


# ----------------------------------------------------------------------
# RQA params
# ----------------------------------------------------------------------
def _embedding() -> EmbeddingParams:
    return EmbeddingParams(tau=C.TAU, m=C.M)


def auto_params() -> RqaParams:
    return RqaParams.from_embedding(
        _embedding(), radius_mode="fixed_radius", radius=C.RADIUS, rescale=C.RESCALE,
        theiler=C.AUTO_THEILER, min_line=C.MIN_LINE, norm=C.NORM)


def cross_params() -> RqaParams:
    return RqaParams.from_embedding(
        _embedding(), radius_mode="fixed_radius", radius=C.RADIUS, rescale=C.RESCALE,
        min_line=C.MIN_LINE, norm=C.NORM)


# ----------------------------------------------------------------------
# Individual-level analysis (runnable with one participant)
# ----------------------------------------------------------------------
def run_individual(files: list[str | Path], conditions_csv: str | Path | None = None,
                   progress: bool = True) -> pd.DataFrame:
    """Per-window individual ROI linear metrics + auto-RQA (one participant)."""
    cond_map = load_condition_map(conditions_csv)
    ap = auto_params()
    rows = []
    for f in files:
        seq, roi_map = load_mosaic_file(f)
        cond = cond_map.get((seq.meta["session"], seq.meta["trial"]))
        feats = roi_velocity_signals(seq, roi_map)
        windows = make_windows(feats.n_frames, feats.frame_rate, C.WINDOW_S, C.OVERLAP)
        for roi in roi_map:
            sig = feats.get(f"{roi}_speed")
            for w in windows:
                s = sig[w.start:w.stop]
                if not np.all(np.isfinite(s)):
                    continue
                auto = run_auto_rqa(s, ap)
                rows.append({
                    "session": seq.meta["session"], "trial": seq.meta["trial"],
                    "condition": cond, "roi": roi, "window": w.index,
                    "rms": float(np.sqrt(np.mean(s ** 2))),
                    "mean_vel": float(np.mean(s)), "sd_vel": float(np.std(s)),
                    "perc_recur": auto.metrics["perc_recur"],
                    "perc_determ": auto.metrics["perc_determ"],
                })
        if progress:
            print(f"  {Path(f).name}: {cond}")
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Dyadic analysis (needs BOTH partners; the paper's figure)
# ----------------------------------------------------------------------
def process_dyad(right: PoseSequence, left: PoseSequence, roi_map: dict[str, list[int]],
                 condition: str) -> list[dict]:
    """Windowed interpersonal CRQA + linear cross-correlation between two partners."""
    fr = roi_velocity_signals(right, roi_map)
    fl = roi_velocity_signals(left, roi_map)
    n = min(fr.n_frames, fl.n_frames)
    cp = cross_params()
    rows = []
    for roi in roi_map:
        a = fr.get(f"{roi}_speed")[:n]
        b = fl.get(f"{roi}_speed")[:n]
        windows = make_windows(n, fr.frame_rate, C.WINDOW_S, C.OVERLAP)
        for w in windows:
            aw, bw = a[w.start:w.stop], b[w.start:w.stop]
            if not (np.all(np.isfinite(aw)) and np.all(np.isfinite(bw))):
                continue
            cross = run_cross_rqa(aw, bw, cp)
            # linear coupling: zero-lag cross-correlation of z-scored velocity mag
            za = (aw - aw.mean()) / (aw.std() + 1e-8)
            zb = (bw - bw.mean()) / (bw.std() + 1e-8)
            rows.append({
                "condition": condition, "roi": roi, "window": w.index,
                "cross_perc_recur": cross.metrics["perc_recur"],
                "cross_perc_determ": cross.metrics["perc_determ"],
                "cross_lmax": cross.metrics["maxl_found"],
                "xcorr_lag0": float(np.mean(za * zb)),
            })
    return rows


def run_reproduction(data_dir: str | Path, conditions_csv: str | Path | None = None,
                     progress: bool = True) -> pd.DataFrame:
    """Full dyadic reproduction. Requires both camera files per session-trial."""
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    files = [p for p in data_dir.glob("S*_T*_*.csv") if not p.name.startswith("._")]
    by_key: dict[tuple[int, int], dict[str, Path]] = {}
    for f in files:
        s, t, cam = parse_file(f)
        by_key.setdefault((s, t), {})[cam] = f

    rows = []
    keys = sorted(k for k, v in by_key.items() if {"left", "right"} <= set(v))
    if not keys:
        raise FileNotFoundError(
            "No session-trial has BOTH 'left' and 'right' camera files; the dyadic "
            "reproduction needs both partners. Only individual-level analysis is "
            "possible with single-camera data (see run_individual)."
        )
    for (s, t) in keys:
        cond = cond_map.get((s, t))
        if cond not in C.CONDITION_ORDER:
            continue
        right, roi_map = load_mosaic_file(by_key[(s, t)]["right"])
        left, _ = load_mosaic_file(by_key[(s, t)]["left"])
        if progress:
            print(f"  session {s} trial {t} ({cond})")
        rows.extend(process_dyad(right, left, roi_map, cond))
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_individual_figure(df: pd.DataFrame, roi: str = "arms", axes=None):
    """Individual ROI metrics by condition (mean +/- SEM) for one ROI."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = np.asarray(axes).flatten()
    sub = df[df["roi"] == roi]
    for (metric, ylab), ax in zip(
        [("rms", f"{roi} RMS velocity"), ("perc_recur", f"{roi} %REC"),
         ("perc_determ", f"{roi} %DET")], axes):
        stats = (sub[["condition", metric]].dropna()
                 .groupby("condition", observed=True)[metric].agg(["mean", "sem"])
                 .reindex(C.CONDITION_ORDER))
        ax.bar(range(len(C.CONDITION_ORDER)), stats["mean"], yerr=stats["sem"],
               color=COLORS, edgecolor="black", linewidth=2, capsize=5)
        ax.set_xticks(range(len(C.CONDITION_ORDER)))
        ax.set_xticklabels(C.CONDITION_ORDER, rotation=30, ha="right")
        ax.set_ylabel(ylab, fontsize=11)
    axes[0].figure.tight_layout()
    return axes
