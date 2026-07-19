"""
Case 1 (MATB) reproduction entry point.

Regenerates the paper's Case-1 figure (pupil velocity, pupil %REC, cross gaze-head
%REC, pupil %DET across Low/Moderate/High load) from the raw OpenPose exports,
using the recovered Case-1 configuration. Notebooks call these functions; they
contain no analysis logic themselves (build plan §10).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.pose_sequence import PoseSequence
from ...embedding import EmbeddingParams
from ...features import FeaturePipeline
from ...preprocessing import butterworth_filter, interpolate_gaps, mask_low_confidence
from ...rqa import RqaParams, run_auto_rqa, run_cross_rqa
from ...windowing import make_windows
from . import config as C

_COL_RE = re.compile(r"^(x|y|prob)(\d+)$")
CONDITION_ORDER = ["L", "M", "H"]
CONDITION_LABELS = {"L": "Low", "M": "Mod", "H": "High"}
COLORS = ["#d3e5f2", "#8fbcdb", "#3c7ab3"]


# ----------------------------------------------------------------------
# Loading / preprocessing
# ----------------------------------------------------------------------
def load_matb_file(path: str | Path) -> PoseSequence:
    """Load a raw MATB OpenPose CSV (``x{n},y{n},prob{n}``, 1-based) as a PoseSequence."""
    path = Path(path)
    df = pd.read_csv(path)

    def rename(col: str) -> str:
        m = _COL_RE.match(col)
        axis, n = m.group(1), int(m.group(2))
        axis = "c" if axis == "prob" else axis
        return f"{axis}{n - 1}"

    df = df.rename(columns=rename)
    # Reuse the schema-driven loader by writing nothing to disk: build arrays here.
    from ...data.schema import parse_header
    schema = parse_header(list(df.columns))
    T = len(df)
    coords = np.empty((T, schema.n_keypoints, schema.dims))
    conf = np.empty((T, schema.n_keypoints)) if schema.has_confidence else None
    for k in range(schema.n_keypoints):
        cols = schema.columns_for[k]
        for i, ax in enumerate(schema.spatial_axes):
            coords[:, k, i] = df[cols[ax]].to_numpy(float)
        if conf is not None:
            conf[:, k] = df[cols["c"]].to_numpy(float)

    participant, _, condition = path.stem.partition("_")
    return PoseSequence(
        coords=coords, keypoint_names=[f"kp{i}" for i in range(schema.n_keypoints)],
        frame_rate=C.FRAME_RATE, confidence=conf, source_file=str(path),
        meta={"participant": participant, "condition": condition.upper()},
    )


def preprocess(seq: PoseSequence) -> PoseSequence:
    """Confidence mask -> provisional interpolation -> Butterworth (Case 1 settings)."""
    seq = mask_low_confidence(seq, threshold=C.CONF_THRESHOLD)
    seq = interpolate_gaps(seq, max_gap=C.INTERP_CAP)
    seq = butterworth_filter(seq, cutoff_hz=C.FILTER_CUTOFF, order=C.FILTER_ORDER)
    return seq


def build_global_template(sequences: list[PoseSequence]) -> np.ndarray:
    """Global template = mean pose (in normalized coords) across all frames/participants."""
    scale = np.array([C.IMG_WIDTH, C.IMG_HEIGHT], float)
    per_file = [np.nanmean(seq.coords / scale, axis=0) for seq in sequences]  # each (K, 2)
    return np.nanmean(np.stack(per_file, axis=0), axis=0)


# ----------------------------------------------------------------------
# RQA parameters
# ----------------------------------------------------------------------
def _embedding() -> EmbeddingParams:
    return EmbeddingParams(tau=C.TAU, m=C.M, chosen_by="AMI/FNN committed")


def auto_params() -> RqaParams:
    return RqaParams.from_embedding(
        _embedding(), radius_mode="fixed_radius", radius=C.AUTO_RADIUS,
        rescale=C.RESCALE, theiler=C.AUTO_THEILER, min_line=C.AUTO_MINL, norm=C.NORM,
    )


def cross_params() -> RqaParams:
    return RqaParams.from_embedding(
        _embedding(), radius_mode="fixed_radius", radius=C.CROSS_RADIUS,
        rescale=C.RESCALE, min_line=C.CROSS_MINL, norm=C.NORM,
    )


# ----------------------------------------------------------------------
# Per-sequence processing
# ----------------------------------------------------------------------
def process_sequence(seq: PoseSequence, template: np.ndarray) -> list[dict]:
    """Run the Case-1 pipeline on one preprocessed sequence, returning window rows."""
    pipe = FeaturePipeline.from_config(C.feature_pipeline_config(template.tolist()))
    ctx = pipe.run(seq)
    fs = ctx.features
    pupil = fs.get("pupil_metric_mag")
    head = fs.get("head_motion_mag")

    # linear: pupil velocity (whole-signal derivative), summarized per window
    vel = np.gradient(pupil) * seq.frame_rate

    ap, cp = auto_params(), cross_params()
    windows = make_windows(
        seq.n_frames, seq.frame_rate, C.WINDOW_S, C.OVERLAP,
        valid=np.isfinite(pupil), max_missing=0.5,
    )

    rows = []
    for w in windows:
        pw = pupil[w.start:w.stop]
        hw = head[w.start:w.stop]
        vw = vel[w.start:w.stop]
        if not np.all(np.isfinite(pw)) or not np.all(np.isfinite(hw)):
            continue  # skip windows spanning a residual gap
        auto = run_auto_rqa(pw, ap)
        cross = run_cross_rqa(hw, pw, cp)
        rows.append({
            "participant": seq.meta["participant"],
            "condition": seq.meta["condition"],
            "window_index": w.index,
            "flagged": w.flagged,
            "pupil_metric_vel_rms": float(np.sqrt(np.mean(vw ** 2))),
            "pupil_metric_perc_recur": auto.metrics["perc_recur"],
            "pupil_metric_perc_determ": auto.metrics["perc_determ"],
            "crqa_head_pupil_mag_perc_recur": cross.metrics["perc_recur"],
        })
    return rows


def run_reproduction(
    paths: list[str | Path],
    template_sample: int = 24,
    progress: bool = True,
) -> pd.DataFrame:
    """Load, preprocess, and process a set of MATB files into a tidy results table.

    Streams one file at a time to bound memory. The global template is built from
    the first ``template_sample`` files (a mean pose is stable from a modest
    sample); pass ``template_sample >= len(paths)`` to use all files.
    """
    paths = list(paths)

    # Pass 1: global template from a sample (loaded then discarded).
    sample = paths[: min(template_sample, len(paths))]
    sample_seqs = []
    for i, p in enumerate(sample):
        if progress:
            print(f"[template {i + 1}/{len(sample)}] {Path(p).name}")
        sample_seqs.append(preprocess(load_matb_file(p)))
    template = build_global_template(sample_seqs)
    del sample_seqs

    # Pass 2: process every file, one at a time.
    rows: list[dict] = []
    for i, p in enumerate(paths):
        seq = preprocess(load_matb_file(p))
        if progress:
            print(f"[analyze {i + 1}/{len(paths)}] {seq.meta['participant']}_{seq.meta['condition']}")
        rows.extend(process_sequence(seq, template))
        del seq
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
_PANELS = [
    ("pupil_metric_vel_rms", "Avg Pupil Velocity"),
    ("pupil_metric_perc_recur", "Avg Pupil %REC"),
    ("crqa_head_pupil_mag_perc_recur", "Cross Gaze-Head %REC"),
    ("pupil_metric_perc_determ", "Avg Pupil %DET"),
]


def plot_case1_figure(df: pd.DataFrame, axes=None):
    """Reproduce the 2x2 Case-1 figure (mean +/- SEM across windows, by condition)."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = np.asarray(axes).flatten()

    for (metric, ylab), ax in zip(_PANELS, axes):
        stats = (
            df[["condition", metric]].dropna()
            .groupby("condition", observed=True)[metric]
            .agg(["mean", "sem"]).reindex(CONDITION_ORDER)
        )
        ax.bar(
            range(len(CONDITION_ORDER)), stats["mean"], yerr=stats["sem"],
            color=COLORS, edgecolor="black", linewidth=2, capsize=5,
        )
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
        ax.set_ylabel(ylab, fontsize=12)
        lo = float((stats["mean"] - stats["sem"]).min())
        hi = float((stats["mean"] + stats["sem"]).max())
        span = max(hi - lo, abs(hi) * 0.15, 1e-9)
        ax.set_ylim(max(0, lo - span * 0.2), hi + span * 0.2)
    axes[0].figure.tight_layout()
    return axes
