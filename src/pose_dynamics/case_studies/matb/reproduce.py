"""
Case 1 (MATB) reproduction entry point.

Regenerates the paper's Case-1 figure (pupil velocity, pupil %REC, cross gaze-head
%REC, pupil %DET across Low/Moderate/High load) from the raw OpenPose exports,
using the recovered Case-1 configuration. Notebooks call these functions; they
contain no analysis logic themselves (build plan §10).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.pose_sequence import PoseSequence
from ...embedding import EmbeddingParams
from ...preprocessing import butterworth_filter, interpolate_gaps, mask_low_confidence
from ...rqa import RqaParams, run_auto_rqa, run_cross_rqa
from ...windowing import make_windows
from . import config as C
from .features import REL_IDXS, load_global_template, matb_features

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
    seq = interpolate_gaps(seq, max_gap=C.INTERP_CAP, edge_fill=True)
    seq = butterworth_filter(seq, cutoff_hz=C.FILTER_CUTOFF, order=C.FILTER_ORDER, by_segment=True)
    return seq


def build_global_template(sequences: list[PoseSequence] | None = None,
                          path: str | Path | None = None) -> np.ndarray:
    """The (23, 2) global template in screen-normalized coordinates.

    Prefers the parent analysis repository's saved ``global_template.csv`` when
    ``path`` is given (or ``MATB_GLOBAL_TEMPLATE`` is set), since the published
    results were produced against that exact template. Falls back to the mean
    relevant-landmark pose across ``sequences``.
    """
    import os
    path = path or os.environ.get("MATB_GLOBAL_TEMPLATE")
    if path and Path(path).exists():
        return load_global_template(path)
    if not sequences:
        raise ValueError("supply either a template path or sequences to average.")
    scale = np.array([C.IMG_WIDTH, C.IMG_HEIGHT], float)
    rel0 = [i - 1 for i in REL_IDXS]
    per_file = [np.nanmean(seq.coords[:, rel0, :] / scale, axis=0) for seq in sequences]
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


def cross_params(min_line: int | None = None) -> RqaParams:
    """Cross-RQA parameters, optionally overriding the committed ``l_min``.

    The parent analysis repository sets ``minl: 2`` for CRQA while the published
    methods text quotes a single minimum line length of 4, so the two values need
    to be comparable side by side.
    """
    return RqaParams.from_embedding(
        _embedding(), radius_mode="fixed_radius", radius=C.CROSS_RADIUS,
        rescale=C.RESCALE, min_line=C.CROSS_MINL if min_line is None else min_line,
        norm=C.NORM,
    )


# ----------------------------------------------------------------------
# Per-sequence processing
# ----------------------------------------------------------------------
def _assemble_features(seq: PoseSequence, template: np.ndarray) -> dict[str, np.ndarray]:
    """Derive the Case-1 analysis signals from a preprocessed sequence.

    Delegates to :mod:`.features`, which is a faithful port of the parent
    analysis repository's feature extraction (verified to ~1e-15 against its
    saved per-frame output). ``template`` is the ``(23, 2)`` global template in
    screen-normalized coordinates, ordered by ``features.REL_IDXS``.
    """
    coords = seq.coords / np.array([C.IMG_WIDTH, C.IMG_HEIGHT], float)
    return matb_features(coords, template)


def _summarise(x: np.ndarray, prefix: str, frame_rate: float) -> dict[str, float]:
    """Linear kinematic summary of one window: stats x position/velocity/acceleration."""
    vel = np.gradient(x) * frame_rate
    acc = np.gradient(vel) * frame_rate
    out: dict[str, float] = {}
    for arr, order in ((x, "pos"), (vel, "vel"), (acc, "accel")):
        for stat in C.LINEAR_STATS:
            key = f"{prefix}_{order}_{stat}"
            if stat == "rms":
                out[key] = float(np.sqrt(np.mean(arr ** 2)))
            else:
                out[key] = float(getattr(np, stat)(arr))
    return out


def process_sequence(
    seq: PoseSequence,
    template: np.ndarray,
    cross_min_lines: Sequence[int] = (C.CROSS_MINL,),
) -> list[dict]:
    """Run the Case-1 pipeline on one preprocessed sequence, returning window rows.

    Each row is one (trial, window) and carries, for every feature in
    ``C.AUTO_FEATURES``: the linear kinematic summaries and the full auto-RQA
    metric set; plus the full cross-RQA metric set for every pair in
    ``C.CROSS_PAIRS``.

    ``cross_min_lines`` gives the cross-RQA minimum line lengths to evaluate. A
    single value (the default) yields the usual ``crqa_{pair}_{metric}`` columns;
    several values yield ``crqa_l{n}_{pair}_{metric}`` for each, so that competing
    ``l_min`` choices can be compared without re-running the auto-RQA.
    """
    feats = _assemble_features(seq, template)
    ap = auto_params()
    cross_min_lines = tuple(cross_min_lines)
    cps = [(n, cross_params(n)) for n in cross_min_lines]
    multi = len(cps) > 1

    # Window inclusion is decided PER SIGNAL, not jointly: a window is analysed
    # for a given feature whenever that feature is gap-free across it, even if a
    # different feature has a gap there. Requiring every signal to be
    # simultaneously valid would restrict all features to the intersection of
    # their coverage, which changes each feature's effective sample and shifts
    # its coefficients. This matches the per-column loop in the parent analysis
    # repository (Pose/process_pose_recurrence.py).
    windows = make_windows(
        seq.n_frames, seq.frame_rate, C.WINDOW_S, C.OVERLAP,
        valid=np.ones(seq.n_frames, bool), max_missing=1.0,
    )

    rows = []
    for w in windows:
        row: dict[str, float | str | bool | int] = {
            "participant": seq.meta["participant"],
            "condition": seq.meta["condition"],
            "window_index": w.index,
            "flagged": w.flagged,
        }
        # A non-zero err_code means the recurrence routine could not produce a
        # valid result for that window (e.g. a degenerate distance matrix). Those
        # windows are dropped rather than carried forward: retaining them admits
        # metrics the library itself flagged as invalid, and because such windows
        # are not distributed evenly across conditions, keeping them biases the
        # condition means. This matches the `if err != 0: continue` guard in the
        # parent analysis repository.
        for name in C.AUTO_FEATURES:
            sig = feats[name][w.start:w.stop]
            if not np.all(np.isfinite(sig)):
                continue
            auto = run_auto_rqa(sig, ap)
            if auto.err_code != 0:
                continue
            row.update(_summarise(sig, name, seq.frame_rate))
            for key, val in auto.metrics.items():
                row[f"{name}_{key}"] = val
        for gaze, head in C.CROSS_PAIRS:
            gw = feats[gaze][w.start:w.stop]
            hw = feats[head][w.start:w.stop]
            if not (np.all(np.isfinite(gw)) and np.all(np.isfinite(hw))):
                continue
            for n, cp in cps:
                cross = run_cross_rqa(hw, gw, cp)
                if cross.err_code != 0:
                    continue
                tag = f"crqa_l{n}" if multi else "crqa"
                for key, val in cross.metrics.items():
                    row[f"{tag}_{gaze}_{key}"] = val
        if len(row) > 4:      # at least one signal contributed
            rows.append(row)
    return rows


def run_reproduction(
    paths: list[str | Path],
    template_sample: int = 24,
    progress: bool = True,
    cross_min_lines: Sequence[int] = (C.CROSS_MINL,),
) -> pd.DataFrame:
    """Load, preprocess, and process a set of MATB files into a tidy results table.

    Streams one file at a time to bound memory. The global template is built from
    the first ``template_sample`` files (a mean pose is stable from a modest
    sample); pass ``template_sample >= len(paths)`` to use all files.

    ``cross_min_lines`` is forwarded to :func:`process_sequence`; pass several
    values to emit cross-RQA at each minimum line length in one pass.
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
        rows.extend(process_sequence(seq, template, cross_min_lines))
        del seq
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Parameter sensitivity sweeps
# ----------------------------------------------------------------------
def _sweep_windows(paths, template_sample: int, progress: bool):
    """Yield (participant, condition, window_index, {name: window signal})."""
    paths = list(paths)
    sample = paths[: min(template_sample, len(paths))]
    template = build_global_template([preprocess(load_matb_file(p)) for p in sample])
    needed = set(C.AUTO_FEATURES) | {b for _, b in C.CROSS_PAIRS}
    for i, p in enumerate(paths):
        seq = preprocess(load_matb_file(p))
        if progress:
            print(f"[sweep {i + 1}/{len(paths)}] {seq.meta['participant']}_{seq.meta['condition']}")
        feats = _assemble_features(seq, template)
        for w in make_windows(seq.n_frames, seq.frame_rate, C.WINDOW_S, C.OVERLAP,
                              valid=np.isfinite(feats["pupil_metric_mag"]), max_missing=0.5):
            cut = {n: feats[n][w.start:w.stop] for n in needed}
            if all(np.all(np.isfinite(v)) for v in cut.values()):
                yield seq.meta["participant"], seq.meta["condition"], w.index, cut


def run_embedding_sweep(
    paths: list[str | Path],
    taus: list[int] | None = None,
    ms: list[int] | None = None,
    features: list[str] | None = None,
    template_sample: int = 24,
    progress: bool = True,
) -> pd.DataFrame:
    """Auto- and cross-RQA across a grid of embedding delays and dimensions.

    Asks whether the Case-1 condition effects depend on the committed
    ``(tau, m) = (20, 4)``. Radii are held at the committed values (``AUTO_RADIUS``
    for auto-RQA, ``CROSS_RADIUS`` for cross-RQA); ``run_radius_sweep`` varies
    those instead. One row per (participant, condition, window, feature, tau, m).
    """
    taus = taus or [10, 15, 20, 25, 30]
    ms = ms or [3, 4, 5, 6]
    features = features or ["pupil_metric_mag"]
    rows = []
    for participant, condition, widx, cut in _sweep_windows(paths, template_sample, progress):
        n = len(cut["pupil_metric_mag"])
        for tau in taus:
            for m in ms:
                if n - (m - 1) * tau < 100:      # too little series left to embed
                    continue
                emb = EmbeddingParams(tau=tau, m=m)
                ap = RqaParams.from_embedding(
                    emb, radius_mode="fixed_radius", radius=C.AUTO_RADIUS,
                    rescale=C.RESCALE, theiler=C.AUTO_THEILER, min_line=C.AUTO_MINL, norm=C.NORM)
                cp = RqaParams.from_embedding(
                    emb, radius_mode="fixed_radius", radius=C.CROSS_RADIUS,
                    rescale=C.RESCALE, min_line=C.CROSS_MINL, norm=C.NORM)
                base = {"participant": participant, "condition": condition,
                        "window_index": widx, "tau": tau, "m": m}
                for name in features:
                    res = run_auto_rqa(cut[name], ap)
                    if res.err_code != 0:      # same guard as ``process_sequence``
                        continue
                    rows.append({**base, "analysis": "auto", "feature": name,
                                 **{k: float(v) for k, v in res.metrics.items()}})
                for gaze, head in C.CROSS_PAIRS:
                    if gaze not in features:
                        continue
                    res = run_cross_rqa(cut[head], cut[gaze], cp)
                    if res.err_code != 0:
                        continue
                    rows.append({**base, "analysis": "cross", "feature": gaze,
                                 **{k: float(v) for k, v in res.metrics.items()}})
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df


def run_radius_sweep(
    paths: list[str | Path],
    radii: list[float] | None = None,
    features: list[str] | None = None,
    template_sample: int = 24,
    progress: bool = True,
) -> pd.DataFrame:
    """Auto- and cross-RQA across a grid of recurrence radii at the committed (tau, m).

    The companion to :func:`run_embedding_sweep`. Because a fixed-radius result
    depends on the threshold chosen, this asks whether the condition effects hold
    across the plausible range rather than only at the committed radius.
    """
    radii = radii or [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    features = features or ["pupil_metric_mag"]
    emb = _embedding()
    rows = []
    for participant, condition, widx, cut in _sweep_windows(paths, template_sample, progress):
        for radius in radii:
            ap = RqaParams.from_embedding(
                emb, radius_mode="fixed_radius", radius=radius, rescale=C.RESCALE,
                theiler=C.AUTO_THEILER, min_line=C.AUTO_MINL, norm=C.NORM)
            cp = RqaParams.from_embedding(
                emb, radius_mode="fixed_radius", radius=radius, rescale=C.RESCALE,
                min_line=C.CROSS_MINL, norm=C.NORM)
            base = {"participant": participant, "condition": condition,
                    "window_index": widx, "radius": radius}
            for name in features:
                res = run_auto_rqa(cut[name], ap)
                if res.err_code != 0:          # same guard as ``process_sequence``
                    continue
                rows.append({**base, "analysis": "auto", "feature": name,
                             **{k: float(v) for k, v in res.metrics.items()}})
            for gaze, head in C.CROSS_PAIRS:
                if gaze not in features:
                    continue
                res = run_cross_rqa(cut[head], cut[gaze], cp)
                if res.err_code != 0:
                    continue
                rows.append({**base, "analysis": "cross", "feature": gaze,
                             **{k: float(v) for k, v in res.metrics.items()}})
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
_PANELS = [
    ("pupil_metric_mag_vel_rms", "Avg Pupil Velocity"),
    ("pupil_metric_mag_perc_recur", "Avg Pupil %REC"),
    ("crqa_pupil_metric_mag_perc_recur", "Cross Gaze-Head %REC"),
    ("pupil_metric_mag_perc_determ", "Avg Pupil %DET"),
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
