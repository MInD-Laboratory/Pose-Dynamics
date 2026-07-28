"""
Case 2 (MOSAIC) reproduction entry point.

Case 2 quantifies *interpersonal* coordination during conversation across four
background-noise levels, via cross-recurrence between the two partners' ROI
velocity-magnitude signals. That dyadic analysis needs **both** partners (the
left- and right-camera files) and many pairs; :func:`run_reproduction` implements
it but requires the full dataset.

The individual-level pieces — ROI velocity-magnitude linear metrics only, no
recurrence analysis — need only one participant and are exercised by
:func:`run_individual`. The paper's Case 2 results report individual-level
*linear* metrics and dyadic CRQA; it never reports individual-level auto-RQA, so
:func:`run_individual` doesn't compute any.

Notebooks call these functions; they hold no analysis logic themselves.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.pose_sequence import PoseSequence
from ...features import FeaturePipeline
from ...features.geometry import procrustes_uniform
from ...preprocessing import butterworth_filter, interpolate_gaps, mask_low_confidence
from ...rqa import RqaParams, run_cross_rqa
from ...embedding import EmbeddingParams
from ...windowing import Window, make_windows
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
    """Resolve the curated alignment keypoint set present in a header, plus the
    ROI membership map.

    Returns ``(keypoint_names, roi_index_map)``. ``keypoint_names`` is
    ``C.SELECTED_KEYPOINTS`` restricted to what's actually present in this file's
    header -- a curated alignment/feature selection that deliberately
    excludes lower-body points (hips, knees, ankles, toes, heels) that are
    occluded/unreliable in a seated conversation and must not influence the
    Procrustes fit in :func:`windowed_align`. In this design every entry here
    ends up in one of the three ROIs (the face landmarks all match a
    ``C.CENTRE_FACE_SUBSTRINGS`` substring), so this list and the ROI union
    happen to coincide exactly. ``roi_index_map`` gives, per ROI, the positions
    of its keypoints within ``keypoint_names``, used later to reduce the
    *aligned* pose down to ROI centroids.
    """
    present = {c.replace("_x_offset", "").replace("_y_offset", "").replace("_confidence", "")
               for c in columns if c.endswith(("_x_offset", "_y_offset", "_confidence"))}
    keypoint_names = [n for n in C.SELECTED_KEYPOINTS if n in present]
    base = set(keypoint_names)
    pos = {n: i for i, n in enumerate(keypoint_names)}

    roi_index_map: dict[str, list[int]] = {}
    for roi, exact in C.ROI_EXACT.items():
        names = [n for n in exact if n in base]
        if roi == "centre_face":
            names += [n for n in keypoint_names if any(s in n for s in C.CENTRE_FACE_SUBSTRINGS)
                      and n not in names]
        if names:
            roi_index_map[roi] = [pos[n] for n in names]
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


def preprocess_pose(seq: PoseSequence) -> PoseSequence:
    """Mask -> interpolate -> filter -> downsample -> centre-on-nose -> normalize
    (Case 2 settings).

    The nose-anchor centering subtracts *that same frame's* nose position from
    every keypoint, per frame -- not a single per-window constant -- removing
    frame-to-frame head-translation jitter continuously throughout the trial,
    before :func:`windowed_align` ever sees the data. (Order relative to
    filtering/scaling doesn't matter -- both are linear and commute with this
    per-frame subtraction -- so it's placed here for one shared code path rather
    than split across two stages.) A frame
    where the nose itself is invalid propagates NaN to every keypoint in that
    frame, same principle as :func:`windowed_align`'s whole-window NaN when the
    nose is missing for an entire window.

    Shared by the template-building pass and the per-file analysis pass, so both
    operate in the same normalized coordinate space before alignment.
    """
    seq = mask_low_confidence(seq, C.CONF_THRESHOLD)
    seq = interpolate_gaps(seq, C.INTERP_CAP)
    seq = butterworth_filter(seq, C.FILTER_CUTOFF, C.FILTER_ORDER)
    seq = _downsample(seq, int(round(C.FRAME_RATE / C.TARGET_RATE)))
    nose_idx = seq.keypoint_names.index("Nose")
    pipe = FeaturePipeline.from_config([
        {"step": "center", "params": {"reference": nose_idx}},
        {"step": "coordinate_normalization",
         "params": {"width": C.VIDEO_WIDTH, "height": C.VIDEO_HEIGHT, "mode": C.NORMALIZE_MODE}},
    ])
    return pipe.run(seq).pose


def build_global_template(sequences: list[PoseSequence]) -> np.ndarray:
    """Global template = mean ROI-keypoint pose (normalized coords), pooled across
    every valid frame in every given sequence (paper: "a single global template by
    averaging the position of each ROI keypoint across all valid frames in the
    dataset"). This is a single frame-count-weighted average over all sequences
    combined, not a mean of per-file means -- a file with more valid frames
    contributes proportionally more, matching "across all valid frames" literally
    rather than giving every file equal weight regardless of length. ``sequences``
    must already be preprocessed (see :func:`preprocess_pose`) and share the same
    ``keypoint_names``.
    """
    names = sequences[0].keypoint_names
    for seq in sequences[1:]:
        if seq.keypoint_names != names:
            raise ValueError(
                "MOSAIC files must resolve to the same keypoint set to share a "
                "global Procrustes template; got a mismatch against the first "
                "sampled file."
            )
    d = sequences[0].coords.shape[-1]
    total = np.zeros((len(names), d))
    count = np.zeros((len(names), d))
    for seq in sequences:
        finite = np.isfinite(seq.coords)
        total += np.where(finite, seq.coords, 0.0).sum(axis=0)
        count += finite.sum(axis=0)
    return total / count


def windowed_align(seq: PoseSequence, template: np.ndarray) -> list[tuple[Window, np.ndarray]]:
    """Fit one rigid+uniform-scale Procrustes transform per analysis window (from
    that window's mean pose to the global ``template``) and apply it to every frame
    in the window.

    Because RQA windows overlap 50%, a frame's aligned position is only well-defined
    within a specific window, so alignment happens per-window rather than once over
    the whole sequence (contrast Case 1's single global-template, per-frame fit).

    Coordinates are also centred here on the window's mean nose position before
    the fit -- a per-window constant offset that is mathematically absorbed by
    ``procrustes_uniform``'s own centering (any constant shift of the input cancels
    out of the fit) and, separately, already zeroed out by :func:`preprocess_pose`'s
    whole-trial per-frame nose-anchor centering -- the paper's actual jitter-removal
    step -- by the time this function runs, making it a harmless no-op in practice.
    It's kept only so this function stays correct in isolation if it's ever called
    on data that skipped that upstream centering, not because the published method
    describes it as a distinct step. If the nose has zero valid frames across the
    whole window, centering is undefined and the *entire* window is set to NaN.

    A keypoint is used in the fit only if it is finite in at least
    ``C.MIN_VALID_FRAC_PER_KP`` of the window's frames; windows with fewer than
    ``C.MIN_KEYPOINTS_FOR_FIT`` such keypoints are left nose-centred but not
    rotated/scaled.
    """
    nose_idx = seq.keypoint_names.index("Nose")
    windows = make_windows(seq.n_frames, seq.frame_rate, C.WINDOW_S, C.OVERLAP)
    out: list[tuple[Window, np.ndarray]] = []
    for w in windows:
        coords = seq.coords[w.start:w.stop]                    # (L, K, 2)
        nose_pos = np.nanmean(coords[:, nose_idx, :], axis=0)   # (2,)
        if not np.all(np.isfinite(nose_pos)):
            out.append((w, np.full_like(coords, np.nan)))
            continue
        centred = coords - nose_pos
        finite_per_frame = np.all(np.isfinite(centred), axis=2)  # (L, K)
        valid_frac = finite_per_frame.mean(axis=0)               # (K,)
        valid = valid_frac >= C.MIN_VALID_FRAC_PER_KP
        if valid.sum() < C.MIN_KEYPOINTS_FOR_FIT:
            out.append((w, centred))
            continue
        window_mean = np.nanmean(centred[:, valid, :], axis=0)  # (n_valid, 2)
        tp = procrustes_uniform(window_mean, template[valid], allow_scale=True)
        out.append((w, centred @ tp.L + tp.t))
    return out


def _window_roi_speeds(aligned_coords: np.ndarray, keypoint_names: list[str],
                       roi_index_map: dict[str, list[int]], frame_rate: float):
    """ROI centroid -> velocity magnitude for one window's aligned coordinates."""
    win_seq = PoseSequence(coords=aligned_coords, keypoint_names=keypoint_names, frame_rate=frame_rate)
    pipe = FeaturePipeline.from_config([
        {"step": "roi_centroid", "params": {"rois": roi_index_map}},
        {"step": "velocity_magnitude", "params": {"method": "diff"}},
    ])
    return pipe.run(win_seq).features  # columns: {roi}_speed


def roi_velocity_signals(seq: PoseSequence, roi_index_map: dict[str, list[int]]):
    """Preprocess and reduce to one ROI velocity-magnitude signal per ROI --
    an *unaligned* preview over the whole sequence, for exploration/visualization.

    mask -> interpolate -> filter -> normalize -> downsample to 30 Hz, then ROI
    centroid -> velocity magnitude. :func:`run_individual` and
    :func:`run_reproduction` additionally apply windowed Procrustes alignment
    (:func:`windowed_align`) before this reduction, matching the published method;
    that alignment has no single well-defined whole-sequence form (RQA windows
    overlap), so this preview is left unaligned.
    """
    seq = preprocess_pose(seq)
    pipe = FeaturePipeline.from_config([
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
def run_individual(
    files: list[str | Path],
    conditions_csv: str | Path | None = None,
    template: np.ndarray | None = None,
    template_sample: int | None = C.TEMPLATE_SAMPLE,
    progress: bool = True,
) -> pd.DataFrame:
    """Per-window individual ROI linear metrics (one participant) -- RMS, mean,
    and SD of velocity magnitude. No recurrence analysis: the paper's Case 2
    individual-level results are linear-metrics only; dyadic CRQA lives in
    :func:`process_dyad`/:func:`run_reproduction`.

    Each output row carries ``session``/``camera`` -- since a session's ``camera``
    (left/right) is a dedicated webcam per participant, ``session`` + ``camera``
    together identify one participant, e.g. for a mixed-effects model's random
    intercept (the paper: "random intercepts for pair and individual-within-pair").

    Applies windowed Procrustes alignment before ROI reduction (see
    :func:`windowed_align`), matching the published method. If ``template`` isn't
    supplied, it's built from ``files`` (see :func:`build_global_template`) -- by
    default every file, pooling all valid frames across the dataset (the paper's
    "across all valid frames in the dataset"); pass ``template_sample`` to cap this
    to the first N files for faster iteration, or pass an explicit ``template``
    (e.g. built once and shared with :func:`run_reproduction`) for a single
    dataset-wide template.
    """
    cond_map = load_condition_map(conditions_csv)

    if template is None:
        sample = files if template_sample is None else files[: min(template_sample, len(files))]
        sample_seqs = [preprocess_pose(load_mosaic_file(f)[0]) for f in sample]
        template = build_global_template(sample_seqs)
        del sample_seqs

    rows = []
    for f in files:
        seq, roi_map = load_mosaic_file(f)
        session, trial, camera = seq.meta["session"], seq.meta["trial"], seq.meta["camera"]
        cond = cond_map.get((session, trial))
        seq = preprocess_pose(seq)
        for w, aligned in windowed_align(seq, template):
            feats = _window_roi_speeds(aligned, seq.keypoint_names, roi_map, seq.frame_rate)
            for roi in roi_map:
                s = feats.get(f"{roi}_speed")
                if not np.all(np.isfinite(s)):
                    continue
                rows.append({
                    "session": session, "trial": trial, "camera": camera,
                    "condition": cond, "roi": roi, "window": w.index,
                    "rms": float(np.sqrt(np.mean(s ** 2))),
                    "mean_vel": float(np.mean(s)), "sd_vel": float(np.std(s)),
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
                 condition: str, template: np.ndarray,
                 session: int | None = None, trial: int | None = None) -> list[dict]:
    """Windowed interpersonal CRQA + linear cross-correlation between two partners.

    Both partners are preprocessed and trimmed to a shared frame count first (so
    window boundaries line up in time), then each is aligned independently, per
    window, against the same shared ``template`` (see :func:`windowed_align`).

    ``session``/``trial`` are carried through into each output row (not used
    internally) so a mixed-effects model can group by pair (``session`` -- the
    paper: "random intercepts for pair") -- pass them from :func:`run_reproduction`.
    """
    right = preprocess_pose(right)
    left = preprocess_pose(left)
    n = min(right.n_frames, left.n_frames)
    right = PoseSequence(coords=right.coords[:n], keypoint_names=right.keypoint_names,
                         frame_rate=right.frame_rate)
    left = PoseSequence(coords=left.coords[:n], keypoint_names=left.keypoint_names,
                        frame_rate=left.frame_rate)

    cp = cross_params()
    rows = []
    right_windows = windowed_align(right, template)
    left_windows = windowed_align(left, template)
    for (w, aligned_r), (_, aligned_l) in zip(right_windows, left_windows):
        feats_r = _window_roi_speeds(aligned_r, right.keypoint_names, roi_map, right.frame_rate)
        feats_l = _window_roi_speeds(aligned_l, left.keypoint_names, roi_map, left.frame_rate)
        for roi in roi_map:
            aw = feats_r.get(f"{roi}_speed")
            bw = feats_l.get(f"{roi}_speed")
            if not (np.all(np.isfinite(aw)) and np.all(np.isfinite(bw))):
                continue
            cross = run_cross_rqa(aw, bw, cp)
            # linear coupling: zero-lag cross-correlation of z-scored velocity mag
            za = (aw - aw.mean()) / (aw.std() + 1e-8)
            zb = (bw - bw.mean()) / (bw.std() + 1e-8)
            rows.append({
                "session": session, "trial": trial,
                "condition": condition, "roi": roi, "window": w.index,
                "cross_perc_recur": cross.metrics["perc_recur"],
                "cross_perc_determ": cross.metrics["perc_determ"],
                "cross_lmax": cross.metrics["maxl_found"],
                "xcorr_lag0": float(np.mean(za * zb)),
            })
    return rows


def run_reproduction(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    template: np.ndarray | None = None,
    template_sample: int | None = C.TEMPLATE_SAMPLE,
    progress: bool = True,
) -> pd.DataFrame:
    """Full dyadic reproduction. Requires both camera files per session-trial.

    Applies windowed Procrustes alignment (see :func:`process_dyad`) before ROI
    reduction. If ``template`` isn't supplied, it's built from the discovered
    session-trials (see :func:`build_global_template`) -- by default every one,
    pooling all valid frames across the dataset (the paper's "across all valid
    frames in the dataset"); pass ``template_sample`` to cap this to the first N
    session-trials for faster iteration, or pass an explicit ``template`` (e.g.
    shared with :func:`run_individual`) for a single dataset-wide template.
    """
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    files = [p for p in data_dir.glob("S*_T*_*.csv") if not p.name.startswith("._")]
    by_key: dict[tuple[int, int], dict[str, Path]] = {}
    for f in files:
        s, t, cam = parse_file(f)
        by_key.setdefault((s, t), {})[cam] = f

    keys = sorted(k for k, v in by_key.items() if {"left", "right"} <= set(v))
    if not keys:
        raise FileNotFoundError(
            "No session-trial has BOTH 'left' and 'right' camera files; the dyadic "
            "reproduction needs both partners. Only individual-level analysis is "
            "possible with single-camera data (see run_individual)."
        )

    if template is None:
        sample_keys = keys if template_sample is None else keys[:template_sample]
        sample_files = [by_key[k][cam] for k in sample_keys for cam in ("left", "right")]
        sample_seqs = [preprocess_pose(load_mosaic_file(f)[0]) for f in sample_files]
        template = build_global_template(sample_seqs)
        del sample_seqs

    rows = []
    for (s, t) in keys:
        cond = cond_map.get((s, t))
        if cond not in C.CONDITION_ORDER:
            continue
        right, roi_map = load_mosaic_file(by_key[(s, t)]["right"])
        left, _ = load_mosaic_file(by_key[(s, t)]["left"])
        if progress:
            print(f"  session {s} trial {t} ({cond})")
        rows.extend(process_dyad(right, left, roi_map, cond, template, session=s, trial=t))
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_individual_figure(df: pd.DataFrame, roi: str = "arms", axes=None):
    """Individual ROI linear metrics by condition (mean +/- SEM) for one ROI --
    RMS, mean, and SD of velocity magnitude (no recurrence metrics; see module
    docstring)."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = np.asarray(axes).flatten()
    sub = df[df["roi"] == roi]
    for (metric, ylab), ax in zip(
        [("rms", f"{roi} RMS velocity"), ("mean_vel", f"{roi} mean velocity"),
         ("sd_vel", f"{roi} SD velocity")], axes):
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


def plot_dyadic_figure(df: pd.DataFrame, roi: str = "arms", axes=None):
    """Dyadic interpersonal-coordination metrics by condition (mean +/- SEM) for
    one ROI -- linear cross-correlation (paper's Fig. 9) and cross-RQA %REC/%DET/
    Lmax (Fig. 10). ``df`` is :func:`run_reproduction`'s output."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = np.asarray(axes).flatten()
    sub = df[df["roi"] == roi]
    for (metric, ylab), ax in zip(
        [("xcorr_lag0", f"{roi} cross-correlation"), ("cross_perc_recur", f"{roi} %REC"),
         ("cross_perc_determ", f"{roi} %DET"), ("cross_lmax", f"{roi} cross Lmax")], axes):
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
