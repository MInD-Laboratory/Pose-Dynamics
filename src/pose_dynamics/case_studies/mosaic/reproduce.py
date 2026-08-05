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

import hashlib
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

#: Digest of this file as it was when the module was imported. A notebook kernel keeps
#: modules in ``sys.modules``, so editing this file has no effect on a session that already
#: imported it -- and a plain "Run All" then reuses the old code while looking like a fresh
#: run. Comparing this against a fresh read of the file detects that. ``inspect.getsource``
#: cannot: it reads from disk too, so it always agrees with disk no matter what is loaded.
_SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

_FILE_RE = re.compile(r"^S(\d+)_T(\d+)_(left|right)$")
COLORS = ["#cfe0ef", "#93b8db", "#4f8cc0", "#20517e"]
# Categorical (not sequential): distinguishes alignment *strategy* identity (the
# paper's uniform Procrustes vs. no alignment), not a magnitude/condition ramp, so
# it deliberately doesn't reuse COLORS above.
STRATEGY_COLORS = ["#2a78d6", "#8e5fd6"]


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
    """Mask -> interpolate -> filter -> downsample -> normalize (Case 2 settings).

    Deliberately does **no** nose centering. The paper centres "on the mean nose
    position" *within each window* -- a single per-window constant -- which is
    :func:`windowed_align`'s job, not this function's.

    An earlier version subtracted *that same frame's* nose position from every
    keypoint here, per frame, intending to remove head-translation jitter. That is
    not a translation-invariance fix but a change of reference frame: it makes
    every keypoint's velocity ``v_k - v_nose``, so each ROI's velocity magnitude
    becomes movement *relative to the nose* rather than movement in the aligned
    frame. Two consequences motivated its removal. For ``centre_face`` the ROI
    centroid co-moves with the nose, so ``mean_k(v_k) ~= v_nose`` and the signal
    collapsed to residual facial deformation -- reversing the direction of the
    published %DET/Lmax condition effects and shrinking RMS roughly threefold. For
    ``arms``/``upper_body`` it subtracted a common-mode nuisance that compressed
    the condition effects without flipping them. A per-window *constant* offset,
    by contrast, differentiates to zero and cannot do any of this, which is why
    the published prototype's equivalent step (subtracting each trial's mean pose
    after alignment) left its velocity signals untouched.

    Shared by the template-building pass and the per-file analysis pass, so both
    operate in the same normalized coordinate space before alignment.
    """
    seq = mask_low_confidence(seq, C.CONF_THRESHOLD)
    seq = interpolate_gaps(seq, C.INTERP_CAP)
    seq = butterworth_filter(seq, C.FILTER_CUTOFF, C.FILTER_ORDER)
    seq = _downsample(seq, int(round(C.FRAME_RATE / C.TARGET_RATE)))
    pipe = FeaturePipeline.from_config([
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


def compute_reference_limb_lengths(template: np.ndarray, keypoint_names: list[str]
                                   ) -> list[tuple[tuple[int, int], float]]:
    """Segment lengths of the upper-limb chain, measured on the global ``template``.

    Returned as an ordered list rather than a dict so the distal composition order
    (shoulder->elbow before elbow->wrist) is explicit rather than resting on dict
    insertion order, as it did in the prototype. Segments whose endpoints are absent
    from ``keypoint_names`` are skipped.
    """
    idx = {n: i for i, n in enumerate(keypoint_names)}
    out: list[tuple[tuple[int, int], float]] = []
    for proximal, distal in C.LIMB_CHAIN:
        if proximal in idx and distal in idx:
            i, j = idx[proximal], idx[distal]
            out.append(((i, j), float(np.linalg.norm(template[j] - template[i]))))
    return out


def apply_fixed_limb_lengths(coords: np.ndarray,
                             ref_lengths: list[tuple[tuple[int, int], float]]) -> np.ndarray:
    """Force each upper-limb segment to its template length, preserving its direction.

    For every frame the distal keypoint is moved along the segment direction until the
    segment matches ``target``, so direction is kept while length is overwritten.

    Precisely: segments are corrected in chain order, and each one preserves the
    direction measured from its *already-corrected* proximal joint to the distal
    joint's original position. For the proximal-most segment of a chain
    (shoulder->elbow) the parent never moves, so the original joint angle is preserved
    exactly; for elbow->wrist the parent has already shifted, so the wrist's direction
    is preserved relative to the corrected elbow rather than the observed one. That is
    inherent to a chained correction and matches the prototype.

    This is the prototype's ``batch_apply_fixed_lengths``, reproduced for parity. Be
    clear about what it costs: it deletes *all* radial motion of the elbow and wrist
    relative to the shoulder. Reaching toward or away from the camera -- which in 2-D
    projection appears as the arm lengthening and shortening -- is removed along with
    the tracking noise it is presumably meant to suppress, leaving only swing about
    the joints. It touches the elbows and wrists and nothing else, so it can only
    affect the ``arms`` ROI; ``upper_body`` (neck, shoulders, mid-hip) and
    ``centre_face`` are untouched by construction.
    """
    out = coords.copy()
    for (i, j), target in ref_lengths:
        v = out[:, j] - out[:, i]                               # (L, 2)
        length = np.linalg.norm(v, axis=1, keepdims=True)        # (L, 1)
        out[:, j] = out[:, i] + v * (target / (length + 1e-12))
    return out


def windowed_align(seq: PoseSequence, template: np.ndarray | None, align: bool = True
                   ) -> list[tuple[Window, np.ndarray]]:
    """Fit one rigid+uniform-scale Procrustes transform per analysis window (from
    that window's mean pose to the global ``template``) and apply it to every frame
    in the window.

    Because RQA windows overlap 50%, a frame's aligned position is only well-defined
    within a specific window, so alignment happens per-window rather than once over
    the whole sequence (contrast Case 1's single global-template, per-frame fit).

    Coordinates are centred here on the window's **mean** nose position before the
    fit -- the paper's "centred on the mean nose position" step, and the only nose
    centering in the pipeline (:func:`preprocess_pose` deliberately does none; see
    its docstring for why per-frame centering is not equivalent).

    Being a per-window *constant* offset, it has no effect on either of the things
    downstream code reads. ``procrustes_uniform`` centres both configurations
    internally, so any constant shift of the input cancels out of the fitted
    rotation/scale; and a constant differentiates to zero, so it cannot alter the
    ROI velocity magnitudes computed from the returned coordinates. What it does
    fix is the origin of those coordinates when no fit is applied -- ``align=False``
    or too few valid keypoints -- so the returned window is nose-relative rather
    than in raw scaled-screen coordinates. If the nose has zero valid frames across
    the whole window, centering is undefined and the *entire* window is set to NaN.

    Window inclusion follows ``C.WINDOW_COMPLETENESS``. Under the default
    ``"all_keypoints"`` -- the published prototype's rule -- a window containing any
    missing value in any selected keypoint is returned as all-NaN and therefore dropped
    downstream for every ROI. Under ``"per_roi"`` such windows are retained and only the
    affected ROIs drop out later; see the config note for the artifact that admits.

    A keypoint is used in the fit only if it is finite in at least
    ``C.MIN_VALID_FRAC_PER_KP`` of the window's frames; windows with fewer than
    ``C.MIN_KEYPOINTS_FOR_FIT`` such keypoints are left nose-centred but not
    rotated/scaled. Both thresholds are inactive under ``"all_keypoints"``, which admits
    only fully-observed windows.

    ``align=False`` skips the Procrustes fit (rotation/scale) entirely and returns
    the window's nose-centred coordinates -- a reviewer-requested comparison against
    the published, aligned pipeline. Limb rescaling still applies in that case (it is
    a separate stage of the published pipeline), so ``template`` is still required
    whenever ``C.APPLY_LIMB_RESCALE`` is set; pass ``None`` only to disable both.
    """
    nose_idx = seq.keypoint_names.index("Nose")
    windows = make_windows(seq.n_frames, seq.frame_rate, C.WINDOW_S, C.OVERLAP)
    ref_lengths = (compute_reference_limb_lengths(template, seq.keypoint_names)
                   if C.APPLY_LIMB_RESCALE and template is not None else [])
    out: list[tuple[Window, np.ndarray]] = []
    complete_only = C.WINDOW_COMPLETENESS == "all_keypoints"
    for w in windows:
        coords = seq.coords[w.start:w.stop]                    # (L, K, 2)
        # Prototype rule: one missing value in one keypoint at one frame voids the whole
        # window, for every ROI. Equivalent to its `window.isnull().any().any()` over the
        # selected keypoint set, which is exactly what ``seq`` holds at this point.
        if complete_only and not np.all(np.isfinite(coords)):
            out.append((w, np.full_like(coords, np.nan)))
            continue
        nose_pos = np.nanmean(coords[:, nose_idx, :], axis=0)   # (2,)
        if not np.all(np.isfinite(nose_pos)):
            out.append((w, np.full_like(coords, np.nan)))
            continue
        centred = coords - nose_pos
        if not align:
            result = centred
        else:
            finite_per_frame = np.all(np.isfinite(centred), axis=2)  # (L, K)
            valid_frac = finite_per_frame.mean(axis=0)               # (K,)
            valid = valid_frac >= C.MIN_VALID_FRAC_PER_KP
            if valid.sum() < C.MIN_KEYPOINTS_FOR_FIT:
                result = centred
            else:
                window_mean = np.nanmean(centred[:, valid, :], axis=0)  # (n_valid, 2)
                tp = procrustes_uniform(window_mean, template[valid], allow_scale=True)
                result = centred @ tp.L + tp.t
        # Prototype order: limb rescaling runs *after* the Procrustes fit, so it
        # never feeds back into the transform that was estimated from the raw pose.
        if ref_lengths:
            result = apply_fixed_limb_lengths(result, ref_lengths)
        out.append((w, result))
    return out


def _window_roi_speeds(aligned_coords: np.ndarray, keypoint_names: list[str],
                       roi_index_map: dict[str, list[int]], frame_rate: float):
    """ROI centroid -> velocity magnitude for one window's aligned coordinates.

    Under ``C.WINDOW_COMPLETENESS == "roi_complete"`` (the default) an ROI's signal is
    voided for this window unless every one of *that ROI's own* keypoints is finite
    throughout it. Voiding here rather than in :func:`windowed_align` is what keeps the
    rule per-ROI: the ROIs share keypoints (both shoulders belong to ``arms`` and
    ``upper_body``), so a missing member cannot simply be nulled in the coordinate array
    without silently affecting the other ROI that uses it.
    """
    win_seq = PoseSequence(coords=aligned_coords, keypoint_names=keypoint_names, frame_rate=frame_rate)
    pipe = FeaturePipeline.from_config([
        {"step": "roi_centroid", "params": {"rois": roi_index_map}},
        {"step": "velocity_magnitude", "params": {"method": "diff"}},
    ])
    feats = pipe.run(win_seq).features  # columns: {roi}_speed
    if C.WINDOW_COMPLETENESS == "roi_complete":
        for roi, idx in roi_index_map.items():
            if np.all(np.isfinite(aligned_coords[:, np.asarray(idx, int), :])):
                continue
            name = f"{roi}_speed"
            if name in feats.names:
                feats.values[:, feats.names.index(name)] = np.nan
    return feats


def roi_velocity_signals(seq: PoseSequence, roi_index_map: dict[str, list[int]]):
    """Preprocess and reduce to one ROI velocity-magnitude signal per ROI --
    an *unaligned* preview over the whole sequence, for exploration/visualization.

    mask -> interpolate -> filter -> downsample to 30 Hz -> normalize, then ROI
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
    align: bool = True,
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

    ``align=False`` skips Procrustes alignment entirely (see :func:`windowed_align`)
    for a reviewer-requested comparison against the aligned pipeline above -- no
    template is needed or built in that case, regardless of ``template``/
    ``template_sample``.
    """
    cond_map = load_condition_map(conditions_csv)

    if (align or C.APPLY_LIMB_RESCALE) and template is None:
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
        for w, aligned in windowed_align(seq, template, align=align):
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
                 condition: str, template: np.ndarray | None,
                 session: int | None = None, trial: int | None = None,
                 align: bool = True) -> list[dict]:
    """Windowed interpersonal CRQA + linear cross-correlation between two partners.

    Both partners are preprocessed and trimmed to a shared frame count first (so
    window boundaries line up in time), then each is aligned independently, per
    window, against the same shared ``template`` (see :func:`windowed_align`).

    ``session``/``trial`` are carried through into each output row (not used
    internally) so a mixed-effects model can group by pair (``session`` -- the
    paper: "random intercepts for pair") -- pass them from :func:`run_reproduction`.

    ``align=False`` skips Procrustes alignment entirely (see :func:`windowed_align`);
    ``template`` is unused and may be ``None`` in that case.
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
    right_windows = windowed_align(right, template, align=align)
    left_windows = windowed_align(left, template, align=align)
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
                # the seven CRQA measures Section 3.2.1 reports extracting
                "cross_perc_recur": cross.metrics["perc_recur"],
                "cross_perc_determ": cross.metrics["perc_determ"],
                "cross_laminarity": cross.metrics["laminarity"],
                "cross_mean_line_length": cross.metrics["mean_line_length"],
                "cross_lmax": cross.metrics["maxl_found"],
                "cross_entropy": cross.metrics["entropy"],
                "cross_trapping_time": cross.metrics["trapping_time"],
                # linear coupling, not a recurrence measure
                "xcorr_lag0": float(np.mean(za * zb)),
            })
    return rows


def run_reproduction(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    template: np.ndarray | None = None,
    template_sample: int | None = C.TEMPLATE_SAMPLE,
    sessions: list[int] | None = None,
    align: bool = True,
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

    ``sessions`` restricts processing to the given session numbers -- e.g. for a
    fast subset run over a handful of pairs instead of the whole dataset. This
    filter is applied *before* the default template is built, so a subset run
    also builds its template from just that subset, not the full dataset.

    ``align=False`` skips Procrustes alignment entirely (see :func:`windowed_align`)
    for a reviewer-requested comparison against the aligned pipeline above -- no
    template is needed or built in that case, regardless of ``template``/
    ``template_sample``.
    """
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    files = [p for p in data_dir.glob("S*_T*_*.csv") if not p.name.startswith("._")]
    by_key: dict[tuple[int, int], dict[str, Path]] = {}
    for f in files:
        s, t, cam = parse_file(f)
        by_key.setdefault((s, t), {})[cam] = f

    keys = sorted(k for k, v in by_key.items() if {"left", "right"} <= set(v))
    if sessions is not None:
        session_set = set(sessions)
        keys = [k for k in keys if k[0] in session_set]
    if not keys:
        raise FileNotFoundError(
            "No session-trial has BOTH 'left' and 'right' camera files; the dyadic "
            "reproduction needs both partners. Only individual-level analysis is "
            "possible with single-camera data (see run_individual)."
        )

    if (align or C.APPLY_LIMB_RESCALE) and template is None:
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
        rows.extend(process_dyad(right, left, roi_map, cond, template,
                                 session=s, trial=t, align=align))
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _despine(ax) -> None:
    """Drop the top and right spines, leaving just the x- and y-axis lines.

    Called by every figure helper below so the panels match: a full box around each
    panel adds ink without adding information.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _condition_bar_panels(panels: list[tuple[pd.DataFrame, str, str, str]], axes=None):
    """Shared renderer for the paper-style summary figures: each entry in
    ``panels`` is ``(df, roi, metric, title)``; bars are mean +/- SEM by
    condition, colored by :data:`COLORS`. ``title`` is placed on the y-axis
    (not as a panel title), and one legend is shared below all panels instead
    of per-panel condition tick labels.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "Times New Roman"}):
        if axes is None:
            fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
        else:
            fig = np.atleast_1d(axes).flatten()[0].figure
        axes = np.atleast_1d(axes).flatten()

        for (df, roi, metric, title), ax in zip(panels, axes):
            sub = df[df["roi"] == roi]
            stats = (sub[["condition", metric]].dropna()
                     .groupby("condition", observed=True)[metric].agg(["mean", "sem"])
                     .reindex(C.CONDITION_ORDER))
            ax.bar(range(len(C.CONDITION_ORDER)), stats["mean"], yerr=stats["sem"],
                   color=COLORS, edgecolor="black", linewidth=2, capsize=5)
            ax.set_xticks([])
            ax.set_ylabel(title, fontsize=12)
            _despine(ax)

        handles = [mpatches.Patch(facecolor=c, edgecolor="black", label=cond)
                   for c, cond in zip(COLORS, C.CONDITION_ORDER)]
        fig.legend(handles=handles, loc="lower center", ncol=len(C.CONDITION_ORDER),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.08, 1, 1])
    return axes


def plot_case2_figure(individual_df: pd.DataFrame, dyad_df: pd.DataFrame,
                      arm_roi: str = "arms", axes=None):
    """Reproduce the paper's Case 2 linear-metrics figure: RMS of arm-ROI velocity
    magnitude (individual, left panel) alongside cross-correlation of arm-ROI
    velocity magnitude (dyadic, right panel), by condition.

    ``individual_df`` is :func:`run_individual`'s output; ``dyad_df`` is
    :func:`run_reproduction`'s output.
    """
    return _condition_bar_panels([
        (individual_df, arm_roi, "rms", "RMS of Arm Magnitude"),
        (dyad_df, arm_roi, "xcorr_lag0", "Cross-Correlation of Arm Magnitude"),
    ], axes=axes)


def plot_case2_crqa_figure(dyad_df: pd.DataFrame, roi: str = "arms", axes=None):
    """Cross-RQA %REC and %DET of arm-ROI velocity magnitude (dyadic), by
    condition. ``dyad_df`` is :func:`run_reproduction`'s output.
    """
    return _condition_bar_panels([
        (dyad_df, roi, "cross_perc_recur", "Arm Magnitude %REC"),
        (dyad_df, roi, "cross_perc_determ", "Arm Magnitude %DET"),
    ], axes=axes)


def _alignment_comparison_panels(panels: list[tuple[dict[str, pd.DataFrame], str, str, str]],
                                 axes=None):
    """Shared renderer comparing alignment *strategies* (the paper's uniform
    Procrustes vs. no alignment): each entry in
    ``panels`` is ``(dfs, roi, metric, title)``, where ``dfs`` maps a strategy
    label to its results dataframe (same keys/order for every panel in one call).

    Unlike :func:`_condition_bar_panels` (one bar per condition, colored by
    condition), this draws one bar *group* per condition, with one bar per
    strategy inside each group, colored by :data:`STRATEGY_COLORS` -- strategy is
    the identity being compared here, condition is just the x-axis grouping.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "Times New Roman"}):
        if axes is None:
            fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.5))
        else:
            fig = np.atleast_1d(axes).flatten()[0].figure
        axes = np.atleast_1d(axes).flatten()

        strategies = list(panels[0][0])
        n_strat = len(strategies)
        width = 0.8 / n_strat
        x = np.arange(len(C.CONDITION_ORDER))

        for (dfs, roi, metric, title), ax in zip(panels, axes):
            for i, strat in enumerate(strategies):
                sub = dfs[strat]
                sub = sub[sub["roi"] == roi]
                stats = (sub[["condition", metric]].dropna()
                         .groupby("condition", observed=True)[metric].agg(["mean", "sem"])
                         .reindex(C.CONDITION_ORDER))
                offset = (i - (n_strat - 1) / 2) * width
                ax.bar(x + offset, stats["mean"], width=width, yerr=stats["sem"],
                       color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                       edgecolor="black", linewidth=1.5, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(C.CONDITION_ORDER, rotation=30, ha="right")
            ax.set_ylabel(title, fontsize=12)
            _despine(ax)

        handles = [mpatches.Patch(facecolor=STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                                  edgecolor="black", label=strat)
                   for i, strat in enumerate(strategies)]
        fig.legend(handles=handles, loc="lower center", ncol=n_strat,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.08, 1, 1])
    return axes


def plot_alignment_comparison_linear(individual_dfs: dict[str, pd.DataFrame], axes=None):
    """RMS of velocity magnitude for all three ROIs, aligned vs. unaligned.

    The linear metrics are reported in absolute units, so they are the measures that
    alignment genuinely affects: the Procrustes scale step corrects between-participant
    differences in pixel scale arising when participants sit at different distances from
    their cameras.
    """
    return _alignment_comparison_panels([
        (individual_dfs, "arms", "rms", "Arms RMS"),
        (individual_dfs, "upper_body", "rms", "Upper Body RMS"),
        (individual_dfs, "centre_face", "rms", "Centre Face RMS"),
    ], axes=axes)


def plot_alignment_comparison_crqa_invariant(dyad_dfs: dict[str, pd.DataFrame], axes=None):
    """Cross-RQA for the two ROIs alignment cannot affect -- centre-face and upper body.

    The bars are identical by construction, not merely close. A rotation plus uniform
    scale multiplies each window's velocity-magnitude series by one constant, and the
    per-window z-scoring inside the recurrence computation removes exactly that constant,
    so the recurrence plot is unchanged. Measured difference across every window and
    metric is 0.
    """
    return _alignment_comparison_panels([
        (dyad_dfs, "centre_face", "cross_perc_recur", "Centre Face %REC"),
        (dyad_dfs, "centre_face", "cross_perc_determ", "Centre Face %DET"),
        (dyad_dfs, "upper_body", "cross_perc_recur", "Upper Body %REC"),
        (dyad_dfs, "upper_body", "cross_perc_determ", "Upper Body %DET"),
    ], axes=axes)


def plot_alignment_comparison_crqa_arms(dyad_dfs: dict[str, pd.DataFrame], axes=None):
    """Cross-RQA for the one ROI alignment does affect -- the arms.

    The contrast with :func:`plot_alignment_comparison_crqa_invariant` is the point. The
    arms are not geometrically special; the difference arises because limb-length
    normalisation rescales upper-limb segments to fixed template lengths, which
    reintroduces absolute scale and so restores sensitivity to the alignment's scale
    factor. Disabling that step makes these bars identical too.
    """
    return _alignment_comparison_panels([
        (dyad_dfs, "arms", "cross_perc_recur", "Arms %REC"),
        (dyad_dfs, "arms", "cross_perc_determ", "Arms %DET"),
    ], axes=axes)
