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
from ...rqa import METRIC_KEYS, RqaParams, run_auto_rqa, run_cross_rqa, run_multivariate_cross_rqa
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


def load_leader_map(conditions_csv: str | Path | None = None) -> dict[tuple[int, int], int]:
    """Map ``(pair, trial 1..12)`` -> the person index (1 or 2) leading that trial.

    ``block1_lead`` names the block-1 leader; the roles swap for block 2 (trials
    7-12), which is how the design assigns each participant one leader block.
    """
    df = pd.read_csv(conditions_csv or default_conditions_csv())
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    out: dict[tuple[int, int], int] = {}
    for _, row in df.iterrows():
        pair = int(row["Pair"])
        lead1 = int(str(row["block1_lead"]).strip().upper().lstrip("P"))
        lead2 = 2 if lead1 == 1 else 1
        for t in range(1, 7):
            out[(pair, t)] = lead1
            out[(pair, t + 6)] = lead2
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


def md_cross_params() -> RqaParams:
    """MdCRQA parameters. ``multivariate=True`` -> the library does no embedding;
    we embed explicitly (see :func:`delay_embed`) and pass ``norm='none'`` because
    each dimension is z-scored first (one normalization decision, never two)."""
    return RqaParams(
        eDim=C.M, tLag=C.TAU, multivariate=True, radius_mode="fixed_rrec",
        target_rec=C.MD_TARGET_REC, rescale=C.RESCALE, min_line=C.MD_MIN_LINE,
        norm="none",
    )


def md_cross_params_fixed(radius: float) -> RqaParams:
    """MdCRQA at a supplied radius, so %REC is an outcome rather than pinned."""
    return RqaParams(
        eDim=C.M, tLag=C.TAU, multivariate=True, radius_mode="fixed_radius",
        radius=radius, rescale=C.RESCALE, min_line=C.MD_MIN_LINE, norm="none",
    )


def delay_embed(X: np.ndarray, tau: int, m: int) -> np.ndarray:
    """Delay-embed a ``(T, d)`` multivariate signal into ``(T-(m-1)tau, d*m)``.

    Every dimension shares the same ``(tau, m)``, so the result is the collective
    state vector of the whole system at each time point -- the construction
    Md(C)RQA operates on.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0] - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"signal too short to embed: {X.shape[0]} frames, need > {(m - 1) * tau}.")
    return np.concatenate([X[i * tau: i * tau + n] for i in range(m)], axis=1)


def _zscore(X: np.ndarray) -> np.ndarray:
    """Z-score each column independently (per-dimension standardization)."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


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
def kinematic_summary(coords: np.ndarray, fps: float = C.TARGET_RATE) -> dict[str, float]:
    """Displacement / velocity / acceleration summaries for a ``(T, k, 3)`` subset.

    Displacement is the Euclidean distance between successive frames, velocity its
    first temporal derivative, and acceleration the derivative of velocity. Each is
    summarized by its mean and its RMS, averaged over the ``k`` keypoints.
    """
    disp = np.linalg.norm(np.diff(coords, axis=0), axis=2)        # (T-1, k)
    vel = np.linalg.norm(np.gradient(coords, axis=0) * fps, axis=2)
    acc = np.linalg.norm(np.gradient(np.gradient(coords, axis=0) * fps, axis=0) * fps, axis=2)

    out: dict[str, float] = {}
    for name, sig in (("disp", disp), ("vel", vel), ("accel", acc)):
        out[f"{name}_mean"] = float(sig.mean(axis=0).mean())
        out[f"{name}_rms"] = float(np.sqrt(np.mean(sig ** 2, axis=0)).mean())
    return out


def process_dyad(p1: PoseSequence, p2: PoseSequence, condition: str,
                 leader: int | None = None) -> dict:
    """Kinematics, per-keypoint CRQA, and whole-body MdCRQA for one dyad-trial.

    ``leader`` (1 or 2) tags which person led, so per-role movement summaries can
    be reported separately; when it is ``None`` the role columns are omitted.
    """
    a = _prep_subset(p1)
    b = _prep_subset(p2)
    n = min(a.n_frames, b.n_frames)               # shared-clock: trim to overlap
    ac, bc = a.coords[:n], b.coords[:n]
    fps = C.TARGET_RATE

    # kinematics: per person, then averaged over the two for the dyad-level column
    kin = {1: kinematic_summary(ac, fps), 2: kinematic_summary(bc, fps)}
    row: dict = {
        "pair": p1.meta["pair"], "trial": p1.meta["trial"], "condition": condition,
    }
    row.update({k: float(np.mean([kin[1][k], kin[2][k]])) for k in kin[1]})

    if leader in (1, 2):
        follower = 2 if leader == 1 else 1
        row["leader_person"] = leader
        row.update({f"leader_{k}": v for k, v in kin[leader].items()})
        row.update({f"follower_{k}": v for k, v in kin[follower].items()})

    # per-keypoint CRQA on magnitude time series, averaged over the five keypoints.
    # Under fixed-rec mode %REC is pinned to the target (a convergence check); the
    # achieved radius is the informative density measure.
    cp = cross_params()
    m1 = np.stack([np.linalg.norm(ac[:, k, :], axis=1) for k in range(len(C.SUBSET_INDICES))], axis=1)
    m2 = np.stack([np.linalg.norm(bc[:, k, :], axis=1) for k in range(len(C.SUBSET_INDICES))], axis=1)
    # Order the two streams as (leader, follower). Diagonal-line metrics are
    # invariant to swapping the axes -- transposing a cross-recurrence matrix
    # preserves its diagonals -- but VERTICAL-line metrics (trapping time,
    # laminarity, vmax) are not: they depend on which signal indexes which axis.
    # Pairing by recording order (person 1 vs person 2) would compute those on an
    # arbitrary assignment, so the meaningful direction is used instead.
    mags_a, mags_b = (m1, m2) if leader != 2 else (m2, m1)

    recs, radii, per_metric = [], [], []
    for k in range(len(C.SUBSET_INDICES)):
        res = run_cross_rqa(mags_a[:, k], mags_b[:, k], cp)
        recs.append(res.rec_rate)
        radii.append(res.radius_used)
        per_metric.append(res.metrics)

    row.update({
        "cross_perc_recur": float(np.mean(recs)),   # pinned ~2.5% (convergence check)
        "cross_radius": float(np.mean(radii)),       # informative density measure
        "cross_lmax": float(np.mean([m["maxl_found"] for m in per_metric])),
    })
    # the full metric family, averaged across the five keypoints
    for key in METRIC_KEYS:
        if key in per_metric[0]:
            row[f"cross_{key}"] = float(np.mean([m[key] for m in per_metric]))

    # Primary CRQA: the same per-keypoint analysis at a **fixed radius**, where %REC
    # is an outcome rather than pinned. These `crossfx_*` columns are the ones the
    # manuscript reports; the `cross_*` columns above are the fixed-2.5%-REC
    # counterpart, retained so the two recurrence modes can be compared directly.
    cpf = cross_params_fixed(C.CROSS_RADIUS)
    fx_recs, fx_metrics = [], []
    for k in range(len(C.SUBSET_INDICES)):
        res = run_cross_rqa(mags_a[:, k], mags_b[:, k], cpf)
        fx_recs.append(res.rec_rate)
        fx_metrics.append(res.metrics)
    row["crossfx_radius"] = float(C.CROSS_RADIUS)
    row["crossfx_perc_recur"] = float(np.mean(fx_recs))
    for key in METRIC_KEYS:
        if key in fx_metrics[0]:
            row[f"crossfx_{key}"] = float(np.mean([m[key] for m in fx_metrics]))
    row["crossfx_lmax"] = row.get("crossfx_maxl_found", np.nan)

    # MdCRQA: the two bodies as whole five-dimensional systems, compared once
    # rather than keypoint-by-keypoint. Each dimension is z-scored, then the
    # multivariate signal is delay-embedded with the shared (tau, m).
    row.update(md_cross_rqa_row(mags_a, mags_b))
    # ...and again at a fixed radius, where %REC is an outcome, not a constant.
    row.update(md_cross_rqa_fixed_row(mags_a, mags_b))
    return row


def _md_embed_pair(mags_a: np.ndarray, mags_b: np.ndarray):
    return (delay_embed(_zscore(mags_a), C.TAU, C.M),
            delay_embed(_zscore(mags_b), C.TAU, C.M))


def md_cross_rqa_fixed_row(mags_a: np.ndarray, mags_b: np.ndarray,
                           radius: float | None = None, prefix: str = "mdfx") -> dict:
    """MdCRQA at a fixed radius -> ``mdfx_*`` columns (``mdfx_perc_recur`` is a result)."""
    Xa, Xb = _md_embed_pair(mags_a, mags_b)
    res = run_multivariate_cross_rqa(Xa, Xb, md_cross_params_fixed(radius or C.MD_RADIUS))
    row = {f"{prefix}_radius": float(res.radius_used),
           f"{prefix}_perc_recur": float(res.rec_rate)}
    for key in METRIC_KEYS:
        if key in res.metrics:
            row[f"{prefix}_{key}"] = float(res.metrics[key])
    return row


def md_radius_sweep(mags_a: np.ndarray, mags_b: np.ndarray,
                    radii: list[float] | None = None) -> list[dict]:
    """MdCRQA across a grid of fixed radii -> one dict per radius (sensitivity check)."""
    Xa, Xb = _md_embed_pair(mags_a, mags_b)
    out = []
    for r in (radii or C.MD_RADIUS_GRID):
        res = run_multivariate_cross_rqa(Xa, Xb, md_cross_params_fixed(r))
        rec = {"radius": float(r), "perc_recur": float(res.rec_rate)}
        rec.update({k: float(v) for k, v in res.metrics.items()})
        out.append(rec)
    return out


def md_cross_rqa_row(mags_a: np.ndarray, mags_b: np.ndarray) -> dict:
    """MdCRQA of two ``(T, d)`` keypoint-magnitude streams -> ``md_*`` columns."""
    Xa = delay_embed(_zscore(mags_a), C.TAU, C.M)
    Xb = delay_embed(_zscore(mags_b), C.TAU, C.M)
    res = run_multivariate_cross_rqa(Xa, Xb, md_cross_params())
    m = res.metrics
    row = {
        "md_perc_recur": float(res.rec_rate),       # pinned ~2.5% (convergence check)
        "md_radius": float(res.radius_used),        # informative density measure
        "md_mean_line": float(m.get("mean_line_length", np.nan)),
        "md_lmax": float(m.get("maxl_found", np.nan)),
        "md_converged": bool(res.converged),
    }
    # the full metric family, so every metric the manuscript reports for CRQA has
    # its MdCRQA counterpart
    for key in METRIC_KEYS:
        if key in m:
            row[f"md_{key}"] = float(m[key])
    return row


def cross_params_fixed(radius: float) -> RqaParams:
    """Per-keypoint CRQA at a supplied radius, so %REC is an outcome."""
    return RqaParams.from_embedding(
        EmbeddingParams(tau=C.TAU, m=C.M), radius_mode="fixed_radius",
        radius=radius, rescale=C.RESCALE, min_line=C.MIN_LINE, norm=C.NORM,
    )


def run_crqa_radius_sweep(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    pairs: list[int] | None = None,
    radii: list[float] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Per-keypoint CRQA across a grid of fixed radii, kept **per keypoint**.

    One row per (pair, trial, keypoint, radius) -- the five keypoints are not
    averaged, so condition effects can be examined for each landmark separately as
    well as pooled.
    """
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    radii = radii or C.CROSS_RADIUS_GRID
    rows = []
    for pr, tr, cond, p1, p2 in _iter_dyads(data_dir, cond_map, pairs, progress):
        a, b = _prep_subset(p1), _prep_subset(p2)
        n = min(a.n_frames, b.n_frames)
        for k, name in enumerate(C.SUBSET_NAMES):
            mag_a = np.linalg.norm(a.coords[:n, k, :], axis=1)
            mag_b = np.linalg.norm(b.coords[:n, k, :], axis=1)
            for r in radii:
                res = run_cross_rqa(mag_a, mag_b, cross_params_fixed(r))
                rec = {"pair": pr, "trial": tr, "condition": cond,
                       "keypoint": name, "radius": float(r),
                       "perc_recur": float(res.rec_rate)}
                rec.update({key: float(v) for key, v in res.metrics.items()})
                rows.append(rec)
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


def run_embedding_estimates(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    pairs: list[int] | None = None,
    ami_max_lag: int = 60,
    fnn_max_dim: int = 10,
    fnn_threshold: float = 10.0,
    progress: bool = True,
) -> pd.DataFrame:
    """Per-trial, per-keypoint AMI and FNN estimates, tagged by condition.

    Answers whether the *ideal* embedding parameters differ systematically between
    visual-coupling conditions -- i.e. whether committing to one shared (tau, m)
    across conditions papers over a real difference. Returns one row per
    (pair, trial, person, keypoint) with the AMI first-minimum, the AMI 1/e
    crossing, and the smallest dimension whose false-neighbour fraction falls below
    ``fnn_threshold`` at that delay.
    """
    from ...embedding import ami_curve, fnn_curve

    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    rows = []
    for pr, tr, cond, p1, p2 in _iter_dyads(data_dir, cond_map, pairs, progress):
        a, b = _prep_subset(p1), _prep_subset(p2)
        n = min(a.n_frames, b.n_frames)
        for person, seq in ((1, a), (2, b)):
            for k, name in enumerate(C.SUBSET_NAMES):
                x = np.linalg.norm(seq.coords[:n, k, :], axis=1)
                rec = {"pair": pr, "trial": tr, "condition": cond,
                       "person": person, "keypoint": name,
                       "ami_first_min": np.nan, "ami_1e": np.nan, "fnn_dim": np.nan}
                curve = ami_curve(x, max_lag=ami_max_lag)
                if curve is not None:
                    v, lags = curve.ami, curve.lags
                    # first local minimum
                    loc = np.where((v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:]))[0]
                    if loc.size:
                        rec["ami_first_min"] = int(lags[loc[0] + 1])
                    # first crossing of 1/e of the lag-1 value (relative criterion)
                    below = np.where(v <= v[0] / np.e)[0]
                    if below.size:
                        rec["ami_1e"] = int(lags[below[0]])
                tau_here = int(rec["ami_first_min"]) if np.isfinite(rec["ami_first_min"]) else C.TAU
                fc = fnn_curve(x, tau=tau_here, max_dim=fnn_max_dim)
                if fc is not None:
                    ok = np.where(fc.pct_false <= fnn_threshold)[0]
                    if ok.size:
                        rec["fnn_dim"] = int(fc.dims[ok[0]])
                rows.append(rec)
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


def run_embedding_sweep(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    pairs: list[int] | None = None,
    taus: list[int] | None = None,
    ms: list[int] | None = None,
    radius: float | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """CRQA across a grid of embedding delays and dimensions, at a fixed radius.

    The companion to the radius sweep: it asks whether the condition effects depend
    on the committed ``(tau, m)``. One row per (pair, trial, keypoint, tau, m), with
    the five keypoints kept separate so they can be averaged or inspected per
    landmark downstream.
    """
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    taus = taus or [10, 15, 20, 25, 30]
    ms = ms or [3, 4, 5, 6]
    radius = radius if radius is not None else C.CROSS_RADIUS

    rows = []
    for pr, tr, cond, p1, p2 in _iter_dyads(data_dir, cond_map, pairs, progress):
        a, b = _prep_subset(p1), _prep_subset(p2)
        n = min(a.n_frames, b.n_frames)
        mags_a = np.stack([np.linalg.norm(a.coords[:n, k, :], axis=1) for k in range(len(C.SUBSET_INDICES))], axis=1)
        mags_b = np.stack([np.linalg.norm(b.coords[:n, k, :], axis=1) for k in range(len(C.SUBSET_INDICES))], axis=1)
        for tau in taus:
            for m in ms:
                if n - (m - 1) * tau < 100:      # too little series left to embed
                    continue
                p = RqaParams.from_embedding(
                    EmbeddingParams(tau=tau, m=m), radius_mode="fixed_radius",
                    radius=radius, rescale=C.RESCALE, min_line=C.MIN_LINE, norm=C.NORM,
                )
                for k, name in enumerate(C.SUBSET_NAMES):
                    res = run_cross_rqa(mags_a[:, k], mags_b[:, k], p)
                    rec = {"pair": pr, "trial": tr, "condition": cond, "keypoint": name,
                           "tau": tau, "m": m, "perc_recur": float(res.rec_rate)}
                    rec.update({key: float(v) for key, v in res.metrics.items()})
                    rows.append(rec)
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


def auto_params_fixed(radius: float) -> RqaParams:
    """Per-person auto-RQA at a supplied radius (Theiler window defaults to tau)."""
    return RqaParams.from_embedding(
        EmbeddingParams(tau=C.TAU, m=C.M), radius_mode="fixed_radius",
        radius=radius, rescale=C.RESCALE, min_line=C.MIN_LINE, norm=C.NORM,
    )


def run_auto_rqa_analysis(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    pairs: list[int] | None = None,
    radius: float | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Auto-RQA of each **individual's** keypoint signal, per person and keypoint.

    This is a within-person measure: it asks whether one participant's own
    trajectory revisits its previously visited states, with no partner involved.
    A condition effect here is therefore a change in individual movement structure,
    not in coordination -- the check that distinguishes the two for the head
    keypoint, whose cross-recurrence effect appears identically in surrogate pairs.

    Returns one row per (pair, trial, person, keypoint).
    """
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    lead_map = load_leader_map(conditions_csv)
    ap = auto_params_fixed(radius if radius is not None else C.CROSS_RADIUS)

    rows = []
    for pr, tr, cond, p1, p2 in _iter_dyads(data_dir, cond_map, pairs, progress):
        a, b = _prep_subset(p1), _prep_subset(p2)
        n = min(a.n_frames, b.n_frames)
        leader = lead_map.get((pr, tr), 1)
        for person, seq in ((1, a), (2, b)):
            for k, name in enumerate(C.SUBSET_NAMES):
                res = run_auto_rqa(np.linalg.norm(seq.coords[:n, k, :], axis=1), ap)
                row = {"pair": pr, "trial": tr, "condition": cond, "person": person,
                       "role": "leader" if person == leader else "follower",
                       "keypoint": name, "perc_recur": float(res.rec_rate)}
                row.update({key: float(v) for key, v in res.metrics.items()})
                rows.append(row)
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


def run_surrogate_analysis(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    pairs: list[int] | None = None,
    radius: float | None = None,
    n_surrogates: int = 5,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """Real vs. surrogate-pair CRQA at a fixed radius, per keypoint.

    A surrogate pair takes this trial's leader and a *different dyad's* follower
    from the **same condition**, so the two people never interacted but performed
    the same task under the same visual-coupling instruction. Recurrence surviving
    in surrogates reflects shared task and movement statistics rather than genuine
    interpersonal coupling.

    The condition contrast is the decisive part: if surrogates show the same
    condition effect as real dyads, the effect is driven by how people move rather
    than by how they coordinate.

    Returns one row per (pair, trial, keypoint, kind), ``kind`` in {real, surrogate}.
    """
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    lead_map = load_leader_map(conditions_csv)
    radius = radius if radius is not None else C.CROSS_RADIUS
    cp = cross_params_fixed(radius)
    rng = np.random.default_rng(seed)

    # Pass 1: cache each trial's leader/follower keypoint-magnitude streams.
    store: dict[tuple[int, int], dict] = {}
    for pr, tr, cond, p1, p2 in _iter_dyads(data_dir, cond_map, pairs, progress):
        a, b = _prep_subset(p1), _prep_subset(p2)
        n = min(a.n_frames, b.n_frames)
        mags = {}
        for person, seq in ((1, a), (2, b)):
            mags[person] = np.stack(
                [np.linalg.norm(seq.coords[:n, k, :], axis=1) for k in range(len(C.SUBSET_INDICES))],
                axis=1)
        leader = lead_map.get((pr, tr), 1)
        follower = 2 if leader == 1 else 1
        store[(pr, tr)] = {"condition": cond, "leader": mags[leader], "follower": mags[follower]}

    by_cond: dict[str, list[tuple[int, int]]] = {}
    for key, rec in store.items():
        by_cond.setdefault(rec["condition"], []).append(key)

    def _crqa_rows(x: np.ndarray, y: np.ndarray, base: dict) -> list[dict]:
        n = min(len(x), len(y))
        out = []
        for k, name in enumerate(C.SUBSET_NAMES):
            res = run_cross_rqa(x[:n, k], y[:n, k], cp)
            row = {**base, "keypoint": name, "perc_recur": float(res.rec_rate)}
            row.update({key: float(v) for key, v in res.metrics.items()})
            out.append(row)
        return out

    rows = []
    for (pr, tr), rec in store.items():
        base = {"pair": pr, "trial": tr, "condition": rec["condition"]}
        rows += _crqa_rows(rec["leader"], rec["follower"], {**base, "kind": "real", "surrogate_of": None})

        # surrogate partners: same condition, different dyad
        candidates = [k for k in by_cond[rec["condition"]] if k[0] != pr]
        if not candidates:
            continue
        take = min(n_surrogates, len(candidates))
        picks = rng.choice(len(candidates), size=take, replace=False)
        for i in picks:
            other = candidates[int(i)]
            rows += _crqa_rows(rec["leader"], store[other]["follower"],
                               {**base, "kind": "surrogate", "surrogate_of": f"{other[0]}_{other[1]}"})

    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


def run_md_radius_sweep(
    data_dir: str | Path,
    conditions_csv: str | Path | None = None,
    pairs: list[int] | None = None,
    radii: list[float] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """MdCRQA across a grid of fixed radii for every dyad-trial (tidy long table).

    One row per (pair, trial, radius), so the condition effects can be refitted at
    each radius to check that conclusions do not depend on the threshold.
    """
    data_dir = Path(data_dir)
    cond_map = load_condition_map(conditions_csv)
    rows = []
    for pr, tr, cond, p1, p2 in _iter_dyads(data_dir, cond_map, pairs, progress):
        a, b = _prep_subset(p1), _prep_subset(p2)
        n = min(a.n_frames, b.n_frames)
        ma = np.stack([np.linalg.norm(a.coords[:n, k, :], axis=1) for k in range(len(C.SUBSET_INDICES))], axis=1)
        mb = np.stack([np.linalg.norm(b.coords[:n, k, :], axis=1) for k in range(len(C.SUBSET_INDICES))], axis=1)
        for rec in md_radius_sweep(ma, mb, radii):
            rows.append({"pair": pr, "trial": tr, "condition": cond, **rec})
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


def _iter_dyads(data_dir: Path, cond_map: dict, pairs: list[int] | None, progress: bool):
    """Yield ``(pair, trial, condition, seq1, seq2)`` for every usable dyad-trial."""
    files = [p for p in data_dir.glob("P*_T*_P*_pose_3d.csv") if not p.name.startswith("._")]
    by_key: dict[tuple[int, int], dict[int, Path]] = {}
    for f in files:
        pr, tr, pe = parse_file(f)
        by_key.setdefault((pr, tr), {})[pe] = f

    keys = sorted(k for k, v in by_key.items() if {1, 2} <= set(v))
    if pairs is not None:
        keys = [k for k in keys if k[0] in pairs]

    min_frames = (C.M + 1) * C.TAU + 50
    for i, (pr, tr) in enumerate(keys):
        cond = cond_map.get((pr, tr))
        if cond not in C.CONDITION_ORDER:
            continue
        if progress:
            print(f"[{i + 1}/{len(keys)}] pair {pr} trial {tr} ({cond})")
        p1 = load_and_resample(by_key[(pr, tr)][1])
        p2 = load_and_resample(by_key[(pr, tr)][2])
        if min(p1.n_frames, p2.n_frames) < min_frames:
            if progress:
                print(f"    skipped (too short: {min(p1.n_frames, p2.n_frames)} frames)")
            continue
        yield pr, tr, cond, p1, p2


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
    lead_map = load_leader_map(conditions_csv)

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
        rows.append(process_dyad(p1, p2, cond, leader=lead_map.get((pr, tr))))
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return df


# ----------------------------------------------------------------------
# Movement summaries by condition (and by role)
# ----------------------------------------------------------------------
KINEMATIC_FEATURES = ["disp_mean", "disp_rms", "vel_mean", "vel_rms", "accel_mean", "accel_rms"]


def movement_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the dyad-level table to one row per person-trial with a ``role``.

    Requires the ``leader_*``/``follower_*`` columns emitted when the leader is
    known. Used for the role x condition movement summaries.
    """
    if "leader_person" not in df.columns:
        raise ValueError("no role columns; run_reproduction needs the conditions table with block1_lead.")
    keep = ["pair", "trial", "condition"]
    parts = []
    for role in ("leader", "follower"):
        sub = df[keep + [f"{role}_{f}" for f in KINEMATIC_FEATURES]].copy()
        sub.columns = keep + KINEMATIC_FEATURES
        sub["role"] = role
        if role == "leader":
            sub["person"] = df["leader_person"].to_numpy()
        else:
            sub["person"] = np.where(df["leader_person"].to_numpy() == 1, 2, 1)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out["condition"] = pd.Categorical(out["condition"], categories=C.CONDITION_ORDER, ordered=True)
    return out


def movement_summary(df: pd.DataFrame, by_role: bool = True,
                     features: list[str] | None = None) -> pd.DataFrame:
    """Descriptive movement statistics (mean, SD, n) by condition, optionally by role.

    This is the descriptive companion to the inferential models: it answers how
    much participants moved in each condition, and whether leaders and followers
    differ, rather than only whether condition coefficients are reliable.
    """
    features = features or KINEMATIC_FEATURES
    if by_role:
        long = movement_long(df)
        grouped = long.groupby(["condition", "role"], observed=True)[features]
    else:
        grouped = df.groupby("condition", observed=True)[features]
    out = grouped.agg(["mean", "std", "count"])
    return out


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
_PANELS = [
    ("accel_rms", "Acceleration RMS"),
    # Primary mode is a fixed radius, so %REC is a result and is what the figure
    # (and its caption) reports. Under the fixed-2.5%-REC mode %REC is pinned and
    # the achieved radius would be plotted here instead.
    ("crossfx_perc_recur", "Cross-recurrence rate (%REC)"),
    ("crossfx_lmax", "Cross Lmax"),
]


# MdCRQA counterpart of ``_PANELS``: the two bodies compared as whole
# five-dimensional systems at the fixed radius (C.MD_RADIUS), so %REC is an
# outcome. These are the ``mdfx_*`` columns the manuscript reports.
_MD_PANELS = [
    ("mdfx_perc_recur", "MdCRQA recurrence rate (%REC)"),
    ("mdfx_maxl_found", "MdCRQA Lmax"),
    ("mdfx_trapping_time", "MdCRQA trapping time"),
]


def _bar_panels(df: pd.DataFrame, panels, axes=None, figsize=(12, 4)):
    """Draw one mean +/- SEM bar panel per (column, ylabel) in ``panels``."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, len(panels), figsize=figsize)
    axes = np.asarray(axes).flatten()
    for (metric, ylab), ax in zip(panels, axes):
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


def plot_case3_figure(df: pd.DataFrame, axes=None):
    """Group-averaged (mean +/- SEM) kinematics and CRQA by visual-coupling condition."""
    return _bar_panels(df, _PANELS, axes=axes)


def plot_md_figure(df: pd.DataFrame, axes=None):
    """Group-averaged (mean +/- SEM) MdCRQA metrics by visual-coupling condition.

    Same format as :func:`plot_case3_figure`, for the whole-body multivariate
    analysis: %REC, maximum diagonal line length, and trapping time.
    """
    return _bar_panels(df, _MD_PANELS, axes=axes)
