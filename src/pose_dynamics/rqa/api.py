from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pose_dynamics.progress import stage_progress_with_total
from pose_dynamics.rqa.rqa_config import ConfigError, RQAConfig

try:
    from pose_dynamics.rqa.utils import norm_utils, rqa_utils_cpp
except ImportError:
    rqa_utils_cpp = None


@dataclass(frozen=True)
class RQAOutputs:
    stats_path: Path
    qc_path: Path
    provenance_path: Path
    plots_dir: Path


def _time_col(df: pd.DataFrame) -> str:
    return "time" if "time" in df.columns else "frame"


def _embed(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = X.shape[0] - (m - 1) * tau
    if n <= 0:
        return np.empty((0, X.shape[1] * m))
    parts = [X[i : i + n] for i in range(0, m * tau, tau)]
    return np.concatenate(parts, axis=1)


def _rr_to_epsilon(dist: np.ndarray, rr_target: float) -> float:
    d = dist[np.triu_indices_from(dist, k=1)]
    if d.size == 0:
        return float("nan")
    return float(np.percentile(d, rr_target * 100))


def _epsilon(dist: np.ndarray, cfg: RQAConfig) -> float:
    if cfg.epsilon.mode == "absolute":
        return float(cfg.epsilon.value)
    if cfg.epsilon.mode == "percentile":
        d = dist[np.triu_indices_from(dist, k=1)]
        return float(np.percentile(d, cfg.epsilon.value)) if d.size else float("nan")
    if cfg.epsilon.mode == "rr_target":
        return _rr_to_epsilon(dist, cfg.epsilon.value)
    if cfg.epsilon.mode == "mean_scaled":
        d = dist[np.triu_indices_from(dist, k=1)]
        if d.size == 0:
            return float("nan")
        return float(np.mean(d) * cfg.epsilon.value)
    raise ConfigError("unknown epsilon.mode")


def _line_lengths(mat: np.ndarray, min_len: int, axis: int = 0) -> list[int]:
    # axis=0 for diagonals, axis=1 for vertical
    lengths = []
    if axis == 0:
        # diagonals
        for k in range(-mat.shape[0] + 1, mat.shape[1]):
            diag = np.diagonal(mat, offset=k)
            lengths += _run_lengths(diag, min_len)
    else:
        # vertical
        for j in range(mat.shape[1]):
            col = mat[:, j]
            lengths += _run_lengths(col, min_len)
    return lengths


def _run_lengths(arr: np.ndarray, min_len: int) -> list[int]:
    lengths = []
    run = 0
    for v in arr:
        if v:
            run += 1
        else:
            if run >= min_len:
                lengths.append(run)
            run = 0
    if run >= min_len:
        lengths.append(run)
    return lengths


def _rqa_stats(mat: np.ndarray, l_min: int, v_min: int, theiler: int) -> dict:
    N = mat.shape[0]
    if N == 0:
        return {}
    # exclude main diagonal for RQA
    mask = np.ones_like(mat, dtype=bool)
    for k in range(-theiler, theiler + 1):
        if k == 0:
            np.fill_diagonal(mask, False)
        else:
            i = np.arange(max(0, -k), min(N, N - k))
            mask[i, i + k] = False
    recur_points = int(np.sum(mat & mask))
    total_points = int(np.sum(mask))
    rr = recur_points / total_points if total_points else 0.0

    diag_lengths = _line_lengths(mat, l_min, axis=0)
    vert_lengths = _line_lengths(mat, v_min, axis=1)

    det = (np.sum(diag_lengths) / recur_points) if recur_points else 0.0
    lam = (np.sum(vert_lengths) / recur_points) if recur_points else 0.0

    maxl = max(diag_lengths) if diag_lengths else 0
    meanl = float(np.mean(diag_lengths)) if diag_lengths else 0.0
    stdl = float(np.std(diag_lengths)) if diag_lengths else 0.0
    countl = int(len(diag_lengths))
    ent = 0.0
    if diag_lengths:
        vals, counts = np.unique(diag_lengths, return_counts=True)
        p = counts / counts.sum()
        ent = float(-np.sum(p * np.log(p)))

    vmax = max(vert_lengths) if vert_lengths else 0
    tt = float(np.mean(vert_lengths)) if vert_lengths else 0.0
    div = 1.0 / maxl if maxl else 0.0

    return {
        "perc_recur": rr * 100.0,
        "perc_determ": det * 100.0,
        "maxl_found": float(maxl),
        "mean_line_length": meanl,
        "std_line_length": stdl,
        "count_line": countl,
        "entropy": ent,
        "laminarity": lam * 100.0,
        "trapping_time": tt,
        "vmax": float(vmax),
        "divergence": div,
    }


def _drp(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = mat.shape[0]
    lags = np.arange(-(N - 1), N)
    drp = []
    for k in lags:
        diag = np.diagonal(mat, offset=k)
        rr = diag.mean() * 100.0 if diag.size else 0.0
        drp.append(rr)
    return lags, np.array(drp)


def _pivot_xy(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    val_cols = [c for c in ["x", "y", "z"] if c in df.columns]
    return df.pivot_table(
        index=tcol, columns="keypoint", values=val_cols, aggfunc="mean"
    )


def _compute_derived_series(
    pivot: pd.DataFrame,
    df_head: Optional[pd.DataFrame] = None,
) -> dict[str, np.ndarray]:
    """Derived signals (blink, mouth, pupil, head motion) on pivoted trial data.
    
    Args:
        pivot: Pivoted dataframe with time index and (dim, keypoint) columns
        df_head: Optional head motion transforms dataframe
    """
    # Minimal defaults mirror examples/case_study_1_matb/configs/features.yaml
    blink_left_upper = ["38", "39"]
    blink_left_lower = ["41", "42"]
    blink_right_upper = ["44", "45"]
    blink_right_lower = ["47", "48"]
    mouth_upper = ["63"]
    mouth_lower = ["67"]
    left_pupil = ["69"]
    right_pupil = ["70"]
    left_eye_contour = ["37", "38", "39", "41", "42"]
    right_eye_contour = ["44", "45", "46", "47", "48"]

    def _xy_for_ref(
        piv: pd.DataFrame, ref: list[str]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for kp in ref:
            if ("x", kp) not in piv.columns or ("y", kp) not in piv.columns:
                continue
            xs.append(piv[("x", kp)].to_numpy(dtype=float))
            ys.append(piv[("y", kp)].to_numpy(dtype=float))
        if not xs or not ys:
            return None
        return np.nanmean(np.vstack(xs), axis=0), np.nanmean(np.vstack(ys), axis=0)

    def _vertical_gap(
        piv: pd.DataFrame, upper: list[str], lower: list[str]
    ) -> np.ndarray | None:
        upper_xy = _xy_for_ref(piv, upper)
        lower_xy = _xy_for_ref(piv, lower)
        if upper_xy is None or lower_xy is None:
            return None
        return np.abs(upper_xy[1] - lower_xy[1])

    def _dist(
        x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
    ) -> np.ndarray:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    out: dict[str, np.ndarray] = {}
    if pivot.empty:
        return out

    # Blink aperture (mean of both eyes when available)
    left_gap = _vertical_gap(pivot, blink_left_upper, blink_left_lower)
    right_gap = _vertical_gap(pivot, blink_right_upper, blink_right_lower)
    if left_gap is not None and right_gap is not None:
        out["blink"] = np.nanmean(np.vstack([left_gap, right_gap]), axis=0)
    elif left_gap is not None:
        out["blink"] = left_gap
    elif right_gap is not None:
        out["blink"] = right_gap

    # Mouth aperture
    upper = _xy_for_ref(pivot, mouth_upper)
    lower = _xy_for_ref(pivot, mouth_lower)
    if upper is not None and lower is not None:
        out["mouth_aperture"] = _dist(upper[0], upper[1], lower[0], lower[1])

    # Pupil displacement (averaged across eyes)
    pupil_components: dict[str, list[np.ndarray]] = {"mag": [], "dx": [], "dy": []}
    for suffix, pupil_ref, contour in [
        ("left", left_pupil, left_eye_contour),
        ("right", right_pupil, right_eye_contour),
    ]:
        center = _xy_for_ref(pivot, contour)
        pupil = _xy_for_ref(pivot, pupil_ref)
        if center is None or pupil is None:
            continue
        dx = pupil[0] - center[0]
        dy = pupil[1] - center[1]
        mag = np.sqrt(dx**2 + dy**2)
        out[f"pupil_{suffix}_dx"] = dx
        out[f"pupil_{suffix}_dy"] = dy
        out[f"pupil_{suffix}_mag"] = mag
        pupil_components["mag"].append(mag)
        pupil_components["dx"].append(dx)
        pupil_components["dy"].append(dy)

    if pupil_components["mag"]:
        out["pupil_mag"] = np.nanmean(np.vstack(pupil_components["mag"]), axis=0)
    if pupil_components["dx"]:
        out["pupil_dx"] = np.nanmean(np.vstack(pupil_components["dx"]), axis=0)
    if pupil_components["dy"]:
        out["pupil_dy"] = np.nanmean(np.vstack(pupil_components["dy"]), axis=0)

    # Head motion from alignment transforms (already aligned by time)
    if df_head is not None and not df_head.empty:
        if "scale" in df_head.columns:
            out["head_scale"] = df_head["scale"].to_numpy(dtype=float)
        if "rotation_angle" in df_head.columns:
            out["head_rotation"] = df_head["rotation_angle"].to_numpy(dtype=float)
        tx = df_head["translation_x"].to_numpy(dtype=float)
        ty = df_head["translation_y"].to_numpy(dtype=float)
        out["head_tx"] = tx
        out["head_ty"] = ty
        out["head_translation_mag"] = np.sqrt(tx**2 + ty**2)
        components = []
        if "head_translation_mag" in out:
            components.append(out["head_translation_mag"])
        if "head_rotation" in out:
            components.append(out["head_rotation"])
        if "head_scale" in out:
            components.append(out["head_scale"] - 1.0)
        if components:
            stack = np.vstack(components)
            out["head_motion_mag"] = np.sqrt(np.nansum(stack**2, axis=0))

    return out


def _series_matrix(
    pivot: pd.DataFrame,
    keypoints: list[str],
    signal: str,
    derived: Optional[dict[str, np.ndarray]] = None,
    # Slice indices (start, end)
    indices: slice = slice(None),
) -> np.ndarray:
    dims = [
        c
        for c in ["x", "y", "z"]
        if ("x", keypoints[0]) in pivot.columns or c in ["x", "y", "z"]
    ]
    # Note: dims detection above is imperfect if first kp misses some dims.
    # Better: just look at pivot.columns levels
    if ("z", keypoints[0]) in pivot.columns:
        dims = ["x", "y", "z"]
    else:
        dims = ["x", "y"]

    rows: list[np.ndarray] = []

    for kp in keypoints:
        if derived and kp in derived:
            vals = derived[kp][indices]
            if vals.ndim == 1:
                vals = vals.reshape(-1, 1)
            rows.append(vals)
            continue

        # Check availability
        found_dims = []
        for d in dims:
            if (d, kp) in pivot.columns:
                found_dims.append((d, kp))

        if not found_dims:
            continue

        # Extract columns for this keypoint slice
        vals = pivot.loc[pivot.index[indices], found_dims].to_numpy(dtype=float)

        if vals.size == 0:
            continue
        if signal == "magnitude":
            vals = np.linalg.norm(vals, axis=1, keepdims=True)
        rows.append(vals)

    if not rows:
        return np.empty((0, 1))

    # Align lengths (truncate to the shortest to keep sample sync)
    min_len = min(r.shape[0] for r in rows)
    rows = [r[:min_len] for r in rows]
    # Ensure 2D then concatenate feature-wise
    rows = [r if r.ndim == 2 else r.reshape(-1, 1) for r in rows]
    return np.concatenate(rows, axis=1)


def run_rqa(
    pose_clean_path: str | Path,
    windows_path: str | Path,
    config: RQAConfig | str | Path | None = None,
    out_dir: str | Path | None = None,
    *,
    config_dict: dict | None = None,
    pose_y_path: Optional[str | Path] = None,
    overwrite: bool = False,
    progress_title: str | None = None,
) -> RQAOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config_dict:
        # Map flat dict to RQAConfig structure
        # This mapping depends on RQAConfig validation, which we might need to bypass or construct carefully
        # Alternatively, we just use the dict directly if we refactor, but to keep type safety let's try to map
        # Simplified mapping logic for now:
        from pose_dynamics.rqa.rqa_config import RQAConfig

        # determine analysis type
        atype = config_dict.get("analysis_type", "rqa")
        is_crqa = atype == "crqa"

        # keypoints
        if is_crqa:
            # We need to handle X and Y. RQAConfig might assume single set.
            # If RQAConfig supports separate X/Y keypoints, good. If not, we might need to update RQAConfig
            # OR logic below needs to handle 'keypoint_x' vs 'keypoint_y'.
            # For now, let's assume we pass X keypoints here and handle Y separately or update RQAConfig
            # Let's see how RQAConfig is defined. Assuming it needs 'keypoints' list.
            kp = config_dict.get("keypoint_x", [])
            kp_y = config_dict.get(
                "keypoint_y", []
            )  # We'll need to pass this somehow or assume config holds it
        else:
            kp = config_dict.get("keypoints", [])
            kp_y = None

        # radius
        rad_val = config_dict.get("radius", 0.0)
        rad_mode = config_dict.get("radius_mode", "absolute")

        # embed
        emb = config_dict.get("embedding", {})
        m = emb.get("dim", 1)
        tau = emb.get("tau", 1)

        @dataclass
        class SimpleConfig:
            analysis: str
            keypoints: list
            keypoints_y: list | None
            signal: str
            m: int
            tau: int
            l_min: int
            v_min: int
            theiler: int
            epsilon: Any
            plots: Any
            norm: str
            rescale_norm: bool

        @dataclass
        class Epsilon:
            value: float
            mode: str

        @dataclass
        class Plots:
            enabled: bool
            max_plots: int

        cfg = SimpleConfig(
            analysis=atype,
            keypoints=kp,
            keypoints_y=kp_y,
            signal=config_dict.get(
                "signal", "magnitude"
            ),  # default to magnitude for example
            m=m,
            tau=tau,
            l_min=config_dict.get("line_min", 2),
            v_min=config_dict.get(
                "line_min", 2
            ),  # assume same for vertical for now unless specified
            theiler=config_dict.get("theiler", 1),
            epsilon=Epsilon(value=rad_val, mode=rad_mode),
            plots=Plots(
                enabled=False, max_plots=0
            ),  # disable plots for batch run mostly
            norm=config_dict.get("norm", "none"),
            rescale_norm=bool(config_dict.get("rescale_norm", False)),
        )

    elif isinstance(config, RQAConfig):
        cfg = config
        # Backwards compat hack if RQAConfig doesn't have keypoints_y
        if not hasattr(cfg, "keypoints_y"):
            cfg.keypoints_y = None
    elif config:
        cfg = RQAConfig.from_yaml(str(config))
        if not hasattr(cfg, "keypoints_y"):
            cfg.keypoints_y = None
    else:
        raise ValueError("Must provide config or config_dict")

    pose_clean_path = Path(pose_clean_path)
    windows_path = Path(windows_path)
    df = pd.read_parquet(pose_clean_path)
    # Optional alignment transforms (for head motion derived signals)
    align_path = pose_clean_path.parent / "alignment_transforms.parquet"
    df_head = pd.read_parquet(align_path) if align_path.exists() else None
    
    # Load pre-computed ROI series if available (from feature extraction)
    # Look in features directory (sibling to preprocess directory)
    preprocess_dir = pose_clean_path.parent
    features_dir = preprocess_dir.parent.parent / "features" / preprocess_dir.name
    roi_series_path = features_dir / "roi_series.parquet"
    df_roi_series: Optional[pd.DataFrame] = None
    if roi_series_path.exists():
        df_roi_series = pd.read_parquet(roi_series_path)
    
    windows = pd.read_parquet(windows_path)

    if windows.empty:
        raise ConfigError("windows.parquet is empty.")

    if "dropped" in windows.columns:
        windows = windows[~windows["dropped"]].copy()

    df_y = None  # For CRQA Y-data

    # Identify unique keypoints available
    all_kps = sorted(df["keypoint"].dropna().unique().tolist())

    # Helper to resolve keypoints list
    def resolve_kps(kp_list):
        if not kp_list:
            return []
        if kp_list == "all":
            return all_kps
        return list(kp_list)

    kps_x = resolve_kps(cfg.keypoints)
    kps_y = resolve_kps(cfg.keypoints_y) if cfg.keypoints_y else []

    tcol = _time_col(df)
    stats_rows = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # For CRQA: detect trial pairs (e.g., S002_T1_left <-> S002_T1_right)
    # Build a mapping of base_id -> {left_trial_id, right_trial_id}
    trial_pairs: dict[str, dict[str, str]] = {}
    all_trial_ids = windows["trial_id"].unique().tolist()
    
    for tid in all_trial_ids:
        tid_str = str(tid)
        if tid_str.endswith("_left"):
            base = tid_str[:-5]  # Remove "_left"
            if base not in trial_pairs:
                trial_pairs[base] = {}
            trial_pairs[base]["left"] = tid_str
        elif tid_str.endswith("_right"):
            base = tid_str[:-6]  # Remove "_right"
            if base not in trial_pairs:
                trial_pairs[base] = {}
            trial_pairs[base]["right"] = tid_str

    # Check if we have valid pairs for CRQA
    is_paired_crqa = cfg.analysis == "crqa" and any(
        "left" in p and "right" in p for p in trial_pairs.values()
    )

    # Group windows by trial to avoid repeatedly slicing full dataframes
    windows_by_trial = windows.groupby("trial_id", sort=False)

    total_windows = len(windows)
    processed = 0

    # Pre-compute derived series for all trials (needed for paired CRQA)
    trial_pivots: dict[str, pd.DataFrame] = {}
    trial_derived: dict[str, dict[str, np.ndarray]] = {}
    
    for trial_id in all_trial_ids:
        df_trial = df[df["trial_id"] == trial_id]
        if df_trial.empty:
            continue
        pivot = _pivot_xy(df_trial, tcol)
        if pivot.empty:
            continue
        trial_pivots[str(trial_id)] = pivot
        
        # Align head data
        df_head_filtered = None
        if df_head is not None:
            dh = df_head[df_head["trial_id"] == trial_id]
            if not dh.empty and tcol in dh.columns:
                dh = dh.drop_duplicates(subset=[tcol])
                dh = dh.set_index(tcol).reindex(pivot.index)
                df_head_filtered = dh
        
        # Compute derived series (blink, mouth, pupil, head motion, and fallback ROI)
        derived = _compute_derived_series(pivot, df_head_filtered)
        
        # Merge in pre-computed ROI series if available (takes precedence)
        if df_roi_series is not None:
            roi_trial = df_roi_series[df_roi_series["trial_id"] == trial_id]
            if not roi_trial.empty:
                # Get the ROI columns (exclude trial_id and time)
                roi_cols = [c for c in roi_trial.columns if c not in ["trial_id", tcol, "time", "frame"]]
                for col in roi_cols:
                    series_values = roi_trial[col].to_numpy()
                    if series_values.size > 0:
                        derived[col] = series_values
        
        trial_derived[str(trial_id)] = derived

    with stage_progress_with_total(
        progress_title or "RQA", total_windows
    ) as update_progress:
        # For paired CRQA, process by session pairs
        if is_paired_crqa:
            for base_id, pair in trial_pairs.items():
                if "left" not in pair or "right" not in pair:
                    continue
                
                left_tid = pair["left"]
                right_tid = pair["right"]
                
                if left_tid not in trial_pivots or right_tid not in trial_pivots:
                    continue
                
                pivot_left = trial_pivots[left_tid]
                pivot_right = trial_pivots[right_tid]
                derived_left = trial_derived[left_tid]
                derived_right = trial_derived[right_tid]
                
                # Get windows for left trial (use left as reference for window timing)
                if left_tid not in windows_by_trial.groups:
                    continue
                win_df = windows_by_trial.get_group(left_tid)
                
                times_left = pivot_left.index.to_numpy()
                times_right = pivot_right.index.to_numpy()
                
                for _, w in win_df.iterrows():
                    s = float(w["start"])
                    e = float(w["end"])
                    
                    istart_l = np.searchsorted(times_left, s)
                    iend_l = np.searchsorted(times_left, e)
                    istart_r = np.searchsorted(times_right, s)
                    iend_r = np.searchsorted(times_right, e)
                    
                    if istart_l >= iend_l or istart_r >= iend_r:
                        processed += 1
                        update_progress(f"{base_id} | empty window", advance=1)
                        continue
                    
                    # Get X series from left participant
                    X_win = _series_matrix(
                        pivot_left, kps_x, cfg.signal, derived_left, 
                        indices=slice(istart_l, iend_l)
                    )
                    # Get Y series from right participant
                    Y_win = _series_matrix(
                        pivot_right, kps_y if kps_y else kps_x, cfg.signal, derived_right,
                        indices=slice(istart_r, iend_r)
                    )
                    
                    if X_win.size == 0 or Y_win.size == 0:
                        processed += 1
                        update_progress(f"{base_id} | empty signal", advance=1)
                        continue
                    
                    # Align lengths (truncate to shorter)
                    min_len = min(X_win.shape[0], Y_win.shape[0])
                    X_win = X_win[:min_len]
                    Y_win = Y_win[:min_len]
                    
                    Xn_win = norm_utils.normalize_data(X_win, cfg.norm)
                    Yn_win = norm_utils.normalize_data(Y_win, cfg.norm)
                    
                    ds = rqa_utils_cpp.rqa_dist(Xn_win, Yn_win, dim=cfg.m, lag=cfg.tau)
                    dist_win = ds.get("d")
                    
                    if dist_win is None or dist_win.size == 0:
                        processed += 1
                        update_progress(f"{base_id} | empty dist", advance=1)
                        continue
                    
                    eps = _epsilon(dist_win, cfg)
                    
                    # CRQA stats
                    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(
                        dist_win.astype(float),
                        rescale=bool(cfg.rescale_norm),
                        rad=float(eps),
                        diag_ignore=0,  # No theiler for cross-recurrence
                        minl=int(cfg.l_min),
                        rqa_mode="cross",
                    )
                    
                    if err_code != 0 or rs is None:
                        processed += 1
                        update_progress(f"{base_id} | rqa err {err_code}", advance=1)
                        continue
                    
                    stats_rows.append(
                        {
                            "trial_id": base_id,  # Use base session ID
                            "window_id": w["window_id"],
                            "epsilon": float(eps),
                            **{k: float(v) for k, v in rs.items()},
                        }
                    )
                    
                    processed += 1
                    label = f"{base_id} | {w['window_id']} ({processed}/{total_windows})"
                    update_progress(label, advance=1)
        
        else:
            # Original single-trial processing (RQA or same-trial CRQA)
            for trial_id, win_df in windows_by_trial:
                df_trial = df[df["trial_id"] == trial_id]
                if df_trial.empty:
                    processed += len(win_df)
                    update_progress(f"{trial_id} | no data", advance=len(win_df))
                    continue

                # Per-Trial Optimization: Pivot once (Time x Keypoints)
                pivot = _pivot_xy(df_trial, tcol)
                if pivot.empty:
                    processed += len(win_df)
                    update_progress(f"{trial_id} | empty pivot", advance=len(win_df))
                    continue

                # Align head data to pivot index
                df_head_filtered = None
                if df_head is not None:
                    dh = df_head[df_head["trial_id"] == trial_id]
                    if not dh.empty:
                        if tcol in dh.columns:
                            dh = dh.drop_duplicates(subset=[tcol])
                            dh = dh.set_index(tcol).reindex(pivot.index)
                            df_head_filtered = dh

                # Compute derived signals for the whole trial
                derived_trial = _compute_derived_series(pivot, df_head_filtered)

                # Pre-calc times for binary search
                times = pivot.index.to_numpy()

                for _, w in win_df.iterrows():
                    s = float(w["start"])
                    e = float(w["end"])

                    istart = np.searchsorted(times, s)
                    iend = np.searchsorted(times, e)

                    if istart >= iend:
                        processed += 1
                        update_progress(f"{trial_id} | empty window", advance=1)
                        continue

                    # Slice series to the window and compute distance just for that window
                    X_win = _series_matrix(
                        pivot, kps_x, cfg.signal, derived_trial, indices=slice(istart, iend)
                    )
                    if X_win.size == 0:
                        processed += 1
                        update_progress(f"{trial_id} | empty signal", advance=1)
                        continue

                    Xn_win = norm_utils.normalize_data(X_win, cfg.norm)

                    Yn_win = None
                    if cfg.analysis == "crqa":
                        if kps_y:
                            Y_win = _series_matrix(
                                pivot,
                                kps_y,
                                cfg.signal,
                                derived_trial,
                                indices=slice(istart, iend),
                            )
                        elif df_y is not None:
                            processed += 1
                            update_progress(f"{trial_id} | missing Y", advance=1)
                            continue
                        else:
                            processed += 1
                            update_progress(f"{trial_id} | missing Y", advance=1)
                            continue

                        if Y_win.size == 0:
                            processed += 1
                            update_progress(f"{trial_id} | empty Y", advance=1)
                            continue

                        Yn_win = norm_utils.normalize_data(Y_win, cfg.norm)

                    if cfg.analysis == "crqa":
                        ds = rqa_utils_cpp.rqa_dist(Xn_win, Yn_win, dim=cfg.m, lag=cfg.tau)
                    else:
                        ds = rqa_utils_cpp.rqa_dist(Xn_win, Xn_win, dim=cfg.m, lag=cfg.tau)

                    dist_win = ds.get("d")
                    if dist_win is None or dist_win.size == 0:
                        processed += 1
                        update_progress(f"{trial_id} | empty dist", advance=1)
                        continue

                    eps = _epsilon(dist_win, cfg)

                    # Delegate stats to C++ backend
                    rqa_mode = "cross" if cfg.analysis == "crqa" else "auto"
                    diag_ignore = 0 if rqa_mode == "cross" else cfg.theiler
                    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(
                        dist_win.astype(float),
                        rescale=bool(cfg.rescale_norm),
                        rad=float(eps),
                        diag_ignore=int(diag_ignore),
                        minl=int(cfg.l_min),
                        rqa_mode=rqa_mode,
                    )

                    if err_code != 0 or rs is None:
                        processed += 1
                        update_progress(f"{trial_id} | rqa err {err_code}", advance=1)
                        continue

                    stats_rows.append(
                        {
                            "trial_id": trial_id,
                            "window_id": w["window_id"],
                            "epsilon": float(eps),
                            **{k: float(v) for k, v in rs.items()},
                        }
                    )

                    processed += 1
                    label = f"{trial_id} | {w['window_id']} ({processed}/{total_windows})"
                    update_progress(label, advance=1)

    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / "rqa_stats.parquet"
    # Ensure target directory exists (defensive even though created above)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    # Save CSV too for easy reading
    stats_path_csv = out_dir / "rqa_stats.csv"

    if not overwrite and stats_path.exists():
        raise FileExistsError(f"Output already exists: {stats_path}")

    if not stats_df.empty:
        stats_df.to_parquet(stats_path, index=False)
        stats_df.to_csv(stats_path_csv, index=False)

    # ... generate json provenance ...

    return RQAOutputs(
        stats_path=stats_path,
        qc_path=out_dir / "qc.json",  # filler
        provenance_path=out_dir / "prov.json",  # filler
        plots_dir=plots_dir,
    )
