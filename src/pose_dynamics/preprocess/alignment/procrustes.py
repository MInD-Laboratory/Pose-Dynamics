from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig

_ALIGNMENT_DEBUG = os.environ.get("POSE_DYNAMICS_DEBUG_ALIGNMENT", "").lower() in (
    "1",
    "true",
    "yes",
)


def _resolve_alignment_flags(cfg: PreprocessConfig) -> Tuple[bool, bool, bool]:
    # Defaults derived from transform if explicit toggles are not provided
    if cfg.alignment.transform == "rigid":
        defaults = (True, False, True)  # rotation, scaling, translation
    else:
        defaults = (True, True, True)

    rot = (
        defaults[0] if cfg.alignment.rotation is None else bool(cfg.alignment.rotation)
    )
    scale = (
        defaults[1] if cfg.alignment.scaling is None else bool(cfg.alignment.scaling)
    )
    trans = (
        defaults[2]
        if cfg.alignment.translation is None
        else bool(cfg.alignment.translation)
    )
    return rot, scale, trans


def _valid_keypoints_for_alignment(
    df: pd.DataFrame, cfg: PreprocessConfig
) -> List[str]:
    kps = sorted(df["keypoint"].dropna().unique().tolist())
    if cfg.alignment.keypoints == "all":
        return kps

    missing = sorted(set(cfg.alignment.keypoints) - set(kps))
    if missing:
        raise ConfigError(
            "alignment.keypoints must be selected for preprocessing; missing: "
            f"{missing}"
        )
    return list(cfg.alignment.keypoints)


def _compute_mean_pose(
    df_trial: pd.DataFrame, keypoints: List[str], dims: List[str], cfg: PreprocessConfig
) -> Tuple[np.ndarray, List[str]]:
    means = []
    used = []
    coverage_debug: List[str] = []
    for kp in keypoints:
        g = df_trial[df_trial["keypoint"] == kp]
        if g.empty:
            coverage_debug.append(f"{kp}: total=0, valid_frac=0.000")
            continue
        vals = g[dims]
        valid = vals.notna().all(axis=1)
        valid_frac = float(valid.mean()) if len(valid) else 0.0
        coverage_debug.append(f"{kp}: total={len(vals):,}, valid_frac={valid_frac:.3f}")
        if valid_frac < cfg.alignment.min_valid_frac_per_kp:
            continue
        means.append(vals.mean(skipna=True).to_numpy(dtype=float))
        used.append(kp)

    if len(used) < cfg.alignment.min_kps_for_fit:
        detail = ""
        if _ALIGNMENT_DEBUG:
            coverage = "; ".join(coverage_debug)
            detail = f" | coverage: {coverage}"
        raise ConfigError(
            "not enough valid keypoints to fit alignment for trial "
            f"{df_trial['trial_id'].iloc[0]} (have {len(used)})." + detail
        )
    return np.vstack(means), used


def _procrustes_transform(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    allow_reflection: bool,
    rotation: bool,
    scaling: bool,
    translation: bool,
) -> Tuple[np.ndarray, float, np.ndarray]:
    # X, Y: (k, d)
    if X.shape != Y.shape:
        raise ConfigError("procrustes requires X and Y to have same shape.")

    muX = X.mean(axis=0) if translation else np.zeros(X.shape[1])
    muY = Y.mean(axis=0) if translation else np.zeros(Y.shape[1])
    Xc = X - muX
    Yc = Y - muY

    if rotation:
        H = Xc.T @ Yc
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if not allow_reflection and np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
    else:
        R = np.eye(X.shape[1])

    if scaling:
        denom = float((Xc**2).sum())
        if denom <= 0:
            s = 1.0
        else:
            s = float(np.trace((Xc @ R).T @ Yc) / denom)
    else:
        s = 1.0

    if translation:
        t = muY - s * (muX @ R)
    else:
        t = np.zeros(X.shape[1])

    return R, s, t


def _apply_transform(
    df_trial: pd.DataFrame,
    dims: List[str],
    R: np.ndarray,
    s: float,
    t: np.ndarray,
) -> pd.DataFrame:
    df_out = df_trial.copy()
    vals = df_out[dims].to_numpy(dtype=float)
    mask = np.isnan(vals).any(axis=1)
    vals_clean = vals.copy()
    vals_clean[mask] = 0.0
    transformed = s * (vals_clean @ R) + t
    transformed[mask] = np.nan
    for i, d in enumerate(dims):
        df_out[d] = transformed[:, i]
    return df_out


def _rotation_angle_2d(R: np.ndarray) -> float:
    if R.shape[0] < 2 or R.shape[1] < 2:
        return float("nan")
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _time_col(df: pd.DataFrame) -> str:
    return "time" if "time" in df.columns else "frame"


def _frame_pose(
    df_frame: pd.DataFrame, kps: List[str], dims: List[str]
) -> Tuple[np.ndarray, List[str]]:
    rows = []
    used = []
    for kp in kps:
        g = df_frame[df_frame["keypoint"] == kp]
        if g.empty:
            continue
        vals = g[dims]
        if vals.notna().all(axis=1).any():
            rows.append(vals.mean(skipna=True).to_numpy(dtype=float))
            used.append(kp)
    if not rows:
        return np.empty((0, len(dims))), []
    return np.vstack(rows), used


def align_procrustes(
    df: pd.DataFrame, cfg: PreprocessConfig
) -> Tuple[pd.DataFrame, List[dict], pd.DataFrame | None]:
    if not cfg.alignment.enabled:
        return df, [], None
    if cfg.alignment.method != "procrustes":
        raise ConfigError("alignment.method currently supports only 'procrustes'.")

    dims = ["x", "y"] + (["z"] if "z" in df.columns else [])
    keypoints = _valid_keypoints_for_alignment(df, cfg)
    rotation, scaling, translation = _resolve_alignment_flags(cfg)

    # Compute per-trial mean poses
    trial_means: Dict[str, np.ndarray] = {}
    trial_used: Dict[str, List[str]] = {}
    for trial_id, df_trial in df.groupby("trial_id", sort=False):
        mean_pose, used = _compute_mean_pose(df_trial, keypoints, dims, cfg)
        trial_means[trial_id] = mean_pose
        trial_used[trial_id] = used

    # Determine template
    if cfg.alignment.template_scope == "trial":
        template_by_trial = {tid: trial_means[tid] for tid in trial_means}
        kps_by_trial = {tid: trial_used[tid] for tid in trial_used}
    else:
        # global template using intersection of valid keypoints across trials
        kp_sets = [set(kps) for kps in trial_used.values()]
        if not kp_sets:
            raise ConfigError("no trials available for alignment.")
        kp_intersection = sorted(set.intersection(*kp_sets))
        if len(kp_intersection) < cfg.alignment.min_kps_for_fit:
            raise ConfigError("not enough shared keypoints for global template.")

        # Build global template as mean over trials (equal weight per trial)
        template_rows = []
        for tid, mean_pose in trial_means.items():
            idx = [trial_used[tid].index(kp) for kp in kp_intersection]
            template_rows.append(mean_pose[idx, :])
        template = np.mean(np.stack(template_rows, axis=0), axis=0)
        template_by_trial = {tid: template for tid in trial_means}
        kps_by_trial = {tid: kp_intersection for tid in trial_means}

    # Fit and apply transforms
    aligned_parts = []
    transforms: List[dict] = []
    transforms_df: pd.DataFrame | None = None

    if not cfg.alignment.framewise:
        for trial_id, df_trial in df.groupby("trial_id", sort=False):
            mean_pose = trial_means[trial_id]
            used = trial_used[trial_id]
            template = template_by_trial[trial_id]
            kps = kps_by_trial[trial_id]

            if kps != used:
                # Use subset that matches template
                idx_trial = [used.index(kp) for kp in kps]
                X = mean_pose[idx_trial, :]
                Y = template
            else:
                X = mean_pose
                Y = template

            R, s, t = _procrustes_transform(
                X,
                Y,
                allow_reflection=cfg.alignment.reflection,
                rotation=rotation,
                scaling=scaling,
                translation=translation,
            )

            df_aligned = _apply_transform(df_trial, dims, R, s, t)
            aligned_parts.append(df_aligned)

            # Build homogeneous transform matrix
            d = len(dims)
            T = np.eye(d + 1)
            T[:d, :d] = s * R
            T[:d, d] = t

            transforms.append(
                {
                    "trial_id": trial_id,
                    "dims": dims,
                    "keypoints_used": kps,
                    "rotation": rotation,
                    "scaling": scaling,
                    "translate": translation,
                    "reflection": cfg.alignment.reflection,
                    "transform": cfg.alignment.transform,
                    "template_scope": cfg.alignment.template_scope,
                    "framewise": False,
                    "scale": float(s),
                    "rotation_matrix": R.tolist(),
                    "translation": t.tolist(),
                    "transform_matrix": T.tolist(),
                }
            )
    else:
        transform_rows: List[dict] = []
        time_col = _time_col(df)
        for trial_id, df_trial in df.groupby("trial_id", sort=False):
            template = template_by_trial[trial_id]
            kps = kps_by_trial[trial_id]

            for tval, df_frame in df_trial.groupby(time_col, sort=False):
                Xf, used = _frame_pose(df_frame, kps, dims)
                if len(used) < cfg.alignment.min_kps_for_fit:
                    transform_rows.append(
                        {
                            "trial_id": trial_id,
                            time_col: tval,
                            "scale": float("nan"),
                            "rotation_angle": float("nan"),
                            **{f"translation_{d}": float("nan") for d in dims},
                            "keypoints_used": used,
                        }
                    )
                    df_frame = df_frame.copy()
                    for d in dims:
                        df_frame[d] = np.nan
                    aligned_parts.append(df_frame)
                    continue

                idx_template = [kps.index(kp) for kp in used]
                Y = template[idx_template, :]

                R, s, t = _procrustes_transform(
                    Xf,
                    Y,
                    allow_reflection=cfg.alignment.reflection,
                    rotation=rotation,
                    scaling=scaling,
                    translation=translation,
                )

                df_aligned = _apply_transform(df_frame, dims, R, s, t)
                aligned_parts.append(df_aligned)

                row = {
                    "trial_id": trial_id,
                    time_col: tval,
                    "scale": float(s),
                    "rotation_angle": _rotation_angle_2d(R),
                    **{f"translation_{d}": float(t[i]) for i, d in enumerate(dims)},
                    "keypoints_used": used,
                }
                transform_rows.append(row)

            transforms.append(
                {
                    "trial_id": trial_id,
                    "dims": dims,
                    "keypoints_used": kps,
                    "rotation": rotation,
                    "scaling": scaling,
                    "translate": translation,
                    "reflection": cfg.alignment.reflection,
                    "transform": cfg.alignment.transform,
                    "template_scope": cfg.alignment.template_scope,
                    "framewise": True,
                }
            )

        transforms_df = pd.DataFrame(transform_rows)

    df_out = pd.concat(aligned_parts, ignore_index=True)
    return df_out, transforms, transforms_df


def _center_on_keypoint(
    df: pd.DataFrame,
    center_keypoint: str,
    dims: List[str],
    group_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Center coordinates on the mean position of a specific keypoint within groups.
    
    Args:
        df: Pose dataframe with keypoint, x, y, etc.
        center_keypoint: Keypoint to center on (e.g., 'Nose')
        dims: Coordinate dimensions ['x', 'y'] or ['x', 'y', 'z']
        group_cols: Columns to group by (e.g., ['trial_id', 'window_id'])
        
    Returns:
        df_centered: Dataframe with centered coordinates
        offsets_df: Dataframe with offset values per group
    """
    # Get the center keypoint data
    center_df = df[df["keypoint"] == center_keypoint].copy()
    if center_df.empty:
        return df, pd.DataFrame()
    
    # Compute mean position of center keypoint per group
    offsets = center_df.groupby(group_cols, as_index=False)[dims].mean()
    offsets = offsets.rename(columns={d: f"{d}_offset" for d in dims})
    
    # Merge offsets back to full dataframe
    df_out = df.merge(offsets, on=group_cols, how="left")
    
    # Subtract offsets to center
    for d in dims:
        df_out[d] = df_out[d] - df_out[f"{d}_offset"]
        df_out = df_out.drop(columns=[f"{d}_offset"])
    
    return df_out, offsets


def _compute_window_mean_pose(
    df_window: pd.DataFrame,
    keypoints: List[str],
    dims: List[str],
    min_valid_frac: float,
    min_kps: int,
) -> Tuple[np.ndarray, List[str]]:
    """Compute mean pose for a window (average position across all frames)."""
    means = []
    used = []
    
    for kp in keypoints:
        kp_data = df_window[df_window["keypoint"] == kp]
        if kp_data.empty:
            continue
        vals = kp_data[dims]
        valid = vals.notna().all(axis=1)
        valid_frac = float(valid.mean()) if len(valid) else 0.0
        if valid_frac < min_valid_frac:
            continue
        means.append(vals.mean(skipna=True).to_numpy(dtype=float))
        used.append(kp)
    
    if len(used) < min_kps:
        return np.empty((0, len(dims))), []
    
    return np.vstack(means), used


def align_procrustes_windowed(
    df: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, List[dict], pd.DataFrame]:
    """
    Window-based Procrustes alignment.
    
    For each window:
    1. Center coordinates on mean position of center_keypoint (if specified)
    2. Compute mean pose of the window
    3. Fit Procrustes transform to align window mean pose to global template
    4. Apply the same transform to ALL frames in the window
    
    This preserves within-window movement dynamics while correcting for
    gross positional bias and inter-window differences.
    """
    if not cfg.alignment.enabled:
        return df, [], pd.DataFrame()
    if cfg.alignment.method != "procrustes":
        raise ConfigError("alignment.method currently supports only 'procrustes'.")

    dims = ["x", "y"] + (["z"] if "z" in df.columns else [])
    keypoints = _valid_keypoints_for_alignment(df, cfg)
    rotation, scaling, translation = _resolve_alignment_flags(cfg)
    time_col = _time_col(df)

    # Step 1: Center on keypoint if specified
    center_kp = cfg.alignment.center_keypoint
    if center_kp:
        if center_kp not in df["keypoint"].unique():
            raise ConfigError(
                f"alignment.center_keypoint '{center_kp}' not found in data."
            )
        # We need window_id in df to center per-window
        # Merge window info to df
        df = _assign_frames_to_windows(df, windows, time_col)
        df, offsets_df = _center_on_keypoint(
            df, center_kp, dims, ["trial_id", "window_id"]
        )
    else:
        df = _assign_frames_to_windows(df, windows, time_col)

    # Step 2: Compute global template (mean pose across all windows)
    window_means: Dict[Tuple[str, str], np.ndarray] = {}
    window_used: Dict[Tuple[str, str], List[str]] = {}
    
    for (trial_id, window_id), df_win in df.groupby(["trial_id", "window_id"], sort=False):
        mean_pose, used = _compute_window_mean_pose(
            df_win, keypoints, dims,
            cfg.alignment.min_valid_frac_per_kp,
            cfg.alignment.min_kps_for_fit,
        )
        if len(used) >= cfg.alignment.min_kps_for_fit:
            window_means[(trial_id, window_id)] = mean_pose
            window_used[(trial_id, window_id)] = used

    if not window_means:
        raise ConfigError("No windows have enough valid keypoints for alignment.")

    # Find intersection of keypoints across all windows
    kp_sets = [set(kps) for kps in window_used.values()]
    kp_intersection = sorted(set.intersection(*kp_sets))
    if len(kp_intersection) < cfg.alignment.min_kps_for_fit:
        raise ConfigError(
            f"Not enough shared keypoints across windows for alignment "
            f"(have {len(kp_intersection)}, need {cfg.alignment.min_kps_for_fit})."
        )

    # Build global template as mean over all windows
    template_rows = []
    for (trial_id, window_id), mean_pose in window_means.items():
        used = window_used[(trial_id, window_id)]
        idx = [used.index(kp) for kp in kp_intersection]
        template_rows.append(mean_pose[idx, :])
    template = np.mean(np.stack(template_rows, axis=0), axis=0)

    # Step 3 & 4: Fit and apply transform per window
    aligned_parts = []
    transform_rows: List[dict] = []

    for (trial_id, window_id), df_win in df.groupby(["trial_id", "window_id"], sort=False):
        key = (trial_id, window_id)
        
        if key not in window_means:
            # Not enough keypoints - set coordinates to NaN
            df_win = df_win.copy()
            for d in dims:
                df_win[d] = np.nan
            aligned_parts.append(df_win)
            transform_rows.append({
                "trial_id": trial_id,
                "window_id": window_id,
                "scale": float("nan"),
                "rotation_angle": float("nan"),
                **{f"translation_{d}": float("nan") for d in dims},
                "keypoints_used": [],
            })
            continue

        # Get window mean pose for shared keypoints
        mean_pose = window_means[key]
        used = window_used[key]
        idx = [used.index(kp) for kp in kp_intersection]
        X = mean_pose[idx, :]
        Y = template

        # Fit Procrustes
        R, s, t = _procrustes_transform(
            X, Y,
            allow_reflection=cfg.alignment.reflection,
            rotation=rotation,
            scaling=scaling,
            translation=translation,
        )

        # Apply transform to all frames in window
        df_aligned = _apply_transform(df_win, dims, R, s, t)
        aligned_parts.append(df_aligned)

        transform_rows.append({
            "trial_id": trial_id,
            "window_id": window_id,
            "scale": float(s),
            "rotation_angle": _rotation_angle_2d(R),
            **{f"translation_{d}": float(t[i]) for i, d in enumerate(dims)},
            "keypoints_used": kp_intersection,
        })

    transforms_df = pd.DataFrame(transform_rows)
    
    # Summary transforms (one per trial for compatibility)
    transforms = []
    for trial_id in df["trial_id"].unique():
        transforms.append({
            "trial_id": trial_id,
            "dims": dims,
            "keypoints_used": kp_intersection,
            "rotation": rotation,
            "scaling": scaling,
            "translate": translation,
            "reflection": cfg.alignment.reflection,
            "transform": cfg.alignment.transform,
            "template_scope": "global",
            "scope": "window",
            "center_keypoint": center_kp,
        })

    df_out = pd.concat(aligned_parts, ignore_index=True)
    
    # Remove temporary window_id column if it was added
    if "window_id" in df_out.columns and "window_id" not in df.columns:
        df_out = df_out.drop(columns=["window_id"])
    
    return df_out, transforms, transforms_df


def _assign_frames_to_windows(
    df: pd.DataFrame,
    windows: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    """Assign each frame to its corresponding window based on time/frame."""
    if "window_id" in df.columns:
        return df
    
    df = df.copy()
    df["window_id"] = None
    
    for _, win in windows.iterrows():
        trial_id = win["trial_id"]
        window_id = win["window_id"]
        start = win["start"]
        end = win["end"]
        
        mask = (
            (df["trial_id"] == trial_id) &
            (df[time_col] >= start) &
            (df[time_col] < end)
        )
        df.loc[mask, "window_id"] = window_id
    
    # Drop frames not in any window
    df = df[df["window_id"].notna()]
    
    return df
