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
