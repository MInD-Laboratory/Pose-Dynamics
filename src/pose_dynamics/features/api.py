from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from pose_dynamics.features.facial import facial_feature_series
from pose_dynamics.features.geometry import pairwise_distance_features
from pose_dynamics.features.head_motion import head_motion_series
from pose_dynamics.features.kinematics import kinematics_features
from pose_dynamics.features.roi import roi_feature_series
from pose_dynamics.features.schema import ConfigError, FeaturesConfig
from pose_dynamics.features.stats import derivative_series, summary_stats
from pose_dynamics.progress import stage_progress_with_total


def _estimate_dt(values: pd.Series) -> float:
    uniq = pd.Series(values.dropna().unique()).sort_values()
    if uniq.empty or uniq.shape[0] <= 1:
        return float("nan")
    diffs = uniq.diff().dropna().to_numpy(dtype=float)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float("nan")
    return float(np.median(diffs))


def _append_feature_stats(
    rows: list[dict],
    *,
    trial_id: str,
    window_id: str | int,
    keypoint: str,
    base_name: str,
    series: np.ndarray,
    stats: list[str],
    derivatives: list[str],
    dt: float,
) -> None:
    if series.size == 0:
        return

    base_stats = summary_stats(series, stats)
    for stat_name, val in base_stats.items():
        rows.append(
            {
                "trial_id": trial_id,
                "window_id": window_id,
                "keypoint": keypoint,
                "feature": f"{base_name}_{stat_name}",
                "value": val,
            }
        )

    if not derivatives or not (isfinite(dt) and dt > 0):
        return

    for deriv in derivatives:
        if deriv == "velocity":
            derived = derivative_series(series, dt, order=1)
            suffix = "vel"
        elif deriv == "acceleration":
            derived = derivative_series(series, dt, order=2)
            suffix = "acc"
        else:
            continue
        if derived.size == 0:
            continue
        d_stats = summary_stats(derived, stats)
        for stat_name, val in d_stats.items():
            rows.append(
                {
                    "trial_id": trial_id,
                    "window_id": window_id,
                    "keypoint": keypoint,
                    "feature": f"{base_name}_{suffix}_{stat_name}",
                    "value": val,
                }
            )


@dataclass(frozen=True)
class FeatureOutputs:
    features_path: Path
    qc_path: Path
    provenance_path: Path


def run_feature_extract(
    pose_clean_path: str | Path,
    windows_path: str | Path,
    config: FeaturesConfig | str | Path,
    out_dir: str | Path,
    *,
    alignment_transforms_path: str | Path | None = None,
    overwrite: bool = False,
) -> FeatureOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(config, FeaturesConfig):
        cfg = config
    else:
        cfg = FeaturesConfig.from_yaml(str(config))

    pose_clean_path = Path(pose_clean_path)
    windows_path = Path(windows_path)

    df = pd.read_parquet(pose_clean_path)
    windows = pd.read_parquet(windows_path)
    transforms_df = None
    if alignment_transforms_path is not None:
        transforms_df = pd.read_parquet(Path(alignment_transforms_path))

    if df.empty:
        raise ConfigError("pose_clean.parquet is empty.")
    if windows.empty:
        features_rows: list[dict] = []
        features_df = pd.DataFrame(features_rows)

        features_path = out_dir / "features.parquet"
        qc_path = out_dir / "qc_features.json"
        provenance_path = out_dir / "provenance_features.json"

        if not overwrite:
            for p in [features_path, qc_path, provenance_path]:
                if p.exists():
                    raise FileExistsError(f"Output already exists: {p}")

        features_df.to_parquet(features_path, index=False)

        qc_payload = {
            "n_rows": 0,
            "n_windows": 0,
            "n_keypoints": int(df["keypoint"].nunique())
            if "keypoint" in df.columns
            else 0,
            "warning": "windows.parquet is empty",
        }
        qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

        provenance_payload = {
            "pose_clean_path": str(pose_clean_path.resolve()),
            "windows_path": str(windows_path.resolve()),
            "config": cfg.to_dict(),
        }
        provenance_path.write_text(
            json.dumps(provenance_payload, indent=2), encoding="utf-8"
        )

        return FeatureOutputs(
            features_path=features_path,
            qc_path=qc_path,
            provenance_path=provenance_path,
        )

    # Legacy pipeline kept windows even if QC flagged them; keep all here
    # to mirror that behavior for this dataset.

    all_kps = sorted(df["keypoint"].dropna().unique().tolist())
    if cfg.keypoints == "all":
        kps = all_kps
    else:
        missing = sorted(set(cfg.keypoints) - set(all_kps))
        if missing:
            raise ConfigError(
                "features.keypoints must be a subset of preprocessed keypoints; missing: "
                f"{missing}"
            )
        kps = list(cfg.keypoints)

    dims = [c for c in ["x", "y", "z"] if c in df.columns]
    time_col = "time" if "time" in df.columns else "frame"

    features_rows = []
    total_windows = len(windows)

    # Group by trial_id to avoid repeatedly slicing the full dataframe for each window.
    windows_by_trial = windows.groupby("trial_id", sort=False)

    with stage_progress_with_total(
        "Extracting features", total=total_windows
    ) as update_progress:
        processed = 0
        for trial_id, win_df in windows_by_trial:
            df_trial = df[df["trial_id"] == trial_id]
            if df_trial.empty:
                processed += len(win_df)
                update_progress(f"{trial_id} | no data", advance=len(win_df))
                continue

            # Pre-filter by keypoints once per trial
            df_trial_kps = df_trial[df_trial["keypoint"].isin(kps)]

            # If head motion is enabled, grab trial transforms once.
            df_trans_trial = None
            if cfg.head_motion.enabled:
                if transforms_df is None or transforms_df.empty:
                    raise ConfigError(
                        "head_motion.enabled requires alignment_transforms_path with framewise transforms."
                    )
                df_trans_trial = transforms_df[transforms_df["trial_id"] == trial_id]

            for _, w in win_df.iterrows():
                s = float(w["start"])
                e = float(w["end"])
                units = w.get("units", "seconds")

                if units == "seconds":
                    mask = (df_trial_kps[time_col] >= s) & (df_trial_kps[time_col] < e)
                else:
                    mask = (df_trial_kps[time_col] >= s) & (df_trial_kps[time_col] < e)

                df_win = df_trial_kps.loc[mask]
                dt_pose = (
                    _estimate_dt(df_win[time_col]) if not df_win.empty else float("nan")
                )

                if cfg.kinematics.enabled:
                    for kp in kps:
                        df_kp = df_win[df_win["keypoint"] == kp]
                        feats = kinematics_features(
                            df_kp, time_col, dims, cfg.kinematics.metrics
                        )
                        for name, val in feats.items():
                            features_rows.append(
                                {
                                    "trial_id": trial_id,
                                    "window_id": w["window_id"],
                                    "keypoint": kp,
                                    "feature": name,
                                    "value": val,
                                }
                            )

                if cfg.geometry.enabled and cfg.geometry.pairwise_distances:
                    feats = pairwise_distance_features(df_win, time_col, dims, kps)
                    for name, val in feats.items():
                        features_rows.append(
                            {
                                "trial_id": trial_id,
                                "window_id": w["window_id"],
                                "keypoint": "pairwise",
                                "feature": name,
                                "value": val,
                            }
                        )

                if cfg.facial.enabled:
                    series_map = facial_feature_series(df_win, time_col, cfg.facial)
                    for base, series in series_map.items():
                        _append_feature_stats(
                            features_rows,
                            trial_id=trial_id,
                            window_id=w["window_id"],
                            keypoint="composite",
                            base_name=base,
                            series=series,
                            stats=cfg.facial.stats,
                            derivatives=cfg.facial.derivatives,
                            dt=dt_pose,
                        )

                if (
                    cfg.head_motion.enabled
                    and df_trans_trial is not None
                    and not df_trans_trial.empty
                ):
                    if time_col not in df_trans_trial.columns:
                        raise ConfigError(
                            f"alignment transforms missing '{time_col}' column."
                        )
                    if units == "seconds":
                        mask_t = (df_trans_trial[time_col] >= s) & (
                            df_trans_trial[time_col] < e
                        )
                    else:
                        mask_t = (df_trans_trial[time_col] >= s) & (
                            df_trans_trial[time_col] < e
                        )
                    df_trans_win = df_trans_trial.loc[mask_t]
                    if not df_trans_win.empty:
                        series_map = head_motion_series(df_trans_win, time_col)
                        dt_trans = _estimate_dt(df_trans_win[time_col])
                        for base, series in series_map.items():
                            _append_feature_stats(
                                features_rows,
                                trial_id=trial_id,
                                window_id=w["window_id"],
                                keypoint="head_motion",
                                base_name=base,
                                series=series,
                                stats=cfg.head_motion.stats,
                                derivatives=cfg.head_motion.derivatives,
                                dt=dt_trans,
                            )

                processed += 1
                label = f"{trial_id} | window {w['window_id']} ({processed}/{total_windows})"
                update_progress(label, advance=1)

                if cfg.roi.enabled:
                    for region in cfg.roi.regions:
                        series_map = roi_feature_series(
                            df_win,
                            time_col,
                            region.name,
                            region.keypoints,
                            cfg.roi.derivatives,
                            dt_pose,
                        )
                        for base, series in series_map.items():
                            _append_feature_stats(
                                features_rows,
                                trial_id=trial_id,
                                window_id=w["window_id"],
                                keypoint=region.name,  # Use ROI name as keypoint identifier
                                base_name=base,
                                series=series,
                                stats=cfg.roi.stats,
                                derivatives=[],  # Already computed in roi_feature_series
                                dt=dt_pose,
                            )

    features_df = pd.DataFrame(features_rows)

    features_path = out_dir / "features.parquet"
    features_csv_path = out_dir / "features.csv"
    qc_path = out_dir / "qc_features.json"
    provenance_path = out_dir / "provenance_features.json"

    if not overwrite:
        for p in [features_path, features_csv_path, qc_path, provenance_path]:
            if p.exists():
                raise FileExistsError(f"Output already exists: {p}")

    features_df.to_parquet(features_path, index=False)
    features_df.to_csv(features_csv_path, index=False)

    qc_payload = {
        "n_rows": int(features_df.shape[0]),
        "n_windows": int(windows.shape[0]),
        "n_keypoints": int(len(kps)),
    }
    qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

    provenance_payload = {
        "pose_clean_path": str(pose_clean_path.resolve()),
        "windows_path": str(windows_path.resolve()),
        "config": cfg.to_dict(),
    }
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2), encoding="utf-8"
    )

    return FeatureOutputs(
        features_path=features_path, qc_path=qc_path, provenance_path=provenance_path
    )
