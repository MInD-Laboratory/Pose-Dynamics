from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pose_dynamics.features.geometry import pairwise_distance_features
from pose_dynamics.features.kinematics import kinematics_features
from pose_dynamics.features.schema import ConfigError, FeaturesConfig


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

    if "dropped" in windows.columns:
        windows = windows[~windows["dropped"]].copy()

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
    for _, w in windows.iterrows():
        trial_id = w["trial_id"]
        s = float(w["start"])
        e = float(w["end"])
        units = w.get("units", "seconds")

        df_trial = df[df["trial_id"] == trial_id]
        if units == "seconds":
            mask = (df_trial[time_col] >= s) & (df_trial[time_col] < e)
        else:
            mask = (df_trial[time_col] >= s) & (df_trial[time_col] < e)

        df_win = df_trial.loc[mask]
        df_win = df_win[df_win["keypoint"].isin(kps)]

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
