from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pose_dynamics.pca.schema import ConfigError, PCAConfig


@dataclass(frozen=True)
class PCAOutputs:
    scores_path: Path
    components_path: Path
    qc_path: Path
    provenance_path: Path


def _zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    Xz = (X - mean) / std
    return Xz, mean, std


def _pca(
    X: np.ndarray, n_components: Optional[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns scores, components, explained_variance_ratio
    Xc = X - np.nanmean(X, axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if n_components is None:
        n_components = Vt.shape[0]
    Vt = Vt[:n_components]
    S = S[:n_components]
    scores = Xc @ Vt.T
    # explained variance
    var = (S**2) / (Xc.shape[0] - 1) if Xc.shape[0] > 1 else S**2
    total_var = (
        np.sum((np.linalg.svd(Xc, full_matrices=False)[1] ** 2) / (Xc.shape[0] - 1))
        if Xc.shape[0] > 1
        else np.sum(S**2)
    )
    evr = var / total_var if total_var > 0 else var
    return scores, Vt, evr


def _select_components_by_variance(evr: np.ndarray, threshold: float) -> int:
    if evr.size == 0:
        return 0
    cum = np.cumsum(evr)
    return int(np.searchsorted(cum, threshold, side="left") + 1)


def _pose_summary_matrix(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    time_col = "time" if "time" in df.columns else "frame"
    dims = [c for c in ["x", "y", "z"] if c in df.columns]

    rows = []
    for _, w in windows.iterrows():
        trial_id = w["trial_id"]
        s = float(w["start"])
        e = float(w["end"])
        df_trial = df[df["trial_id"] == trial_id]
        mask = (df_trial[time_col] >= s) & (df_trial[time_col] < e)
        df_win = df_trial.loc[mask]
        if df_win.empty:
            continue
        means = df_win.groupby("keypoint")[dims].mean().reset_index()
        for _, row in means.iterrows():
            kp = row["keypoint"]
            for d in dims:
                rows.append(
                    {
                        "trial_id": trial_id,
                        "window_id": w["window_id"],
                        f"pose_mean_{d}_{kp}": row[d],
                    }
                )

    if not rows:
        return pd.DataFrame()

    df_rows = pd.DataFrame(rows)
    df_wide = df_rows.groupby(["trial_id", "window_id"]).first().reset_index()
    return df_wide


def _features_matrix(features: pd.DataFrame) -> pd.DataFrame:
    # features: trial_id, window_id, keypoint, feature, value
    df = features.copy()
    df["col"] = df["keypoint"].astype(str) + "__" + df["feature"].astype(str)
    wide = df.pivot_table(
        index=["trial_id", "window_id"], columns="col", values="value"
    )
    wide = wide.reset_index()
    return wide


def run_pca(
    pose_clean_path: str | Path,
    windows_path: str | Path,
    features_path: str | Path,
    config: PCAConfig | str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> PCAOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(config, PCAConfig):
        cfg = config
    else:
        cfg = PCAConfig.from_yaml(str(config))

    pose_clean_path = Path(pose_clean_path)
    windows_path = Path(windows_path)
    features_path = Path(features_path)

    df_pose = pd.read_parquet(pose_clean_path)
    windows = pd.read_parquet(windows_path)

    if windows.empty:
        raise ConfigError("windows.parquet is empty.")

    if "dropped" in windows.columns:
        windows = windows[~windows["dropped"]].copy()

    mats = []
    if cfg.include_pose_summary:
        mats.append(_pose_summary_matrix(df_pose, windows))

    if cfg.include_features:
        if not features_path.exists():
            raise ConfigError("features.parquet not found but include_features=true.")
        df_feat = pd.read_parquet(features_path)
        mats.append(_features_matrix(df_feat))

    if not mats:
        raise ConfigError("No inputs selected for PCA.")

    # Merge on trial_id/window_id
    base = mats[0]
    for m in mats[1:]:
        base = base.merge(m, on=["trial_id", "window_id"], how="outer")

    if base.empty:
        raise ConfigError("PCA matrix is empty.")

    id_cols = ["trial_id", "window_id"]
    X = base.drop(columns=id_cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)

    n_components = cfg.n_components

    scores_rows = []
    components_payload = []

    def _run(scope_df: pd.DataFrame, scope_label: str) -> None:
        nonlocal scores_rows, components_payload
        X_scope = scope_df.drop(columns=id_cols).to_numpy(dtype=float)
        if cfg.standardize:
            X_scope, mean, std = _zscore(X_scope)
        else:
            mean = np.nanmean(X_scope, axis=0)
            std = np.nanstd(X_scope, axis=0)
        scores, comps, evr = _pca(X_scope, n_components)
        if cfg.variance_threshold is not None:
            n_keep = _select_components_by_variance(evr, cfg.variance_threshold)
            comps = comps[:n_keep]
            evr = evr[:n_keep]
            scores = scores[:, :n_keep]
        for i, row in scope_df[id_cols].iterrows():
            entry = {"trial_id": row["trial_id"], "window_id": row["window_id"]}
            for j in range(scores.shape[1]):
                entry[f"pc{j + 1}"] = float(scores[i, j])
            scores_rows.append(entry)
        components_payload.append(
            {
                "scope": scope_label,
                "columns": list(scope_df.drop(columns=id_cols).columns),
                "components": comps.tolist(),
                "explained_variance_ratio": evr.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        )

    if cfg.scope == "global":
        _run(base, "global")
    else:
        for trial_id, g in base.groupby("trial_id", sort=False):
            _run(g, f"trial:{trial_id}")

    scores_df = pd.DataFrame(scores_rows)

    scores_path = out_dir / "pca_scores.parquet"
    components_path = out_dir / "pca_components.json"
    qc_path = out_dir / "qc_pca.json"
    provenance_path = out_dir / "provenance_pca.json"
    scree_path = out_dir / "pca_scree.png"

    if not overwrite:
        for p in [scores_path, components_path, qc_path, provenance_path, scree_path]:
            if p.exists():
                raise FileExistsError(f"Output already exists: {p}")

    scores_df.to_parquet(scores_path, index=False)
    components_path.write_text(
        json.dumps(components_payload, indent=2), encoding="utf-8"
    )

    qc_payload = {
        "n_rows": int(scores_df.shape[0]),
        "n_windows": int(base.shape[0]),
        "n_features": int(base.drop(columns=id_cols).shape[1]),
        "scope": cfg.scope,
        "variance_threshold": cfg.variance_threshold,
    }
    qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

    provenance_payload = {
        "pose_clean_path": str(pose_clean_path.resolve()),
        "windows_path": str(windows_path.resolve()),
        "features_path": str(features_path.resolve()),
        "config": cfg.to_dict(),
    }
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2), encoding="utf-8"
    )

    # Scree plot from global components if available
    global_entry = next(
        (c for c in components_payload if c.get("scope") == "global"),
        components_payload[0] if components_payload else None,
    )
    if global_entry is not None:
        evr = np.array(global_entry.get("explained_variance_ratio", []), dtype=float)
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, evr.size + 1), evr, marker="o")
        plt.title("PCA Scree Plot")
        plt.xlabel("Component")
        plt.ylabel("Explained Variance Ratio")
        plt.tight_layout()
        plt.savefig(scree_path)
        plt.close()

    return PCAOutputs(
        scores_path=scores_path,
        components_path=components_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
    )
