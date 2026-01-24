from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pose_dynamics.rqa.rqa_config import ConfigError, RQAConfig
from pose_dynamics.rqa.utils.plot_utils import plot_drp_results, plot_rqa_results


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


def _rqa_stats(mat: np.ndarray, l_min: int, v_min: int) -> dict:
    N = mat.shape[0]
    if N == 0:
        return {}
    # exclude main diagonal for RQA
    mask = np.ones_like(mat, dtype=bool)
    np.fill_diagonal(mask, False)
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


def _series_matrix(
    df_win: pd.DataFrame, keypoints: list[str], signal: str
) -> np.ndarray:
    dims = [c for c in ["x", "y", "z"] if c in df_win.columns]
    if signal == "magnitude":
        rows = []
        for kp in keypoints:
            vals = df_win[df_win["keypoint"] == kp][dims].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(np.linalg.norm(vals, axis=1))
        if not rows:
            return np.empty((0, 1))
        return np.stack(rows, axis=1)
    # coords
    rows = []
    for kp in keypoints:
        vals = df_win[df_win["keypoint"] == kp][dims].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(vals)
    if not rows:
        return np.empty((0, 1))
    return np.concatenate(rows, axis=1)


def run_rqa(
    pose_clean_path: str | Path,
    windows_path: str | Path,
    config: RQAConfig | str | Path,
    out_dir: str | Path,
    *,
    pose_y_path: Optional[str | Path] = None,
    overwrite: bool = False,
) -> RQAOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(config, RQAConfig):
        cfg = config
    else:
        cfg = RQAConfig.from_yaml(str(config))

    pose_clean_path = Path(pose_clean_path)
    windows_path = Path(windows_path)
    df = pd.read_parquet(pose_clean_path)
    windows = pd.read_parquet(windows_path)

    if windows.empty:
        raise ConfigError("windows.parquet is empty.")

    if "dropped" in windows.columns:
        windows = windows[~windows["dropped"]].copy()

    df_y = None
    if cfg.analysis == "crqa":
        if pose_y_path is None:
            raise ConfigError("crqa requires pose_y_path.")
        df_y = pd.read_parquet(Path(pose_y_path))

    all_kps = sorted(df["keypoint"].dropna().unique().tolist())
    if cfg.keypoints == "all":
        kps = all_kps
    else:
        missing = sorted(set(cfg.keypoints) - set(all_kps))
        if missing:
            raise ConfigError(f"rqa.keypoints not found: {missing}")
        kps = list(cfg.keypoints)

    tcol = _time_col(df)
    stats_rows = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for i, w in windows.iterrows():
        trial_id = w["trial_id"]
        s = float(w["start"])
        e = float(w["end"])
        df_trial = df[df["trial_id"] == trial_id]
        mask = (df_trial[tcol] >= s) & (df_trial[tcol] < e)
        df_win = df_trial.loc[mask]

        if df_win.empty:
            continue

        X = _series_matrix(df_win, kps, cfg.signal)
        if X.size == 0:
            continue

        if cfg.analysis == "crqa":
            df_trial_y = df_y[df_y["trial_id"] == trial_id]
            mask_y = (df_trial_y[tcol] >= s) & (df_trial_y[tcol] < e)
            df_win_y = df_trial_y.loc[mask_y]
            Y = _series_matrix(df_win_y, kps, cfg.signal)
            if Y.size == 0:
                continue
            X_emb = _embed(X, cfg.m, cfg.tau)
            Y_emb = _embed(Y, cfg.m, cfg.tau)
            if X_emb.size == 0 or Y_emb.size == 0:
                continue
            dist = np.linalg.norm(X_emb[:, None, :] - Y_emb[None, :, :], axis=2)
            eps = _epsilon(dist, cfg)
            mat = dist <= eps
        else:
            X_emb = _embed(X, cfg.m, cfg.tau)
            if X_emb.size == 0:
                continue
            dist = np.linalg.norm(X_emb[:, None, :] - X_emb[None, :, :], axis=2)
            eps = _epsilon(dist, cfg)
            mat = dist <= eps

        stats = _rqa_stats(mat, cfg.l_min, cfg.v_min)
        stats_rows.append(
            {
                "trial_id": trial_id,
                "window_id": w["window_id"],
                "epsilon": float(eps),
                **stats,
            }
        )

        if cfg.plots.enabled and i < cfg.plots.max_plots:
            plot_rqa_results(
                dataX=X_emb[:, 0],
                dataY=Y_emb[:, 0] if cfg.analysis == "crqa" else None,
                td=mat.astype(int),
                plot_mode="rp",
                save_path=str(plots_dir / f"rp_{trial_id}_{w['window_id']}.png"),
                stats=stats,
            )
            lags, drp = _drp(mat)
            plot_drp_results(
                lags,
                drp,
                save_path=str(plots_dir / f"drp_{trial_id}_{w['window_id']}.png"),
            )

    stats_df = pd.DataFrame(stats_rows)

    stats_path = out_dir / "rqa_stats.parquet"
    qc_path = out_dir / "qc_rqa.json"
    provenance_path = out_dir / "provenance_rqa.json"

    if not overwrite:
        for p in [stats_path, qc_path, provenance_path]:
            if p.exists():
                raise FileExistsError(f"Output already exists: {p}")

    stats_df.to_parquet(stats_path, index=False)

    qc_payload = {
        "n_rows": int(stats_df.shape[0]),
        "n_windows": int(windows.shape[0]),
        "analysis": cfg.analysis,
    }
    qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

    provenance_payload = {
        "pose_clean_path": str(pose_clean_path.resolve()),
        "windows_path": str(windows_path.resolve()),
        "pose_y_path": str(Path(pose_y_path).resolve()) if pose_y_path else None,
        "config": cfg.to_dict(),
    }
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2), encoding="utf-8"
    )

    return RQAOutputs(
        stats_path=stats_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
        plots_dir=plots_dir,
    )
