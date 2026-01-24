from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pose_dynamics.rqa.schema import ConfigError, RQAParamsConfig


@dataclass(frozen=True)
class RQAParamOutputs:
    params_path: Path
    qc_path: Path
    provenance_path: Path


def _time_col(df: pd.DataFrame) -> str:
    return "time" if "time" in df.columns else "frame"


def _series_for_keypoint(
    df_win: pd.DataFrame, keypoint: str, signal: str
) -> np.ndarray:
    df_kp = df_win[df_win["keypoint"] == keypoint].copy()
    if df_kp.empty:
        return np.array([])
    dims = [c for c in ["x", "y", "z"] if c in df_kp.columns]
    if not dims:
        return np.array([])
    vals = df_kp[dims].to_numpy(dtype=float)
    if signal == "magnitude":
        s = np.linalg.norm(vals, axis=1)
    else:
        s = vals[:, 0]
    return s


def _ami(x: np.ndarray, max_lag: int, bins: int) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size < max_lag + 2:
        return np.array([])
    ami_vals = []
    for lag in range(1, max_lag + 1):
        x1 = x[:-lag]
        x2 = x[lag:]
        hist2d, _, _ = np.histogram2d(x1, x2, bins=bins)
        pxy = hist2d / np.sum(hist2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        nz = pxy > 0
        mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz]))
        ami_vals.append(mi)
    return np.array(ami_vals, dtype=float)


def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = x.size - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    return np.column_stack([x[i : i + n] for i in range(0, m * tau, tau)])


def _fnn(x: np.ndarray, max_dim: int, tau: int, rtol: float, atol: float) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size < (max_dim + 1) * tau + 2:
        return np.array([])

    fnn_rates = []
    for m in range(1, max_dim + 1):
        Xm = _embed(x, m, tau)
        Xm1 = _embed(x, m + 1, tau)
        if Xm.size == 0 or Xm1.size == 0:
            fnn_rates.append(np.nan)
            continue
        n2 = Xm1.shape[0]
        Xm = Xm[:n2]
        Xm1 = Xm1[:n2]
        # nearest neighbor in m-dim
        dists = np.linalg.norm(Xm[:, None, :] - Xm[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        nn_idx = np.argmin(dists, axis=1)
        Rm = dists[np.arange(dists.shape[0]), nn_idx]
        # distance in m+1 dim
        Rm1 = np.linalg.norm(Xm1 - Xm1[nn_idx], axis=1)
        ratio = np.full_like(Rm, np.inf, dtype=float)
        valid = Rm > 0
        ratio[valid] = (Rm1[valid] - Rm[valid]) / Rm[valid]
        fnn = (ratio > rtol) | (Rm1 > atol)
        fnn_rates.append(float(np.mean(fnn)))
    return np.array(fnn_rates, dtype=float)


def _epsilon_sensitivity(
    x: np.ndarray, m: int, tau: int, percentiles: Iterable[int]
) -> dict:
    x = x[np.isfinite(x)]
    emb = _embed(x, m, tau)
    if emb.size == 0:
        return {"percentiles": [], "rr": []}
    dists = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    iu = np.triu_indices_from(dists, k=1)
    d = dists[iu]
    out_perc = []
    out_rr = []
    for p in percentiles:
        eps = np.percentile(d, p)
        rr = float(np.mean(d <= eps))
        out_perc.append(p)
        out_rr.append(rr)
    return {"percentiles": out_perc, "rr": out_rr}


def run_rqa_params(
    pose_clean_path: str | Path,
    windows_path: str | Path,
    config: RQAParamsConfig | str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> RQAParamOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(config, RQAParamsConfig):
        cfg = config
    else:
        cfg = RQAParamsConfig.from_yaml(str(config))

    pose_clean_path = Path(pose_clean_path)
    windows_path = Path(windows_path)

    df = pd.read_parquet(pose_clean_path)
    windows = pd.read_parquet(windows_path)

    if windows.empty:
        raise ConfigError("windows.parquet is empty.")

    if "dropped" in windows.columns:
        windows = windows[~windows["dropped"]].copy()

    all_kps = sorted(df["keypoint"].dropna().unique().tolist())
    if cfg.keypoints == "all":
        kps = all_kps
    else:
        missing = sorted(set(cfg.keypoints) - set(all_kps))
        if missing:
            raise ConfigError(f"rqa_params.keypoints not found: {missing}")
        kps = list(cfg.keypoints)

    if cfg.n_keypoints is not None and len(kps) > cfg.n_keypoints:
        kps = kps[: cfg.n_keypoints]

    tcol = _time_col(df)
    win_subset = windows.head(cfg.n_windows)

    results = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for kp in kps:
        series_parts = []
        for _, w in win_subset.iterrows():
            trial_id = w["trial_id"]
            s = float(w["start"])
            e = float(w["end"])
            df_trial = df[df["trial_id"] == trial_id]
            mask = (df_trial[tcol] >= s) & (df_trial[tcol] < e)
            df_win = df_trial.loc[mask]
            s_kp = _series_for_keypoint(df_win, kp, cfg.signal)
            if s_kp.size:
                series_parts.append(s_kp)
        if not series_parts:
            continue
        x = np.concatenate(series_parts)

        ami = _ami(x, cfg.ami.max_lag, cfg.ami.bins)
        fnn = _fnn(x, cfg.fnn.max_dim, cfg.fnn.tau, cfg.fnn.rtol, cfg.fnn.atol)
        eps = _epsilon_sensitivity(
            x, cfg.epsilon.m, cfg.epsilon.tau, cfg.epsilon.percentiles
        )

        results.append(
            {
                "keypoint": kp,
                "ami": ami.tolist(),
                "ami_lags": list(range(1, len(ami) + 1)),
                "fnn": fnn.tolist(),
                "fnn_dims": list(range(1, len(fnn) + 1)),
                "epsilon": eps,
            }
        )

        if ami.size:
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(ami) + 1), ami, marker="o")
            plt.title(f"AMI: {kp}")
            plt.xlabel("Lag")
            plt.ylabel("AMI")
            plt.tight_layout()
            plt.savefig(plots_dir / f"ami_{kp}.png", dpi=200)
            plt.close()

        if fnn.size:
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(fnn) + 1), fnn, marker="o")
            plt.title(f"FNN: {kp}")
            plt.xlabel("Embedding dimension")
            plt.ylabel("FNN rate")
            plt.tight_layout()
            plt.savefig(plots_dir / f"fnn_{kp}.png", dpi=200)
            plt.close()

        if eps["percentiles"]:
            plt.figure(figsize=(6, 4))
            plt.plot(eps["percentiles"], eps["rr"], marker="o")
            plt.title(f"Epsilon sensitivity: {kp}")
            plt.xlabel("Percentile of distances")
            plt.ylabel("Recurrence rate")
            plt.tight_layout()
            plt.savefig(plots_dir / f"epsilon_{kp}.png", dpi=200)
            plt.close()

    params_path = out_dir / "rqa_params.json"
    qc_path = out_dir / "qc_rqa_params.json"
    provenance_path = out_dir / "provenance_rqa_params.json"

    if not overwrite:
        for p in [params_path, qc_path, provenance_path]:
            if p.exists():
                raise FileExistsError(f"Output already exists: {p}")

    params_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    qc_payload = {
        "n_keypoints": len(kps),
        "n_windows": int(win_subset.shape[0]),
        "plots_dir": str(plots_dir),
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

    return RQAParamOutputs(
        params_path=params_path, qc_path=qc_path, provenance_path=provenance_path
    )
