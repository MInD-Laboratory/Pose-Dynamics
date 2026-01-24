from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "examples" / "case_study_4_synthetic" / "data"
CFG_DIR = ROOT / "examples" / "case_study_4_synthetic" / "configs"

INGEST_OUT = ROOT / "artifacts" / "ingest" / "run_synth_001"
PRE_OUT = ROOT / "artifacts" / "preprocess" / "run_synth_001"
FEAT_OUT = ROOT / "artifacts" / "features" / "run_synth_001"
PCA_OUT = ROOT / "artifacts" / "pca" / "run_synth_001"
RQA_PARAM_OUT = ROOT / "artifacts" / "rqa_params" / "run_synth_001"
RQA_OUT = ROOT / "artifacts" / "rqa" / "run_synth_001"


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def _pd_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "pose_dynamics.cli.main", *args]


def generate_csvs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_frames = 300
    t = np.arange(n_frames) / 30.0

    for trial in range(2):
        noise = rng.normal(scale=0.02, size=(n_frames, 2))
        x1 = np.sin(2 * np.pi * 0.5 * t) + noise[:, 0]
        y1 = np.cos(2 * np.pi * 0.5 * t) + noise[:, 1]
        x2 = np.sin(2 * np.pi * 0.75 * t + 0.5) + noise[:, 0]
        y2 = np.cos(2 * np.pi * 0.75 * t + 0.5) + noise[:, 1]

        df = pd.DataFrame(
            {
                "time": t,
                "x_kp1": x1,
                "y_kp1": y1,
                "x_kp2": x2,
                "y_kp2": y2,
            }
        )
        df.to_csv(DATA_DIR / f"trial_{trial + 1:03d}.csv", index=False)


def main() -> None:
    generate_csvs()

    _run(
        _pd_cmd(
            "ingest-csv",
            "--in",
            str(DATA_DIR),
            "--out",
            str(INGEST_OUT),
        )
    )

    _run(
        _pd_cmd(
            "preprocess",
            "--in",
            str(INGEST_OUT / "pose.parquet"),
            "--recording",
            str(INGEST_OUT / "recording.json"),
            "--config",
            str(CFG_DIR / "preprocess.yaml"),
            "--out",
            str(PRE_OUT),
            "--overwrite",
        )
    )

    _run(
        _pd_cmd(
            "feature-extract",
            "--pose",
            str(PRE_OUT / "pose_clean.parquet"),
            "--windows",
            str(PRE_OUT / "windows.parquet"),
            "--config",
            str(CFG_DIR / "features.yaml"),
            "--out",
            str(FEAT_OUT),
            "--overwrite",
        )
    )

    _run(
        _pd_cmd(
            "pca",
            "--pose",
            str(PRE_OUT / "pose_clean.parquet"),
            "--windows",
            str(PRE_OUT / "windows.parquet"),
            "--features",
            str(FEAT_OUT / "features.parquet"),
            "--config",
            str(CFG_DIR / "pca.yaml"),
            "--out",
            str(PCA_OUT),
            "--overwrite",
        )
    )

    _run(
        _pd_cmd(
            "rqa-params",
            "--pose",
            str(PRE_OUT / "pose_clean.parquet"),
            "--windows",
            str(PRE_OUT / "windows.parquet"),
            "--config",
            str(CFG_DIR / "rqa_params.yaml"),
            "--out",
            str(RQA_PARAM_OUT),
            "--overwrite",
        )
    )

    _run(
        _pd_cmd(
            "rqa",
            "--pose",
            str(PRE_OUT / "pose_clean.parquet"),
            "--windows",
            str(PRE_OUT / "windows.parquet"),
            "--config",
            str(CFG_DIR / "rqa.yaml"),
            "--out",
            str(RQA_OUT),
            "--overwrite",
        )
    )


if __name__ == "__main__":
    main()
