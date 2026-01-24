from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pose_dynamics.preprocess.pipeline import run_pipeline
from pose_dynamics.preprocess.schema import PreprocessConfig


@dataclass(frozen=True)
class PreprocessOutputs:
    pose_clean_path: Path
    windows_path: Path
    qc_path: Path
    provenance_path: Path
    alignment_transforms_path: Path | None = None


def run_preprocess(
    pose_path: str | Path,
    recording_path: str | Path,
    config: PreprocessConfig | str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> PreprocessOutputs:
    """
    Run preprocessing for canonical pose data.

    Inputs:
      - pose_path: canonical pose.parquet from ingest
      - recording_path: recording.json from ingest
      - config: PreprocessConfig OR path to preprocess.yml
      - out_dir: output directory

    Outputs (written to out_dir):
      - pose_clean.parquet
      - windows.parquet
      - qc_preprocess.json
      - provenance.json (resolved config + hashes + run metadata)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(config, PreprocessConfig):
        cfg = config
    else:
        cfg = PreprocessConfig.from_yaml(str(config))

    pose_path = Path(pose_path)
    recording_path = Path(recording_path)

    df = pd.read_parquet(pose_path)
    with recording_path.open("r", encoding="utf-8") as f:
        recording = json.load(f)

    df_clean, windows, qc, transforms = run_pipeline(df, recording, cfg)

    pose_clean_path = out_dir / "pose_clean.parquet"
    windows_path = out_dir / "windows.parquet"
    qc_path = out_dir / "qc_preprocess.json"
    provenance_path = out_dir / "provenance.json"
    alignment_transforms_path = out_dir / "alignment_transforms.json"

    if not overwrite:
        required = [pose_clean_path, windows_path, qc_path, provenance_path]
        if cfg.alignment.enabled:
            required.append(alignment_transforms_path)
        for p in required:
            if p.exists():
                raise FileExistsError(f"Output already exists: {p}")

    df_clean.to_parquet(pose_clean_path, index=False)
    windows.to_parquet(windows_path, index=False)

    qc_payload = {
        "summary": qc,
        "n_rows_clean": int(df_clean.shape[0]),
    }
    qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

    provenance_payload = {
        "pose_path": str(pose_path.resolve()),
        "recording_path": str(recording_path.resolve()),
        "config": cfg.to_dict(),
    }
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2), encoding="utf-8"
    )

    if cfg.alignment.enabled:
        alignment_transforms_path.write_text(
            json.dumps(transforms, indent=2), encoding="utf-8"
        )
        align_path = alignment_transforms_path
    else:
        align_path = None

    return PreprocessOutputs(
        pose_clean_path=pose_clean_path,
        windows_path=windows_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
        alignment_transforms_path=align_path,
    )
