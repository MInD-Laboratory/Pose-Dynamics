from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

STEP_CHOICES = ["ingest", "preprocess", "features", "rqa", "analysis"]

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "examples" / "case_study_1_matb" / "data" / "raw_test"  # raw_pose_csv
CFG_DIR = ROOT / "examples" / "case_study_1_matb" / "configs"
PREPROCESS_CFG = CFG_DIR / "preprocess.yaml"
FEATURE_CFG = CFG_DIR / "features.yaml"
RQA_CFG = CFG_DIR / "rqa.yaml"
INGEST_OUT = ROOT / "artifacts" / "ingest" / "matb"
PRE_OUT = ROOT / "artifacts" / "preprocess" / "matb"
FEAT_OUT = ROOT / "artifacts" / "features" / "matb"
RQA_OUT = ROOT / "artifacts" / "rqa" / "matb"

INGEST_OUTPUTS = [
    INGEST_OUT / "pose.parquet",
    INGEST_OUT / "recording.json",
    INGEST_OUT / "qc_ingest.json",
]

PREPROCESS_OUTPUTS = [
    PRE_OUT / "pose_clean.parquet",
    PRE_OUT / "windows.parquet",
    PRE_OUT / "qc_preprocess.json",
    PRE_OUT / "provenance.json",
    PRE_OUT / "alignment_transforms.json",
    PRE_OUT / "alignment_transforms.parquet",
]

FEATURE_OUTPUTS = [
    FEAT_OUT / "features.parquet",
    FEAT_OUT / "qc_features.json",
    FEAT_OUT / "provenance_features.json",
]


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def _pd_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "pose_dynamics.cli.main", *args]


def _all_exist(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)


def run_ingest(force: bool = False) -> None:
    if not force and _all_exist(INGEST_OUTPUTS):
        print("[skip] ingest (artifacts already exist)")
        return
    INGEST_OUT.mkdir(parents=True, exist_ok=True)
    cmd = _pd_cmd(
        "ingest-csv",
        "--in",
        str(DATA_DIR),
        "--out",
        str(INGEST_OUT),
        "--fps",
        "60",
    )
    _run(cmd)


def _snapshot_preprocess_state(stage_label: str) -> None:
    stage_dir = PRE_OUT / "_last_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage.txt").write_text(f"{stage_label}\n", encoding="utf-8")
    if PREPROCESS_CFG.exists():
        shutil.copy2(PREPROCESS_CFG, stage_dir / "preprocess.yaml")
    for name in [
        "pose_clean.parquet",
        "windows.parquet",
        "qc_preprocess.json",
        "provenance.json",
        "alignment_transforms.json",
        "alignment_transforms.parquet",
    ]:
        src = PRE_OUT / name
        if src.exists():
            shutil.copy2(src, stage_dir / name)


def run_preprocess(stage_label: str, force: bool = False) -> None:
    if not PREPROCESS_CFG.exists():
        raise FileNotFoundError(f"Missing preprocess config: {PREPROCESS_CFG}")
    if not _all_exist(INGEST_OUTPUTS):
        raise RuntimeError("Ingest outputs missing. Run 'ingest' first.")
    if not force and _all_exist(PREPROCESS_OUTPUTS):
        print("[skip] preprocess (pose_clean + windows already exist)")
        return
    PRE_OUT.mkdir(parents=True, exist_ok=True)
    cmd = _pd_cmd(
        "preprocess",
        "--in",
        str(INGEST_OUT / "pose.parquet"),
        "--recording",
        str(INGEST_OUT / "recording.json"),
        "--config",
        str(PREPROCESS_CFG),
        "--out",
        str(PRE_OUT),
        "--overwrite",
    )
    _run(cmd)
    _snapshot_preprocess_state(stage_label)


def run_features(force: bool = False) -> None:
    if not FEATURE_CFG.exists():
        raise FileNotFoundError(f"Missing features config: {FEATURE_CFG}")
    if not _all_exist(PREPROCESS_OUTPUTS):
        raise RuntimeError("Preprocess outputs missing. Run 'preprocess' first.")
    if not force and _all_exist(FEATURE_OUTPUTS):
        print("[skip] features (artifacts already exist)")
        return
    FEAT_OUT.mkdir(parents=True, exist_ok=True)
    cmd = _pd_cmd(
        "feature-extract",
        "--pose",
        str(PRE_OUT / "pose_clean.parquet"),
        "--windows",
        str(PRE_OUT / "windows.parquet"),
        "--alignment-transforms",
        str(PRE_OUT / "alignment_transforms.parquet"),
        "--config",
        str(FEATURE_CFG),
        "--out",
        str(FEAT_OUT),
        "--overwrite",
    )
    _run(cmd)


def run_rqa(force: bool = False) -> None:
    if not RQA_CFG.exists():
        raise FileNotFoundError(f"Missing rqa config: {RQA_CFG}")
    if not _all_exist(PREPROCESS_OUTPUTS):
        raise RuntimeError("Preprocess outputs missing. Run 'preprocess' first.")

    if not force and RQA_OUT.exists() and any(RQA_OUT.iterdir()):
        print("[skip] rqa (artifacts appear to exist)")
        return

    RQA_OUT.mkdir(parents=True, exist_ok=True)
    cmd = _pd_cmd(
        "rqa",
        "--pose",
        str(PRE_OUT / "pose_clean.parquet"),
        "--windows",
        str(PRE_OUT / "windows.parquet"),
        "--config",
        str(RQA_CFG),
        "--out",
        str(RQA_OUT),
        "--overwrite",
    )
    _run(cmd)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="matb-run",
        description="MATB pipeline helper (incremental setup)",
    )
    parser.add_argument(
        "steps",
        nargs="*",
        choices=STEP_CHOICES,
        help="Steps to run in order.",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Re-run ingest even if outputs already exist.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Re-run preprocess even if pose_clean/windows exist.",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Re-run feature extraction even if outputs already exist.",
    )
    parser.add_argument(
        "--force-rqa",
        action="store_true",
        help="Re-run rqa even if outputs already exist.",
    )
    parser.add_argument(
        "--preprocess-stage-label",
        default="selection_confidence",
        help="Label stored with the snapshot of the most recent preprocess run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    steps = args.steps or []
    if not steps:
        print("No steps selected yet. Add a step name when invoking matb-run.")
        return
    for step in steps:
        if step == "ingest":
            run_ingest(force=args.force_ingest)
        elif step == "preprocess":
            run_preprocess(
                stage_label=args.preprocess_stage_label,
                force=args.force_preprocess,
            )
        elif step == "features":
            run_features(force=args.force_features)
        elif step == "rqa":
            run_rqa(force=args.force_rqa)
        else:
            print(f"[placeholder] would run step: {step}")


if __name__ == "__main__":
    main()
