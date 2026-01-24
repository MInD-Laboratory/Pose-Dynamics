"""
pose_dynamics.cli.main

A tiny dispatcher so `pose-dynamics <command>` can work without external CLI frameworks.

Commands:
- ingest-csv
- preprocess
- feature-extract
- pca
- list-keypoints
- rqa-params
- rqa
"""

from __future__ import annotations

import argparse

from pose_dynamics.cli.feature_extract import main as feature_extract_main
from pose_dynamics.cli.ingest_csv import main as ingest_csv_main
from pose_dynamics.cli.list_keypoints import main as list_keypoints_main
from pose_dynamics.cli.pca import main as pca_main
from pose_dynamics.cli.preprocess import main as preprocess_main
from pose_dynamics.cli.rqa import main as rqa_main
from pose_dynamics.cli.rqa_params import main as rqa_params_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pose-dynamics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest-csv subcommand
    p_ing = sub.add_parser(
        "ingest-csv", help="Ingest wide pose CSV files into canonical parquet."
    )
    p_ing.add_argument("--in", dest="in_path", required=True)
    p_ing.add_argument("--out", dest="out_dir", required=True)
    p_ing.add_argument("--fps", dest="fps", default=None)
    p_ing.add_argument("--pattern", dest="pattern", default="*.csv")

    # preprocess subcommand
    p_pre = sub.add_parser("preprocess", help="Run preprocessing on pose.parquet.")
    p_pre.add_argument("--in", dest="pose_path", required=True)
    p_pre.add_argument("--recording", dest="recording_path", required=True)
    p_pre.add_argument("--config", dest="config_path", required=True)
    p_pre.add_argument("--out", dest="out_dir", required=True)
    p_pre.add_argument("--overwrite", action="store_true")

    # feature-extract subcommand
    p_feat = sub.add_parser(
        "feature-extract", help="Extract features from preprocessed data."
    )
    p_feat.add_argument("--pose", dest="pose_path", required=True)
    p_feat.add_argument("--windows", dest="windows_path", required=True)
    p_feat.add_argument("--config", dest="config_path", required=True)
    p_feat.add_argument("--out", dest="out_dir", required=True)
    p_feat.add_argument("--overwrite", action="store_true")

    # pca subcommand
    p_pca = sub.add_parser("pca", help="Run PCA on pose summaries and/or features.")
    p_pca.add_argument("--pose", dest="pose_path", required=True)
    p_pca.add_argument("--windows", dest="windows_path", required=True)
    p_pca.add_argument("--features", dest="features_path", required=True)
    p_pca.add_argument("--config", dest="config_path", required=True)
    p_pca.add_argument("--out", dest="out_dir", required=True)
    p_pca.add_argument("--overwrite", action="store_true")

    # list-keypoints subcommand
    p_kp = sub.add_parser("list-keypoints", help="List available keypoints.")
    p_kp.add_argument("--pose", dest="pose_path", required=True)

    # rqa-params subcommand
    p_rqa = sub.add_parser("rqa-params", help="Estimate AMI/FNN/epsilon for RQA.")
    p_rqa.add_argument("--pose", dest="pose_path", required=True)
    p_rqa.add_argument("--windows", dest="windows_path", required=True)
    p_rqa.add_argument("--config", dest="config_path", required=True)
    p_rqa.add_argument("--out", dest="out_dir", required=True)
    p_rqa.add_argument("--overwrite", action="store_true")

    # rqa subcommand
    p_rqa_run = sub.add_parser("rqa", help="Run RQA/CRQA on preprocessed windows.")
    p_rqa_run.add_argument("--pose", dest="pose_path", required=True)
    p_rqa_run.add_argument("--windows", dest="windows_path", required=True)
    p_rqa_run.add_argument("--config", dest="config_path", required=True)
    p_rqa_run.add_argument("--out", dest="out_dir", required=True)
    p_rqa_run.add_argument("--pose-y", dest="pose_y_path", required=False)
    p_rqa_run.add_argument("--overwrite", action="store_true")

    args, unknown = parser.parse_known_args(argv)

    if args.cmd == "ingest-csv":
        # Rebuild argv for the subcommand parser
        sub_argv = [
            "--in",
            args.in_path,
            "--out",
            args.out_dir,
            "--pattern",
            args.pattern,
        ]
        if args.fps is not None:
            sub_argv += ["--fps", str(args.fps)]
        # Pass through any unknown args
        sub_argv += unknown
        return ingest_csv_main(sub_argv)

    if args.cmd == "preprocess":
        sub_argv = [
            "--in",
            args.pose_path,
            "--recording",
            args.recording_path,
            "--config",
            args.config_path,
            "--out",
            args.out_dir,
        ]
        if args.overwrite:
            sub_argv += ["--overwrite"]
        sub_argv += unknown
        return preprocess_main(sub_argv)

    if args.cmd == "feature-extract":
        sub_argv = [
            "--pose",
            args.pose_path,
            "--windows",
            args.windows_path,
            "--config",
            args.config_path,
            "--out",
            args.out_dir,
        ]
        if args.overwrite:
            sub_argv += ["--overwrite"]
        sub_argv += unknown
        return feature_extract_main(sub_argv)

    if args.cmd == "pca":
        sub_argv = [
            "--pose",
            args.pose_path,
            "--windows",
            args.windows_path,
            "--features",
            args.features_path,
            "--config",
            args.config_path,
            "--out",
            args.out_dir,
        ]
        if args.overwrite:
            sub_argv += ["--overwrite"]
        sub_argv += unknown
        return pca_main(sub_argv)

    if args.cmd == "list-keypoints":
        sub_argv = ["--pose", args.pose_path]
        sub_argv += unknown
        return list_keypoints_main(sub_argv)

    if args.cmd == "rqa-params":
        sub_argv = [
            "--pose",
            args.pose_path,
            "--windows",
            args.windows_path,
            "--config",
            args.config_path,
            "--out",
            args.out_dir,
        ]
        if args.overwrite:
            sub_argv += ["--overwrite"]
        sub_argv += unknown
        return rqa_params_main(sub_argv)

    if args.cmd == "rqa":
        sub_argv = [
            "--pose",
            args.pose_path,
            "--windows",
            args.windows_path,
            "--config",
            args.config_path,
            "--out",
            args.out_dir,
        ]
        if args.pose_y_path is not None:
            sub_argv += ["--pose-y", args.pose_y_path]
        if args.overwrite:
            sub_argv += ["--overwrite"]
        sub_argv += unknown
        return rqa_main(sub_argv)

    raise RuntimeError("Unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
