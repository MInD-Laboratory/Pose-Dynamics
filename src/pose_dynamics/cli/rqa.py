"""
pose_dynamics.cli.rqa

CLI entrypoint for RQA/CRQA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_dynamics.rqa.api import run_rqa


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pose-dynamics rqa",
        description="Run RQA or CRQA on preprocessed windows.",
    )
    p.add_argument("--pose", dest="pose_path", required=True, type=Path)
    p.add_argument("--windows", dest="windows_path", required=True, type=Path)
    p.add_argument("--config", dest="config_path", required=True, type=Path)
    p.add_argument("--out", dest="out_dir", required=True, type=Path)
    p.add_argument("--pose-y", dest="pose_y_path", required=False, type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load config and iterate
    import yaml

    with open(args.config_path, "r") as f:
        conf = yaml.safe_load(f)

    rqa_list = conf.get("rqa", [])
    if not rqa_list:
        print("No RQA configurations found in yaml.")
        return 0

    for i, item in enumerate(rqa_list):
        name = item.get("output_name", f"analysis_{i}")
        print(f"[RQA] starting {name} ({i + 1}/{len(rqa_list)})")
        run_rqa(
            pose_clean_path=args.pose_path,
            windows_path=args.windows_path,
            config_dict=item,
            out_dir=args.out_dir / name,
            pose_y_path=args.pose_y_path,
            overwrite=bool(args.overwrite),
            progress_title=f"RQA: {name}",
        )
        print(f"[RQA] finished {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
