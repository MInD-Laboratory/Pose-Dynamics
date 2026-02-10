"""
pose_dynamics.cli.preprocess

CLI entrypoint for preprocessing canonical pose data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from pose_dynamics.preprocess.api import run_preprocess


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pose-dynamics preprocess",
        description="Preprocess canonical pose parquet into clean signals and windows.",
    )
    p.add_argument(
        "--in",
        dest="pose_path",
        required=True,
        type=Path,
        help="Input pose.parquet from ingest.",
    )
    p.add_argument(
        "--recording",
        dest="recording_path",
        required=True,
        type=Path,
        help="Input recording.json from ingest.",
    )
    p.add_argument(
        "--config",
        dest="config_path",
        required=True,
        type=Path,
        help="Preprocess config YAML file (preprocess.yml).",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        help="Output directory for preprocess artifacts.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    console = Console()
    columns = (
        SpinnerColumn(),
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console) as progress:
        stage_task = progress.add_task("pose-dynamics preprocess", total=None)
        template_task = progress.add_task(
            "alignment templates",
            total=0,
            visible=False,
        )

        def report_stage(label: str, _advance: int = 0) -> None:
            progress.update(
                stage_task,
                description=f"pose-dynamics preprocess: {label}",
            )

        def report_alignment(advance: int, total: int) -> None:
            if total <= 0:
                return
            task = progress.tasks[template_task]
            if (not task.visible) or (task.total != total):
                progress.update(
                    template_task,
                    total=total,
                    completed=0,
                    visible=True,
                    description=f"alignment templates: 0/{total}",
                )
            if advance:
                progress.advance(template_task, advance)
            completed = int(progress.tasks[template_task].completed)
            progress.update(
                template_task,
                description=f"alignment templates: {completed}/{total}",
            )
            if completed >= total:
                progress.update(
                    template_task,
                    description="alignment templates: complete",
                    visible=False,
                )

        run_preprocess(
            pose_path=args.pose_path,
            recording_path=args.recording_path,
            config=args.config_path,
            out_dir=args.out_dir,
            overwrite=bool(args.overwrite),
            stage_callback=report_stage,
            alignment_progress_callback=report_alignment,
        )
        progress.update(
            stage_task,
            description="pose-dynamics preprocess: complete",
        )
        progress.stop_task(stage_task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
