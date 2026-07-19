"""
Command-line interface: run the standard pipeline from a config file.

    pose-dynamics run study.yaml          # extract features / metrics over a folder
    pose-dynamics new-config out.yaml     # write a template config to edit
    python -m pose_dynamics run study.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .study import StudyConfig, run_study

_TEMPLATE = """\
# pose-dynamics study config. Edit the values, then:  pose-dynamics run this_file.yaml
# Every setting below shows its default; delete a line to accept the default.
# Full explanations + the paper's guidance: docs/configuration.md

data: ./data                 # folder of canonical CSVs (one file per person per trial)
frame_rate: 60               # Hz — how your data was recorded (required)

# --- embedding (choose these in the quickstart notebook) ---
tau: 20                      # embedding delay
m: 4                         # embedding dimension

# --- preprocessing ---
conf_threshold: 0.30         # mask keypoints below this confidence
interp_cap: null             # max gap (frames) to interpolate; null -> (m-1)*tau
filter_cutoff: 10.0          # Hz low-pass (use ~5 for 3D full body)
filter_order: 4              # Butterworth order (zero-phase)

# --- feature pipeline: a list of steps applied in order (docs/feature_steps.md) ---
# Delete this whole block to fall back to per-keypoint speed.
features:
  - step: coordinate_normalization
    params: {width: 720, height: 720}   # your video resolution
  # - step: procrustes                  # <- UNCOMMENT to align (remove head/body motion)
  #   params: {template: self_mean, scale: uniform, emit: geometry}
  - step: roi_centroid
    params:
      rois:
        arms: [2, 3, 4, 5, 6, 7]        # indices from `pose-dynamics inspect`
        upper_body: [1, 2, 5, 8]
  - step: velocity_magnitude
    params: {method: diff}
default_signal: speed          # used only when the `features` block is deleted

# --- what to compute ---
compute_linear: true           # magnitude summaries (mean/std/rms/max)
compute_recurrence: true       # recurrence metrics

# --- recurrence settings ---
radius_mode: fixed_rrec        # fixed_rrec = target a %REC (radius solved) | fixed_radius
target_rec: 5.0                # % target, for fixed_rrec
radius: null                   # threshold, for fixed_radius mode
rescale: mean                  # rescale distances by mean | max | none
norm: zscore                   # normalise each signal: zscore | minmax | center | none
min_line: 2                    # l_min: minimum line length (raise if %DET ceilings)
theiler: null                  # Theiler window; null -> tau (cross-RQA forces 0)

# --- windowing ---
window_s: 30                   # analysis window (seconds); null -> whole trial
overlap: 0.5                   # fractional overlap between windows

# --- data quality ---
max_missing_frac: 0.30         # trials above this missingness get `on_exceed`
on_exceed: flag                # retain | flag | exclude (never dropped silently)

# --- output ---
features_dir: ./features       # per-file feature time series (null to skip)
output_csv: ./metrics.csv      # tidy metrics table (null to skip)
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pose-dynamics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run the standard pipeline from a config file")
    run_p.add_argument("config", help="path to a .yaml/.json study config")
    run_p.add_argument("--quiet", action="store_true")

    new_p = sub.add_parser("new-config", help="write a template config to edit")
    new_p.add_argument("path", help="where to write the template (e.g. study.yaml)")

    insp_p = sub.add_parser(
        "inspect", help="plot a file's keypoints with their index numbers")
    insp_p.add_argument("file", help="a canonical CSV")
    insp_p.add_argument("-o", "--output", help="save the plot here (default <file>_keypoints.png)")
    insp_p.add_argument("--frame", type=int, default=None, help="frame to plot (default: mean pose)")

    args = parser.parse_args(argv)

    if args.cmd == "new-config":
        Path(args.path).write_text(_TEMPLATE)
        print(f"wrote template config to {args.path} — edit it, then: pose-dynamics run {args.path}")
        return 0

    if args.cmd == "run":
        config = StudyConfig.from_file(args.config)
        results, quality = run_study(config, progress=not args.quiet)
        n_ex = int((quality["status"] == "exclude").sum()) if len(quality) else 0
        print(f"\nDone: {len(quality)} trials ({n_ex} excluded), {len(results)} metric rows.")
        return 0

    if args.cmd == "inspect":
        import matplotlib
        matplotlib.use("Agg")
        from .data.loader import load_pose_csv

        seq = load_pose_csv(args.file, frame_rate=1.0)   # rate irrelevant for inspection
        print(f"{args.file}: {seq.n_keypoints} keypoints, {seq.dims}D, "
              f"confidence={'yes' if seq.has_confidence else 'no'}, {seq.n_frames} frames")
        print(f"keypoint indices: 0 .. {seq.n_keypoints - 1} "
              "(use these numbers in a feature pipeline's ROI lists)")
        # default: write into the current directory (always writable), not next to
        # the input (which may be read-only, e.g. a mounted volume or site-packages).
        out = Path(args.output) if args.output else Path.cwd() / f"{Path(args.file).stem}_keypoints.png"
        ax = seq.plot_keypoints(frame=args.frame)
        ax.figure.savefig(out, dpi=110, bbox_inches="tight")
        print(f"saved labelled skeleton to {out.resolve()}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
