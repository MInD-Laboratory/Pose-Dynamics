"""
pose_dynamics.io.csv_pose

Ingest "Pose CSV (wide)" files into the repo's canonical "Pose Parquet (long)" format.

INPUT CONTRACT (per CSV = one trial/session):
- Each row is a time sample.
- Required time axis:
    - Either a `time` column (float seconds), OR
    - a `frame` column (int) AND the user provides --fps
- Keypoints are represented in *wide* columns:
    - x_<kp>, y_<kp> are required for a keypoint to be valid
    - optional: z_<kp>
    - optional: prob_<kp> or conf_<kp>
- <kp> can be any suffix (e.g., "1", "knee", "left_wrist", etc.).
- Any other columns are ignored (and logged).

OUTPUT (across a directory of CSVs):
- pose.parquet   : canonical long-form table for all trials
- recording.json : run metadata + per-trial metadata (including ignored columns)
- qc_ingest.json : per-trial QC summary table

- One stream per CSV: no multi-person tracking. If users have multiple people, they must split upstream.
- If both `time` and `frame` exist, we use `time` and log `frame` under recognized_but_unused_columns.
"""

from __future__ import annotations

import json
import re
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# -------------------------
# Column parsing utilities
# -------------------------

# Regex patterns for recognized pose columns.
# We accept:
#   x_<kp>, y_<kp>, z_<kp>
#   prob_<kp> or conf_<kp>
_X_RE = re.compile(r"^x_(.+)$")
_Y_RE = re.compile(r"^y_(.+)$")
_Z_RE = re.compile(r"^z_(.+)$")
_PROB_RE = re.compile(r"^prob_(.+)$")
_CONF_RE = re.compile(r"^conf_(.+)$")


@dataclass(frozen=True)
class PoseColumnSpec:
    """
    Parsed description of pose columns found in a CSV.

    keypoints: list of keypoint names (suffixes) that are considered valid (must have x & y).
    colmap: mapping of (prefix, keypoint) -> actual column name in the CSV.
            prefix in {"x","y","z","conf"}; conf may originate from prob_* or conf_*.
    ignored_columns: columns that are not recognized by the ingest system.
    recognized_but_unused_columns: columns that are recognized but not used (e.g., frame when time exists).
    """

    keypoints: List[str]
    colmap: Dict[Tuple[str, str], str]
    ignored_columns: List[str]
    recognized_but_unused_columns: List[str]


def parse_pose_columns(columns: Sequence[str]) -> PoseColumnSpec:
    """
    Parse columns of a wide pose CSV and identify keypoints and which columns correspond
    to x/y/z/conf for each keypoint.

    Rules:
    - A keypoint is considered "valid" if it has BOTH x_<kp> and y_<kp>.
    - z and conf/prob are optional.
    - We unify prob_<kp> and conf_<kp> into canonical 'conf'.
    - Unknown columns are ignored.
    """
    # Track which keypoints have which components.
    x_cols: Dict[str, str] = {}
    y_cols: Dict[str, str] = {}
    z_cols: Dict[str, str] = {}
    conf_cols: Dict[str, str] = {}  # canonical name: "conf"

    recognized_cols: set[str] = set()
    recognized_but_unused: List[
        str
    ] = []  # populated elsewhere; keep here for API symmetry

    for c in columns:
        m = _X_RE.match(c)
        if m:
            kp = m.group(1)
            x_cols[kp] = c
            recognized_cols.add(c)
            continue

        m = _Y_RE.match(c)
        if m:
            kp = m.group(1)
            y_cols[kp] = c
            recognized_cols.add(c)
            continue

        m = _Z_RE.match(c)
        if m:
            kp = m.group(1)
            z_cols[kp] = c
            recognized_cols.add(c)
            continue

        m = _PROB_RE.match(c)
        if m:
            kp = m.group(1)
            conf_cols[kp] = c
            recognized_cols.add(c)
            continue

        m = _CONF_RE.match(c)
        if m:
            kp = m.group(1)
            conf_cols[kp] = c
            recognized_cols.add(c)
            continue

        # Not a recognized pose column (time/frame handled elsewhere).
        # We don't decide ignored here yet, because time/frame are recognized outside this function.

    # Valid keypoints must have x and y.
    kps = sorted(set(x_cols.keys()) & set(y_cols.keys()))

    colmap: Dict[Tuple[str, str], str] = {}
    for kp in kps:
        colmap[("x", kp)] = x_cols[kp]
        colmap[("y", kp)] = y_cols[kp]
        if kp in z_cols:
            colmap[("z", kp)] = z_cols[kp]
        if kp in conf_cols:
            colmap[("conf", kp)] = conf_cols[kp]

    # We can only determine ignored columns once we also consider time/frame.
    ignored_columns = []  # filled by caller

    return PoseColumnSpec(
        keypoints=kps,
        colmap=colmap,
        ignored_columns=ignored_columns,
        recognized_but_unused_columns=recognized_but_unused,
    )


# -------------------------
# Ingest core
# -------------------------


@dataclass(frozen=True)
class TrialIngestResult:
    """
    Result of ingesting a single trial CSV.
    """

    trial_id: str
    source_file: str
    df_long: pd.DataFrame
    trial_meta: dict
    qc: dict


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """
    In-place numeric coercion for the specified columns.
    Non-convertible values become NaN. This is intentional: downstream QC will expose missingness.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def wide_pose_csv_to_long(
    csv_path: Path,
    *,
    fps: Optional[float],
) -> TrialIngestResult:
    """
    Read one wide-format pose CSV and convert it into canonical long format.

    Canonical long output columns:
    - trial_id (str)
    - source_file (str)
    - time OR frame (one of them)
    - keypoint (str)
    - x, y (float)
    - z (float, optional if any z_* exists)
    - conf (float, optional if any prob_*/conf_* exists)
    """
    csv_path = Path(csv_path)
    trial_id = csv_path.stem
    source_file = csv_path.name

    # Read with pandas. We assume headers exist.
    # Keep default dtype inference; we'll coerce relevant numeric columns below.
    df = pd.read_csv(csv_path)

    if df.shape[0] == 0:
        raise ValueError(
            f"{source_file}: CSV has 0 rows. Each CSV must contain at least one sample."
        )

    # Recognize time/frame columns first (they are special and not part of the keypoint regex scanning).
    has_time = "time" in df.columns
    has_frame = "frame" in df.columns

    # Enforce timing rules:
    # - If time exists, we use it.
    # - Else we require frame + fps.
    recognized_but_unused: List[str] = []
    if has_time:
        time_col = "time"
        if has_frame:
            recognized_but_unused.append("frame")
    else:
        # No time -> must have frame and fps provided
        if not has_frame:
            raise ValueError(
                f"{source_file}: missing 'time' column and missing 'frame' column. "
                "Provide either a 'time' column, or a 'frame' column with --fps."
            )
        if fps is None:
            raise ValueError(
                f"{source_file}: 'time' column not found. You must provide --fps when using 'frame'."
            )
        time_col = "frame"

    # Parse pose columns (x_/y_/z_/prob_*/conf_*)
    spec = parse_pose_columns(list(df.columns))

    # Determine ignored columns: anything not time/frame and not a recognized pose column pattern.
    recognized_cols = set()
    recognized_cols.add("time") if has_time else None
    recognized_cols.add("frame") if has_frame else None
    # Add all columns referenced by colmap
    for (_, _kp), colname in spec.colmap.items():
        recognized_cols.add(colname)

    ignored_columns = sorted([c for c in df.columns if c not in recognized_cols])

    # Identify malformed keypoints: suffixes where x or y is missing.
    # We do this by scanning x_* and y_* columns present, then checking pairs.
    # If user has x_knee but no y_knee, that is almost always a data mistake.
    x_suffixes = {m.group(1) for c in df.columns if (m := _X_RE.match(c))}
    y_suffixes = {m.group(1) for c in df.columns if (m := _Y_RE.match(c))}
    malformed = sorted((x_suffixes ^ y_suffixes))  # symmetric difference
    if malformed:
        raise ValueError(
            f"{source_file}: malformed keypoints detected (missing x_ or y_ partner): {malformed}. "
            "Each keypoint must have both x_<kp> and y_<kp>."
        )

    if len(spec.keypoints) == 0:
        raise ValueError(
            f"{source_file}: no valid keypoints found. Expected columns like x_<kp> and y_<kp>."
        )

    # Coerce numeric types for time/frame and pose columns we will use.
    numeric_cols = []
    numeric_cols.append("time") if has_time else None
    numeric_cols.append("frame") if has_frame else None
    for (prefix, kp), colname in spec.colmap.items():
        numeric_cols.append(colname)
    _coerce_numeric(df, numeric_cols)

    # Build long-form rows.
    long_parts: List[pd.DataFrame] = []

    # We'll keep either `time` or `frame` in the output; never both.
    t = df[time_col].to_numpy()

    # Determine whether any z/conf columns exist at all (controls whether we include them in output).
    any_z = any((("z", kp) in spec.colmap) for kp in spec.keypoints)
    any_conf = any((("conf", kp) in spec.colmap) for kp in spec.keypoints)

    # Reuse trial_id/source_file vectors for speed.
    n = df.shape[0]
    trial_vec = pd.Series([trial_id] * n, dtype="string")
    file_vec = pd.Series([source_file] * n, dtype="string")

    for kp in spec.keypoints:
        x = df[spec.colmap[("x", kp)]].to_numpy()
        y = df[spec.colmap[("y", kp)]].to_numpy()

        part = pd.DataFrame(
            {
                "trial_id": trial_vec,
                "source_file": file_vec,
                time_col: t,
                "keypoint": kp,
                "x": x,
                "y": y,
            }
        )

        if any_z:
            if ("z", kp) in spec.colmap:
                part["z"] = df[spec.colmap[("z", kp)]].to_numpy()
            else:
                part["z"] = pd.NA  # explicit missing if other keypoints have z

        if any_conf:
            if ("conf", kp) in spec.colmap:
                part["conf"] = df[spec.colmap[("conf", kp)]].to_numpy()
            else:
                part["conf"] = pd.NA  # explicit missing if other keypoints have conf

        long_parts.append(part)

    df_long = pd.concat(long_parts, ignore_index=True)

    # Standardize column name: if time existed, output column is `time`; else output is `frame`.
    # If we used `frame`, `time_col` == "frame" already, which is what we want.
    # If we used `time`, `time_col` == "time" already.

    # Sort for determinism and to make downstream operations predictable.
    sort_cols = ["trial_id", "keypoint", time_col]
    df_long = df_long.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Basic QC metrics
    # Missingness in x/y is critical; z/conf are optional.
    qc = {
        "trial_id": trial_id,
        "source_file": source_file,
        "n_rows_input": int(df.shape[0]),
        "n_keypoints": int(len(spec.keypoints)),
        "n_rows_long": int(df_long.shape[0]),
        "has_time": bool(has_time),
        "has_frame": bool(has_frame),
        "timing_mode": "time" if has_time else "frame",
        "fps_used": float(fps) if (not has_time and fps is not None) else None,
        "has_z": bool(any_z),
        "has_conf": bool(any_conf),
        "x_missing_frac": float(pd.isna(df_long["x"]).mean()),
        "y_missing_frac": float(pd.isna(df_long["y"]).mean()),
        "z_missing_frac": float(pd.isna(df_long["z"]).mean())
        if "z" in df_long.columns
        else None,
        "conf_missing_frac": float(pd.isna(df_long["conf"]).mean())
        if "conf" in df_long.columns
        else None,
        "time_min": float(pd.to_numeric(df[time_col], errors="coerce").min())
        if has_time
        else None,
        "time_max": float(pd.to_numeric(df[time_col], errors="coerce").max())
        if has_time
        else None,
        "frame_min": int(pd.to_numeric(df[time_col], errors="coerce").min())
        if (not has_time)
        else None,
        "frame_max": int(pd.to_numeric(df[time_col], errors="coerce").max())
        if (not has_time)
        else None,
        "ignored_columns": ignored_columns,
        "recognized_but_unused_columns": recognized_but_unused,
        "keypoints_detected": spec.keypoints,
    }

    # Trial metadata recorded in recording.json
    trial_meta = {
        "trial_id": trial_id,
        "source_file": source_file,
        "has_time": bool(has_time),
        "has_frame": bool(has_frame),
        "timing_mode": "time" if has_time else "frame",
        "fps_used": float(fps) if (not has_time and fps is not None) else None,
        "has_z": bool(any_z),
        "has_conf": bool(any_conf),
        "keypoints_detected": spec.keypoints,
        "ignored_columns": ignored_columns,
        "recognized_but_unused_columns": recognized_but_unused,
        "n_rows_input": int(df.shape[0]),
        "n_rows_long": int(df_long.shape[0]),
    }

    return TrialIngestResult(
        trial_id=trial_id,
        source_file=source_file,
        df_long=df_long,
        trial_meta=trial_meta,
        qc=qc,
    )


def ingest_pose_csv_dir(
    in_path: Path,
    out_dir: Path,
    *,
    fps: Optional[float],
    glob_pattern: str = "*.csv",
) -> None:
    """
    Ingest a directory (or glob root) of wide pose CSV files.

    Parameters
    ----------
    in_path:
        Directory containing CSVs OR a path that will be used as a directory.
    out_dir:
        Output directory where pose.parquet, recording.json, qc_ingest.json will be written.
    fps:
        Required if any file lacks a `time` column and uses `frame` instead.
        If time exists, fps is ignored for that file.
    glob_pattern:
        Pattern used to locate CSVs within in_path.

    Notes
    -----
    - Each CSV is treated as one trial/session.
    - We concatenate all trials into a single canonical parquet.
    """
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather input CSV files deterministically.
    # Sorting matters for reproducibility and for stable outputs across machines.
    csv_files = sorted(in_path.glob(glob_pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {in_path} matching pattern {glob_pattern!r}."
        )

    trial_results: List[TrialIngestResult] = []
    trials_meta: List[dict] = []
    qc_rows: List[dict] = []

    # Ingest each file
    for csv_path in csv_files:
        res = wide_pose_csv_to_long(csv_path, fps=fps)
        trial_results.append(res)
        trials_meta.append(res.trial_meta)
        qc_rows.append(res.qc)

    # Concatenate into one big canonical dataframe.
    df_all = pd.concat([r.df_long for r in trial_results], ignore_index=True)

    # Determine whether the run is time-mode or frame-mode:
    # It's allowed to mix (some trials with time, some with frame).
    has_any_time = any(t["timing_mode"] == "time" for t in trials_meta)
    has_any_frame = any(t["timing_mode"] == "frame" for t in trials_meta)

    run_meta = {
        "schema_version": "pose_dynamics_csv_wide_v1",
        "ingest_time_utc": time_module.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time_module.gmtime()
        ),
        "input_path": str(in_path.resolve()),
        "glob_pattern": glob_pattern,
        "fps_arg": float(fps) if fps is not None else None,
        "timing_modes_present": {
            "time": bool(has_any_time),
            "frame": bool(has_any_frame),
        },
        "pose_dynamics_version": None,
        "git_commit": None,
        "trials": trials_meta,
    }

    # Write outputs.
    pose_parquet = out_dir / "pose.parquet"
    recording_json = out_dir / "recording.json"
    qc_json = out_dir / "qc_ingest.json"

    df_all.to_parquet(pose_parquet, index=False)

    with recording_json.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    with qc_json.open("w", encoding="utf-8") as f:
        json.dump(qc_rows, f, indent=2)
