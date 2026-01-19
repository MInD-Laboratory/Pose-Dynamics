# src/pose_dynamics/io/openpose.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .validation import coerce_pose_df, validate_pose_df

# Accept a few common naming styles:
#   x0, y0, prob0
#   x_0, y_0, c_0
#   x[0], y[0], conf[0]
_X_RE = re.compile(r"^x(?:_|\[)?(?P<kp>\d+)\]?$")
_Y_RE = re.compile(r"^y(?:_|\[)?(?P<kp>\d+)\]?$")
_C_RE = re.compile(r"^(?:prob|conf|c)(?:_|\[)?(?P<kp>\d+)\]?$")
_PACKED_2D_COLS = ("pose_keypoints_2d", "keypoints_2d", "pose_keypoints")


def load_openpose_csv(
    path: str | Path,
    *,
    fps: float | None = None,
    subject: str | None = None,
    trial: str | None = None,
) -> pd.DataFrame:
    """
    Load an OpenPose CSV and return canonical long-form pose format.

    Supported input formats:
      1) Wide columns: xN,yN,probN (or x_N,y_N,c_N, etc.)
      2) Packed column: a single column like "pose_keypoints_2d" with 3K numbers per row

    Time base:
      - If "timestamp_ns" exists: seconds relative to first sample
      - Else if "dt_ms" exists: cumulative seconds (first sample at 0)
      - Else require fps=...

    Canonical output columns:
      t (seconds), kp (int), x, y, z (NaN), conf, optional subject/trial
    """
    path = Path(path)
    wide = pd.read_csv(path)

    t = _infer_timebase(wide, fps=fps)

    if _has_wide_triplets(wide.columns):
        df = _from_wide_triplets(wide, t)
    else:
        packed_col = _find_first_existing_col(wide.columns, _PACKED_2D_COLS)
        if packed_col is None:
            raise ValueError(
                "Unrecognized OpenPose CSV format. Expected wide xN/yN/probN columns "
                "or a packed keypoints column like 'pose_keypoints_2d'."
            )
        df = _from_packed_2d_series(wide[packed_col], t)

    if subject is not None:
        df["subject"] = subject
    if trial is not None:
        df["trial"] = trial

    df = coerce_pose_df(df)
    validate_pose_df(df)
    return df


def load_openpose_json_dir(
    dir_path: str | Path,
    *,
    fps: float,
    person: int | str = "best",
    subject: str | None = None,
    trial: str | None = None,
    glob_pattern: str = "*.json",
) -> pd.DataFrame:
    """
    Load an OpenPose JSON output directory and return canonical long-form pose format.

    Parameters
    ----------
    dir_path:
        Directory containing per-frame OpenPose JSON files.
    fps:
        Frames per second. Required (OpenPose JSON files usually do not include timestamps).
    person:
        - int: choose that person index from "people"
        - "best": choose person with max mean confidence per frame (default)
    subject, trial:
        Optional identifiers added to output.
    glob_pattern:
        Which JSON files to read (default "*.json").

    Returns
    -------
    Canonical long-form DataFrame with columns: t, kp, x, y, z (NaN), conf, optional subject/trial.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"OpenPose JSON directory not found: {dir_path}")

    files = sorted(dir_path.glob(glob_pattern))
    if not files:
        raise ValueError(
            f"No OpenPose JSON files found in {dir_path} (pattern={glob_pattern!r})"
        )

    if fps <= 0:
        raise ValueError("fps must be > 0")

    # time base from frame index
    t = pd.Series(np.arange(len(files), dtype=float) / float(fps))

    frames: list[pd.DataFrame] = []
    for i, fp in enumerate(files):
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        people = data.get("people", [])
        if not people:
            # No detections: create an empty frame (we'll just skip it)
            continue

        if isinstance(person, int):
            p = people[person] if person < len(people) else None
        elif person == "best":
            p = _pick_best_person_2d(people)
        else:
            raise ValueError("person must be an int or 'best'")

        if p is None:
            continue

        arr = p.get("pose_keypoints_2d", None)
        if arr is None:
            # sometimes it's named differently, but pose_keypoints_2d is the standard
            raise ValueError(f"Missing 'pose_keypoints_2d' in {fp.name}")

        xyconf = np.asarray(arr, dtype=float)
        if xyconf.size % 3 != 0:
            raise ValueError(
                f"pose_keypoints_2d length not divisible by 3 in {fp.name}"
            )

        k = xyconf.size // 3
        xyconf = xyconf.reshape(k, 3)

        frame_df = pd.DataFrame(
            {
                "t": float(t.iloc[i]),
                "kp": np.arange(k, dtype=int),
                "x": xyconf[:, 0],
                "y": xyconf[:, 1],
                "z": np.nan,
                "conf": xyconf[:, 2],
            }
        )
        frames.append(frame_df)

    if not frames:
        raise ValueError("No usable pose frames found (all frames empty or invalid).")

    df = pd.concat(frames, ignore_index=True)

    if subject is not None:
        df["subject"] = subject
    if trial is not None:
        df["trial"] = trial

    df = coerce_pose_df(df)
    validate_pose_df(df)
    return df


# -------------------------
# helpers
# -------------------------


def _find_first_existing_col(
    cols: Iterable[str], candidates: Iterable[str]
) -> str | None:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _infer_timebase(wide: pd.DataFrame, *, fps: float | None) -> pd.Series:
    if "timestamp_ns" in wide.columns:
        ts = pd.to_numeric(wide["timestamp_ns"], errors="raise")
        t0 = ts.iloc[0]
        return (ts - t0) / 1e9

    if "dt_ms" in wide.columns:
        dt = pd.to_numeric(wide["dt_ms"], errors="raise")
        if (dt < 0).any():
            raise ValueError("dt_ms must be nonnegative.")
        # dt describes time between samples: t[0]=0, t[i]=sum(dt[:i])
        return pd.Series(np.r_[0.0, np.cumsum(dt.to_numpy()[:-1]) / 1000.0])

    if fps is None:
        raise ValueError(
            "No time base found. Provide timestamp_ns or dt_ms column, or pass fps=..."
        )
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return pd.Series(np.arange(len(wide), dtype=float) / float(fps))


def _has_wide_triplets(cols: Iterable[str]) -> bool:
    # True if we can find at least one x, y, and confidence/prob column with same kp id
    kp_x = set()
    kp_y = set()
    kp_c = set()
    for c in cols:
        mx = _X_RE.match(c)
        if mx:
            kp_x.add(int(mx.group("kp")))
            continue
        my = _Y_RE.match(c)
        if my:
            kp_y.add(int(my.group("kp")))
            continue
        mc = _C_RE.match(c)
        if mc:
            kp_c.add(int(mc.group("kp")))
            continue
    return len(kp_x & kp_y & kp_c) > 0


def _from_wide_triplets(wide: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    # Collect kp ids that have x,y,conf
    kp_x = {}
    kp_y = {}
    kp_c = {}
    for c in wide.columns:
        mx = _X_RE.match(c)
        if mx:
            kp_x[int(mx.group("kp"))] = c
            continue
        my = _Y_RE.match(c)
        if my:
            kp_y[int(my.group("kp"))] = c
            continue
        mc = _C_RE.match(c)
        if mc:
            kp_c[int(mc.group("kp"))] = c
            continue

    kp_ids = sorted(set(kp_x) & set(kp_y) & set(kp_c))
    if not kp_ids:
        raise ValueError(
            "No complete (x,y,conf) keypoint triplets found in CSV columns."
        )

    out = []
    t_np = t.to_numpy()
    for kp in kp_ids:
        out.append(
            pd.DataFrame(
                {
                    "t": t_np,
                    "kp": kp,
                    "x": pd.to_numeric(wide[kp_x[kp]], errors="coerce").to_numpy(),
                    "y": pd.to_numeric(wide[kp_y[kp]], errors="coerce").to_numpy(),
                    "z": np.nan,
                    "conf": pd.to_numeric(wide[kp_c[kp]], errors="coerce").to_numpy(),
                }
            )
        )

    return pd.concat(out, ignore_index=True)


def _parse_packed_numbers(v: Any) -> np.ndarray:
    """
    Parse a packed 2D keypoints value:
      - list/tuple -> array
      - string like "[1,2,3,...]" or "1,2,3" -> array
    """
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=float)

    if isinstance(v, str):
        s = v.strip()
        # remove brackets if present
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if not s:
            return np.asarray([], dtype=float)
        # split on commas or whitespace
        parts = re.split(r"[,\s]+", s)
        parts = [p for p in parts if p]
        return np.asarray([float(p) for p in parts], dtype=float)

    # pandas might give us a scalar number in weird cases
    try:
        return np.asarray([float(v)], dtype=float)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Could not parse packed keypoints value: {v!r}") from e


def _from_packed_2d_series(packed: pd.Series, t: pd.Series) -> pd.DataFrame:
    frames = []
    for i, v in enumerate(packed.to_list()):
        arr = _parse_packed_numbers(v)
        if arr.size == 0:
            continue
        if arr.size % 3 != 0:
            raise ValueError(
                f"Packed 2D keypoints length not divisible by 3 at row {i} (len={arr.size})."
            )
        k = arr.size // 3
        xyconf = arr.reshape(k, 3)
        frames.append(
            pd.DataFrame(
                {
                    "t": float(t.iloc[i]),
                    "kp": np.arange(k, dtype=int),
                    "x": xyconf[:, 0],
                    "y": xyconf[:, 1],
                    "z": np.nan,
                    "conf": xyconf[:, 2],
                }
            )
        )

    if not frames:
        raise ValueError(
            "Packed keypoints column exists but contained no usable frames."
        )
    return pd.concat(frames, ignore_index=True)


def _pick_best_person_2d(people: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    best_score = -np.inf
    for p in people:
        arr = p.get("pose_keypoints_2d", None)
        if arr is None:
            continue
        xyconf = np.asarray(arr, dtype=float)
        if xyconf.size < 3 or xyconf.size % 3 != 0:
            continue
        conf = xyconf.reshape(-1, 3)[:, 2]
        score = float(np.nanmean(conf))
        if score > best_score:
            best_score = score
            best = p
    return best
