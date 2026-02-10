from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd

from pose_dynamics.features.schema import FacialConfig


def _pivot_xy(
    df_win: pd.DataFrame, time_col: str, x_col: str = "x", y_col: str = "y"
) -> pd.DataFrame:
    pivot = df_win.pivot_table(
        index=time_col, columns="keypoint", values=[x_col, y_col], aggfunc="mean"
    )
    # Normalize column labels so downstream helpers continue to look for ("x", kp)/("y", kp).
    if (x_col, y_col) != ("x", "y"):
        pivot = pivot.rename(columns={x_col: "x", y_col: "y"}, level=0)
    return pivot


def _xy_for_keypoint(
    pivot: pd.DataFrame, kp: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if ("x", kp) not in pivot.columns or ("y", kp) not in pivot.columns:
        return None
    x = pivot[("x", kp)].to_numpy(dtype=float)
    y = pivot[("y", kp)].to_numpy(dtype=float)
    return x, y


KeypointRef = Union[str, Sequence[str]]


def _xy_for_ref(
    pivot: pd.DataFrame, ref: KeypointRef | None
) -> tuple[np.ndarray, np.ndarray] | None:
    if ref is None:
        return None
    if isinstance(ref, str):
        return _xy_for_keypoint(pivot, ref)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for kp in ref:
        xy = _xy_for_keypoint(pivot, kp)
        if xy is None:
            continue
        xs.append(xy[0])
        ys.append(xy[1])
    if not xs or not ys:
        return None
    return np.nanmean(np.vstack(xs), axis=0), np.nanmean(np.vstack(ys), axis=0)


def _dist(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _vertical_gap(
    pivot: pd.DataFrame, upper: KeypointRef | None, lower: KeypointRef | None
) -> np.ndarray | None:
    xy_u = _xy_for_ref(pivot, upper)
    xy_l = _xy_for_ref(pivot, lower)
    if xy_u is None or xy_l is None:
        return None
    return np.abs(xy_u[1] - xy_l[1])


def _center_from_contour(
    pivot: pd.DataFrame, contour: Iterable[str]
) -> tuple[np.ndarray, np.ndarray] | None:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for kp in contour:
        xy = _xy_for_keypoint(pivot, kp)
        if xy is None:
            continue
        x, y = xy
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    x_stack = np.vstack(xs)
    y_stack = np.vstack(ys)
    return np.nanmean(x_stack, axis=0), np.nanmean(y_stack, axis=0)


def facial_feature_series(
    df_win: pd.DataFrame, time_col: str, cfg: FacialConfig
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if df_win.empty or not cfg.enabled:
        return out

    pivot = _pivot_xy(df_win, time_col)
    pivot_screen = None
    if {"x_screen", "y_screen"}.issubset(df_win.columns):
        pivot_screen = _pivot_xy(df_win, time_col, x_col="x_screen", y_col="y_screen")
    if pivot.empty:
        return out

    interocular: np.ndarray | None = None
    if cfg.scale_by_interocular and "interocular_screen" in df_win.columns:
        io_series = (
            df_win[[time_col, "interocular_screen"]]
            .groupby(time_col)["interocular_screen"]
            .mean()
            .reindex(pivot.index)
        )
        interocular = io_series.to_numpy(dtype=float)

    def _interocular_distance() -> np.ndarray | None:
        ref_pivot = pivot_screen if pivot_screen is not None else pivot
        cL = _center_from_contour(ref_pivot, cfg.pupil.left_eye_contour)
        cR = _center_from_contour(ref_pivot, cfg.pupil.right_eye_contour)
        if cL is None or cR is None:
            return None
        return _dist(cL[0], cL[1], cR[0], cR[1])

    if interocular is None and cfg.scale_by_interocular:
        interocular = _interocular_distance()

    def _maybe_scale(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        if interocular is None:
            return arr
        io = interocular
        valid = np.isfinite(io) & (io != 0)
        scaled = np.full_like(arr, np.nan, dtype=float)
        scaled[valid] = arr[valid] / io[valid]
        return scaled

    # Blink aperture (average across eyes)
    if cfg.blink.enabled:
        ref_pivot = pivot_screen if pivot_screen is not None else pivot
        left = None
        right = None
        if cfg.blink.left_upper and cfg.blink.left_lower:
            left = _vertical_gap(ref_pivot, cfg.blink.left_upper, cfg.blink.left_lower)
        if cfg.blink.right_upper and cfg.blink.right_lower:
            right = _vertical_gap(
                ref_pivot, cfg.blink.right_upper, cfg.blink.right_lower
            )
        if left is not None and right is not None:
            out["blink_aperture"] = _maybe_scale(
                np.nanmean(np.vstack([left, right]), axis=0)
            )
        elif left is not None:
            out["blink_aperture"] = _maybe_scale(left)
        elif right is not None:
            out["blink_aperture"] = _maybe_scale(right)

    # Mouth aperture
    if cfg.mouth.enabled and cfg.mouth.upper and cfg.mouth.lower:
        ref_pivot = pivot_screen if pivot_screen is not None else pivot
        xy_u = _xy_for_ref(ref_pivot, cfg.mouth.upper)
        xy_l = _xy_for_ref(ref_pivot, cfg.mouth.lower)
        if xy_u is not None and xy_l is not None:
            out["mouth_aperture"] = _maybe_scale(
                _dist(xy_u[0], xy_u[1], xy_l[0], xy_l[1])
            )

    # Pupil displacement
    if cfg.pupil.enabled:
        if cfg.pupil.left_pupil and cfg.pupil.left_eye_contour:
            center = _center_from_contour(pivot, cfg.pupil.left_eye_contour)
            pupil = _xy_for_ref(pivot, cfg.pupil.left_pupil)
            if center is not None and pupil is not None:
                dx = pupil[0] - center[0]
                dy = pupil[1] - center[1]
                out["pupil_left_dx"] = _maybe_scale(dx)
                out["pupil_left_dy"] = _maybe_scale(dy)
                out["pupil_left_mag"] = _maybe_scale(np.sqrt(dx**2 + dy**2))

        if cfg.pupil.right_pupil and cfg.pupil.right_eye_contour:
            center = _center_from_contour(pivot, cfg.pupil.right_eye_contour)
            pupil = _xy_for_ref(pivot, cfg.pupil.right_pupil)
            if center is not None and pupil is not None:
                dx = pupil[0] - center[0]
                dy = pupil[1] - center[1]
                out["pupil_right_dx"] = _maybe_scale(dx)
                out["pupil_right_dy"] = _maybe_scale(dy)
                out["pupil_right_mag"] = _maybe_scale(np.sqrt(dx**2 + dy**2))

        mags: List[np.ndarray] = []
        dx_components: List[np.ndarray] = []
        dy_components: List[np.ndarray] = []
        if "pupil_left_mag" in out:
            mags.append(out["pupil_left_mag"])
        if "pupil_right_mag" in out:
            mags.append(out["pupil_right_mag"])
        if "pupil_left_dx" in out:
            dx_components.append(out["pupil_left_dx"])
        if "pupil_right_dx" in out:
            dx_components.append(out["pupil_right_dx"])
        if "pupil_left_dy" in out:
            dy_components.append(out["pupil_left_dy"])
        if "pupil_right_dy" in out:
            dy_components.append(out["pupil_right_dy"])

        if mags:
            out["pupil_mag"] = np.nanmean(np.vstack(mags), axis=0)
        if dx_components:
            out["pupil_dx"] = np.nanmean(np.vstack(dx_components), axis=0)
        if dy_components:
            out["pupil_dy"] = np.nanmean(np.vstack(dy_components), axis=0)

    # Center-face dispersion (RMS spread of specified keypoints)
    if cfg.center_face:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for kp in cfg.center_face:
            xy = _xy_for_keypoint(pivot, kp)
            if xy is None:
                continue
            xs.append(xy[0])
            ys.append(xy[1])
        if xs and ys:
            x_mat = np.vstack(xs)
            y_mat = np.vstack(ys)
            mean_x = np.nanmean(x_mat, axis=1, keepdims=True)
            mean_y = np.nanmean(y_mat, axis=1, keepdims=True)
            dx = x_mat - mean_x
            dy = y_mat - mean_y
            mag = np.sqrt(dx**2 + dy**2)
            out["center_face_magnitude"] = _maybe_scale(
                np.sqrt(np.nanmean(mag**2, axis=0))
            )
            out["center_face_x"] = _maybe_scale(np.sqrt(np.nanmean(dx**2, axis=0)))
            out["center_face_y"] = _maybe_scale(np.sqrt(np.nanmean(dy**2, axis=0)))

    return out
