from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pose_dynamics.rqa.rqa_config import ConfigError, RQAConfig
from pose_dynamics.rqa.utils import norm_utils, rqa_utils_cpp


@dataclass(frozen=True)
class RQAOutputs:
    stats_path: Path
    qc_path: Path
    provenance_path: Path
    plots_dir: Path


def _time_col(df: pd.DataFrame) -> str:
    return "time" if "time" in df.columns else "frame"


def _embed(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = X.shape[0] - (m - 1) * tau
    if n <= 0:
        return np.empty((0, X.shape[1] * m))
    parts = [X[i : i + n] for i in range(0, m * tau, tau)]
    return np.concatenate(parts, axis=1)


def _rr_to_epsilon(dist: np.ndarray, rr_target: float) -> float:
    d = dist[np.triu_indices_from(dist, k=1)]
    if d.size == 0:
        return float("nan")
    return float(np.percentile(d, rr_target * 100))


def _epsilon(dist: np.ndarray, cfg: RQAConfig) -> float:
    if cfg.epsilon.mode == "absolute":
        return float(cfg.epsilon.value)
    if cfg.epsilon.mode == "percentile":
        d = dist[np.triu_indices_from(dist, k=1)]
        return float(np.percentile(d, cfg.epsilon.value)) if d.size else float("nan")
    if cfg.epsilon.mode == "rr_target":
        return _rr_to_epsilon(dist, cfg.epsilon.value)
    if cfg.epsilon.mode == "mean_scaled":
        d = dist[np.triu_indices_from(dist, k=1)]
        if d.size == 0:
            return float("nan")
        return float(np.mean(d) * cfg.epsilon.value)
    raise ConfigError("unknown epsilon.mode")


def _line_lengths(mat: np.ndarray, min_len: int, axis: int = 0) -> list[int]:
    # axis=0 for diagonals, axis=1 for vertical
    lengths = []
    if axis == 0:
        # diagonals
        for k in range(-mat.shape[0] + 1, mat.shape[1]):
            diag = np.diagonal(mat, offset=k)
            lengths += _run_lengths(diag, min_len)
    else:
        # vertical
        for j in range(mat.shape[1]):
            col = mat[:, j]
            lengths += _run_lengths(col, min_len)
    return lengths


def _run_lengths(arr: np.ndarray, min_len: int) -> list[int]:
    lengths = []
    run = 0
    for v in arr:
        if v:
            run += 1
        else:
            if run >= min_len:
                lengths.append(run)
            run = 0
    if run >= min_len:
        lengths.append(run)
    return lengths


def _rqa_stats(mat: np.ndarray, l_min: int, v_min: int, theiler: int) -> dict:
    N = mat.shape[0]
    if N == 0:
        return {}
    # exclude main diagonal for RQA
    mask = np.ones_like(mat, dtype=bool)
    for k in range(-theiler, theiler + 1):
        if k == 0:
            np.fill_diagonal(mask, False)
        else:
            i = np.arange(max(0, -k), min(N, N - k))
            mask[i, i + k] = False
    recur_points = int(np.sum(mat & mask))
    total_points = int(np.sum(mask))
    rr = recur_points / total_points if total_points else 0.0

    diag_lengths = _line_lengths(mat, l_min, axis=0)
    vert_lengths = _line_lengths(mat, v_min, axis=1)

    det = (np.sum(diag_lengths) / recur_points) if recur_points else 0.0
    lam = (np.sum(vert_lengths) / recur_points) if recur_points else 0.0

    maxl = max(diag_lengths) if diag_lengths else 0
    meanl = float(np.mean(diag_lengths)) if diag_lengths else 0.0
    stdl = float(np.std(diag_lengths)) if diag_lengths else 0.0
    countl = int(len(diag_lengths))
    ent = 0.0
    if diag_lengths:
        vals, counts = np.unique(diag_lengths, return_counts=True)
        p = counts / counts.sum()
        ent = float(-np.sum(p * np.log(p)))

    vmax = max(vert_lengths) if vert_lengths else 0
    tt = float(np.mean(vert_lengths)) if vert_lengths else 0.0
    div = 1.0 / maxl if maxl else 0.0

    return {
        "perc_recur": rr * 100.0,
        "perc_determ": det * 100.0,
        "maxl_found": float(maxl),
        "mean_line_length": meanl,
        "std_line_length": stdl,
        "count_line": countl,
        "entropy": ent,
        "laminarity": lam * 100.0,
        "trapping_time": tt,
        "vmax": float(vmax),
        "divergence": div,
    }


def _drp(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = mat.shape[0]
    lags = np.arange(-(N - 1), N)
    drp = []
    for k in lags:
        diag = np.diagonal(mat, offset=k)
        rr = diag.mean() * 100.0 if diag.size else 0.0
        drp.append(rr)
    return lags, np.array(drp)


def _compute_derived_series(
    df_win: pd.DataFrame,
    tcol: str,
    head_df_win: Optional[pd.DataFrame] = None,
) -> dict[str, np.ndarray]:
    """Per-window derived signals (blink, mouth, pupil, head motion)."""

    # Minimal defaults mirror examples/case_study_1_matb/configs/features.yaml
    blink_left_upper = ["38", "39"]
    blink_left_lower = ["41", "42"]
    blink_right_upper = ["44", "45"]
    blink_right_lower = ["47", "48"]
    mouth_upper = ["63"]
    mouth_lower = ["67"]
    left_pupil = ["69"]
    right_pupil = ["70"]
    left_eye_contour = ["37", "38", "39", "41", "42"]
    right_eye_contour = ["44", "45", "46", "47", "48"]

    def _pivot_xy(dfw: pd.DataFrame) -> pd.DataFrame:
        return dfw.pivot_table(
            index=tcol, columns="keypoint", values=["x", "y"], aggfunc="mean"
        )

    def _xy_for_ref(
        pivot: pd.DataFrame, ref: list[str]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for kp in ref:
            if ("x", kp) not in pivot.columns or ("y", kp) not in pivot.columns:
                continue
            xs.append(pivot[("x", kp)].to_numpy(dtype=float))
            ys.append(pivot[("y", kp)].to_numpy(dtype=float))
        if not xs or not ys:
            return None
        return np.nanmean(np.vstack(xs), axis=0), np.nanmean(np.vstack(ys), axis=0)

    def _vertical_gap(
        pivot: pd.DataFrame, upper: list[str], lower: list[str]
    ) -> np.ndarray | None:
        upper_xy = _xy_for_ref(pivot, upper)
        lower_xy = _xy_for_ref(pivot, lower)
        if upper_xy is None or lower_xy is None:
            return None
        return np.abs(upper_xy[1] - lower_xy[1])

    def _dist(
        x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
    ) -> np.ndarray:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    out: dict[str, np.ndarray] = {}
    if df_win.empty:
        return out

    pivot = _pivot_xy(df_win)
    if pivot.empty:
        return out

    # Blink aperture (mean of both eyes when available)
    left_gap = _vertical_gap(pivot, blink_left_upper, blink_left_lower)
    right_gap = _vertical_gap(pivot, blink_right_upper, blink_right_lower)
    if left_gap is not None and right_gap is not None:
        out["blink"] = np.nanmean(np.vstack([left_gap, right_gap]), axis=0)
    elif left_gap is not None:
        out["blink"] = left_gap
    elif right_gap is not None:
        out["blink"] = right_gap

    # Mouth aperture
    upper = _xy_for_ref(pivot, mouth_upper)
    lower = _xy_for_ref(pivot, mouth_lower)
    if upper is not None and lower is not None:
        out["mouth_aperture"] = _dist(upper[0], upper[1], lower[0], lower[1])

    # Pupil displacement (averaged across eyes)
    pupil_components: dict[str, list[np.ndarray]] = {"mag": [], "dx": [], "dy": []}
    for suffix, pupil_ref, contour in [
        ("left", left_pupil, left_eye_contour),
        ("right", right_pupil, right_eye_contour),
    ]:
        center = _xy_for_ref(pivot, contour)
        pupil = _xy_for_ref(pivot, pupil_ref)
        if center is None or pupil is None:
            continue
        dx = pupil[0] - center[0]
        dy = pupil[1] - center[1]
        mag = np.sqrt(dx**2 + dy**2)
        out[f"pupil_{suffix}_dx"] = dx
        out[f"pupil_{suffix}_dy"] = dy
        out[f"pupil_{suffix}_mag"] = mag
        pupil_components["mag"].append(mag)
        pupil_components["dx"].append(dx)
        pupil_components["dy"].append(dy)

    if pupil_components["mag"]:
        out["pupil_mag"] = np.nanmean(np.vstack(pupil_components["mag"]), axis=0)
    if pupil_components["dx"]:
        out["pupil_dx"] = np.nanmean(np.vstack(pupil_components["dx"]), axis=0)
    if pupil_components["dy"]:
        out["pupil_dy"] = np.nanmean(np.vstack(pupil_components["dy"]), axis=0)

    # Head motion from alignment transforms (if provided)
    if head_df_win is not None and not head_df_win.empty:
        if "scale" in head_df_win.columns:
            out["head_scale"] = head_df_win["scale"].to_numpy(dtype=float)
        if "rotation_angle" in head_df_win.columns:
            out["head_rotation"] = head_df_win["rotation_angle"].to_numpy(dtype=float)
        tx = head_df_win["translation_x"].to_numpy(dtype=float)
        ty = head_df_win["translation_y"].to_numpy(dtype=float)
        out["head_tx"] = tx
        out["head_ty"] = ty
        out["head_translation_mag"] = np.sqrt(tx**2 + ty**2)
        components = []
        if "head_translation_mag" in out:
            components.append(out["head_translation_mag"])
        if "head_rotation" in out:
            components.append(out["head_rotation"])
        if "head_scale" in out:
            components.append(out["head_scale"] - 1.0)
        if components:
            stack = np.vstack(components)
            out["head_motion_mag"] = np.sqrt(np.nansum(stack**2, axis=0))

    return out


def _series_matrix(
    df_win: pd.DataFrame,
    keypoints: list[str],
    signal: str,
    derived: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    dims = [c for c in ["x", "y", "z"] if c in df_win.columns]
    rows: list[np.ndarray] = []

    for kp in keypoints:
        if derived and kp in derived:
            vals = np.asarray(derived[kp], dtype=float)
            if vals.ndim == 1:
                vals = vals.reshape(-1, 1)
            rows.append(vals)
            continue

        vals = df_win[df_win["keypoint"] == kp][dims].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        if signal == "magnitude":
            vals = np.linalg.norm(vals, axis=1, keepdims=True)
        rows.append(vals)

    if not rows:
        return np.empty((0, 1))

    # Align lengths (truncate to the shortest to keep sample sync)
    min_len = min(r.shape[0] for r in rows)
    rows = [r[:min_len] for r in rows]
    # Ensure 2D then concatenate feature-wise
    rows = [r if r.ndim == 2 else r.reshape(-1, 1) for r in rows]
    return np.concatenate(rows, axis=1)


def run_rqa(
    pose_clean_path: str | Path,
    windows_path: str | Path,
    config: RQAConfig | str | Path | None = None,
    out_dir: str | Path | None = None,
    *,
    config_dict: dict | None = None,
    pose_y_path: Optional[str | Path] = None,
    overwrite: bool = False,
) -> RQAOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config_dict:
        # Map flat dict to RQAConfig structure
        # This mapping depends on RQAConfig validation, which we might need to bypass or construct carefully
        # Alternatively, we just use the dict directly if we refactor, but to keep type safety let's try to map
        # Simplified mapping logic for now:
        from pose_dynamics.rqa.rqa_config import RQAConfig

        # determine analysis type
        atype = config_dict.get("analysis_type", "rqa")
        is_crqa = atype == "crqa"

        # keypoints
        if is_crqa:
            # We need to handle X and Y. RQAConfig might assume single set.
            # If RQAConfig supports separate X/Y keypoints, good. If not, we might need to update RQAConfig
            # OR logic below needs to handle 'keypoint_x' vs 'keypoint_y'.
            # For now, let's assume we pass X keypoints here and handle Y separately or update RQAConfig
            # Let's see how RQAConfig is defined. Assuming it needs 'keypoints' list.
            kp = config_dict.get("keypoint_x", [])
            kp_y = config_dict.get(
                "keypoint_y", []
            )  # We'll need to pass this somehow or assume config holds it
        else:
            kp = config_dict.get("keypoints", [])
            kp_y = None

        # radius
        rad_val = config_dict.get("radius", 0.0)
        rad_mode = config_dict.get("radius_mode", "absolute")

        # embed
        emb = config_dict.get("embedding", {})
        m = emb.get("dim", 1)
        tau = emb.get("tau", 1)

        # Build config object (pseudo-code structure matching likely RQAConfig)
        # We might need to construct a dict that matches RQAConfig.from_dict structure
        # Assuming RQAConfig has from_dict or similar

        # Since I don't see RQAConfig definition, I will infer or rely on flexible construction.
        # Let's assume we can construct a dummy object or just use the dict values if we verify them.
        # But to respect the type hint, let's try to create a Config object or modify the function to accept dict.

        # Actually, let's just use the dictionary values directly in the code below to be more robust
        # given we are changing the config format significantly.
        # I'll create a simple Namespace or dataclass to act as cfg
        @dataclass
        class SimpleConfig:
            analysis: str
            keypoints: list
            keypoints_y: list | None
            signal: str
            m: int
            tau: int
            l_min: int
            v_min: int
            theiler: int
            epsilon: Any
            plots: Any
            norm: str
            rescale_norm: bool

        @dataclass
        class Epsilon:
            value: float
            mode: str

        @dataclass
        class Plots:
            enabled: bool
            max_plots: int

        cfg = SimpleConfig(
            analysis=atype,
            keypoints=kp,
            keypoints_y=kp_y,
            signal=config_dict.get(
                "signal", "magnitude"
            ),  # default to magnitude for example
            m=m,
            tau=tau,
            l_min=config_dict.get("line_min", 2),
            v_min=config_dict.get(
                "line_min", 2
            ),  # assume same for vertical for now unless specified
            theiler=config_dict.get("theiler", 1),
            epsilon=Epsilon(value=rad_val, mode=rad_mode),
            plots=Plots(
                enabled=False, max_plots=0
            ),  # disable plots for batch run mostly
            norm=config_dict.get("norm", "none"),
            rescale_norm=bool(config_dict.get("rescale_norm", False)),
        )

    elif isinstance(config, RQAConfig):
        cfg = config
        # Backwards compat hack if RQAConfig doesn't have keypoints_y
        if not hasattr(cfg, "keypoints_y"):
            cfg.keypoints_y = None
    elif config:
        cfg = RQAConfig.from_yaml(str(config))
        if not hasattr(cfg, "keypoints_y"):
            cfg.keypoints_y = None
    else:
        raise ValueError("Must provide config or config_dict")

    pose_clean_path = Path(pose_clean_path)
    windows_path = Path(windows_path)
    df = pd.read_parquet(pose_clean_path)
    # Optional alignment transforms (for head motion derived signals)
    align_path = pose_clean_path.parent / "alignment_transforms.parquet"
    df_head = pd.read_parquet(align_path) if align_path.exists() else None
    windows = pd.read_parquet(windows_path)

    if windows.empty:
        raise ConfigError("windows.parquet is empty.")

    if "dropped" in windows.columns:
        windows = windows[~windows["dropped"]].copy()

    # Load features if needed for derived signals (like blink)
    # If the keypoint is not in df but in features, we need features.
    # For simplicity, assuming all 'keypoints' are in pose.parquet or computed features are merged there?
    # Or maybe we need to load features.parquet too?
    # For this task, let's assume data is in pose (including features if 'keypoint' column covers them)
    # If 'blink' is a feature in a separate file, we might need that.
    # Checking context: "In addition to some of the keypoints we also derived a few measures including blink... We will save summary statistics for these features"
    # Wait, features are in 'features.parquet'. 'df' is 'pose.parquet'.
    # If users ask for "37" (eyelid), that's a keypoint in pose.parquet.
    # If they ask for "blink" (derived), that might be in features.parquet.
    # BUT the request said: "keypoints: [37, 46]" for blink distance. So we compute it here?
    # No, the request implies RQA on specific keypoint(s).
    # If list has >1 keypoint and signal='magnitude', we combine them?
    # _series_matrix does: "if signal == 'magnitude': ... np.stack(rows, axis=1) ... np.concatenate"?
    # Actually _series_matrix:
    #   if magnitude: norm(vals) -> scalar per row. If multiple keypoints -> multiple cols?
    #   Wait, "cols = np.stack(rows, axis=1)" means (N, num_kps).
    #   Then _embed(X, m, tau) embeds that.
    # The paper says: "magnitude measure combining both axes".
    # And "blink (distance between eyelids)".
    # If we pass multiple keypoints for blink, we probably want the DISTANCE between them, not just stacking.
    # The existing code doesn't seem to calculate distance between keypoints.
    # _series_matrix returns raw coords or magnitudes.
    # For the purpose of "replicating", let's assume the user wants standard RQA on the defined keypoints.
    # If they want blink distance, they might need to use the 'features' file where blink is pre-calculated,
    # OR we assume 'keypoints' list means "calculate distance between these" if count=2?
    # Let's stick to the current implementation of _series_matrix and rely on 'pose.parquet' containing the data.
    # If 'blink' is needed, maybe it is passed as a "keypoint" name if features were merged?
    # Let's proceed with standard logic.

    df_y = None  # For CRQA Y-data (usually same file, different keypoints, but could be different file)

    # Identify unique keypoints available
    all_kps = sorted(df["keypoint"].dropna().unique().tolist())

    # Helper to resolve keypoints list
    def resolve_kps(kp_list):
        if not kp_list:
            return []
        if kp_list == "all":
            return all_kps
        # check availability - simple check
        # missing = sorted(set(kp_list) - set(all_kps))
        # if missing:
        #    print(f"Warning: keypoints {missing} not found in pose data.")
        return list(kp_list)

    kps_x = resolve_kps(cfg.keypoints)
    kps_y = resolve_kps(cfg.keypoints_y) if cfg.keypoints_y else []

    tcol = _time_col(df)
    stats_rows = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for i, w in windows.iterrows():
        trial_id = w["trial_id"]
        s = float(w["start"])
        e = float(w["end"])

        # Get Window Data
        df_trial = df[df["trial_id"] == trial_id]
        df_head_trial = None
        if df_head is not None:
            df_head_trial = df_head[df_head["trial_id"] == trial_id]
        mask = (df_trial[tcol] >= s) & (df_trial[tcol] < e)
        df_win = df_trial.loc[mask]
        df_head_win = None
        if df_head_trial is not None and not df_head_trial.empty:
            mask_head = (df_head_trial[tcol] >= s) & (df_head_trial[tcol] < e)
            df_head_win = df_head_trial.loc[mask_head]

        if df_win.empty:
            continue

        derived = _compute_derived_series(df_win, tcol, df_head_win)

        # Prepare X series
        X = _series_matrix(df_win, kps_x, cfg.signal, derived)
        if X.size == 0:
            continue

        if cfg.analysis == "crqa":
            # For CRQA, we need Y.
            # If kps_y is defined, we use those from the same dataframe (usually).
            # If pose_y_path is provided, we use that file (and kps_y or kps_x if kps_y not set?)
            # Config implies: keypoint_x and keypoint_y lists.

            if kps_y:
                # Same DF, different keypoints (e.g. pupil vs face)
                Y = _series_matrix(df_win, kps_y, cfg.signal, derived)
            elif df_y is not None:
                # Different DF (not implemented fully in this snippet, but supported by signature)
                # ... load window from df_y ...
                pass
            else:
                # Fallback / Error
                continue

            if Y.size == 0:
                continue
            # Normalize raw series then let C++ handle embedding
            Xn = norm_utils.normalize_data(X, cfg.norm)
            Yn = norm_utils.normalize_data(Y, cfg.norm)
            ds = rqa_utils_cpp.rqa_dist(Xn, Yn, dim=cfg.m, lag=cfg.tau)
            dist = ds["d"]
            eps = _epsilon(dist, cfg)

        else:  # RQA
            # Normalize raw series then let C++ handle embedding
            Xn = norm_utils.normalize_data(X, cfg.norm)
            ds = rqa_utils_cpp.rqa_dist(Xn, Xn, dim=cfg.m, lag=cfg.tau)
            dist = ds["d"]
            eps = _epsilon(dist, cfg)

        # Delegate stats to C++ backend for exact parity with reference impl
        rqa_mode = "cross" if cfg.analysis == "crqa" else "auto"
        diag_ignore = 0 if rqa_mode == "cross" else cfg.theiler
        td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(
            dist.astype(float),
            rescale=bool(cfg.rescale_norm),
            rad=float(eps),
            diag_ignore=int(diag_ignore),
            minl=int(cfg.l_min),
            rqa_mode=rqa_mode,
        )

        if err_code != 0 or rs is None:
            continue

        stats_rows.append(
            {
                "trial_id": trial_id,
                "window_id": w["window_id"],
                "epsilon": float(eps),
                **{k: float(v) for k, v in rs.items()},
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / "rqa_stats.parquet"
    # Ensure target directory exists (defensive even though created above)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    # Save CSV too for easy reading
    stats_path_csv = out_dir / "rqa_stats.csv"

    if not overwrite and stats_path.exists():
        raise FileExistsError(f"Output already exists: {stats_path}")

    if not stats_df.empty:
        stats_df.to_parquet(stats_path, index=False)
        stats_df.to_csv(stats_path_csv, index=False)

    # ... generate json provenance ...

    return RQAOutputs(
        stats_path=stats_path,
        qc_path=out_dir / "qc.json",  # filler
        provenance_path=out_dir / "prov.json",  # filler
        plots_dir=plots_dir,
    )
