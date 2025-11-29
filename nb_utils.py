"""Notebook and analysis utilities for pose data exploration.

Provides convenience functions for interactive data analysis in Jupyter notebooks,
including file discovery, data manipulation, visualization, and statistical analysis.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
from .src.pose_dynamics.nonlinear.state_space_recon import fnn,ami,cross_ami
from ..stats.stats_utils import pretty_metric, fmt

# ---------- Output file discovery and status checking ----------
def outputs_exist(base: str | Path) -> dict:
    """Check which pipeline outputs exist on disk.

    Scans the output directory structure to determine what analysis
    results are available from previous pipeline runs.

    Args:
        base: Base output directory path

    Returns:
        Dictionary with structure:
        {
            'per_frame': {source: [list_of_csv_files]},
            'linear': {source: boolean},
            'any_per_frame': boolean,
            'any_linear': boolean
        }

    Note:
        Useful for notebooks to determine what data is available
        without re-running expensive pipeline stages.
    """
    base = Path(base)

    # Check for per-frame feature files (individual trial CSVs)
    per_frame = {
        "procrustes_global": list((base / "features" / "per_frame" / "procrustes_global").glob("*.csv")),
        "procrustes_participant": list((base / "features" / "per_frame" / "procrustes_participant").glob("*.csv")),
        "original": list((base / "features" / "per_frame" / "original").glob("*.csv")),
    }

    # Check for aggregated linear metric files
    linear = {
        "procrustes_global": (base / "linear_metrics" / "procrustes_global_linear.csv").exists(),
        "procrustes_participant": (base / "linear_metrics" / "procrustes_participant_linear.csv").exists(),
        "original": (base / "linear_metrics" / "original_linear.csv").exists(),
    }

    # Summary flags for quick checking
    any_per_frame = any(len(v) > 0 for v in per_frame.values())
    any_linear = any(linear.values())

    return {"per_frame": per_frame, "linear": linear,
            "any_per_frame": any_per_frame, "any_linear": any_linear}

# ---------- File selection utilities ----------
def pick_norm_file(out_base: str | Path, sample_norm: str | Path | None = None) -> Path:
    """Select a normalized CSV file for analysis.

    Provides automatic file selection when exploring pipeline outputs.
    If no specific file is provided, picks the first available normalized file.

    Args:
        out_base: Base output directory
        sample_norm: Specific file path, or None for automatic selection

    Returns:
        Path to a normalized CSV file

    Raises:
        FileNotFoundError: If no normalized files exist

    Note:
        Useful for notebooks where you want to quickly load some
        example data without specifying exact filenames.
    """
    # Use specific file if provided
    if sample_norm:
        return Path(sample_norm)

    # Otherwise, find first available normalized file
    norm_dir = Path(out_base) / "norm_screen"
    files = sorted(norm_dir.glob("*_norm.csv"))

    if not files:
        raise FileNotFoundError(f"No normalized CSVs in {norm_dir}. Run the pipeline first.")

    return files[0]

# ---------- Column access utilities ----------
def find_col(df: pd.DataFrame, axis: str, i: int) -> str | None:
    """Find column name for a landmark coordinate.

    Handles case variations in column naming (x1 vs X1).

    Args:
        df: DataFrame to search
        axis: Coordinate axis ('x' or 'y')
        i: Landmark index number

    Returns:
        Column name if found, None otherwise

    Example:
        >>> find_col(df, 'x', 37)  # Returns 'x37' or 'X37' if present
    """
    c1, c2 = f"{axis}{i}", f"{axis.upper()}{i}"  # Try both cases
    return c1 if c1 in df.columns else (c2 if c2 in df.columns else None)


def series_num(df: pd.DataFrame, axis: str, i: int, n: int) -> pd.Series:
    """Extract landmark coordinate as numeric series.

    Safely extracts and converts landmark coordinates to float,
    returning NaN series if column doesn't exist.

    Args:
        df: DataFrame containing landmark data
        axis: Coordinate axis ('x' or 'y')
        i: Landmark index number
        n: Length for NaN series if column missing

    Returns:
        Numeric series with coordinate values or NaNs
    """
    c = find_col(df, axis, i)
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series([np.nan]*n)

# ---------- Data slicing utilities ----------
def slice_first_seconds(df: pd.DataFrame, fps: int, seconds: int) -> pd.DataFrame:
    """Extract first N seconds of data.

    Useful for quick analysis of data beginnings or creating
    consistent-length samples for comparison.

    Args:
        df: DataFrame with time-series data
        fps: Frames per second (sampling rate)
        seconds: Number of seconds to extract

    Returns:
        DataFrame containing only the first N seconds of data

    Note:
        Resets index to start from 0 for convenience.
    """
    n = len(df)
    end = min(n, fps * seconds)  # Don't exceed available data
    return df.iloc[:end].reset_index(drop=True)

# ---------- Metrics analysis and plotting utilities ----------
# Columns that contain metadata rather than actual measurements
META_COLS = {"source","participant","condition","window_index","t_start_frame","t_end_frame"}

def ensure_condition_order(df: pd.DataFrame, cond_order=("L","M","H")) -> pd.DataFrame:
    """Ensure condition column has consistent ordering.

    Converts condition column to ordered categorical for consistent
    plotting and statistical analysis.

    Args:
        df: DataFrame with 'condition' column
        cond_order: Desired order of conditions (default: Low, Medium, High)

    Returns:
        DataFrame with condition as ordered categorical

    Note:
        Modifies the input DataFrame in-place for the condition column.
    """
    if "condition" in df.columns:
        df["condition"] = pd.Categorical(df["condition"], categories=list(cond_order), ordered=True)
    return df

def candidate_metric_cols(df: pd.DataFrame) -> List[str]:
    """Identify numeric columns suitable for analysis.

    Filters out metadata columns and prioritizes derived metrics
    (velocity, acceleration, RMS) over raw measurements.

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names, with priority metrics first

    Note:
        Priority is given to time-domain metrics (_mean_abs_vel,
        _mean_abs_acc, _rms) as these often show clearer condition effects.
    """
    # Find numeric columns that aren't metadata
    num_cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]

    priority, others = [], []
    for c in num_cols:
        lc = c.lower()
        # Prioritize derived time-domain metrics
        if lc.endswith("_mean_abs_vel") or lc.endswith("_mean_abs_acc") or lc.endswith("_rms"):
            priority.append(c)
        else:
            others.append(c)

    return priority + others  # Priority metrics first

def default_metric(cols: List[str]) -> str | None:
    """Select a sensible default metric for analysis.

    Chooses metrics that typically show clear condition effects,
    preferring blink and mouth RMS metrics.

    Args:
        cols: List of available metric column names

    Returns:
        Recommended metric name, or None if no columns available

    Note:
        Selection priority:
        1. Specific high-impact metrics (blink_aperture_rms, etc.)
        2. Any RMS metric
        3. First available metric
    """
    if not cols:
        return None

    lowers = [c.lower() for c in cols]

    # Preferred metrics known to show strong condition effects
    prefs = [
        "blink_aperture_rms", "mouth_aperture_rms", "center_face_magnitude_rms",
        "blink_aperture_mean_abs_vel", "mouth_aperture_mean_abs_vel",
    ]

    # First, try exact matches with preferred metrics
    for exact in prefs:
        if exact in lowers:
            return cols[lowers.index(exact)]

    # Second, try any RMS metric (good for variability analysis)
    for i, lc in enumerate(lowers):
        if lc.endswith("_rms"):
            return cols[i]

    # Finally, just use the first available metric
    return cols[0]

def sem(series) -> float:
    """Calculate standard error of the mean.

    Args:
        series: Numeric data series

    Returns:
        Standard error of the mean

    Note:
        Uses sample standard deviation (ddof=1) and protects
        against division by zero with max(count, 1).
    """
    s = pd.Series(series).astype(float)
    return s.std(ddof=1) / np.sqrt(max(s.count(), 1))

def bar_by_condition(df: pd.DataFrame, metric: str, cond_order=("L","M","H"),
                     colors=("#1f77b4","#f1c40f","#8B0000"), title_suffix: str = ""):
    """Create bar plot comparing metric across conditions.

    Generates publication-ready bar plot with error bars showing
    mean ± SEM for each experimental condition.

    Args:
        df: DataFrame with 'condition' column and metric data
        metric: Name of the metric column to plot
        cond_order: Order of conditions (default: L, M, H)
        colors: Colors for each condition bar
        title_suffix: Additional text for plot title

    Returns:
        Tuple of (figure, axis) objects

    Note:
        - Error bars represent standard error of the mean (SEM)
        - Values are displayed on top of each bar
        - Grid lines help with value reading
    """
    # Ensure consistent condition ordering
    df = ensure_condition_order(df, cond_order)

    # Calculate means and standard errors by condition
    grouped = df.groupby("condition")[metric].agg(["mean", sem]).reindex(cond_order)
    idx = np.arange(len(cond_order))  # Bar positions
    means = grouped["mean"].to_numpy(dtype=float)
    errs  = grouped["sem"].to_numpy(dtype=float)

    # Create figure and plot bars
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(idx, means, yerr=errs, capsize=4, width=0.75,
           color=list(colors), edgecolor="black", alpha=0.9)

    # Configure axes
    ax.set_xticks(idx)
    ax.set_xticklabels(cond_order)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean ± SEM")

    # Set title
    ttl = f"{metric} by Condition" + (f" — {title_suffix}" if title_suffix else "")
    ax.set_title(ttl)
    ax.set_xlim(-0.5, len(cond_order)-0.5)

    # Format y-axis and add grid
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.grid(axis="y", alpha=0.25)

    # Add value labels on bars
    for x, m in zip(idx, means):
        if np.isfinite(m):
            ax.text(x, m, f"{m:.3f}", ha="center", va="bottom",
                   fontsize=9, fontweight="bold")

    plt.tight_layout()
    return fig, ax

# ---------------- Helpers for AMI batch computation ----------------

def _valid_recordings_with_series(df_all: pd.DataFrame, feature: str, max_lag: int) -> List[Tuple[str, int]]:
    """
    Return list of (recording_id, length_of_valid_series) where the feature exists,
    is numeric, and has enough finite samples for AMI up to max_lag.
    """
    out = []
    if feature not in df_all.columns:
        return out
    for rid, g in df_all.groupby("recording_id"):
        x = pd.to_numeric(g[feature], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if len(x) >= 2 * max_lag:
            out.append((rid, len(x)))
    return out

def pick_recordings(df_all: pd.DataFrame, feature: str, max_lag: int,
                    sample_n: int, strategy: str = "longest", seed: Optional[int] = 42) -> List[str]:
    """
    Choose up to sample_n recording_ids for a feature.
    strategy: "longest" | "random"
    """
    cands = _valid_recordings_with_series(df_all, feature, max_lag)
    if not cands:
        return []
    if strategy == "random":
        rng = random.Random(seed)
        rids = [rid for rid, _ in cands]
        rng.shuffle(rids)
        return rids[:sample_n]
    # default: longest
    cands.sort(key=lambda t: t[1], reverse=True)
    return [rid for rid, _ in cands[:sample_n]]

def compute_ami_for_feature(df_all: pd.DataFrame, feature: str, min_lag: int, max_lag: int,
                            recording_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Compute AMI curves for a feature across selected recording_ids.
    Returns dict: recording_id -> DataFrame(lag, ami).
    """
    curves = {}
    for rid in recording_ids:
        g = df_all[df_all["recording_id"] == rid]
        x = pd.to_numeric(g[feature], errors="coerce")
        arr = ami(x, min_lag=min_lag, max_lag=max_lag)
        if arr is None:
            continue
        curves[rid] = pd.DataFrame({"lag": arr[:, 0].astype(int), "ami": arr[:, 1]})
    return curves

def average_ami(curves: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Average AMI over recordings (inner-join by lag).
    """
    if not curves:
        return None
    dfs = []
    for rid, dfc in curves.items():
        tmp = dfc.copy()
        tmp = tmp.rename(columns={"ami": f"ami__{rid}"})
        dfs.append(tmp)
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on="lag", how="inner")
    if base.empty:
        return None
    ami_cols = [c for c in base.columns if c.startswith("ami__")]
    base["ami_mean"] = base[ami_cols].mean(axis=1)
    base["ami_sem"]  = base[ami_cols].sem(axis=1, ddof=1)
    return base[["lag", "ami_mean", "ami_sem"]].copy()

def first_local_minimum(
    y: np.ndarray,
    min_lag: int = 1,
    lags: np.ndarray | None = None,
    smooth_win: int = 5,
    min_prominence: float = 0.05,
    rel_drop: float = 0.10,
    allow_global_fallback: bool = True,
    cutoff_bits: float | None = 1.0,   # <-- NEW: hard AMI threshold (bits). Set None to disable.
) -> Optional[int]:
    """
    Robust AMI tau finder:
      1) Prefer first lag where smoothed AMI <= cutoff_bits (default 1.0 bit).
      2) Otherwise, return first meaningful local minimum (prominence/relative-drop guarded).
      3) Otherwise, optionally return global minimum if it passes the same test.
    """
    if y is None or len(y) < 3:
        return None

    # convert to float and mask non-finite
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return None

    # simple moving-average smoothing
    if smooth_win is None or smooth_win <= 1:
        ys = y.copy()
    else:
        w = int(max(1, int(smooth_win)))
        if w % 2 == 0:
            w += 1
        pad = w // 2
        y_pad = np.pad(y, pad_width=pad, mode="edge")
        kernel = np.ones(w) / w
        ys = np.convolve(y_pad, kernel, mode="valid")

    L = len(ys)
    if min_lag < 1:
        min_lag = 1
    if min_lag >= L - 1:
        return None

    # ---- 1) Hard threshold criterion --------------------------------------
    if cutoff_bits is not None:
        for i in range(min_lag, L):
            if np.isfinite(ys[i]) and ys[i] <= float(cutoff_bits):
                return int(lags[i]) if (lags is not None and len(lags) == L) else i

    # ---- 2) Meaningful first local minimum --------------------------------
    early_n = max(1, min(min_lag, L // 4))
    early_level = np.nanmean(ys[:early_n])

    abs_thresh = float(min_prominence)
    rel_thresh = float(rel_drop) * max(abs(early_level), 1e-12)

    def is_meaningful_drop(val):
        drop = early_level - val
        return (drop >= abs_thresh) or (drop >= rel_thresh)

    for i in range(min_lag + 1, L - 1):
        if not (np.isfinite(ys[i-1]) and np.isfinite(ys[i]) and np.isfinite(ys[i+1])):
            continue
        # strict/flat local min
        if ((ys[i] < ys[i-1] and ys[i] <= ys[i+1]) or
            (ys[i] <= ys[i-1] and ys[i] < ys[i+1]) or
            (ys[i] == ys[i-1] == ys[i+1])):
            if is_meaningful_drop(ys[i]):
                return int(lags[i]) if (lags is not None and len(lags) == L) else i

    # ---- 3) Global-min fallback -------------------------------------------
    if allow_global_fallback:
        idx = int(np.nanargmin(ys))
        if idx >= min_lag and is_meaningful_drop(ys[idx]):
            return int(lags[idx]) if (lags is not None and len(lags) == L) else idx

    return None

def ami_average_across_features(
    df_all: pd.DataFrame,
    features: List[str],
    sample_n: int = 10,
    min_lag: int = 1,
    max_lag: int = 140,
    strategy: str = "longest",
    seed: int = 42,
    norm_kind: Optional[str] = None,
    plot: bool = True,
    ncols: int = 2,
    title_suffix: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute AMI per-feature (average across recordings) and a combined average across features.

    Returns a dict:
      {
        "per_feature_curves": { feature: [df_curve_rec1, df_curve_rec2, ...], ... },
        "per_feature_avg":    { feature: df_avg, ... },
        "combined_avg":       df_combined_or_None,
        "summary_lines":      [str, ...]
      }

    Notes:
      - `pick_recordings`, `compute_ami_for_feature`, and `average_ami` are expected to
        exist in utils.nl_utils and follow the repo's existing signatures.
      - combined average is equal-weight across features (mean of per-feature averaged curves).
    """
    per_feature_curves: Dict[str, List[pd.DataFrame]] = {}
    per_feature_avg: Dict[str, Optional[pd.DataFrame]] = {}
    summary_lines: List[str] = []

    if verbose:
        feat_names_short = ", ".join(features)
        print(f"Doing AMI on {sample_n} recordings per feature for: {feat_names_short} — plotting." if plot else
              f"Doing AMI on {sample_n} recordings per feature for: {feat_names_short}.")

    for feat in features:
        # pick recordings for this feature
        rids = pick_recordings(df_all, feat, max_lag=max_lag, sample_n=sample_n,
                               strategy=strategy, seed=seed)
        if not rids:
            summary_lines.append(f"[{feat}] No recordings with length ≥ {2*max_lag} frames. Skipped.")
            per_feature_curves[feat] = []
            per_feature_avg[feat] = None
            continue

        # compute AMI curves for each recording (list of DataFrames or arrays depending on repo)
        curves = compute_ami_for_feature(df_all, feat, min_lag=min_lag, max_lag=max_lag, recording_ids=rids)
        if not curves:
            summary_lines.append(f"[{feat}] compute_ami_for_feature returned empty. Skipped.")
            per_feature_curves[feat] = []
            per_feature_avg[feat] = None
            continue

        # average across recordings for this feature
        avg = average_ami(curves)
        per_feature_curves[feat] = curves
        per_feature_avg[feat] = avg

        if avg is None or (isinstance(avg, pd.DataFrame) and avg.empty):
            summary_lines.append(f"[{feat}] Computed on {len(curves)} rec(s), but averaging failed (no common lags?).")
        else:
            try:
                l0, l1 = int(avg["lag"].min()), int(avg["lag"].max())
                summary_lines.append(f"[{feat}] {len(curves)} rec(s); lags {l0}..{l1}")
            except Exception:
                summary_lines.append(f"[{feat}] {len(curves)} rec(s); averaged (lag info not available).")

    # Build combined average across the features we successfully computed
    good_avgs = {k: v for k, v in per_feature_avg.items() if v is not None and not getattr(v, "empty", False)}
    combined_avg = None

    if good_avgs:
        # choose numeric value column automatically (first numeric column other than 'lag')
        sample_df = next(iter(good_avgs.values()))
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [c for c in numeric_cols if str(c).lower() != "lag"]
        if not value_cols:
            summary_lines.append("Couldn't find numeric AMI column in averaged frames; aborting combined average.")
        else:
            value_col = value_cols[0]
            series_list = []
            for feat, df_avg in good_avgs.items():
                if value_col in df_avg.columns:
                    s = df_avg.set_index("lag")[value_col].rename(feat)
                else:
                    other_numeric = [c for c in df_avg.select_dtypes(include=[np.number]).columns if str(c).lower() != "lag"]
                    if not other_numeric:
                        continue
                    s = df_avg.set_index("lag")[other_numeric[0]].rename(feat)
                series_list.append(s)

            if series_list:
                concat = pd.concat(series_list, axis=1)  # columns = features, index = lag
                mean_series = concat.mean(axis=1, skipna=True)
                combined_avg = mean_series.reset_index()
                combined_avg.columns = ["lag", value_col]

                # add combined to the per_feature_avg mapping so plotting & downstream code see it
                per_feature_avg["Combined"] = combined_avg
                summary_lines.append(f"[Combined] Averaged across {len(series_list)} feature-averages.")
            else:
                summary_lines.append("No valid per-feature averages to combine.")
    else:
        summary_lines.append("No successful per-feature averages computed; skipping combined average.")

    # if plot:
    #     try:
    #         # lazy import to avoid circular import at module import time
    #         plot_ami_subplots(per_feature_curves, per_feature_avg, ncols=ncols, title_suffix=title_suffix)
    #     except Exception as e:
    #         summary_lines.append(f"Plot failed: {e}")

    return {
        "per_feature_curves": per_feature_curves,
        "per_feature_avg": per_feature_avg,
        "combined_avg": combined_avg,
        "summary_lines": summary_lines,
    }


# ---- helpers to choose recordings  --------------------
def _valid_recordings(df_all: pd.DataFrame, feature: str, min_len: int) -> List[Tuple[str, int]]:
    """Return (recording_id, usable_length) with enough finite samples for FNN."""
    out: List[Tuple[str, int]] = []
    if feature not in df_all.columns:
        return out
    for rid, g in df_all.groupby("recording_id"):
        x = pd.to_numeric(g[feature], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if len(x) >= min_len:
            out.append((rid, len(x)))
    return out

def _pick_recordings(df_all: pd.DataFrame, feature: str, min_len: int,
                     sample_n: int, strategy: str = "longest", seed: int = 42) -> List[str]:
    cands = _valid_recordings(df_all, feature, min_len)
    if not cands:
        return []
    if strategy == "random":
        rng = np.random.default_rng(seed)
        rids = [rid for rid, _ in cands]
        rng.shuffle(rids)
        return rids[:sample_n]
    # longest
    cands.sort(key=lambda t: t[1], reverse=True)
    return [rid for rid, _ in cands[:sample_n]]


# ---- Orchestrator: average across recordings and features -----------------
def fnn_average_across_features(
    df_all: pd.DataFrame,
    features: List[str],
    tau: int,
    min_dim: int = 1,
    max_dim: int = 10,
    sample_n: int = 10,
    strategy: str = "longest",
    seed: int = 42,
    plot: bool = True,
    ncols: int = 2,
    title_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For each feature:
      - pick `sample_n` recordings with length >= (max_dim+1)*tau
      - run FNN per recording
      - average %FNN across recordings (mean ± SEM)
    Also returns a 'Combined' curve (mean across per-feature means).
    """
    results: Dict[str, Any] = {
        "per_feature_curves": {},   # feature -> list of %FNN arrays (one per recording)
        "per_feature_avg":   {},    # feature -> DataFrame [dim, fnn_mean, fnn_sem]
        "combined_avg":      None,  # DataFrame or None
        "summary_lines":     [],
    }

    dims_ref = np.arange(min_dim, max_dim + 1, dtype=int)
    min_len_required = (max_dim + 1) * tau

    for feat in features:
        rids = _pick_recordings(df_all, feat, min_len_required, sample_n, strategy, seed)
        if not rids:
            results["summary_lines"].append(f"[{feat}] No recordings with ≥ {(max_dim+1)*tau} samples. Skipped.")
            continue

        curves = []
        for rid in rids:
            g = df_all[df_all["recording_id"] == rid]
            s = pd.to_numeric(g[feat], errors="coerce").dropna().values
            d, p = fnn(s, tlag=tau, min_dimension=min_dim, max_dimension=max_dim)
            # guard: if returned NaNs because borderline length, skip
            if np.all(~np.isfinite(p)):
                continue
            curves.append(p)

        if not curves:
            results["summary_lines"].append(f"[{feat}] FNN produced no valid curves. Skipped.")
            continue

        mat = np.vstack(curves)
        mean = np.nanmean(mat, axis=0)
        sem  = np.nanstd(mat, axis=0, ddof=1) / math.sqrt(np.sum(np.isfinite(mat[:, 0])))
        df_avg = pd.DataFrame({"dim": dims_ref, "fnn_mean": mean, "fnn_sem": sem})

        results["per_feature_curves"][feat] = curves
        results["per_feature_avg"][feat]    = df_avg
        results["summary_lines"].append(f"[{feat}] {len(curves)} rec(s); dims {min_dim}..{max_dim}")

    # combined average (equal-weight across features)
    good = [df for df in results["per_feature_avg"].values() if df is not None and not df.empty]
    if good:
        stack = np.stack([df["fnn_mean"].to_numpy() for df in good], axis=0)
        comb_mean = np.nanmean(stack, axis=0)
        comb_sem  = np.nanstd(stack, axis=0, ddof=1) / math.sqrt(stack.shape[0])
        results["combined_avg"] = pd.DataFrame({"dim": dims_ref, "fnn_mean": comb_mean, "fnn_sem": comb_sem})
        results["summary_lines"].append(f"[Combined] Averaged across {len(good)} feature-averages.")
    else:
        results["summary_lines"].append("No per-feature FNN averages to combine.")

    if plot:
        _plot_fnn_subplots(results["per_feature_avg"], results["combined_avg"], ncols=ncols, title_suffix=title_suffix)

    return results

# ---- simple plotting (kept local to avoid viz_utils circular import) ------
def _plot_fnn_subplots(per_feature_avg: Dict[str, pd.DataFrame],
                       combined_avg: Optional[pd.DataFrame],
                       ncols: int = 2,
                       title_suffix: Optional[str] = None):
    import matplotlib.pyplot as plt

    items = list(per_feature_avg.items())
    if combined_avg is not None:
        items.append(("Combined", combined_avg))

    if not items:
        return

    n = len(items)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.6*nrows), squeeze=False)
    axit = iter(axes.ravel())

    for name, df in items:
        ax = next(axit)
        ax.plot(df["dim"], df["fnn_mean"], marker="o", label="Mean %FNN")
        ax.fill_between(df["dim"],
                        df["fnn_mean"] - df["fnn_sem"],
                        df["fnn_mean"] + df["fnn_sem"],
                        alpha=0.2, label="± SEM")
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("% False Nearest Neighbours")
        title = f"{name}" + (f" · {title_suffix}" if title_suffix else "")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="best")

    # hide any extra axes
    for ax in axit:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


# ---------- helpers to batch across recordings / pair averaging -----------
def average_cross_ami_across_recordings(
    df_all: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    min_lag: int = 1,
    max_lag: int = 100,
    recording_col: str = "recording_id",
    require_min_length: Optional[int] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Compute cross_ami for each recording in df_all and return averaged AMI curve.

    Returns dict:
      {
        "lags": np.ndarray,
        "per_recording": { recording_id: ami_array (1D) , ... },
        "mean": np.ndarray (mean across recordings),
        "sem": np.ndarray (sem across recordings),
        "n_recordings": int
      }

    Requirements:
      - df_all must contain recording_col, and columns feat_x, feat_y.
    """
    if feat_x not in df_all.columns or feat_y not in df_all.columns:
        raise ValueError(f"Features not present in df_all: {feat_x}, {feat_y}")

    per_rec = {}
    recs = df_all.groupby(recording_col, sort=False)
    outer = recs if not progress else tqdm(recs, desc="Cross-AMI recordings")

    computed = []
    lags = None
    for rid, g in outer:
        xa = pd.to_numeric(g[feat_x], errors="coerce").values
        ya = pd.to_numeric(g[feat_y], errors="coerce").values
        # aligned finite region inside cross_ami will trim to overlapping finite region
        if require_min_length is not None:
            finite = np.isfinite(xa) & np.isfinite(ya)
            if finite.sum() < require_min_length:
                continue
        arr = cross_ami(xa, ya, min_lag=min_lag, max_lag=max_lag)
        if arr is None or arr.size == 0:
            continue
        per_rec[rid] = arr[:, 1]
        computed.append(arr[:, 1])
        if lags is None:
            lags = arr[:, 0]

    if not computed:
        return {"lags": np.array([], dtype=int), "per_recording": {}, "mean": np.array([]), "sem": np.array([]), "n_recordings": 0}

    stacked = np.vstack(computed)  # n_rec x n_lags
    mean = np.nanmean(stacked, axis=0)
    sem = np.nanstd(stacked, ddof=1, axis=0) / np.sqrt(stacked.shape[0])

    return {"lags": lags.astype(int), "per_recording": per_rec, "mean": mean, "sem": sem, "n_recordings": stacked.shape[0]}


# ---------- plotting / saving helper ------------------------------------
def plot_and_save_cross_ami(lags: np.ndarray, mean_curve: np.ndarray, sem: Optional[np.ndarray] = None,
                            title: Optional[str] = None, out_path: Optional[Path] = None, show: bool = True):
    """
    Plot mean cross-AMI with optional SEM ribbon and optionally save figure.
    """
    plt.figure(figsize=(7, 3.5))
    plt.plot(lags, mean_curve, marker="o", linewidth=1)
    if sem is not None and len(sem) == len(mean_curve):
        plt.fill_between(lags, mean_curve - sem, mean_curve + sem, alpha=0.2)
    if title:
        plt.title(title)
    plt.xlabel("Lag (samples)")
    plt.ylabel("Cross AMI")
    plt.grid(True)
    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()




# ---------- Statistical utilities ----------
def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Step-down procedure that controls family-wise error rate while
    being less conservative than standard Bonferroni correction.

    Args:
        pvals: Dictionary mapping comparison labels to raw p-values

    Returns:
        Dictionary with same keys but corrected p-values

    Note:
        DEPRECATED: Import holm_bonferroni from stats_utils instead.
        This is kept for backward compatibility.
    """
    # Sort p-values in ascending order
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected = {}

    # Apply Holm-Bonferroni formula: p_corrected = p_raw * (m - i + 1)
    for i, (lbl, p) in enumerate(items, start=1):
        corrected[lbl] = min(p * (m - i + 1), 1.0)  # Cap at 1.0

    # Return in original key order
    return {k: corrected[k] for k in pvals.keys()}

def build_rqa_table(results, out_file):
    """Build LaTeX table with β and p-values for each contrast."""
    lines = [
        r"\begin{tabular}{llcc|cc|cc}",
        r"\toprule",
        r"Data Type & Metric & $\beta_{\text{M}}$ & $p_{\text{M}}$ & "
        r"$\beta_{\text{H}}$ & $p_{\text{H}}$ & $\beta_{\text{H--M}}$ & $p_{\text{H--M}}$ \\",
        r"\midrule"
    ]
    
    for col_name in sorted(results.keys()):
        metric_results = results[col_name]
        rows = []
        
        for metric in sorted(metric_results.keys()):
            ests, pvals, _, _ = metric_results[metric]
            
            # Extract contrasts: M-L, H-L, H-M
            b_m = ests.get(("L", "M"))
            p_m = pvals.get(("L", "M"))
            b_h = ests.get(("L", "H"))
            p_h = pvals.get(("L", "H"))
            b_hm = ests.get(("M", "H"))
            p_hm = pvals.get(("M", "H"))
            
            Bm, Pm = fmt(b_m, p_m)
            Bh, Ph = fmt(b_h, p_h)
            Bhm, Phm = fmt(b_hm, p_hm)
            
            rows.append((pretty_metric(metric), Bm, Pm, Bh, Ph, Bhm, Phm))
        
        # Write rows for this data type
        first = True
        for (metric_name, Bm, Pm, Bh, Ph, Bhm, Phm) in rows:
            left = f"\\multirow{{{len(rows)}}}{{*}}{{{col_name}}}" if first else ""
            lines.append(
                f"{left} & {metric_name} & {Bm} & {Pm} & {Bh} & {Ph} & {Bhm} & {Phm} \\\\"
            )
            first = False
        lines.append(r"\midrule")
    
    lines += [r"\bottomrule", r"\end{tabular}"]
    
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Table written to: {out_file}")