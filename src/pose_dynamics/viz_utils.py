# utils/viz_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath("Pose"))
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Any, List
from dataclasses import dataclass, field
from .preprocessing import relevant_indices, detect_conf_prefix_case_insensitive, lm_triplet_colnames
from .features import procrustes_frame_to_template
from .nb_utils import find_col
from .rqa.crossRQA import crossRQA
from .rqa.utils import norm_utils, rqa_utils_cpp, plot_utils

COND_ORDER = ["L", "M", "H"]

# --- Recurrence Plot Configuration ---
@dataclass
class RecurrenceConfig:
    """Configuration for recurrence plot generation."""
    preprocessed_dir: Path
    figs_dir: Path
    rqa_params: Dict[str, Any] = field(default_factory=lambda: {
        "norm": 1,
        "eDim": 4,
        "tLag": 20,
        "rescaleNorm": 1,
        "radius": 0.2,
        "minl": 4,
        "plotMode": "rp",
        "pointSize": 2,
        "showMetrics": True,
        "doStatsFile": False,
        "saveFig": True,
    })
    crqa_params: Dict[str, Any] = field(default_factory=lambda: {
        "norm": 1,
        "eDim": 4,
        "tLag": 40,
        "rescaleNorm": 1,
        "radius": 0.3,
        "minl": 2,
        "plotMode": "rp",
        "pointSize": 2,
        "showMetrics": True,
        "doStatsFile": False,
        "saveFig": True,
    })
    
    def __post_init__(self):
        self.figs_dir.mkdir(parents=True, exist_ok=True)

def find_representative_window(
    df: pd.DataFrame,
    column: str,
    metric: str = "perc_recur",
) -> Optional[Tuple[str, int, int, int, float, float, float]]:
    """
    Find a window that best represents the L→H trend (decrease/increase).
    
    Returns:
        Tuple of (participant, window_index, window_start, window_end, 
                  L_value, H_value, difference)
        or None if no data found
    """
    dsub = df[df["column"] == column].copy()
    
    if dsub.empty:
        return None
    
    # Pivot to compare L vs H for same participant/window
    pivot = dsub.pivot_table(
        values=metric,
        index=["participant", "window_index", "window_start", "window_end"],
        columns="condition"
    )
    
    # Check both conditions exist
    if "L" not in pivot.columns or "H" not in pivot.columns:
        print(f"  ⚠️  Missing L or H condition data")
        return None
    
    # Calculate difference (positive = decrease from L to H)
    pivot["diff"] = pivot["L"] - pivot["H"]
    
    # Remove NaN differences
    pivot = pivot[pivot["diff"].notna()]
    
    if pivot.empty:
        print(f"  ⚠️  No valid L-H comparisons found")
        return None
    
    # Find window closest to the median difference (most representative)
    median_diff = pivot["diff"].median()
    pivot["dist_from_median"] = (pivot["diff"] - median_diff).abs()
    best_idx = pivot["dist_from_median"].idxmin()
    
    if pd.notna(best_idx):
        participant, window_index, window_start, window_end = best_idx
        l_val = pivot.loc[best_idx, "L"]
        h_val = pivot.loc[best_idx, "H"]
        diff = pivot.loc[best_idx, "diff"]
        
        return (
            str(participant),
            int(window_index),
            int(window_start),
            int(window_end),
            float(l_val),
            float(h_val),
            float(diff)
        )
    
    return None

def generate_rqa_plot(
    config: RecurrenceConfig,
    participant: str,
    condition: str,
    column: str,
    window_start: int,
    window_end: int
) -> Optional[Path]:
    """
    Generate RQA plot for a specific window using the library's plotting.
    
    Args:
        config: RecurrenceConfig with paths and parameters
        participant: Participant ID
        condition: Condition code (L/M/H)
        column: Data column name
        window_start: Start frame index
        window_end: End frame index
        
    Returns:
        Path to saved figure or None if error
    """
    # Load the preprocessed data file
    data_file = config.preprocessed_dir / f"{participant}_{condition}_perframe.csv"
    
    if not data_file.exists():
        print(f"  ⚠️  Data file not found: {data_file}")
        return None
    
    df_data = pd.read_csv(data_file)
    
    if column not in df_data.columns:
        print(f"  ⚠️  Column '{column}' not in data file")
        return None
    
    # Extract window
    window = df_data.iloc[window_start:window_end][column].values
    
    if not np.isfinite(window).all():
        print(f"  ⚠️  Window contains non-finite values")
        return None
    
    # Prepare parameters with save path
    params = config.rqa_params.copy()
    out_file = config.figs_dir / f"rqa_{column}_P{participant}_{condition}_w{window_start}-{window_end}.png"
    params['savePath'] = str(out_file)
    
    # Normalize data
    data_norm = norm_utils.normalize_data(window, params["norm"])
    
    # Compute distance matrix
    ds = rqa_utils_cpp.rqa_dist(
        data_norm, data_norm,
        dim=params["eDim"],
        lag=params["tLag"]
    )
    
    # Compute RQA stats and get recurrence plot
    td, rs, mats, err = rqa_utils_cpp.rqa_stats(
        ds["d"],
        rescale=params["rescaleNorm"],
        rad=params["radius"],
        diag_ignore=params["minl"],
        minl=params["minl"],
        rqa_mode="auto",
    )
    
    if err != 0:
        print(f"  ⚠️  RQA error code: {err}")
        return None
    
    # Use the library's plotting function
    plot_utils.plot_rqa_results(
        dataX=data_norm,
        dataY=data_norm,
        td=td,
        plot_mode=params["plotMode"],
        point_size=params["pointSize"],
        save_path=str(out_file)
    )
    
    print(f"  ✅ RQA plot saved: {out_file.name}")
    print(f"      %REC={rs['perc_recur']:.2f}, %DET={rs['perc_determ']:.2f}")
    return out_file


def generate_crqa_plot(
    config: RecurrenceConfig,
    participant: str,
    condition: str,
    col_x: str,
    col_y: str,
    window_start: int,
    window_end: int,
    label: str
) -> Optional[Path]:
    """
    Generate CRQA plot for a specific window using the library's plotting.
    
    Args:
        config: RecurrenceConfig with paths and parameters
        participant: Participant ID
        condition: Condition code (L/M/H)
        col_x: First data column name
        col_y: Second data column name
        window_start: Start frame index
        window_end: End frame index
        label: Label for output filename
        
    Returns:
        Path to saved figure or None if error
    """
    data_file = config.preprocessed_dir / f"{participant}_{condition}_perframe.csv"
    
    if not data_file.exists():
        print(f"  ⚠️  Data file not found: {data_file}")
        return None
    
    df_data = pd.read_csv(data_file)
    
    if col_x not in df_data.columns or col_y not in df_data.columns:
        print(f"  ⚠️  Columns '{col_x}' or '{col_y}' not in data file")
        return None
    
    # Extract window
    window_data = df_data.iloc[window_start:window_end]
    x = window_data[col_x].values
    y = window_data[col_y].values
    
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        print(f"  ⚠️  Window contains non-finite values")
        return None
    
    # Prepare parameters with save path
    params = config.crqa_params.copy()
    out_file = config.figs_dir / f"crqa_{label}_P{participant}_{condition}_w{window_start}-{window_end}.svg"
    params['savePath'] = str(out_file)
    
    # Use the crossRQA function from the library
    td, rs, mats, err = crossRQA(x, y, params)
    
    if err != 0:
        print(f"  ⚠️  CRQA error code: {err}")
        return None
    
    print(f"  ✅ CRQA plot saved: {out_file.name}")
    print(f"      %REC={rs['perc_recur']:.2f}, %DET={rs['perc_determ']:.2f}")
    return out_file


# --- Build Procrustes-aligned coordinates for a slice (and a couple features) ---
def procrustes_transform_series(
    df_norm: pd.DataFrame,
    template_df: pd.DataFrame,
    rel_idxs: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    For each frame in df_norm, align relevant landmarks to template_df (single-row).
    Returns:
      - DataFrame with columns x1..y68 (NaN where unavailable)
      - dict of simple per-frame features: head_rotation_angle, blink_dist, mouth_dist
    """
    if rel_idxs is None:
        rel_idxs = relevant_indices()

    # map columns present
    def _cols(df, axis):
        cols = {}
        for i in rel_idxs:
            c = find_col(df, axis, i)
            if c: cols[i] = c
        return cols

    xcols = _cols(df_norm, "x"); ycols = _cols(df_norm, "y")
    templ_xy = np.column_stack([
        [template_df.iloc[0][find_col(template_df, "x", i)] if find_col(template_df, "x", i) else np.nan for i in rel_idxs],
        [template_df.iloc[0][find_col(template_df, "y", i)] if find_col(template_df, "y", i) else np.nan for i in rel_idxs],
    ])

    n = len(df_norm)
    cols_out = []
    for i in range(1, 69):
        cols_out += [f"x{i}", f"y{i}"]
    data_out = np.full((n, len(cols_out)), np.nan, float)

    head_rotation_angle = np.full(n, np.nan, float)
    blink_dist = np.full(n, np.nan, float)
    mouth_dist = np.full(n, np.nan, float)

    def _idx(i):
        try: return rel_idxs.index(i)
        except ValueError: return -1

    Ltop = [_idx(38), _idx(39)]; Lbot = [_idx(41), _idx(42)]
    Rtop = [_idx(44), _idx(45)]; Rbot = [_idx(47), _idx(48)]
    eye_pair = (_idx(37), _idx(46))
    mouth_pair = (_idx(63), _idx(67))

    def _angle(p1, p2): return float(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
    def _blink_ap(top2, bot2):
        top_mean = top2.mean(axis=0); bot_mean = bot2.mean(axis=0)
        return abs(float(top_mean[1] - bot_mean[1]))
    def _mouth_ap(p63, p67): return float(np.linalg.norm(p67 - p63))

    for t in range(n):
        fx = [pd.to_numeric(df_norm.iloc[t][xcols[i]], errors="coerce") if i in xcols else np.nan for i in rel_idxs]
        fy = [pd.to_numeric(df_norm.iloc[t][ycols[i]], errors="coerce") if i in ycols else np.nan for i in rel_idxs]
        frame_xy = np.column_stack([np.asarray(fx, float), np.asarray(fy, float)])
        avail = np.isfinite(frame_xy).all(axis=1) & np.isfinite(templ_xy).all(axis=1)
        ok, s, tx, ty, R, X = procrustes_frame_to_template(frame_xy, templ_xy, avail)  # imported from utils
        if not ok:
            continue

        # fill transformed pose for that frame
        for i in range(1, 69):
            if i in rel_idxs:
                j = rel_idxs.index(i)
                if np.isfinite(X[j]).all():
                    data_out[t, 2*(i-1)  ] = X[j, 0]
                    data_out[t, 2*(i-1)+1] = X[j, 1]

        # basic features
        i37, i46 = eye_pair
        if i37 >= 0 and i46 >= 0 and np.isfinite(X[i37]).all() and np.isfinite(X[i46]).all():
            head_rotation_angle[t] = _angle(X[i37], X[i46])

        def _safe_pts(idxs):
            pts = [X[k] for k in idxs if k >= 0 and np.isfinite(X[k]).all()]
            return np.vstack(pts) if len(pts) == 2 else None

        vals = []
        ltop, lbot = _safe_pts(Ltop), _safe_pts(Lbot)
        rtop, rbot = _safe_pts(Rtop), _safe_pts(Rbot)
        if ltop is not None and lbot is not None: vals.append(_blink_ap(ltop, lbot))
        if rtop is not None and rbot is not None: vals.append(_blink_ap(rtop, rbot))
        if vals: blink_dist[t] = float(np.mean(vals))

        m63, m67 = mouth_pair
        if m63 >= 0 and m67 >= 0 and np.isfinite(X[m63]).all() and np.isfinite(X[m67]).all():
            mouth_dist[t] = _mouth_ap(X[m63], X[m67])

    df_pose = pd.DataFrame(data_out, columns=cols_out)
    feats = {
        "head_rotation_angle": head_rotation_angle,
        "blink_dist": blink_dist,
        "mouth_dist": mouth_dist,
    }
    return df_pose, feats


def plot_face_landmarks(row: pd.Series, title: str | None = None, ax=None, annotate: bool=False):
    """
    Plot one OpenPose facial frame given a pandas Series `row`
    with columns x1,y1,prob1,...,x70,y70,prob70.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    def color_for(i):
        if 37 <= i <= 48: return 'C0'   # eyes
        if 49 <= i <= 68: return 'C3'   # mouth
        if 28 <= i <= 36: return 'C2'   # nose
        return '0.5'

    xs, ys, cols, present = [], [], [], []
    for i in range(1, 71):
        xk, yk = f"x{i}", f"y{i}"
        if xk in row.index and yk in row.index:
            x, y = float(row[xk]), float(row[yk])
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x); ys.append(y); cols.append(color_for(i)); present.append(i)
    xs = np.asarray(xs); ys = np.asarray(ys)
    ax.scatter(xs, ys, c=cols, s=20, alpha=0.9)

    def poly(seq, c):
        pts = []
        for j in seq:
            if j in present:
                k = present.index(j)
                pts.append([xs[k], ys[k]])
        if len(pts) > 1:
            pts = np.asarray(pts)
            ax.plot(pts[:,0], pts[:,1], c=c, lw=1.2, alpha=0.8)

    poly([37,38,39,40,41,42,37], 'C0')
    poly([43,44,45,46,47,48,43], 'C0')
    poly([49,50,51,52,53,54,55,56,57,58,59,60,49], 'C3')
    poly([28,29,30,31,32,33,34,35,36], 'C2')

    if len(xs):
        cx, cy = xs.mean(), ys.mean()
        rng = max(xs.max()-xs.min(), ys.max()-ys.min())
        pad = 0.2*rng
        ax.set_xlim(cx-rng/2-pad, cx+rng/2+pad)
        ax.set_ylim(cy+rng/2+pad, cy-rng/2-pad)

    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    if title: ax.set_title(title)
    if annotate:
        for i, (x,y) in zip(present, zip(xs,ys)):
            ax.text(x, y, str(i), fontsize=7)


# --- Interactive viewer ---
def create_interactive_pose_timeseries_viewer(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    features_to_plot: List[str] = ('blink_dist', 'mouth_dist'),
    landmark_subset: Optional[List[int]] = None,
    figsize: tuple = (16, 10),
    fps: Optional[int] = None,   # computed if None
    pose_sampling_hz: int = 60,
    plot_downsample: int = 2,
    window_seconds: int = 240
):
    WINDOW_FRAMES = int(window_seconds * pose_sampling_hz)
    HALF_WIN = WINDOW_FRAMES // 2
    STEP_FRAMES = max(1, plot_downsample)
    if fps is None:
        fps = max(1, pose_sampling_hz // plot_downsample)
    interval_ms = int(1000 / fps)

    if landmark_subset is None:
        landmark_subset = list(range(1, 69))  # 1..68 inclusive
    avail = [lm for lm in landmark_subset if f'x{lm}' in df_raw.columns and f'y{lm}' in df_raw.columns]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(features_to_plot), 2, width_ratios=[1, 1.5],
                          height_ratios=[1] * len(features_to_plot),
                          hspace=0.3, wspace=0.3)
    ax_pose = fig.add_subplot(gs[:, 0])
    ax_times = []
    for i in range(len(features_to_plot)):
        ax = fig.add_subplot(gs[i, 1]) if i == 0 else fig.add_subplot(gs[i, 1], sharex=ax_times[0])
        ax_times.append(ax)

    ax_slider = plt.axes([0.1, 0.02, 0.65, 0.03])
    ax_play   = plt.axes([0.81, 0.02, 0.08, 0.04])

    n_frames = len(df_raw)
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valfmt='%d')
    play_button = Button(ax_play, 'Play')
    state = {'playing': False, 'current_frame': 0}

    def get_bounds(xs, ys):
        vx, vy = xs[~np.isnan(xs)], ys[~np.isnan(ys)]
        if len(vx) == 0: return -1, 1, -1, 1
        cx, cy = np.mean(vx), np.mean(vy)
        rng = max(np.ptp(vx), np.ptp(vy))
        pad = rng * 0.2
        return (cx - rng/2 - pad, cx + rng/2 + pad,
                cy - rng/2 - pad, cy + rng/2 + pad)

    def update_pose(i):
        ax_pose.clear()
        xs, ys, colors = [], [], []
        for lm in avail:
            xs.append(df_raw.loc[i, f'x{lm}']); ys.append(df_raw.loc[i, f'y{lm}'])
            colors.append('blue' if 37 <= lm <= 48 else
                          'red'  if 49 <= lm <= 68 else
                          'green' if 28 <= lm <= 36 else 'gray')
        xs = np.array(xs); ys = np.array(ys); mask = ~np.isnan(xs) & ~np.isnan(ys)
        ax_pose.scatter(xs[mask], ys[mask], c=np.array(colors)[mask], s=20, alpha=0.8)

        def draw(seq, c):
            pts = []
            for lm in seq:
                if lm in avail:
                    j = avail.index(lm)
                    if mask[j]: pts.append([xs[j], ys[j]])
            if len(pts) > 1:
                pts = np.array(pts); ax_pose.plot(pts[:, 0], pts[:, 1], color=c, alpha=0.6, linewidth=1.5)

        draw([37,38,39,40,41,42,37], 'blue')
        draw([43,44,45,46,47,48,43], 'blue')
        draw([49,50,51,52,53,54,55,56,57,58,59,60,49], 'red')
        draw([28,29,30,31,32,33,34,35,36], 'green')

        x_min, x_max, y_min, y_max = get_bounds(xs, ys)
        ax_pose.set_xlim(x_min, x_max); ax_pose.set_ylim(y_max, y_min)  # flip Y
        ax_pose.set_aspect('equal'); ax_pose.set_title(f'Facial Pose - Frame {i}', fontweight='bold')
        ax_pose.grid(True, alpha=0.3)
        ax_pose.legend(handles=[Patch(facecolor='blue', label='Eyes'),
                                Patch(facecolor='red', label='Mouth'),
                                Patch(facecolor='green', label='Nose'),
                                Patch(facecolor='gray', label='Face')],
                       loc='upper right', fontsize=8)

    def update_times(i):
        start = max(0, i - HALF_WIN); end = min(len(df_features), i + HALF_WIN)
        idx = np.arange(start, end); idx_ds = idx[::plot_downsample]
        for ax, feat in zip(ax_times, features_to_plot):
            ax.clear()
            if feat in df_features.columns:
                y = df_features[feat].values
                ax.plot(idx_ds, y[start:end:plot_downsample], '-', alpha=0.8, linewidth=1)
                ax.axvline(i, linestyle='--', linewidth=2, alpha=0.9)
                if i < len(df_features) and np.isfinite(y[i]):
                    ax.scatter(i, y[i], s=80, zorder=5)
                    ax.text(i, y[i], f'{y[i]:.3f}', ha='center', va='bottom',
                            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                ax.set_xlim(start, end)
                ax.set_ylabel(feat, fontweight='bold'); ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Feature "{feat}" not found', transform=ax.transAxes, ha='center', va='center')
        ax_times[-1].set_xlabel('Frame Number', fontweight='bold')

    def update_all(i, *, from_slider=False):
        i = int(np.clip(i, 0, n_frames - 1))
        update_pose(i); update_times(i); state['current_frame'] = i
        if not from_slider and slider.val != i:
            slider.eventson = False; slider.set_val(i); slider.eventson = True
        fig.canvas.draw_idle()

    def on_slider(val):
        if not state['playing']: update_all(int(val), from_slider=True)

    def toggle(_):
        state['playing'] = not state['playing']
        play_button.label.set_text('Pause' if state['playing'] else 'Play')
        (fig._timer.start() if state['playing'] else fig._timer.stop())

    def step():
        if not state['playing']: return
        update_all((state['current_frame'] + max(1, plot_downsample)) % n_frames, from_slider=False)

    update_all(0, from_slider=True)
    slider.on_changed(on_slider); play_button.on_clicked(toggle)
    timer = fig.canvas.new_timer(interval=interval_ms); timer.add_callback(step)
    fig._widgets = {'slider': slider, 'play_button': play_button}; fig._timer = timer

    fig.text(0.02, 0.98,
             f'Instructions:\n• Plot decimation ×{plot_downsample}\n'
             f'• Showing {window_seconds}s window (~{WINDOW_FRAMES} frames @ {pose_sampling_hz} Hz)\n'
             f'• Slider = full-res; Play = +{max(1, plot_downsample)} frame(s)/tick @ ~{fps} Hz',
             transform=fig.transFigure, va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    plt.show()
    return fig


def sem(series):
    """Calculate standard error of the mean."""
    s = pd.Series(series).astype(float)
    return s.std(ddof=1) / np.sqrt(s.count())




def create_2x2_figure(df, stats_results, plot_specs=None):
    """
    Create a 2x2 figure showing key RQA/CRQA results.
    
    Args:
        df: DataFrame with RQA results
        stats_results: Dictionary of statistical results
        plot_specs: List of 4 tuples, each containing:
            (col_name, metric, title, ylabel, ylim)
            If None, uses default specifications
    """
    # Default specifications if none provided
    if plot_specs is None:
        plot_specs = [
            ("center_face_magnitude", "perc_recur", 
             "Face Movement — % Recurrence", "% Recurrence", None),
            ("blink_aperture", "perc_determ",
             "Blink — % Determinism", "% Determinism", None),
            ("crqa_head_pupil_mag", "perc_recur",
             "Head–Pupil CRQA (Magnitude)", "% Recurrence", None),
            ("crqa_head_pupil_x", "perc_recur",
             "Head–Pupil CRQA (X-axis)", "% Recurrence", None),
        ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axes_flat = axes.flatten()
    
    def barplot(ax, col_name, metric, title, ylabel, ylim=None):
        """Helper to create one barplot panel."""
        dsub = df[df["column"] == col_name].copy()
        agg = dsub.groupby("condition")[metric].agg(["mean", sem]).reindex(COND_ORDER)
        
        idx = np.arange(len(COND_ORDER))
        means = agg["mean"].to_numpy(dtype=float)
        errs = agg["sem"].to_numpy(dtype=float)
        
        ax.bar(idx, means, yerr=errs, capsize=5, width=0.7, 
               color=['#4575b4', "#ffffbf", '#d73027'])
        ax.set_xticks(idx)
        ax.set_xticklabels(COND_ORDER, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, len(COND_ORDER) - 0.5)
        ax.spines[['top', 'right']].set_visible(False)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Add significance stars if available
        if col_name in stats_results and metric in stats_results[col_name]:
            _, pvals, _, _ = stats_results[col_name][metric]
            
            # Get y position for stars
            y_max = max(means + errs)
            y_range = y_max - min(means - errs)
            
            # Check each pairwise comparison
            comparisons = [
                (0, 1, pvals.get(("L", "M"))),  # L vs M
                (1, 2, pvals.get(("M", "H"))),  # M vs H
                (0, 2, pvals.get(("L", "H")))   # L vs H
            ]
            
            offset = 0
            for i, j, p in comparisons:
                if p is not None and not np.isnan(p) and p < 0.05:
                    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                    y = y_max + 0.1 * y_range + offset * 0.15 * y_range
                    ax.plot([idx[i], idx[j]], [y, y], 'k-', lw=1.5)
                    ax.text((idx[i] + idx[j]) / 2, y, stars, 
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
                    offset += 1
    
    # Create each panel based on specifications
    for i, spec in enumerate(plot_specs[:4]):  # Limit to 4 plots
        col_name, metric, title, ylabel, ylim = spec
        
        # Check if column exists in data
        if col_name in df["column"].unique():
            barplot(axes_flat[i], col_name, metric, title, ylabel, ylim)
        else:
            # If column doesn't exist, show warning text
            axes_flat[i].text(0.5, 0.5, f'Column "{col_name}" not found', 
                            transform=axes_flat[i].transAxes, 
                            ha='center', va='center', fontsize=12)
            axes_flat[i].set_title(title, fontsize=13, fontweight='bold')
    
    return fig
    """
    Create a 2x2 figure showing key RQA/CRQA results.
    Adjust the specific columns and metrics as needed for your analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    
    def barplot(ax, col_name, metric, title, ylabel, ylim=None):
        """Helper to create one barplot panel."""
        dsub = df[df["column"] == col_name].copy()
        agg = dsub.groupby("condition")[metric].agg(["mean", sem]).reindex(COND_ORDER)
        
        idx = np.arange(len(COND_ORDER))
        means = agg["mean"].to_numpy(dtype=float)
        errs = agg["sem"].to_numpy(dtype=float)
        
        ax.bar(idx, means, yerr=errs, capsize=5, width=0.7, color=['#4575b4', "#ffffbf", '#d73027'])
        ax.set_xticks(idx)
        ax.set_xticklabels(COND_ORDER, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, len(COND_ORDER) - 0.5)
        ax.spines[['top', 'right']].set_visible(False)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Add significance stars if available
        if col_name in stats_results and metric in stats_results[col_name]:
            _, pvals, _, _ = stats_results[col_name][metric]
            
            # Get y position for stars
            y_max = max(means + errs)
            y_range = y_max - min(means - errs)
            
            # Check each pairwise comparison
            comparisons = [
                (0, 1, pvals.get(("L", "M"))),  # L vs M
                (1, 2, pvals.get(("M", "H"))),  # M vs H
                (0, 2, pvals.get(("L", "H")))   # L vs H
            ]
            
            offset = 0
            for i, j, p in comparisons:
                if p is not None and not np.isnan(p) and p < 0.05:
                    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                    y = y_max + 0.1 * y_range + offset * 0.15 * y_range
                    ax.plot([idx[i], idx[j]], [y, y], 'k-', lw=1.5)
                    ax.text((idx[i] + idx[j]) / 2, y, stars, 
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
                    offset += 1
    
    # Customize these panels for your specific analysis
    # Example: showing different columns and metrics
    
    # Panel 1: RQA on a specific measurement
    if "center_face_magnitude" in df["column"].unique():
        barplot(axes[0, 0], "center_face_magnitude", "perc_recur",
               "Face Movement — % Recurrence", "% Recurrence")
    
    # Panel 2: RQA on another measurement
    if "blink_aperture" in df["column"].unique():
        barplot(axes[0, 1], "blink_aperture", "perc_determ",
               "Blink — % Determinism", "% Determinism")
    
    # Panel 3: CRQA magnitude
    if "crqa_head_pupil_mag" in df["column"].unique():
        barplot(axes[1, 0], "crqa_head_pupil_mag", "perc_recur",
               "Head–Pupil CRQA (Magnitude)", "% Recurrence")
    
    # Panel 4: CRQA on x-axis
    if "crqa_head_pupil_x" in df["column"].unique():
        barplot(axes[1, 1], "crqa_head_pupil_x", "perc_recur",
               "Head–Pupil CRQA (X-axis)", "% Recurrence")
    
    return fig




import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ===============================
# --- Plot RQA Results ---
# ===============================

import matplotlib.gridspec as gs

def plot_rqa_results(
    dataX=None, dataY=None, td=None,
    plot_mode='rp', point_size=4,
    save_path=None):
    """
    Plot RQA or CRQA results with aligned RP and TS width.
    """

    ax_ts_x = None
    ax_ts_y = None

    N = len(dataX)
    fig = plt.figure(figsize=(8, 9))  # Squarer figure to accommodate equal width
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 12], height_ratios=[1, 12, 2], hspace=0.4, wspace=0.2)

    # === Recurrence Plot ===
    ax_rp = fig.add_subplot(gs[1, 1])
    ax_rp.set_facecolor('#b0c4de')  # Light Steel Blue, a lighter navy shade

    recur_y, recur_x = np.where(td == 1)
    ax_rp.scatter(recur_x, recur_y, c='blue', s=point_size, edgecolors='none')
    ax_rp.set_xlim([0, N])
    ax_rp.set_ylim([0, N])
    ax_rp.set_title("Cross-Recurrence Plot" if dataY is not None else "Recurrence Plot", pad=8)
    ax_rp.set_xlabel("X(i)")
    ax_rp.set_ylabel("Y(j)" if dataY is not None else "X(j)")

    # === Time Series X ===
    if 'timeseries' in plot_mode:
        ax_ts_x = fig.add_subplot(gs[2, 1], sharex=ax_rp)
        ax_ts_x.plot(np.arange(N), dataX[:N], color='tab:blue')
        ax_ts_x.set_xlim([0, N])
        ax_ts_x.set_title("Time Series X", fontsize=10)
        ax_ts_x.set_xlabel("Time")
        ax_ts_x.set_ylabel("X", rotation=0, labelpad=15)

    # === Time Series Y ===
    if dataY is not None and 'timeseries' in plot_mode:
        ax_ts_y = fig.add_subplot(gs[1, 0], sharey=ax_rp)
        ax_ts_y.plot(dataY[:N], np.arange(N), color='tab:blue')
        ax_ts_y.invert_xaxis()
        ax_ts_y.set_ylim([0, N])
        ax_ts_y.set_title("Time Series Y", fontsize=10)
        ax_ts_y.set_ylabel("Time")
        ax_ts_y.set_xlabel("Y", rotation=0, labelpad=15)

    if 'timeseries' in plot_mode:
        if ax_ts_x is not None:
            fig.align_xlabels([ax_rp, ax_ts_x])
        if ax_ts_y is not None:
            fig.align_ylabels([ax_rp, ax_ts_y])
    else:
        fig.align_xlabels([ax_rp])
        fig.align_ylabels([ax_rp])
    
    if save_path:
       import os
       os.makedirs(os.path.dirname(save_path), exist_ok=True)
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
       print(f"Plot saved to: {save_path}")

    plt.show()

# ===============================
# --- Alignment plotting helpers ---
# ===============================

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from io import SKELETON_CONNECTIONS   

def recenter(P, ref_index=0):
    """Center skeleton on reference joint (default pelvis=0)."""
    return P - P[ref_index]

def remap_zed(P):
    # ZED: (X=side, Y=up, Z=depth) → plot coords (X, -Z, Y) for upright skeleton
    return np.stack([P[:,0], -P[:,2], P[:,1]], axis=1)

def flip_180_vertical(P):
    """Flip skeleton 180 degrees around vertical axis (spin around)."""
    P_flipped = P.copy()
    P_flipped[:, 0] = -P[:, 0]  # flip X
    P_flipped[:, 1] = -P[:, 1]  # flip Y
    # Z stays the same (vertical axis)
    return P_flipped

def adjust_floor_level(P):
    """Adjust skeleton so feet are at floor level (z=0)."""
    # Find the lowest Z coordinate (feet)
    min_z = np.min(P[:, 2])
    P_adjusted = P.copy()
    P_adjusted[:, 2] = P[:, 2] - min_z  # Move so lowest point is at z=0
    return P_adjusted

def draw_skeleton(ax, points, color="k", lw=1.5, groups=("body","legs","face")):
    for g in groups:
        edges = SKELETON_CONNECTIONS[g]
        segs = [(points[int(i)], points[int(j)]) for i,j in edges]
        lc = Line3DCollection(segs, colors=color, linewidths=lw)
        ax.add_collection3d(lc)

def draw_floor(ax, w=2, l=3, z0=0.0):
    """Draw floor rectangle centered at origin, like a mat the skeleton stands on."""
    x = np.array([-w/2, w/2, w/2, -w/2, -w/2])
    y = np.array([-l/2, -l/2, l/2, l/2, -l/2])
    z = np.full_like(x, z0)
    ax.plot(x, y, z, color="gray", alpha=0.3)

def style(ax, w, l):
    ax.set_xlim(-w/2, w/2)
    ax.set_ylim(-l/2, l/2)
    ax.set_zlim(0, 2.0)  # Reduced from 2.5 to 2.0 for closer view
    ax.set_box_aspect([w, l, 2.0])  # Match the zlim
    
    # Set viewing angle to look more straight-on (less from above)
    ax.view_init(elev=10, azim=-90)  # Lower elevation (10 degrees), front view
    
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")

def orient_skeleton(P, nose_idx=9, neck_idx=3):
    """
    Orient skeleton so that the nose->neck vector always faces +Y.
    Assumes P is already remapped into display coordinates.
    """
    P_oriented = P.copy()
    
    # Forward direction (nose - neck), projected to XY plane
    forward = P_oriented[nose_idx] - P_oriented[neck_idx]
    forward[2] = 0  # ignore vertical component
    
    # Canonical forward = +Y
    if np.dot(forward, np.array([0,1,0])) < 0:
        # Flip around vertical axis (Z)
        P_oriented[:,0] = -P_oriented[:,0]  # flip X
    return P_oriented

def prep_for_display(P, nose_idx=9, neck_idx=3):
    """
    Apply all display transformations consistently:
    - remap from ZED to plotting coords
    - recenter on pelvis
    - orient so facing direction is +Y
    - adjust floor so feet touch z=0
    """
    P_disp = remap_zed(P)
    P_disp = recenter(P_disp)
    P_disp = orient_skeleton(P_disp, nose_idx, neck_idx)
    P_disp = adjust_floor_level(P_disp)
    return P_disp

# ===============================
# --- Diagnostic alignment plot ---
# ===============================

def diagnostic_plot_2panel(template_xyz, raw_means_xyz, aligned_means_xyz,
                           save_path=None, max_samples=2,
                           box_width=2, box_length=3,
                           flip_template=False):
    """
    Two-panel diagnostic plot: raw vs aligned skeletons with a floor box.
    Keeps everything centered and adds legend at bottom.
    
    Args:
        template_xyz : (n_points, 3) array (global template).
        raw_means_xyz : list of (n_points, 3) arrays (trial mean poses, raw).
        aligned_means_xyz : list of (n_points, 3) arrays (trial mean poses, aligned).
        save_path : str or None, file to save figure.
        max_samples : int, number of trials to plot.
        box_width, box_length : floor box dimensions.
        flip_template : bool, if True flip template 180° around vertical axis.
    """
    # --- optional flip ---
    if flip_template:
        template_xyz = template_xyz.copy()
        # rotate 180° around vertical axis (Z in display space)
        template_xyz[:, [0,1]] *= -1   # flip X and Y together

    fig = plt.figure(figsize=(16, 8))
    grid = gs.GridSpec(1, 2, figure=fig, left=0.02, right=0.98,
                       top=0.95, bottom=0.15, wspace=0.02)
    ax0 = fig.add_subplot(grid[0, 0], projection="3d")
    ax1 = fig.add_subplot(grid[0, 1], projection="3d")

    colors = plt.cm.tab10.colors

    # --- Before alignment ---
    draw_skeleton(ax0, template_xyz, color="k", lw=1.8)
    ax0.scatter(template_xyz[:,0], template_xyz[:,1], template_xyz[:,2],
                color="k", s=15, label="Global Template")
    
    for i, P in enumerate(raw_means_xyz[:max_samples]):
        draw_skeleton(ax0, P, color=colors[i % 10], lw=1.0)
        ax0.scatter(P[:,0], P[:,1], P[:,2],
                    color=colors[i % 10], s=8, label=f"Raw Data (P{i+1})")

    style(ax0, box_width, box_length)
    draw_floor(ax0, box_width, box_length, z0=0)
    ax0.set_title("Before Alignment")

    # --- After alignment ---
    draw_skeleton(ax1, template_xyz, color="k", lw=1.8)
    ax1.scatter(template_xyz[:,0], template_xyz[:,1], template_xyz[:,2],
                color="k", s=15, label="Global Template")
    
    for i, P in enumerate(aligned_means_xyz[:max_samples]):
        draw_skeleton(ax1, P, color=colors[i % 10], lw=1.0)
        ax1.scatter(P[:,0], P[:,1], P[:,2],
                    color=colors[i % 10], s=8, label=f"Aligned Data (P{i+1})")

    style(ax1, box_width, box_length)
    draw_floor(ax1, box_width, box_length, z0=0)
    ax1.set_title("After Alignment")

    # --- Legend ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# =====================
# animation helper
# =====================
import os
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def create_pm_animation_3dgrid(pca, X, mean_vector, edges, save_path,
                               n_components=3, n_frames=200, scale=2.0, fps=40):
    """
    Create 3D animations of PCA movement components (grid of subplots).
    Grey skeleton = mean pose, Black skeleton = animated PC.
    PCA input is body-centred only. Display transforms applied here.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_points = mean_vector.size // 3

    # === Display-only prep ===
    def prep_for_display(P):
        return adjust_floor_level(
            flip_180_vertical(
                recenter(remap_zed(P))
            )
        )

    # === Build PC reconstructions (in canonical space) ===
    pc_scores = pca.transform(X)
    t = np.linspace(0, 2*np.pi, n_frames)
    all_positions = []
    for i in range(n_components):
        coeffs = np.sin(t) * scale * np.std(pc_scores[:, i])
        reconstruction = np.outer(coeffs, pca.components_[i]) + mean_vector
        positions = reconstruction.reshape(n_frames, n_points, 3)
        all_positions.append(positions)
    all_positions = np.array(all_positions)   # (n_components, n_frames, n_points, 3)

    # === Mean pose (display transformed) ===
    base_pose = prep_for_display(mean_vector.reshape(n_points, 3))

    # === Global axis limits (shared across all PCs, in display space) ===
    poses_disp = np.vstack([
        prep_for_display(pose).reshape(1, n_points, 3)
        for pose in all_positions.reshape(-1, n_points, 3)
    ])
    mins = np.min(poses_disp.reshape(-1, 3), axis=0)
    maxs = np.max(poses_disp.reshape(-1, 3), axis=0)

    pad = 0.05 * (maxs - mins)
    mins -= pad
    maxs += pad

    # === Figure with subplots ===
    ncols = min(3, n_components)
    nrows = int(np.ceil(n_components / ncols))
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    axes, artists = [], []

    for i in range(n_components):
        ax = fig.add_subplot(nrows, ncols, i+1, projection="3d")
        axes.append(ax)

        # Apply shared global limits
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.view_init(elev=15, azim=-70)
        ax.set_title(f"PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% var")
        ax.axis("off")

        # --- Grey mean skeleton ---
        segs_mean = [(base_pose[int(a)], base_pose[int(b)]) for a,b in edges]
        lc_mean = Line3DCollection(segs_mean, colors="gray", linewidths=1, alpha=0.6, zorder=1)
        ax.add_collection3d(lc_mean)
        dots_mean = ax.scatter(base_pose[:,0], base_pose[:,1], base_pose[:,2],
                               c="gray", s=10, alpha=0.6, zorder=1)

        # --- Black animated skeleton (init) ---
        segs_init = [(base_pose[int(a)], base_pose[int(b)]) for a,b in edges]
        lc_black = Line3DCollection(segs_init, colors="black", linewidths=1.5, zorder=10)
        ax.add_collection3d(lc_black)
        dots_black = ax.scatter([], [], [], c="black", s=10, zorder=11)

        artists.append((lc_black, dots_black))

    # === Update function ===
    def update(frame):
        updated = []
        for i, ax in enumerate(axes):
            pose = all_positions[i, frame]
            pose_disp = prep_for_display(pose)

            # update black skeleton lines
            segs = [(pose_disp[int(a)], pose_disp[int(b)]) for a,b in edges]
            artists[i][0].set_segments(segs)
            updated.append(artists[i][0])

            # update black dots
            artists[i][1]._offsets3d = (pose_disp[:,0], pose_disp[:,1], pose_disp[:,2])
            updated.append(artists[i][1])

        return updated

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000/fps, blit=False)
    ani.save(save_path, writer="ffmpeg", fps=fps)
    plt.close()
    print(f"[DONE] 3D PCA grid animation saved to {save_path}")
