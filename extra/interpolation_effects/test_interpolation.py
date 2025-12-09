import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from src.pose_dynamics.rqa.utils import rqa_utils_cpp
from src.pose_dynamics.nonlinear.state_space_recon import estimate_embedding_curves
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

mpl.rcParams['font.family'] = 'Times New Roman'


# ---------------------------------------------------------------------
# Gap handling and interpolation
# ---------------------------------------------------------------------
def introduce_single_gap(ts, gap_length, start=None):
    """Introduce a single gap of given length into ts."""
    ts_gap = ts.copy()
    n = len(ts_gap)

    if gap_length >= n:
        raise ValueError("gap_length must be smaller than time series length")

    if start is None:
        # Center the gap by default
        start = n // 2 - gap_length // 2

    # Clamp start so gap stays inside bounds
    start = max(0, min(start, n - gap_length))

    ts_gap[start:start + gap_length] = np.nan
    return ts_gap


def apply_centered_gap_in_window(ts, gap_len, n_rp):
    """
    Take first n_rp samples, insert one centered gap of length gap_len,
    then interpolate it.

    Returns:
        ts_base  : original window (no gaps)
        ts_interp: window with one centered gap interpolated
    """
    ts_base = ts[:n_rp].copy()
    ts_gap = ts_base.copy()

    if gap_len > 0:
        center = n_rp // 2
        start = max(0, center - gap_len // 2)
        end = min(n_rp, start + gap_len)
        ts_gap[start:end] = np.nan
        ts_interp = interpolate_gaps(ts_gap)
    else:
        ts_interp = ts_base

    return ts_base, ts_interp


def introduce_gaps(time_series, gap_length, n_gaps=10):
    """Introduce random gaps of specified length (contiguous NaNs)."""
    ts_with_gaps = time_series.copy()
    valid_range = len(time_series) - gap_length - 200
    if valid_range <= 200:
        raise ValueError(f"Time series too short for gap length {gap_length}")

    gap_starts = np.random.choice(
        range(100, valid_range),
        size=min(n_gaps, max(1, valid_range // (gap_length * 2))),  # avoid overlap
        replace=False
    )
    for start in gap_starts:
        ts_with_gaps[start:start + gap_length] = np.nan
    return ts_with_gaps


def interpolate_gaps(time_series):
    """Linear interpolation of NaN values."""
    ts_interp = time_series.copy()
    mask = ~np.isnan(ts_interp)
    indices = np.arange(len(ts_interp))

    if np.sum(mask) < 2:
        raise ValueError("Not enough valid data points for interpolation")

    interp_func = interp1d(
        indices[mask],
        ts_interp[mask],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    ts_interp[~mask] = interp_func(indices[~mask])
    return ts_interp


# ---------------------------------------------------------------------
# RQA wrappers
# ---------------------------------------------------------------------
def compute_rqa_metrics(time_series, m, tau, eps, theiler=1, lmin=2):
    """Compute standard RQA metrics using autoRQA implementation."""
    dataX = time_series

    ds = rqa_utils_cpp.rqa_dist(dataX, dataX, dim=m, lag=tau)

    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(
        ds["d"],
        rescale=1,
        rad=eps,
        diag_ignore=theiler,
        minl=lmin,
        rqa_mode="auto"
    )

    if err_code != 0:
        raise RuntimeError(f"RQA computation failed with error code {err_code}")

    return {
        'RR': float(rs['perc_recur']),
        'DET': float(rs['perc_determ']),
        'LAM': float(rs['laminarity']),
        'L': float(rs['mean_line_length']),
        'ENTR': float(rs['entropy']),
        'maxL': float(rs['maxl_found']),
        'TT': float(rs['trapping_time']),
        'Vmax': float(rs['vmax']),
        'divergence': float(rs['divergence']),
        'trend_lower': float(rs['trend_lower_diag']),
        'trend_upper': float(rs['trend_upper_diag']),
        'td': td,
        'mats': mats
    }


def compute_recurrence_plot(time_series, m, tau, eps, theiler=1):
    """Compute recurrence plot (thresholded distance matrix)."""
    dataX = time_series
    ds = rqa_utils_cpp.rqa_dist(dataX, dataX, dim=m, lag=tau)

    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(
        ds["d"],
        rescale=1,
        rad=eps,
        diag_ignore=theiler,
        minl=2,
        rqa_mode="auto"
    )

    if err_code != 0:
        raise RuntimeError(f"Recurrence plot computation failed with error code {err_code}")

    return td


# ---------------------------------------------------------------------
# Embedding diagnostics (AMI / FNN from utils)
# ---------------------------------------------------------------------
def compute_embedding_curves(
    ts,
    tau_for_fnn=None,
    min_lag=1,
    max_lag=140,
    min_dim=1,
    max_dim=10,
):
    """
    Convenience wrapper around state_space_recon.estimate_embedding_curves.

    This just computes the AMI and (optionally) FNN curves and returns them
    without making any automatic decisions about τ or m.
    """
    return estimate_embedding_curves(
        ts,
        min_lag=min_lag,
        max_lag=max_lag,
        tau_for_fnn=tau_for_fnn,
        min_dim=min_dim,
        max_dim=max_dim,
    )


# ---------------------------------------------------------------------
# Noisy sine generator
# ---------------------------------------------------------------------
def generate_noisy_sine(duration=100.0, dt=0.01, T_seconds=1.0, noise_std=0.2):
    """
    Generate noisy sine wave:
        x(t) = sin(2πt/T) + N(0, noise_std)
    """
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * t / T_seconds)
    noise = np.random.normal(0, noise_std, len(signal))
    ts = signal + noise

    # Embedding parameters (analytic choice for this toy system)
    T_samples = int(T_seconds / dt)
    m = 3
    tau = T_samples // 4  # τ = T/4 in samples

    return t, ts, m, tau


# ---------------------------------------------------------------------
# Simulation: noisy sine only
# ---------------------------------------------------------------------
def run_interpolation_simulation_noisy_sine(n_trials=30, eps=0.1):
    """
    Run interpolation simulation for noisy sine only.

    Returns:
        df: DataFrame with error metrics vs. gap length.
        params: dict with t, ts, m, tau, eps, and embedding_curves.
    """
    print("Running interpolation simulation for noisy sine...")

    t, ts, m, tau = generate_noisy_sine()
    print(f"System parameters (analytic): m={m}, tau={tau}")

    # Compute AMI / FNN curves from the utils (for diagnostics)
    embedding_curves = compute_embedding_curves(
        ts,
        tau_for_fnn=tau,  # use the analytic τ for FNN curve
        min_lag=1,
        max_lag=140,
        min_dim=1,
        max_dim=10,
    )
    if embedding_curves["ami"] is not None:
        print("Computed AMI curve from utils.")
    if embedding_curves["fnn"] is not None:
        print("Computed FNN curve from utils.")

    # Ground truth RQA (no gaps)
    print("Computing ground truth RQA...")
    rqa_truth = compute_rqa_metrics(ts, m, tau, eps=eps)
    print(f"Ground truth RR: {rqa_truth['RR']:.4f}, DET: {rqa_truth['DET']:.4f}, ENTR: {rqa_truth['ENTR']:.4f}, MaxL: {rqa_truth['maxL']:.2f}, LAM: {rqa_truth['LAM']:.4f}")
    rr_truth = rqa_truth['RR']
    det_truth = rqa_truth['DET']
    entr_truth = rqa_truth['ENTR']
    maxL_truth = rqa_truth['maxL']
    lam_truth = rqa_truth['LAM']

    # Gap lengths in multiples of τ only
    gap_lengths = [
        tau // 2,          # 0.5τ
        tau,               # 1τ
        int(1.5 * tau),    # 1.5τ
        2 * tau,           # 2τ
        (m - 1) * tau,     # (m-1)τ
        3 * tau,           # 3τ
        4 * tau            # 4τ
    ]
    gap_lengths = sorted(list(set(gap_lengths)))  # remove duplicates

    print(f"Testing gap lengths (samples): {gap_lengths}")

    results = {
        'gap_length': [],
        'gap_in_tau': [],
        'RR_error': [],
        'DET_error': [],
        'RR_rel_error': [],
        'DET_rel_error': [],
        'ENTR_rel_error': [],
        'maxL_rel_error': [],
        'LAM_rel_error': [],
    }


    for gap_len in gap_lengths:
        print(f"  Gap length {gap_len} samples ({gap_len / tau:.2f} τ)")

        for trial in range(n_trials):
            try:
                ts_gap = introduce_gaps(ts, gap_len, n_gaps=5)
                ts_interp = interpolate_gaps(ts_gap)

                rqa_interp = compute_rqa_metrics(ts_interp, m, tau, eps=eps)

                rr_err = abs(rqa_interp['RR'] - rr_truth)
                det_err = abs(rqa_interp['DET'] - det_truth)
                entr_err = abs(rqa_interp['ENTR'] - entr_truth)
                maxL_err = abs(rqa_interp['maxL'] - maxL_truth)
                lam_err = abs(rqa_interp['LAM'] - lam_truth)

                results['gap_length'].append(gap_len)
                results['gap_in_tau'].append(gap_len / tau)

                results['RR_error'].append(rr_err)
                results['DET_error'].append(det_err)

                results['RR_rel_error'].append(rr_err / rr_truth * 100 if rr_truth > 0 else 0.0)
                results['DET_rel_error'].append(det_err / det_truth * 100 if det_truth > 0 else 0.0)
                results['ENTR_rel_error'].append(
                    entr_err / entr_truth * 100 if entr_truth > 0 else 0.0
                )
                results['maxL_rel_error'].append(
                    maxL_err / maxL_truth * 100 if maxL_truth > 0 else 0.0
                )
                results['LAM_rel_error'].append(
                    lam_err / lam_truth * 100 if lam_truth > 0 else 0.0
                )


            except Exception as e:
                print(f"    Trial {trial} failed for gap {gap_len}: {e}")
                continue

    df = pd.DataFrame(results)
    print(f"Completed {len(df)} successful trials total.")

    params = {
        't': t,
        'ts': ts,
        'm': m,
        'tau': tau,
        'eps': eps,
        'embedding_curves': embedding_curves,
    }
    return df, params


# ---------------------------------------------------------------------
# Plotting: error vs gap, baseline RP + raw time series, and difference RPs
# ---------------------------------------------------------------------
def plot_error_vs_gap(df, params, save_path="noisy_sine_error_vs_gap.svg"):
    """
    Plot relative error vs. gap length (in τ) for several RQA metrics.
    """
    m = params['m']

    # Aggregate mean/std per gap_in_tau for each metric
    metrics = [
        ('RR_rel_error', 'RR'),
        ('DET_rel_error', 'DET'),
        ('ENTR_rel_error', 'ENTR'),
        ('maxL_rel_error', 'maxL'),
        ('LAM_rel_error', 'LAM'),
    ]

    grouped = df.groupby('gap_in_tau').agg({
        'RR_rel_error': ['mean', 'std'],
        'DET_rel_error': ['mean', 'std'],
        'ENTR_rel_error': ['mean', 'std'],
        'maxL_rel_error': ['mean', 'std'],
        'LAM_rel_error': ['mean', 'std'],
    }).sort_index()

    x = grouped.index.values

    fig, ax = plt.subplots(figsize=(7, 4))

    # Different markers so it's not a spaghetti mess
    marker_cycle = ['o', 's', 'D', '^', 'v']

    for (col, label), marker in zip(metrics, marker_cycle):
        mean_vals = grouped[col]['mean'].values
        std_vals = grouped[col]['std'].values

        ax.errorbar(
            x, mean_vals, yerr=std_vals,
            marker=marker, linestyle='-',
            linewidth=1.5, label=label
        )

    # Mark theoretical (m−1)τ threshold -> x = (m-1)
    ax.axvline(x=(m - 1), linestyle='--', linewidth=2, label='(m−1)τ')

    # 5% error reference
    ax.axhline(y=5, linestyle=':', linewidth=1, label='5% error')

    ax.set_xlabel('Gap Length (multiples of τ)')
    ax.set_ylabel('Relative Error (%)')
    
    ax.set_title('Relative error vs. gap length (RQA metrics)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # Set a reasonable y-limit
    all_means = []
    for col, _ in metrics:
        all_means.append(grouped[col]['mean'].values)
    all_means = np.concatenate(all_means) if len(all_means) > 0 else np.array([10.0])

    ymax = np.nanmax(all_means)
    #ax.set_ylim(0, max(20, ymax * 1.2))
    ax.set_ylim(0,20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")



def plot_baseline_rp(params, n_rp=500, save_path="noisy_sine_rp_baseline.svg"):
    """
    Plot the baseline raw time series (left) and recurrence plot (right)
    for the first n_rp samples.
    """
    ts = params['ts']
    t = params['t']
    m = params['m']
    tau = params['tau']
    eps = params['eps']

    ts_base = ts[:n_rp]
    t_base = t[:n_rp]
    rp_base = compute_recurrence_plot(ts_base, m, tau, eps)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left: raw time series segment
    ax_ts = axes[0]
    ax_ts.plot(t_base, ts_base, linewidth=1.0)
    ax_ts.set_title("Noisy sine (first window)")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("x(t)")

    # Right: baseline RP
    ax_rp = axes[1]
    im = ax_rp.imshow(
        rp_base,
        cmap='binary',
        origin='lower',
        aspect='equal'
    )
    ax_rp.set_title("Baseline RP (no gaps)")
    ax_rp.set_xlabel("Time")
    ax_rp.set_ylabel("Time")

    fig.colorbar(im, ax=ax_rp, shrink=0.8, label="Recurrence")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_rp_difference_maps(params,
                            n_rp=500,
                            gap_multipliers=(0.5, 2.0, 4.0),
                            save_path="noisy_sine_rp_diff_diverging.svg"):
    """
    Plot RP difference maps (gap - baseline) using a discrete diverging colormap.

    Values:
        -1 = recurrence lost (was 1, now 0)
         0 = unchanged
        +1 = recurrence gained (was 0, now 1)
    """
    ts = params['ts']
    m = params['m']
    tau = params['tau']
    eps = params['eps']

    # ----- Baseline RP on the same window -----
    ts_base_window = ts[:n_rp]
    rp_base = compute_recurrence_plot(ts_base_window, m, tau, eps)
    rp_base_bin = (rp_base > 0).astype(int)

    # Prepare gap lengths in samples
    gap_lengths = [int(mult * tau) for mult in gap_multipliers]
    titles = [f"Gap = {mult}τ" for mult in gap_multipliers]

    diff_maps = []

    for gap_len in gap_lengths:
        # One centered gap in the same window, then interpolate
        _, ts_gap_interp = apply_centered_gap_in_window(ts, gap_len, n_rp)

        rp_gap = compute_recurrence_plot(ts_gap_interp, m, tau, eps)
        rp_gap_bin = (rp_gap > 0).astype(int)

        rp_diff = rp_gap_bin - rp_base_bin  # -1, 0, +1
        diff_maps.append(rp_diff)

    # ----- Discrete diverging colormap for -1, 0, 1 -----
    # Colors: [lost, unchanged, gained]
    cmap = ListedColormap(['#313695', '#f7f7f7', '#a50026'])  # dark blue, white, red
    bounds = [-1.5, -0.5, 0.5, 1.5]  # bins for -1, 0, 1
    norm = BoundaryNorm(bounds, cmap.N)

    # ----- Figure: use constrained_layout to avoid overlap -----
    fig, axes = plt.subplots(
        1,
        len(gap_lengths),
        figsize=(4 * len(gap_lengths) + 1, 4),
        constrained_layout=True
    )
    if len(gap_lengths) == 1:
        axes = [axes]

    for ax, rp_diff, title in zip(axes, diff_maps, titles):
        im = ax.imshow(
            rp_diff,
            cmap=cmap,
            norm=norm,
            origin='lower',
            aspect='equal'
        )
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Time")

    # Shared colorbar with only -1, 0, 1 ticks
    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.8,
        pad=0.02,  # small gap so it doesn't overlap
        label="Δ Recurrence (gap - baseline)"
    )
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['-1', '0', '1'])

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")

def plot_embedding_diagnostics(params,
                               save_path="noisy_sine_embedding_diagnostics.svg"):
    """
    Plot AMI (vs lag) and FNN (vs dimension) curves computed via
    estimate_embedding_curves.

    Uses params['embedding_curves'] populated in run_interpolation_simulation_noisy_sine.
    """
    embedding_curves = params.get("embedding_curves", None)
    if embedding_curves is None:
        print("No embedding_curves in params; skipping embedding diagnostics plot.")
        return

    ami_res = embedding_curves.get("ami", None)
    fnn_res = embedding_curves.get("fnn", None)

    if ami_res is None and fnn_res is None:
        print("AMI and FNN results are both None; skipping embedding diagnostics plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ------------------------------------------------
    # Left panel: AMI vs lag
    # ------------------------------------------------
    ax_ami = axes[0]
    if ami_res is not None:
        lags = ami_res["lags"]
        ami_vals = ami_res["ami"]
        ax_ami.plot(lags, ami_vals, marker='o', linewidth=1.5)
        ax_ami.set_xlabel("Lag (samples)")
        ax_ami.set_ylabel("AMI")
        ax_ami.set_title("AMI curve")

        # Optional: mark the analytic tau used in the simulation, if inside range
        tau = params.get("tau", None)
        if tau is not None and lags.min() <= tau <= lags.max():
            ax_ami.axvline(tau, linestyle='--', linewidth=1.5, label=f"τ = {tau}")
            ax_ami.legend(fontsize=8)

    else:
        ax_ami.text(0.5, 0.5, "AMI not available",
                    ha="center", va="center", transform=ax_ami.transAxes)
        ax_ami.set_axis_off()

    # ------------------------------------------------
    # Right panel: FNN vs dimension
    # ------------------------------------------------
    ax_fnn = axes[1]
    if fnn_res is not None:
        dims = fnn_res["dims"]
        pct_fnn = fnn_res["pct_fnn"]
        ax_fnn.plot(dims, pct_fnn, marker='s', linewidth=1.5)
        ax_fnn.set_xlabel("Embedding dimension m")
        ax_fnn.set_ylabel("% False Neighbors")
        ax_fnn.set_title("FNN curve")

        ax_fnn.set_xticks(dims)

    else:
        ax_fnn.text(0.5, 0.5, "FNN not available",
                    ha="center", va="center", transform=ax_fnn.transAxes)
        ax_fnn.set_axis_off()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")

def plot_raw_timeseries(params,
                        n_ts=1000,
                        save_path="noisy_sine_raw_timeseries.svg"):
    """
    Plot the raw noisy sine time series for the first n_ts samples
    in a wide, single-panel figure.
    """
    t = params['t']
    ts = params['ts']

    n_ts = min(n_ts, len(ts))
    t_seg = t[:n_ts]
    ts_seg = ts[:n_ts]

    fig, ax = plt.subplots(figsize=(10, 3))  # wider aspect
    ax.plot(t_seg, ts_seg, linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x(t)")
    ax.set_title(f"Noisy sine (first {n_ts} samples)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    n_trials = 30   # how many trials per gap length
    eps = 0.2       # radius for RQA

    df, params = run_interpolation_simulation_noisy_sine(
        n_trials=n_trials,
        eps=eps
    )
    df.to_csv('interpolation_results_noisy_sine.csv', index=False)

    # 1) Error vs gap length
    plot_error_vs_gap(df, params, save_path="noisy_sine_error_vs_gap.svg")

    # 2) Baseline RP + raw time series
    plot_raw_timeseries(params, n_ts=1000,
                        save_path="noisy_sine_raw_timeseries.svg")


    # 3) Difference maps with diverging colormap
    plot_rp_difference_maps(
        params,
        n_rp=500,
        gap_multipliers=(0.5, 2.0, 4.0),
        save_path="noisy_sine_rp_diff_diverging.svg"
    )

    # 4) AMI + FNN diagnostics from utils
    plot_embedding_diagnostics(
        params,
        save_path="noisy_sine_embedding_diagnostics.svg"
    )

    print("Done.")

