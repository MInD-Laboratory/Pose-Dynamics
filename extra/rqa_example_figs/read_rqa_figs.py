import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Signals that give useful RPs
# =========================================================

def make_signals(N=900, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 30, N)

    # 1. Stable periodic
    s1 = np.sin(2 * np.pi * t / 5.0)

    # 2. Regime shift: same sine, then higher frequency + noise
    s2 = np.sin(2 * np.pi * t / 5.0)
    s2[N // 2:] = 1.5 * np.sin(2 * np.pi * t[N // 2:] / 3.0) \
                  + 0.2 * rng.standard_normal(N - N // 2)

    # 3. Laminar phases: sine with flat "hold" segments
    s3 = np.sin(2 * np.pi * t / 5.0)
    for start in [150, 350, 600]:
        end = start + 80
        s3[start:end] = s3[start]   # hold value

    # 4. Chaotic logistic map, rescaled
    x = np.empty(N)
    x[0] = 0.2
    r = 3.9
    for i in range(N - 1):
        x[i + 1] = r * x[i] * (1.0 - x[i])
    s4 = (x - x.mean()) / x.std()

    signals = [s1, s2, s3, s4]
    labels = ["Periodic", "Regime shift", "Laminar phases", "Chaotic"]

    return t, signals, labels

# =========================================================
# 2. Embedding & recurrence plot with quantile epsilon
# =========================================================

def embed_time_series(x, m=3, tau=4):
    x = np.asarray(x)
    N = len(x) - (m - 1) * tau
    if N <= 0:
        raise ValueError("Time series too short for given m, tau.")
    return np.column_stack([x[i:i + N] for i in range(0, m * tau, tau)])

def recurrence_plot_quantile(x, m=3, tau=4, target_rr=0.03):
    Y = embed_time_series(x, m=m, tau=tau)
    diff = Y[:, None, :] - Y[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # choose epsilon from OFF-diagonal distances only
    iu = np.triu_indices_from(dist, k=1)
    flat = dist[iu]
    eps = np.quantile(flat, target_rr)

    R = dist <= eps
    return R, eps

# =========================================================
# 3. RQA helpers: lines & metrics
# =========================================================

def diagonal_line_lengths(R, min_len=2):
    n = R.shape[0]
    lengths = []

    # start on top row
    for j0 in range(n):
        i, j = 0, j0
        while i < n and j < n:
            if R[i, j] and (i == 0 or j == 0 or not R[i-1, j-1]):
                L = 0
                ii, jj = i, j
                while ii < n and jj < n and R[ii, jj]:
                    L += 1
                    ii += 1
                    jj += 1
                if L >= min_len:
                    lengths.append(L)
                i = ii
                j = jj
            else:
                i += 1
                j += 1

    # start on left column (excluding 0,0)
    for i0 in range(1, n):
        i, j = i0, 0
        while i < n and j < n:
            if R[i, j] and (i == 0 or j == 0 or not R[i-1, j-1]):
                L = 0
                ii, jj = i, j
                while ii < n and jj < n and R[ii, jj]:
                    L += 1
                    ii += 1
                    jj += 1
                if L >= min_len:
                    lengths.append(L)
                i = ii
                j = jj
            else:
                i += 1
                j += 1

    return lengths

def vertical_line_lengths(R, min_len=2):
    n = R.shape[0]
    lengths = []
    for j in range(n):
        i = 0
        while i < n:
            if R[i, j] and (i == 0 or not R[i-1, j]):
                L = 0
                ii = i
                while ii < n and R[ii, j]:
                    L += 1
                    ii += 1
                if L >= min_len:
                    lengths.append(L)
                i = ii
            else:
                i += 1
    return lengths

def rqa_metrics(R, min_len_diag=2, min_len_vert=2):
    total_pts = R.size
    recur_pts = R.sum()
    rr = recur_pts / total_pts

    if recur_pts == 0:
        return rr, np.nan, np.nan, np.nan

    diag_lens = diagonal_line_lengths(R, min_len_diag)
    vert_lens = vertical_line_lengths(R, min_len_vert)

    det = (sum(diag_lens) / recur_pts) if diag_lens else 0.0
    lam = (sum(vert_lens) / recur_pts) if vert_lens else 0.0

    if diag_lens:
        vals, counts = np.unique(diag_lens, return_counts=True)
        p = counts / counts.sum()
        ent_raw = -np.sum(p * np.log(p))
        ent = ent_raw / np.log(len(p)) if len(p) > 1 else 0.0
    else:
        ent = 0.0

    return rr, det, lam, ent

def sliding_metrics(R, window=150, step=20,
                    min_len_diag=2, min_len_vert=2):
    n = R.shape[0]
    centers = []
    RR, DET, LAM, ENT = [], [], [], []

    k = 0
    while k + window <= n:
        sub = R[k:k+window, k:k+window]
        rr, det, lam, ent = rqa_metrics(sub, min_len_diag, min_len_vert)
        centers.append(k + window//2)
        RR.append(rr)
        DET.append(det)
        LAM.append(lam)
        ENT.append(ent)
        k += step

    return (np.array(centers),
            np.array(RR), np.array(DET),
            np.array(LAM), np.array(ENT))

# =========================================================
# 4. Make the figure
# =========================================================

def make_rqa_demo(m=6, tau=10, target_rr=0.03,
                  window=150, step=20,
                  fname="rqa_demo_4signals.png"):
    t, signals, labels = make_signals()
    ncols = len(signals)

    fig, axes = plt.subplots(3, ncols, figsize=(3.2 * ncols, 7),
                             sharex="col")

    for col, (sig, label) in enumerate(zip(signals, labels)):
        # --- Top: time series ---
        ax_ts = axes[0, col]
        ax_ts.plot(t, sig, color="black", linewidth=1)
        ax_ts.set_xlim(t[0], t[-1])
        ax_ts.set_title(label)
        if col == 0:
            ax_ts.set_ylabel("Signal")
        ax_ts.tick_params(axis="x", bottom=False, labelbottom=False)

        # --- Middle: RP ---
        R, eps = recurrence_plot_quantile(
            sig, m=m, tau=tau, target_rr=target_rr
        )
        ax_rp = axes[1, col]
        ax_rp.imshow(R, origin="lower", cmap="binary",
                     aspect="equal", interpolation="nearest")
        if col == 0:
            ax_rp.set_ylabel("Time index j")
        ax_rp.tick_params(axis="both", which="both",
                          bottom=False, left=False,
                          labelbottom=False, labelleft=False)

        # --- Bottom: sliding metrics with offsets ---
        centers, RR, DET, LAM, ENT = sliding_metrics(
            R, window=window, step=step
        )
        t_emb = t[(m - 1)*tau:]
        t_centers = t_emb[centers]

        ax_m = axes[2, col]
        # vertical offsets so traces are separated
        off_rr, off_det, off_lam, off_ent = 0.0, 1.2, 2.4, 3.6
        ax_m.plot(t_centers, RR  + off_rr,  color="tab:blue",  label="RR")
        ax_m.plot(t_centers, DET + off_det, color="tab:orange", label="DET")
        ax_m.plot(t_centers, LAM + off_lam, color="tab:green", label="LAM")
        ax_m.plot(t_centers, ENT + off_ent, color="tab:red",   label="ENT")
        ax_m.set_xlim(t[0], t[-1])
        if col == 0:
            ax_m.set_ylabel("Metrics (offset)")

        ax_m.set_xlabel("Time")
        ax_m.set_yticks([off_rr, off_det, off_lam, off_ent])
        ax_m.set_yticklabels(["RR", "DET", "LAM", "ENT"], fontsize=8)

        if col == 0:
            ax_m.legend(loc="upper right", fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    make_rqa_demo()
