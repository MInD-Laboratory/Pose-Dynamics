"""Add an MdCRQA row to manuscript Figure 3 (the RQA workflow schematic).

The original figure is a hand-assembled image with three rows (Auto-RQA,
Cross-RQA, MdRQA). This script draws a fourth row in the same style and
stacks it underneath:

    time series (two multivariate systems) -> shared state space -> cross-RP

MdCRQA compares two multivariate systems (e.g., leader and follower), with
no delay embedding: each system's channels are the state-space axes, and a
recurrence marks when the follower's state matches a state the leader
occupied (at any time).

It also relabels the delay axes of the Auto-RQA and Cross-RQA rows from
"x(t-T)", "x(t-2T)" to "x(t−τ)", "x(t−2τ)" to match the manuscript's notation.

Panel positions are measured in pixels from the MdRQA row of the original,
shifted down by one row pitch, so the new row lines up with the others.

Usage:
    python examples/fig3_mdcrqa_row.py

Reads examples/assets/rqa_general_method_base.png (the original three-row
figure) and writes figs/rqa_general_method_mdcrqa.png. The signals are
synthetic with a fixed seed, so the output is identical on every run.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # fixed pixel size (no retina doubling)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "examples" / "assets" / "rqa_general_method_base.png"
OUT = REPO / "figs" / "rqa_general_method_mdcrqa.png"

# ---- style taken from the original image -----------------------------------
DPI = 100
ROW_PITCH = 503                     # px between rows in the original
BOX_FILL = "#DCEBF9"
BOX_EDGE = "#186183"
DARK = ["#3C4698", "#2E3A88", "#5463B5"]    # leader channels (navy)
LIGHT = ["#79B4DC", "#9CCAE8", "#5FA3D4"]   # follower channels (light blue)
SPINE_W = 4 * 72 / DPI              # 4 px spines, in points
SERIF = "Times New Roman"
SANS = "Arial"
TICK_FS = 14.5
LABEL_FS = 17
ROW_FS = 22


def make_signals(n=300, t_max=25.0, lag=0.9, seed=3):
    """Leader: 3 noisy, flat-topped oscillations. Follower: leader delayed by
    `lag` time units, slightly damped, with its own noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, t_max, n)
    period = 6.1
    phases = [0.0, 1.57, 0.5]

    def channel(tt, ph, noise):
        base = np.tanh(1.6 * np.sin(2 * np.pi * tt / period + ph)) / np.tanh(1.6)
        wobble = 0.12 * np.sin(2 * np.pi * tt / 1.3 + 2 * ph)
        walk = np.convolve(rng.standard_normal(len(tt)), np.ones(3) / 3, "same")
        return 0.92 * base + wobble + noise * walk

    leader = np.column_stack([channel(t, ph, 0.10) for ph in phases])
    follower = np.column_stack([0.95 * channel(t - lag, ph, 0.13) for ph in phases])
    return t, leader, follower


def cross_recurrence(X, Y, target_rr=0.06):
    """Cross-recurrence matrix R[i, j] = 1 if ||X_j - Y_i|| < radius.
    Rows index the follower (y axis), columns the leader (x axis); the
    radius is set to give a fixed recurrence rate."""
    D = np.linalg.norm(Y[:, None, :] - X[None, :, :], axis=-1)
    radius = np.quantile(D, target_rr)
    return D < radius


def px_rect(x0, y0, x1, y1, W, H):
    """Pixel box (top-left origin) -> matplotlib [left, bottom, w, h]."""
    return [x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H]


def style_2d(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(SPINE_W)
    ax.tick_params(width=1.4, length=6, labelsize=TICK_FS)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(SANS)


def draw_row(W, H):
    t, X, Y = make_signals()
    R = cross_recurrence(X, Y)

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_alpha(0)

    # Row label box (same geometry as the other rows' boxes).
    box_ax = fig.add_axes(px_rect(0, 70, 66, 354, W, H))
    box_ax.set_axis_off()
    box_ax.set_xlim(0, 66)
    box_ax.set_ylim(0, 284)
    box_ax.add_patch(FancyBboxPatch((4.5, 6), 57, 272, boxstyle="square,pad=0",
                                    fc=BOX_FILL, ec=BOX_EDGE, lw=2.6))
    box_ax.text(33, 142, "MdCRQA", rotation=90, ha="center", va="center",
                fontsize=ROW_FS, fontfamily=SERIF)

    # Time series: leader (dark) and follower (light).
    ax = fig.add_axes(px_rect(145, 24, 734, 425, W, H))
    for k in range(3):
        ax.plot(t, X[:, k], color=DARK[k], lw=2.6, zorder=3)
        ax.plot(t, Y[:, k], color=LIGHT[k], lw=2.6, zorder=2)
    ax.set_xlim(-1.22, 26.46)
    ax.set_ylim(-1.23, 1.18)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Time", fontsize=LABEL_FS, fontfamily=SERIF, labelpad=12)
    style_2d(ax)

    # Shared state space: both systems' trajectories, no embedding.
    ax3 = fig.add_axes(px_rect(735, 20, 1235, 490, W, H), projection="3d")
    ax3.patch.set_alpha(0)
    ax3.plot(*X.T, color=DARK[0], lw=2.2)
    ax3.plot(*Y.T, color=LIGHT[0], lw=2.2)
    # Dots: follower states that recur with some leader state.
    rec_i = np.where(R.any(axis=1))[0][::9]
    ax3.scatter(*Y[rec_i].T, c=np.linspace(0, 1, len(rec_i)), cmap="plasma",
                s=36, edgecolors="white", linewidths=0.8, depthshade=False,
                zorder=5)
    ax3.view_init(elev=24, azim=-58)
    ax3.set_box_aspect((1, 0.8, 0.75), zoom=0.92)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color="#BBBBBB", linewidth=1.3)
        axis.line.set_linewidth(SPINE_W * 0.8)
    for setter in (ax3.set_xticks, ax3.set_yticks):
        setter([-1, -0.5, 0, 0.5, 1])
    ax3.set_zticks([-0.5, 0, 0.5])
    ax3.tick_params(labelsize=TICK_FS - 0.5, pad=1)
    for lab in ax3.get_xticklabels() + ax3.get_yticklabels() + ax3.get_zticklabels():
        lab.set_fontfamily(SANS)
    ax3.set_xlabel("x(t)", fontsize=LABEL_FS, fontfamily=SERIF, labelpad=6)
    ax3.set_ylabel("y(t)", fontsize=LABEL_FS, fontfamily=SERIF, labelpad=12,
                   rotation=-60)
    ax3.yaxis.set_rotate_label(False)
    ax3.zaxis.set_rotate_label(False)
    ax3.set_zlabel("z(t)", fontsize=LABEL_FS, fontfamily=SERIF, labelpad=8,
                   rotation=90)

    # Cross-recurrence plot (asymmetric: no line of identity).
    axr = fig.add_axes(px_rect(1341, 47, 1742, 447, W, H))
    n = len(t)
    axr.imshow(R, cmap="binary", origin="lower", interpolation="nearest",
               extent=(0, n, 0, n))
    axr.set_xticks(range(0, 251, 50))
    axr.set_yticks(range(0, 251, 50))
    for side in axr.spines.values():
        side.set_linewidth(SPINE_W)
    axr.tick_params(width=1.4, length=6, labelsize=TICK_FS)
    for lab in axr.get_xticklabels() + axr.get_yticklabels():
        lab.set_fontfamily(SANS)

    fig.canvas.draw()
    row = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return Image.fromarray(row)


# Old delay-axis labels in the base image: pixel boxes (x0, x1, y0, y1) to
# clear, and where to draw the replacement (centre x, centre y, rotation).
TAU_LABELS = [
    # Auto-RQA row
    dict(text="x(t\u22122\u03c4)", clear=[(1230, 1263, 250, 330)], at=(1246, 290, 90)),
    dict(text="x(t\u2212\u03c4)", clear=[(1167, 1200, 452, 478), (1160, 1180, 474, 492),
                                (1147, 1174, 481, 507)], at=(1173, 480, 52)),
    # Cross-RQA row
    dict(text="x(t\u22122\u03c4)", clear=[(1235, 1268, 749, 829)], at=(1251, 789, 90)),
    dict(text="x(t\u2212\u03c4)", clear=[(1154, 1186, 950, 977), (1133, 1167, 971, 1007)],
         at=(1160, 979, 52)),
]


def fix_tau_labels(img):
    """Replace the "T" delay labels with tau (see TAU_LABELS)."""
    W, H = img.size
    px = np.asarray(img).copy()
    for lab in TAU_LABELS:
        for x0, x1, y0, y1 in lab["clear"]:
            px[y0:y1, x0:x1] = 0
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_alpha(0)
    for lab in TAU_LABELS:
        x, y, rot = lab["at"]
        fig.text(x / W, 1 - y / H, lab["text"], rotation=rot, ha="center",
                 va="center", rotation_mode="anchor", fontsize=LABEL_FS + 2,
                 fontfamily=SERIF)
    fig.canvas.draw()
    overlay = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy())
    plt.close(fig)
    out = Image.fromarray(px)
    out.alpha_composite(overlay)
    return out


def main():
    src = fix_tau_labels(Image.open(SRC).convert("RGBA"))
    W, H0 = src.size
    row = draw_row(W, ROW_PITCH)
    out = Image.new("RGBA", (W, H0 + ROW_PITCH), (0, 0, 0, 0))
    out.paste(src, (0, 0))
    out.alpha_composite(row, (0, H0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT)
    print(f"wrote {OUT} ({out.size[0]}x{out.size[1]})")


if __name__ == "__main__":
    main()
