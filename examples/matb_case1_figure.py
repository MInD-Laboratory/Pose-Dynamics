"""
Regenerate the Case 1 (MATB) manuscript figure from a saved results table.

Produces ``figs/matb_results.svg`` and ``figs/matb_results.png``: the four
representative panels of linear and recurrence-based facial dynamics across the
three cognitive-load conditions (mean +/- SEM over analysis windows).

Fonts
-----
The manuscript figures are set in Times New Roman. Matplotlib writes SVG text one
of two ways, chosen with ``--font-mode``:

``outline`` (default)
    ``svg.fonttype = "path"`` -- every glyph becomes a vector outline. The type is
    then part of the drawing, so the figure renders identically wherever it is
    opened and needs no font installed. This is what "embedded font" means for a
    submission, and it is the safer default for typesetting.
``text``
    ``svg.fonttype = "none"`` -- text stays live and merely *references*
    ``Times New Roman``. Editable in Illustrator and searchable, but it renders
    correctly only where that font is available.

Run
---
    python examples/matb_case1_figure.py \
        --results notebooks/matb_case1_results_v2.csv \
        --out-dir figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

FONT_STACK = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]


def apply_style(font_mode: str) -> None:
    """Times New Roman throughout, with SVG text written per ``font_mode``."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_STACK,
        "mathtext.fontset": "stix",          # math glyphs that match Times
        "svg.fonttype": "path" if font_mode == "outline" else "none",
        "pdf.fonttype": 42,                  # TrueType, not Type 3
        "ps.fonttype": 42,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="notebooks/matb_case1_results_v2.csv",
                    help="Case 1 results table (one row per trial-window).")
    ap.add_argument("--out-dir", default="figs")
    ap.add_argument("--stem", default="matb_results")
    ap.add_argument("--font-mode", choices=("outline", "text"), default="outline")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    apply_style(args.font_mode)
    import matplotlib.pyplot as plt
    from pose_dynamics.case_studies.matb.reproduce import CONDITION_ORDER, plot_case1_figure

    df = pd.read_csv(args.results)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)

    # A results table produced with several cross-RQA minimum line lengths names its
    # columns ``crqa_l{n}_*``. The committed configuration is l_min = C.CROSS_MINL, so
    # alias that set to the plain ``crqa_*`` names the figure expects.
    if not any(c.startswith("crqa_") and not c.startswith("crqa_l") for c in df.columns):
        from pose_dynamics.case_studies.matb import config as C
        prefix = f"crqa_l{C.CROSS_MINL}_"
        aliased = {c: "crqa_" + c[len(prefix):] for c in df.columns if c.startswith(prefix)}
        if not aliased:
            raise SystemExit(f"no cross-RQA columns for l_min={C.CROSS_MINL} in {args.results}")
        df = df.rename(columns=aliased)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2))
    plot_case1_figure(df, axes=axes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg, png = out_dir / f"{args.stem}.svg", out_dir / f"{args.stem}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=args.dpi, bbox_inches="tight")

    n_trials = df.groupby(["participant", "condition"], observed=True).ngroups
    print(f"{len(df)} windows / {n_trials} trials / {df.participant.nunique()} participants")
    print(f"wrote {svg} ({args.font_mode}) and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
