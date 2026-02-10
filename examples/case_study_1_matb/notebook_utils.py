from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

try:
    from rpy2.robjects import pandas2ri

    # activate conversion rules (fix ContextVar warning)
    try:
        pandas2ri.activate()
    except Exception:
        # best-effort; localconverter still used in functions
        pass
    _HAVE_RPY2 = True
except Exception:
    _HAVE_RPY2 = False

COND_ORDER = ["L", "M", "H"]
DESIRED_ORDER = ["Vel", "Acc", "Rms"]
# Columns that are not metric values (metadata / identifiers)
NON_METRIC_COLS = {
    "participant",
    "condition",
    "window_index",
    "column",
    "_source_file",
}


# -----------------------
# small helpers
# -----------------------
def split_metric_name(name: str) -> Tuple[str, str]:
    pretty = (
        name.replace("avg", "Average")
        .replace("dist", "Amplitude")
        .replace("mean_abs_", "")
        .replace("mean_abs", "")
        .replace("_", " ")
        .title()
    )
    parts = pretty.split()
    return (pretty, "") if len(parts) < 2 else (" ".join(parts[:-1]), parts[-1])


def fmt(beta: float | None, p: float | None) -> Tuple[str, str]:
    # beta now holds SMD (Cohen's d / Hedges-like)
    if beta is None or (isinstance(beta, float) and not np.isfinite(beta)):
        b = "--"
    else:
        b = f"$d = {beta:.2f}$" if abs(beta) >= 1e-3 else f"$d = {beta:.3g}$"
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return b, "--"
    return b, (r"$p < .001$" if p < 0.001 else f"$p = {p:.3f}$")


# def fmt(beta: float | None, p: float | None) -> Tuple[str,str]:
#     b = "--" if beta is None or (isinstance(beta,float) and not np.isfinite(beta)) else f"$\\beta = {beta:.3f}$"
#     if p is None or (isinstance(p,float) and not np.isfinite(p)):
#         return b, "--"
#     return b, (r"$p < .001$" if p < 0.001 else f"$p = {p:.3f}$")


# -----------------------
# rpy2 wrapper: lmer + emmeans
# -----------------------
def _robust_ci_cols(ci_pd: pd.DataFrame) -> Tuple[str | None, str | None]:
    cand = list(ci_pd.columns)
    lower = next((c for c in cand if c.lower().startswith("lower")), None)
    upper = next((c for c in cand if c.lower().startswith("upper")), None)
    if lower and upper:
        return lower, upper
    lower = next((c for c in cand if "lcl" in c.lower()), None)
    upper = next((c for c in cand if "ucl" in c.lower()), None)
    return lower, upper


def run_rpy2_lmer(df: pd.DataFrame, dv: str, adjust: str = "tukey"):
    """
    Fit: dv ~ condition + widx_c + (1 + widx_c || participant)
    Returns:
      pairs_est: dict[(lo,hi)->float]  standardized mean differences (hi - lo)
      pairs_p:   dict[(lo,hi)->float]  p-values from emmeans::pairs(emm)
      means:     dict[cond->float]     emmeans (raw units)
      cis:       dict[cond->(lo,hi)]   95% CI for emmeans (raw units)
    """
    # ---- prep pandas data ----
    need = ["participant", "condition", "window_index", dv]
    d = df[need].dropna().copy()
    d = d.rename(columns={dv: "dv"})
    d["condition"] = pd.Categorical(
        d["condition"].astype(str).str.strip().str.upper(),
        categories=["L", "M", "H"],
        ordered=True,
    )
    w = pd.to_numeric(d["window_index"], errors="coerce")
    w = (w - np.nanmean(w)) / (np.nanstd(w) if np.nanstd(w) != 0 else 1.0)
    d["widx_c"] = w.fillna(0.0)

    # ---- R bridge ----
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    # load R packages for mixed models and emmeans; bind to underscore to avoid lint
    _ = importr("lme4")
    _ = importr("lmerTest")
    _ = importr("emmeans")

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["dat"] = ro.conversion.py2rpy(d)

    ro.r('emmeans::emm_options(lmer.df = "satterthwaite", lmerTest.limit = 4000)')
    ro.r("dat$participant <- factor(dat$participant)")
    ro.r(
        'dat$condition   <- factor(dat$condition, levels=c("L","M","H"), ordered=TRUE)'
    )
    ro.r("dat$widx_c      <- as.numeric(dat$widx_c)")

    # ---- fit with random slopes; fallback to intercept-only if singular ----
    ro.r('ctrl <- lme4::lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e6))')
    ro.r(
        "fit_try <- suppressMessages(suppressWarnings("
        "  lmerTest::lmer(dv ~ condition + widx_c + (1 + widx_c || participant), data=dat, control=ctrl)))"
    )
    ro.r("is_sing <- lme4::isSingular(fit_try, tol=1e-6)")
    ro.r("bad_vc  <- any(unlist(lme4::VarCorr(fit_try)) < 1e-10, na.rm=TRUE)")
    ro.r(
        "fit <- if (is_sing || bad_vc) { "
        "  suppressMessages(suppressWarnings("
        "    lmerTest::lmer(dv ~ condition + widx_c + (1 | participant), data=dat, control=ctrl))) "
        "} else { fit_try }"
    )

    # ---- emmeans + pairwise + standardized effect sizes ----
    ro.r("emm <- emmeans::emmeans(fit, ~ condition)")
    ro.r(f'pw  <- pairs(emm, adjust = "{adjust}")')
    ro.r(
        "pw_es <- tryCatch(emmeans::eff_size(pw, sigma = sigma(fit), edf = df.residual(fit)), error=function(e) NULL)"
    )
    ro.r("sig <- as.numeric(sigma(fit))")  # model residual sigma for fallback

    # pull frames
    emm_df_r = ro.r("as.data.frame(emm)")
    try:
        ci_df_r = ro.r("as.data.frame(confint(emm, level=0.95))")
    except Exception:
        ci_df_r = None
    pwc_df_r = ro.r("as.data.frame(pw)")
    pwes_df_r = ro.r("if (is.null(pw_es)) data.frame() else as.data.frame(pw_es)")
    sig_r = float(ro.r("sig")[0])

    with localconverter(ro.default_converter + pandas2ri.converter):
        emm_pd = ro.conversion.rpy2py(emm_df_r)
        pwc_pd = ro.conversion.rpy2py(pwc_df_r)
        pwes_pd = ro.conversion.rpy2py(pwes_df_r)
        ci_pd = ro.conversion.rpy2py(ci_df_r) if ci_df_r is not None else pd.DataFrame()

    # ---- emmeans (raw units) ----
    means = {str(r["condition"]): float(r["emmean"]) for _, r in emm_pd.iterrows()}

    # ---- CIs (raw units) ----
    def _robust_ci_cols(ci_pd: pd.DataFrame):
        cand = list(ci_pd.columns)
        lower = next((c for c in cand if c.lower().startswith("lower")), None)
        upper = next((c for c in cand if c.lower().startswith("upper")), None)
        if not (lower and upper):
            lower = next((c for c in cand if "lcl" in c.lower()), None)
            upper = next((c for c in cand if "ucl" in c.lower()), None)
        return lower, upper

    cis = {}
    if not ci_pd.empty and "condition" in ci_pd.columns:
        lower_col, upper_col = _robust_ci_cols(ci_pd)
        if lower_col and upper_col:
            for _, r in ci_pd.iterrows():
                cis[str(r["condition"])] = (float(r[lower_col]), float(r[upper_col]))
    if not cis:
        se_col = next(
            (c for c in emm_pd.columns if c.lower() in ("se", "stderr", "std.error")),
            None,
        )
        if se_col:
            for _, r in emm_pd.iterrows():
                m = float(r["emmean"])
                se = float(r[se_col])
                cis[str(r["condition"])] = (m - 1.96 * se, m + 1.96 * se)
        else:
            for k in means:
                cis[k] = (float("nan"), float("nan"))

    # ---- pairwise SMDs + p-values ----
    # 1) Build p map from pairs(...)
    pcol = (
        "p.value"
        if "p.value" in pwc_pd.columns
        else next((c for c in pwc_pd.columns if c.lower().startswith("p")), None)
    )
    order = {"L": 0, "M": 1, "H": 2}
    pairs_p = {}

    # 2) Robust SMD column detection on pw_es
    smd_col = None
    for candidate in ["effect.size", "SMD", "g", "d", "es", "ES"]:
        if candidate in pwes_pd.columns:
            smd_col = candidate
            break

    # 3) Build maps for SMD (preferred) and raw estimates (fallback)
    smd_map = {}
    if smd_col:
        for _, r in pwes_pd.iterrows():
            contrast = str(r.get("contrast", "")).replace("–", "-").replace(" - ", "-")
            smd_map[contrast] = float(r.get(smd_col, np.nan))

    est_map = {}
    if "estimate" in pwc_pd.columns:
        for _, r in pwc_pd.iterrows():
            contrast = str(r.get("contrast", "")).replace("–", "-").replace(" - ", "-")
            est_map[contrast] = float(r.get("estimate", np.nan))

    # 4) Normalize to (lo,hi) with sign as (hi - lo)
    pairs_est = {}
    for _, r in pwc_pd.iterrows():
        contrast = str(r.get("contrast", "")).replace("–", "-").replace(" - ", "-")
        parts = [p.strip() for p in contrast.split("-")]
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        if a not in order or b not in order or a == b:
            continue

        # get p
        pv = float(r[pcol]) if (pcol and pd.notnull(r[pcol])) else float("nan")

        # preferred: SMD from eff_size; fallback: estimate/sigma
        est_lr = smd_map.get(contrast, np.nan)
        if (
            not np.isfinite(est_lr)
            and np.isfinite(sig_r)
            and sig_r > 0
            and contrast in est_map
        ):
            est_lr = est_map[contrast] / sig_r

        lo, hi = (a, b) if order[a] < order[b] else (b, a)
        est_hi_minus_lo = est_lr if (a == hi and b == lo) else -est_lr

        pairs_est[(lo, hi)] = est_hi_minus_lo
        pairs_p[(lo, hi)] = pv

    return pairs_est, pairs_p, means, cis


# --------------------------
# plotting helper
# --------------------------
def barplot_ax(
    ax,
    means: List[float],
    sems: List[float],
    pvals: List[float],
    ylabel: str,
    metric_name: str,
    colors: List[str] | None = None,
    bar_width: float = 0.80,
    ylim_padding: Tuple[float, float] = (0.4, 0.1),
):
    if colors is None:
        colors = ["#4575b4", "#ffffbf", "#d73027"]

    import numpy as _np

    x = _np.arange(len(means))

    ax.bar(
        x,
        means,
        yerr=sems,
        capsize=4,
        color=colors,
        width=bar_width,
        edgecolor="black",
        linewidth=4,
    )

    lowers = [m - (s if not _np.isnan(s) else 0) for m, s in zip(means, sems)]
    uppers = [m + (s if not _np.isnan(s) else 0) for m, s in zip(means, sems)]
    y_min = min(lowers)
    y_max = max(uppers)
    y_span = y_max - y_min if y_max > y_min else 1.0
    pairs = [(0, 1, pvals[0]), (0, 2, pvals[1]), (1, 2, pvals[2])]
    sig_pairs = [
        (i, j, p)
        for (i, j, p) in pairs
        if (p is not None and not np.isnan(p) and p < 0.05)
    ]
    sig_pairs = sorted(sig_pairs, key=lambda t: (t[1] - t[0]))
    h_step = 0.2 * y_span
    line_h = 0.03 * y_span
    y0 = y_max + 0.04 * y_span
    for idx, (i, j, p) in enumerate(sig_pairs):
        y = y0 + idx * h_step
        ax.plot(
            [x[i], x[i], x[j], x[j]],
            [y, y + line_h, y + line_h, y],
            lw=1.5,
            color="black",
            clip_on=False,
        )
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ax.text(
            (x[i] + x[j]) / 2,
            y + 0.25 * line_h,
            stars,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="black",
            clip_on=False,
        )
    ax.set_xlim(-0.5, len(means) - 0.5)
    ax.set_xticks([])
    ax.set_ylabel(
        "\n".join(textwrap.wrap(ylabel, width=25)), weight="bold", fontsize=12
    )
    ax.set_ylim(
        y_min - ylim_padding[0] * y_span,
        y_max + ylim_padding[1] * y_span + len(sig_pairs) * h_step,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(axis="y", width=1.3, labelsize=11)
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")


# --------------------------
# discovery + loading
# --------------------------
def discover_linear_files(
    root: Path = Path("data/processed_data"),
) -> Dict[str, List[Path]]:
    sessions = {}
    root = Path(root)
    for session_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        lm_dir = session_dir / "linear_metrics"
        if lm_dir.exists() and any(lm_dir.glob("*.csv")):
            sessions[session_dir.name] = sorted(lm_dir.glob("*.csv"))
    return sessions


def load_session_csvs(files: List[Path]) -> pd.DataFrame:
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "participant" not in df.columns and "participant_id" in df.columns:
                df = df.rename(columns={"participant_id": "participant"})
            df["_source_file"] = str(f.name)
            parts.append(df)
        except Exception as e:
            print(f"[WARN] failed to load {f}: {e}")
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


# --------------------------
# table builder + plots (main)
# --------------------------
def build_table_with_emmeans(
    df: pd.DataFrame, out_tex: str | Path, figs_dir: str | Path
):
    from collections import defaultdict

    out_tex = Path(out_tex)
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # prepare df
    df = df.copy()
    df["condition"] = df["condition"].astype(str).str.strip().str.upper()
    df = df[df["condition"].isin(COND_ORDER)].copy()
    if "window_index" not in df.columns:
        df["window_index"] = 0

    for c in df.columns:
        if c not in NON_METRIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    metric_cols = [
        c
        for c in df.columns
        if c not in NON_METRIC_COLS
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
    ]

    # ---- helper: bucket by kinematics (displacement/velocity/acceleration) ----
    def bucket_kind(metric_name: str, metric_type_label: str) -> str:
        mn = metric_name.lower()
        mt = (metric_type_label or "").lower()
        # Prefer column-name cues; fall back to human label
        if ("_vel" in mn) or (" velocity" in mt) or (mt == "velocity"):
            return "vel"
        if ("_acc" in mn) or (" acceleration" in mt) or (mt == "acceleration"):
            return "acc"
        # everything else is displacement/RMS/etc.
        return "disp"

    # storage: region -> list of dict rows (we keep kind to split later)
    grouped = defaultdict(list)

    modeled = skipped = 0
    for metric in metric_cols:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        ser = df[metric]
        n_total = ser.shape[0]
        n_na = ser.isna().sum()
        if n_na == n_total:
            skipped += 1
            print(f"[skip] {metric}: all NA ({n_na}/{n_total})")
            continue

        sub = df.loc[ser.notna(), ["condition", "participant", "window_index", metric]]
        conds = sorted(sub["condition"].unique().tolist())
        n_by_cond = sub.groupby("condition")[metric].size().to_dict()
        if not {"L", "M", "H"}.issubset(set(conds)):
            skipped += 1
            print(
                f"[skip] {metric}: missing condition(s). have={conds}, counts={n_by_cond}"
            )
            continue

        tmp = sub.rename(columns={metric: "dv"})
        try:
            pairs_est, pairs_p, means, cis = run_rpy2_lmer(tmp, "dv", adjust="none")
        except Exception as e:
            skipped += 1
            print(f"[skip] {metric}: model error -> {e}")
            continue

        b_m = pairs_est.get(("L", "M"), np.nan)
        p_m = pairs_p.get(("L", "M"), np.nan)
        b_h = pairs_est.get(("L", "H"), np.nan)
        p_h = pairs_p.get(("L", "H"), np.nan)
        b_hm = pairs_est.get(("M", "H"), np.nan)
        p_hm = pairs_p.get(("M", "H"), np.nan)

        Bm, Pm = fmt(b_m, p_m)
        Bh, Ph = fmt(b_h, p_h)
        Bhm, Phm = fmt(b_hm, p_hm)

        region, metric_type = split_metric_name(metric)
        kind = bucket_kind(metric, metric_type)  # NEW: tag as disp/vel/acc
        grouped[region].append(
            {
                "metric_type": metric_type,
                "kind": kind,
                "Bm": Bm,
                "Pm": Pm,
                "Bh": Bh,
                "Ph": Ph,
                "Bhm": Bhm,
                "Phm": Phm,
            }
        )
        modeled += 1

        # plot per metric (unchanged)
        conds = ["L", "M", "H"]
        mean_vals = [means.get(c, float("nan")) for c in conds]
        sems = []
        for c in conds:
            if c in cis and cis[c] is not None:
                lo, hi = cis[c]
                sems.append(
                    (hi - lo) / 3.92
                    if (not pd.isna(lo) and not pd.isna(hi))
                    else float("nan")
                )
            else:
                sems.append(float("nan"))
        pvals_for_plot = [p_m, p_h, p_hm]
        fig, ax = plt.subplots(figsize=(4, 5))
        barplot_ax(
            ax,
            mean_vals,
            sems,
            pvals_for_plot,
            ylabel=metric.replace("_", " ").title(),
            metric_name=metric,
        )
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=11, weight="bold")
        out_svg = figs_dir / f"{metric}.svg"
        fig.savefig(out_svg, bbox_inches="tight")
        plt.close(fig)

    # ---- helper to write ONE latex table given a bucket (disp/vel/acc) ----
    def write_table_for_bucket(bucket: str, suffix: str):
        lines = [
            r"\begin{tabular}{llcc|cc|cc}",
            r"\toprule",
            r"Region & Metric & $d_{\text{M--L}}$ & $p_{\text{M--L}}$ & $d_{\text{H--L}}$ & $p_{\text{H--L}}$ & $d_{\text{H--M}}$ & $p_{\text{H--M}}$ \\",
            r"\midrule",
        ]
        wrote_any = False
        for region in sorted(grouped.keys()):
            # filter rows for this bucket
            rows_all = grouped[region]
            rows = [r for r in rows_all if r["kind"] == bucket]
            if not rows:
                continue
            # order within region
            rows.sort(
                key=lambda x: DESIRED_ORDER.index(x["metric_type"])
                if x["metric_type"] in DESIRED_ORDER
                else len(DESIRED_ORDER)
            )
            first = True
            for r in rows:
                region_label = (
                    f"\\multirow{{{len(rows)}}}{{*}}{{{region}}}" if first else ""
                )
                lines.append(
                    f"{region_label} & {r['metric_type']} & {r['Bm']} & {r['Pm']} & {r['Bh']} & {r['Ph']} & {r['Bhm']} & {r['Phm']} \\\\"
                )
                first = False
            lines.append(r"\midrule")
            wrote_any = True

        lines += [r"\bottomrule", r"\end{tabular}"]

        # only write a file if we have rows for this bucket
        if wrote_any:
            path = out_tex.with_name(out_tex.stem + suffix + out_tex.suffix)
            path.write_text("\n".join(lines), encoding="utf-8")
            print(f"[OK] wrote {path}")
        else:
            print(f"[note] no rows for bucket={bucket}; no table written.")

    # write 3 separate tables
    write_table_for_bucket("disp", "_disp")
    write_table_for_bucket("vel", "_vel")
    write_table_for_bucket("acc", "_acc")

    print(f"[DONE] modeled={modeled}, skipped={skipped}")
    return modeled, skipped


def pretty_metric(name: str) -> str:
    """Format metric names for display."""
    return name.replace("_", " ").title()


def run_stats_by_column(
    df: pd.DataFrame,
    metrics,
    *,
    adjust: str = "tukey",
    min_per_condition: int = 1,
    verbose: bool = True,
):
    """
    For each data 'column' and each metric, fit:
        dv ~ condition + widx_c + (1 + widx_c || participant)
    with automatic fallback to (1|participant), emmeans, pairwise tests,
    and standardized effect sizes (via run_rpy2_lmer).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ["participant","condition","window_index","column", <metrics...>]
    metrics : Iterable[str]
        Column names to use as DVs.
    adjust : str
        p-value adjustment method passed to emmeans::pairs (default "tukey").
    min_per_condition : int
        Minimum rows per condition to attempt a model (after NA drop).
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict
        Nested dict: results[column][metric] = (pairs_est, pairs_p, means, cis)
          - pairs_est : dict[(lo,hi)->float] standardized mean differences (hi - lo)
          - pairs_p   : dict[(lo,hi)->float] p-values
          - means     : dict[cond->float]     emmeans (raw units)
          - cis       : dict[cond->(lo,hi)]   95% CI for emmeans (raw units)
    """
    results = defaultdict(dict)

    # sanity: ensure required columns exist
    required = {"participant", "condition", "window_index", "column"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    # iterate deterministically by column label
    for col_name in sorted(pd.Series(df["column"]).dropna().unique().tolist()):
        if verbose:
            print(f"\nAnalyzing: {col_name}")
        dsub = df[df["column"] == col_name].copy()

        # must have at least 2 conditions in this slice (after cleaning)
        conds_raw = pd.Series(dsub["condition"]).dropna().astype(str)
        if conds_raw.empty:
            if verbose:
                print("  Skipping (no condition values)")
            continue

        # iterate metrics
        for metric in metrics:
            if metric not in dsub.columns:
                if verbose:
                    print(f"  – {metric}: not in dataframe; skipping")
                continue

            # build minimal frame and drop NAs
            need = ["participant", "condition", "window_index", metric]
            tmp = dsub[need].dropna().copy()
            if tmp.empty:
                if verbose:
                    print(f"  – {metric}: empty after NA drop; skipping")
                continue

            # normalize condition coding early (mirrors run_rpy2_lmer)
            tmp["condition"] = tmp["condition"].astype(str).str.strip().str.upper()

            # keep only conditions that have at least min_per_condition rows
            counts = tmp.groupby("condition", dropna=True).size()
            ok_levels = counts[counts >= max(1, int(min_per_condition))].index.tolist()
            tmp = tmp[tmp["condition"].isin(ok_levels)]

            uniq_conds = sorted(tmp["condition"].unique().tolist())
            if len(uniq_conds) < 2:
                if verbose:
                    n_conds = len(uniq_conds)
                    print(
                        f"  Skipping {metric} (only {n_conds} usable condition(s) after filtering)"
                    )
                continue

            try:
                # delegate all modeling/details to run_rpy2_lmer
                pairs_est, pairs_p, means, cis = run_rpy2_lmer(
                    tmp.rename(columns={metric: "dv"}), dv="dv", adjust=adjust
                )
                results[col_name][metric] = (pairs_est, pairs_p, means, cis)
                if verbose:
                    # quick, informative summary
                    have_es = ", ".join([f"{a}>{b}" for (a, b) in sorted(pairs_est)])
                    print(f"  ✓ {metric} — contrasts: [{have_es}]")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {metric}: {type(e).__name__}: {e}")

    return results


# --------------------------
# path helper to find project root
# --------------------------
def add_project_root_to_path():
    """Find project root and add to sys.path to make pose_dynamics importable."""
    import sys

    # Start from the current file's directory and go up until pyproject.toml is found
    current_dir = Path(__file__).parent
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists():
            project_root = parent
            src_path = project_root / "src"
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            return
    raise FileNotFoundError("Could not find project root with pyproject.toml")


# --------------------------
# AMI/FNN helpers for MATB (notebook)
# --------------------------
def run_ami_fnn_on_random_timeseries(
    df: pd.DataFrame,
    value_col: str,
    n_series: int = 20,
    max_lag: int = 40,
    ami_bins: int = 32,
    max_dim: int = 10,
    tau: int = 1,
    rtol: float = 10.0,
    atol: float = 2.0,
    min_length: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Sample n_series random time series from df[value_col], compute AMI and FNN.
    Returns dict with 'ami', 'fnn', 'ami_lags', 'fnn_dims', and 'series_idx'.
    """
    import numpy as np

    rng = np.random.default_rng(random_state)
    # Find all unique time series (by participant/trial/window if available)
    group_cols = [
        c for c in ["participant", "trial_id", "window_index"] if c in df.columns
    ]
    if group_cols:
        groups = df.groupby(group_cols)
        valid_keys = [k for k, g in groups if g[value_col].notna().sum() >= min_length]
        chosen = rng.choice(
            len(valid_keys), size=min(n_series, len(valid_keys)), replace=False
        )
        selected = [valid_keys[i] for i in chosen]
        series = [groups.get_group(k)[value_col].dropna().values for k in selected]
        series_idx = selected
    else:
        # fallback: treat each row as a time series
        arr = df[value_col].dropna().values
        if arr.size < min_length:
            raise ValueError("Not enough data for a single time series.")
        series = [arr]
        series_idx = [None]

    # Import AMI/FNN from package
    add_project_root_to_path()
    from tqdm import tqdm

    from pose_dynamics.rqa.params import _ami, _fnn

    ami_curves = []
    fnn_curves = []
    for x in tqdm(series, desc="Processing time series"):
        if len(x) < min_length:
            continue
        ami = _ami(x, max_lag=max_lag, bins=ami_bins)
        fnn = _fnn(x, max_dim=max_dim, tau=tau, rtol=rtol, atol=atol)
        ami_curves.append(ami)
        fnn_curves.append(fnn)

    # Pad to same length for mean/std
    def pad(arrs, length):
        return np.array(
            [
                np.pad(a, (0, max(0, length - len(a))), constant_values=np.nan)
                for a in arrs
            ]
        )

    ami_mat = pad(ami_curves, max_lag)
    fnn_mat = pad(fnn_curves, max_dim)

    return {
        "ami": ami_mat,
        "fnn": fnn_mat,
        "ami_lags": np.arange(1, max_lag + 1),
        "fnn_dims": np.arange(1, max_dim + 1),
        "series_idx": series_idx,
    }


def plot_ami_fnn_curves(results: dict, ax_ami=None, ax_fnn=None, label=None):
    """
    Plot mean ± std band for AMI and FNN curves from run_ami_fnn_on_random_timeseries.
    """
    import numpy as np

    # Simple plot: mean ± std band for AMI and FNN across runs
    ami = np.asarray(results.get("ami", []))
    fnn = np.asarray(results.get("fnn", []))
    ami_lags = np.asarray(
        results.get("ami_lags", np.arange(1, ami.shape[1] + 1 if ami.ndim > 1 else 2))
    )
    fnn_dims = np.asarray(
        results.get("fnn_dims", np.arange(1, fnn.shape[1] + 1 if fnn.ndim > 1 else 2))
    )

    # compute mean and std across the first axis (runs)
    if ami.ndim == 1:
        ami_mean = ami
        ami_std = np.zeros_like(ami_mean)
    else:
        ami_mean = np.nanmean(ami, axis=0)
        ami_std = np.nanstd(ami, axis=0)

    if fnn.ndim == 1:
        fnn_mean = fnn
        fnn_std = np.zeros_like(fnn_mean)
    else:
        fnn_mean = np.nanmean(fnn, axis=0)
        fnn_std = np.nanstd(fnn, axis=0)

    # create axes if not provided
    if ax_ami is None or ax_fnn is None:
        fig, (ax_ami, ax_fnn) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    else:
        fig = None

    ax_ami.plot(ami_lags, ami_mean, label=label or "AMI")
    ax_ami.fill_between(ami_lags, ami_mean - ami_std, ami_mean + ami_std, alpha=0.3)
    ax_ami.set_xlabel("lag")
    ax_ami.set_ylabel("AMI")
    ax_ami.set_title("Average Mutual Information")

    ax_fnn.plot(fnn_dims, fnn_mean, label=label or "FNN")
    ax_fnn.fill_between(fnn_dims, fnn_mean - fnn_std, fnn_mean + fnn_std, alpha=0.3)
    ax_fnn.set_xlabel("embedding dim")
    ax_fnn.set_ylabel("FNN")
    ax_fnn.set_title("False Nearest Neighbours")

    if fig is not None:
        fig.tight_layout()
        return fig, (ax_ami, ax_fnn)
    return None, (ax_ami, ax_fnn)
