"""Build the Case 2 (MOSAIC) full-results report as a standalone HTML page.

The point of this file is to be the *only* place Case 2's reporting choices live, so
a reviewer can be pointed at one page instead of a separate supplementary PDF. It
holds no analysis: the numbers come from
:mod:`pose_dynamics.case_studies.mosaic.reproduce` and
:mod:`~pose_dynamics.case_studies.mosaic.stats`, and the rendering from
:mod:`pose_dynamics.reporting`.

Usage::

    # from the notebook's parquet cache (fast, and what the manuscript reports)
    python examples/apa_report_case2.py --cache notebooks/case2_cache --out case2_results.html

    # or recompute everything from the raw recordings
    python examples/apa_report_case2.py --data-dir "G:/mosaic movement files" --out case2_results.html

Cases 1 and 3 follow the same shape: aggregate to the level you intend to describe,
build a list of :class:`~pose_dynamics.reporting.Table`, hand it to
:func:`~pose_dynamics.reporting.render_report`. Only ``LABELS`` and the table
definitions below are Case 2-specific.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pose_dynamics.case_studies.mosaic import config as CFG
from pose_dynamics.case_studies.mosaic.stats import (
    METRICS_DYAD,
    METRICS_IND,
    ROIS,
    fit_dyadic,
    fit_individual,
    to_trial_dyad,
    to_trial_individual,
    tukey_pairwise,
)
from pose_dynamics.reporting import Table, describe_by, fmt_num, fmt_signed, render_report

# ----------------------------------------------------------------------
# Display labels. The frames carry pipeline names; a reader wants the measure.
# ----------------------------------------------------------------------
ROI_LABEL = {"arms": "Arms", "upper_body": "Upper body", "centre_face": "Centre face"}
METRIC_LABEL = {
    "rms": "RMS velocity", "mean_vel": "Mean velocity", "sd_vel": "SD of velocity",
    "xcorr_lag0": "Cross-correlation (lag 0)",
    "cross_perc_recur": "%REC", "cross_perc_determ": "%DET",
    "cross_laminarity": "LAM", "cross_mean_line_length": "L<sub>mean</sub>",
    "cross_lmax": "L<sub>max</sub>", "cross_entropy": "Entropy",
    "cross_trapping_time": "Trapping time",
}
COND = list(CFG.CONDITION_ORDER)

#: Decimal places per measure. These span four orders of magnitude -- a trapping time
#: near 20 frames and a lag-0 cross-correlation near 0.001 sit in the same column of
#: the same table -- so a single column-wide precision would either round the
#: correlations to nothing or pad the percentages with meaningless digits. APA's
#: fixed-decimal convention applies within a measure, which is what this encodes.
METRIC_DP = {
    "rms": 4, "mean_vel": 4, "sd_vel": 4,
    "xcorr_lag0": 4,
    "cross_perc_recur": 3, "cross_perc_determ": 2, "cross_laminarity": 2,
    "cross_mean_line_length": 3, "cross_lmax": 2, "cross_entropy": 3,
    "cross_trapping_time": 3,
}


def by_measure(signed: bool = False, extra: int = 0):
    """Formatter whose decimals follow the row's measure (see :data:`METRIC_DP`)."""
    def fmt(value, row):
        dp = METRIC_DP.get(row.get("_metric"), 3) + extra
        return fmt_signed(value, dp) if signed else fmt_num(value, dp)
    return fmt


def _label(frame: pd.DataFrame) -> pd.DataFrame:
    """Map pipeline names to display labels, keeping the raw measure in ``_metric``.

    The raw name has to survive labelling because :func:`by_measure` selects precision
    from it, and it is the display label that reaches the reader.
    """
    out = frame.copy()
    if "roi" in out:
        out["roi"] = pd.Categorical(out["roi"].map(ROI_LABEL),
                                    categories=[ROI_LABEL[r] for r in ROIS], ordered=True)
    if "metric" in out:
        out["_metric"] = out["metric"]
        order = [m for m in (*METRICS_IND, *METRICS_DYAD) if m in set(out["metric"])]
        out["metric"] = pd.Categorical(out["metric"].map(METRIC_LABEL),
                                       categories=[METRIC_LABEL[m] for m in order],
                                       ordered=True)
    return out.sort_values([c for c in ("roi", "metric") if c in out])


# ----------------------------------------------------------------------
# Result loading
# ----------------------------------------------------------------------
def load_results(cache: Path | None, data_dir: Path | None, fingerprint: str | None):
    """Return the individual and dyadic window-level frames, cached or freshly run."""
    if cache:
        found = {}
        for stem in ("individual_aligned", "dyadic_aligned"):
            hits = sorted(cache.glob(f"{stem}.{fingerprint or '*'}.parquet"))
            if not hits:
                raise FileNotFoundError(f"no cached {stem} in {cache}")
            if len(hits) > 1 and not fingerprint:
                raise SystemExit(
                    f"{len(hits)} cached {stem} files with different pipeline "
                    f"fingerprints:\n  " + "\n  ".join(h.name for h in hits) +
                    "\nPass --fingerprint to choose; mixing fingerprints would report "
                    "tables computed under different pipeline settings side by side.")
            found[stem] = pd.read_parquet(hits[0])
        for f in found.values():
            if "condition" in f:
                f["condition"] = pd.Categorical(f["condition"], categories=COND, ordered=True)
        return found["individual_aligned"], found["dyadic_aligned"]

    from pose_dynamics.case_studies.mosaic import run_individual, run_reproduction
    files = sorted(p for p in data_dir.glob("S*_T*_*.csv") if not p.name.startswith("._"))
    return (run_individual(files, progress=False),
            run_reproduction(data_dir, progress=False))


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def composition_table(ind_w: pd.DataFrame, dyad_w: pd.DataFrame) -> Table:
    rows = []
    for cond in COND:
        i = ind_w[ind_w["condition"] == cond]
        d = dyad_w[dyad_w["condition"] == cond]
        rows.append({
            "condition": cond,
            "dyads": d["session"].nunique(),
            "trials": d.groupby(["session", "trial"], observed=True).ngroups,
            "participants": i.groupby(["session", "camera"], observed=True).ngroups,
            "ind_rows": len(i),
            "dyad_rows": len(d),
            "win_per_trial": (d.groupby(["session", "trial"], observed=True)["window"]
                              .nunique().mean()),
        })
    return Table(
        number="1",
        title="Sample composition by background-noise condition",
        frame=pd.DataFrame(rows),
        columns={"condition": "Condition", "dyads": "Dyads", "participants": "Participants",
                 "trials": "Trials", "win_per_trial": "Windows per trial",
                 "ind_rows": "Individual rows", "dyad_rows": "Dyadic rows"},
        formatters={"dyads": "int", "participants": "int", "trials": "int",
                    "win_per_trial": "num2", "ind_rows": "int", "dyad_rows": "int"},
        align={"condition": "left"},
    )


def descriptive_table(trial: pd.DataFrame, metrics, number: str, title: str) -> Table:
    frame, cols, spanners = describe_by(
        _label(trial), metrics=metrics, rows=["roi"], columns="condition",
        column_order=COND)
    frame["_metric"] = frame["metric"]
    frame["metric"] = frame["metric"].map(METRIC_LABEL)
    columns = {"roi": "ROI", "metric": "Measure"}
    fmts: dict[str, object] = {"metric": "html"}
    for c in cols[2:]:
        columns[c] = "<i>M</i>" if c.endswith(" M") else "<i>SD</i>"
        fmts[c] = by_measure()
    return Table(number=number, title=title, frame=frame, columns=columns,
                 formatters=fmts, spanners=spanners, stub_groups=["roi"],
                 align={"roi": "left", "metric": "left"})


def contrast_table(coefs: pd.DataFrame, number: str, title: str) -> Table:
    f = _label(coefs).copy()
    f["metric"] = f["metric"].astype(str)
    f["roi"] = f["roi"].astype(str)
    return Table(
        number=number, title=title, frame=f,
        columns={"roi": "ROI", "metric": "Measure", "vs_Office": "Contrast",
                 "intercept": "Office <i>M</i>", "beta": "<i>b</i>", "SE": "<i>SE</i>",
                 "t": "<i>t</i>", "df": "<i>df</i>", "p": "<i>p</i>", "n_obs": "<i>N</i>"},
        formatters={"metric": "html",
                    "intercept": by_measure(), "beta": by_measure(signed=True),
                    "SE": by_measure(extra=1), "t": "num2",
                    "df": "int", "p": "p", "n_obs": "int"},
        stub_groups=["roi", "metric"],
        align={"roi": "left", "metric": "left", "vs_Office": "left"})


def tukey_table(tk: pd.DataFrame, number: str, title: str) -> Table:
    f = _label(tk).copy()
    f["metric"] = f["metric"].astype(str)
    f["roi"] = f["roi"].astype(str)
    f["contrast"] = f["contrast"].str.replace(" - ", " − ", regex=False)
    return Table(
        number=number, title=title, frame=f,
        columns={"roi": "ROI", "metric": "Measure", "contrast": "Contrast",
                 "diff": "<i>M</i><sub>diff</sub>", "SE": "<i>SE</i>", "t": "<i>t</i>",
                 "df": "<i>df</i>", "p_tukey": "<i>p</i><sub>Tukey</sub>"},
        formatters={"metric": "html",
                    "diff": by_measure(signed=True), "SE": by_measure(extra=1),
                    "t": "num2", "df": "int", "p_tukey": "p"},
        stub_groups=["roi", "metric"],
        align={"roi": "left", "metric": "left", "contrast": "left"})


# ----------------------------------------------------------------------
def build(ind_w, dyad_w) -> list:
    """The seven tables, and nothing else -- no prose sections, no table notes.

    Everything a note would have said is either already in the manuscript's methods or
    recoverable from the code, so repeating it here would duplicate rather than
    document. Titles carry the identifying information APA requires of them.
    """
    ind_t, dyad_t = to_trial_individual(ind_w), to_trial_dyad(dyad_w)

    coef_ind, coef_dyad = fit_individual(ind_t), fit_dyadic(dyad_t)
    tk_ind = tukey_pairwise(ind_t, METRICS_IND, "individual")
    tk_dyad = tukey_pairwise(dyad_t, METRICS_DYAD, "dyad")

    return [
        composition_table(ind_w, dyad_w),
        descriptive_table(
            ind_t, METRICS_IND, "2",
            "Individual-level velocity-magnitude measures by ROI and condition"),
        descriptive_table(
            dyad_t, METRICS_DYAD, "3",
            "Dyadic cross-recurrence measures by ROI and condition"),
        contrast_table(coef_ind, "4",
                       "Individual-level condition contrasts against Office"),
        contrast_table(coef_dyad, "5",
                       "Dyadic condition contrasts against Office"),
        tukey_table(tk_ind, "6",
                    "Individual-level pairwise condition contrasts, Tukey-adjusted"),
        tukey_table(tk_dyad, "7",
                    "Dyadic pairwise condition contrasts, Tukey-adjusted"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, help="notebook parquet cache directory")
    ap.add_argument("--fingerprint", help="pipeline fingerprint to select within --cache")
    ap.add_argument("--data-dir", type=Path, help="raw MOSAIC directory (recomputes)")
    ap.add_argument("--out", type=Path, default=Path("case2_results.html"))
    ap.add_argument("--fragment", action="store_true",
                    help="emit a <style>+markup fragment instead of a full HTML document")
    ap.add_argument("--no-toc", action="store_true", help="omit the index of tables")
    args = ap.parse_args()
    if not args.cache and not args.data_dir:
        ap.error("give --cache or --data-dir")

    frames = load_results(args.cache, args.data_dir, args.fingerprint)
    blocks = build(*frames)
    html_out = render_report(
        "Case 2 (MOSAIC): full results",
        blocks,
        standalone=not args.fragment,
        toc=not args.no_toc,
    )
    args.out.write_text(html_out, encoding="utf-8")
    print(f"wrote {args.out}  ({len(html_out):,} bytes, {len(blocks)} blocks)")


if __name__ == "__main__":
    main()
