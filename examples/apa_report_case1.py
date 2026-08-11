"""Build the Case 1 (MATB) full-results report as a standalone HTML page.

The companion to ``examples/apa_report_case2.py``, and the same shape: aggregate to
the level you intend to describe, build a list of
:class:`~pose_dynamics.reporting.Table`, hand it to
:func:`~pose_dynamics.reporting.render_report`. Only the labels and the table
definitions below are Case 1-specific.

The point of the page is to be the one place a reviewer can be sent for Case 1's
numbers: the manuscript quotes a dozen representative effects out of several hundred,
and the rest have until now lived only in the notebook's saved CSVs.

Inputs
------
Everything comes from the artifacts the Case 1 notebook writes, all of which ship
with the repository:

``--results``
    window-level results table (one row per participant x load block x window),
    the same file ``examples/matb_case1_figure.py`` draws from. Descriptives and
    the sample composition are computed from it.
``--effects``
    the committed condition contrasts. These are the numbers the manuscript
    reports, so they are rendered as saved rather than refit; ``--refit`` instead
    recomputes them here with ``statsmodels`` (same model, marginally different
    standard errors -- see :func:`refit_effects`).
``--radius-sensitivity`` / ``--tau-sensitivity`` / ``--theiler``
    the robustness checks: condition effects refit across the recurrence radius
    grid and the (tau, m) grid, and the Theiler-window / minimum-line-length
    comparison.

Usage::

    python examples/apa_report_case1.py --out case1_results.html

    # recompute the contrasts instead of rendering the saved ones
    python examples/apa_report_case1.py --refit --out case1_results.html
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pose_dynamics.case_studies.matb import config as CFG
from pose_dynamics.case_studies.matb.reproduce import CONDITION_ORDER
from pose_dynamics.reporting import Table, describe_by, fmt_num, fmt_signed, render_report

NB = Path(__file__).resolve().parent.parent / "notebooks"

# ----------------------------------------------------------------------
# Display labels. The frames carry pipeline names; a reader wants the measure.
# ----------------------------------------------------------------------
COND = list(CONDITION_ORDER)                       # L, M, H
COND_LABEL = {"L": "Low", "M": "Moderate", "H": "High"}

#: The twelve analysis signals, in the order the manuscript introduces them.
SIGNALS = list(CFG.AUTO_FEATURES)
SIGNAL_LABEL = {
    "pupil_metric_dx": "Pupil displacement, horizontal",
    "pupil_metric_dy": "Pupil displacement, vertical",
    "pupil_metric_mag": "Pupil displacement, magnitude",
    "blink_aperture": "Blink aperture",
    "mouth_aperture": "Mouth aperture",
    "head_tx": "Head translation, horizontal",
    "head_ty": "Head translation, vertical",
    "head_translation_mag": "Head translation, magnitude",
    "head_rotation": "Head rotation",
    "head_scale_x": "Head scale, horizontal",
    "head_scale_y": "Head scale, vertical",
    "head_motion_mag": "Head motion, magnitude",
}

#: Kinematic order x summary statistic, as emitted by ``reproduce._summarise``.
LINEAR_MEASURES = [f"{order}_{stat}"
                   for order in ("pos", "vel", "accel")
                   for stat in CFG.LINEAR_STATS]
LINEAR_LABEL = {
    f"{order}_{stat}": f"{oname}, {sname}"
    for order, oname in (("pos", "Position"), ("vel", "Velocity"), ("accel", "Acceleration"))
    for stat, sname in (("mean", "<i>M</i>"), ("std", "<i>SD</i>"), ("min", "minimum"),
                        ("max", "maximum"), ("rms", "RMS"))
}

#: RQA measures. The first eight are the interpreted set; the rest are emitted by
#: the recurrence routine and reported for completeness in the contrast tables.
RQA_CORE = ["perc_recur", "perc_determ", "laminarity", "mean_line_length",
            "std_line_length", "maxl_found", "entropy", "trapping_time"]
RQA_EXTRA = ["vmax", "divergence", "complexity", "trend_lower_diag", "trend_upper_diag"]
RQA_MEASURES = RQA_CORE + RQA_EXTRA
RQA_LABEL = {
    "perc_recur": "%REC", "perc_determ": "%DET", "laminarity": "LAM",
    "mean_line_length": "L<sub>mean</sub>", "std_line_length": "L<sub>SD</sub>",
    "maxl_found": "L<sub>max</sub>", "entropy": "Entropy",
    "trapping_time": "Trapping time", "vmax": "V<sub>max</sub>",
    "divergence": "Divergence", "complexity": "Complexity",
    "trend_lower_diag": "Trend (lower diagonal)",
    "trend_upper_diag": "Trend (upper diagonal)",
}

#: Cross-RQA pairings, labelled by what is actually being coupled.
PAIR_LABEL = {
    "pupil_metric_dx": "Gaze-head, horizontal",
    "pupil_metric_dy": "Gaze-head, vertical",
    "pupil_metric_mag": "Gaze-head, magnitude",
}
CROSS_MINL = [2, 4]                 # committed l_min, and the comparison value

CONTRASTS = ["M-L", "H-L", "H-M"]
CONTRAST_LABEL = {"M-L": "Moderate − Low", "H-L": "High − Low", "H-M": "High − Moderate"}


# ----------------------------------------------------------------------
# Number formatting
# ----------------------------------------------------------------------
#: Case 1's measures span nine orders of magnitude -- a horizontal pupil
#: displacement near 1e-5 and an acceleration RMS near 1e3 both appear in the
#: linear tables -- so no single column-wide precision is defensible. APA's
#: fixed-decimal convention applies *within* a measure, which is what this
#: encodes: each row picks the decimals that give its own values three
#: significant figures, and every cell in that row is then rendered to it.
SIG_FIGS = 3
DP_MIN, DP_MAX = 2, 8


def _sig_dp(values, sig: int = SIG_FIGS, lo: int = DP_MIN, hi: int = DP_MAX) -> int:
    """Decimals giving ``sig`` significant figures for the largest value in a row."""
    finite = [abs(float(v)) for v in values
              if v is not None and isinstance(v, (int, float, np.floating, np.integer))
              and math.isfinite(float(v)) and float(v) != 0.0]
    if not finite:
        return lo
    return int(min(max(sig - 1 - math.floor(math.log10(max(finite))), lo), hi))


def with_dp(frame: pd.DataFrame, source: list[str]) -> pd.DataFrame:
    """Attach the per-row decimal count ``by_row`` reads, derived from ``source``."""
    out = frame.copy()
    out["_dp"] = [_sig_dp(row) for row in out[source].to_numpy()]
    return out


def by_row(signed: bool = False, extra: int = 0):
    """Formatter whose decimals follow the row's own magnitude (see :func:`with_dp`)."""
    def fmt(value, row):
        dp = int(row.get("_dp", DP_MIN)) + extra
        return fmt_signed(value, dp) if signed else fmt_num(value, dp)
    return fmt


# ----------------------------------------------------------------------
# Reshaping
# ----------------------------------------------------------------------
def stack(window: pd.DataFrame, groups: dict[str, str], measures: list[str],
          group_name: str) -> pd.DataFrame:
    """Long frame with one block of rows per group, measures as columns.

    ``groups`` maps a column-name prefix to its display label. The result is what
    :func:`~pose_dynamics.reporting.describe_by` wants: window rows carrying
    ``condition``, the group label, and one column per measure.
    """
    blocks = []
    for prefix, label in groups.items():
        cols = {f"{prefix}_{m}": m for m in measures if f"{prefix}_{m}" in window.columns}
        if not cols:
            continue
        block = window[["condition", *cols]].rename(columns=cols)
        block.insert(0, group_name, label)
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True)
    out[group_name] = pd.Categorical(out[group_name],
                                     categories=list(dict.fromkeys(groups.values())),
                                     ordered=True)
    return out


def split_metric(effects: pd.DataFrame) -> pd.DataFrame:
    """Split the ``{signal}_{measure}`` metric name into its two parts.

    Cross-RQA metrics are named ``crqa_l{n}_{signal}_{measure}``; they carry the
    minimum line length as a third part so the two ``l_min`` settings can sit in
    one table.
    """
    rows = []
    cross_re = re.compile(r"^crqa_l(\d+)_(.+)$")
    for name in effects["metric"]:
        m = cross_re.match(name)
        stem, lmin = (m.group(2), int(m.group(1))) if m else (name, None)
        for sig in sorted(SIGNALS, key=len, reverse=True):
            if stem.startswith(sig + "_"):
                rows.append({"signal": sig, "measure": stem[len(sig) + 1:], "lmin": lmin})
                break
        else:
            raise ValueError(f"cannot parse metric name {name!r}")
    return effects.join(pd.DataFrame(rows, index=effects.index))


def order_by(frame: pd.DataFrame, signal_labels: dict[str, str],
             measure_labels: dict[str, str], measures: list[str]) -> pd.DataFrame:
    """Label and order a contrast frame's signal/measure/contrast columns."""
    out = frame.copy()
    out["signal"] = pd.Categorical(out["signal"].map(signal_labels),
                                   categories=[signal_labels[s] for s in signal_labels],
                                   ordered=True)
    out["measure"] = pd.Categorical(out["measure"].map(measure_labels),
                                    categories=[measure_labels[m] for m in measures],
                                    ordered=True)
    out["contrast"] = pd.Categorical(out["contrast"].map(CONTRAST_LABEL),
                                     categories=[CONTRAST_LABEL[c] for c in CONTRASTS],
                                     ordered=True)
    sort = ["signal"] + (["lmin"] if "lmin" in out and out["lmin"].notna().any() else [])
    return out.dropna(subset=["measure"]).sort_values([*sort, "measure", "contrast"])


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["condition"] = pd.Categorical(df["condition"], categories=COND, ordered=True)
    return df


def refit_effects(window: pd.DataFrame) -> pd.DataFrame:
    """Recompute every condition contrast from the window-level table.

    Fits the same model the notebook does -- load as a fixed effect, a random
    intercept for participant, one model per measure -- at both reference levels,
    because the ``H-M`` contrast the manuscript reports alongside ``M-L`` and
    ``H-L`` is only a model coefficient when Moderate is the reference.

    Standard errors differ from the saved table in the third significant figure
    (statsmodels' profile-likelihood variance against the saved table's REML
    fit); the coefficients agree to numerical precision. ``d`` is the coefficient
    over the model's residual *SD*, matching the saved table's definition.
    """
    import statsmodels.formula.api as smf

    metrics = [c for c in window.columns
               if c not in ("participant", "condition", "window_index", "flagged")]
    rows = []
    for metric in metrics:
        sub = window[["participant", "condition", metric]].dropna()
        if sub.empty or sub[metric].nunique() < 2:
            continue
        for ref, wanted in (("L", ["M", "H"]), ("M", ["H"])):
            fit = smf.mixedlm(f"{metric} ~ C(condition, Treatment('{ref}'))",
                              sub, groups=sub["participant"]).fit()
            resid_sd = math.sqrt(float(fit.scale))
            for lvl in wanted:
                term = f"C(condition, Treatment('{ref}'))[T.{lvl}]"
                rows.append({
                    "metric": metric, "contrast": f"{lvl}-{ref}",
                    "beta": fit.params[term], "SE": fit.bse[term],
                    "df": float(fit.df_resid), "p": fit.pvalues[term],
                    "d": fit.params[term] / resid_sd if resid_sd else np.nan,
                    "n": len(sub),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def composition_table(window: pd.DataFrame) -> Table:
    rows = []
    for cond in COND:
        w = window[window["condition"] == cond]
        rows.append({
            "condition": COND_LABEL[cond],
            "participants": w["participant"].nunique(),
            "blocks": w.groupby(["participant", "condition"], observed=True).ngroups,
            "windows": len(w),
            "win_per_block": w.groupby(["participant", "condition"],
                                       observed=True).size().mean(),
        })
    return Table(
        number="1",
        title="Sample composition by cognitive-load condition",
        frame=pd.DataFrame(rows),
        columns={"condition": "Load", "participants": "Participants", "blocks": "Blocks",
                 "windows": "Analysis windows", "win_per_block": "Windows per block"},
        formatters={"participants": "int", "blocks": "int", "windows": "int",
                    "win_per_block": "num2"},
        align={"condition": "left"},
    )


def descriptive_table(long: pd.DataFrame, measures: list[str], labels: dict[str, str],
                      rows: list[str], row_headers: dict[str, str],
                      number: str, title: str) -> Table:
    frame, cols, spanners = describe_by(long, metrics=measures, rows=rows,
                                        columns="condition", column_order=COND)
    frame = with_dp(frame, [c for c in cols[len(rows) + 1:] if c.endswith(" M")])
    frame["measure"] = frame["metric"].map(labels)
    columns = {**row_headers, "measure": "Measure"}
    # The stub labels are written here, not read from data, so they may carry markup.
    fmts: dict[str, object] = {"measure": "html", **{r: "html" for r in rows}}
    for c in cols[len(rows) + 1:]:
        columns[c] = "<i>M</i>" if c.endswith(" M") else "<i>SD</i>"
        fmts[c] = by_row()
    spanners = [("", len(rows) + 1)] + [(COND_LABEL[c], 2) for c in COND]
    return Table(number=number, title=title, frame=frame, columns=columns,
                 formatters=fmts, spanners=spanners, stub_groups=rows,
                 align={**{r: "left" for r in rows}, "measure": "left"})


def contrast_table(effects: pd.DataFrame, number: str, title: str,
                   stub: dict[str, str], note: str = "") -> Table:
    frame = with_dp(effects, ["beta"])
    for col in stub:
        frame[col] = frame[col].astype(str)
    frame["contrast"] = frame["contrast"].astype(str)
    return Table(
        number=number, title=title, frame=frame,
        columns={**stub, "contrast": "Contrast", "beta": "<i>b</i>", "SE": "<i>SE</i>",
                 "df": "<i>df</i>", "p": "<i>p</i>", "d": "<i>d</i>", "n": "<i>N</i>"},
        formatters={**{c: "html" for c in stub},
                    "beta": by_row(signed=True), "SE": by_row(extra=1),
                    "df": "num2", "p": "p", "d": "signed", "n": "int"},
        note=note,
        stub_groups=[*stub, "contrast"][:-1],
        align={**{c: "left" for c in stub}, "contrast": "left"})


def sensitivity_table(sweep: pd.DataFrame, grid_name: str, grid_cols: dict[str, str],
                      number: str, title: str, note: str) -> Table:
    """Condition effects refit at every setting of a parameter grid.

    The saved sweeps carry the grid as a repr of the tuple it was grouped by
    (``"(np.int64(10), np.int64(3))"``); the numbers are recovered from it rather
    than re-run, since the sweep itself is the expensive part.
    """
    def numbers(text: str) -> list[float]:
        # "(np.int64(10), np.int64(3))" -- the grid value, not the dtype width, is the
        # one immediately before a closing parenthesis.
        found = re.findall(r"-?[\d.]+(?=\))", text) or re.findall(r"-?\d+\.?\d*", text)
        return [float(x) for x in found]

    grid = pd.DataFrame(sweep["grid"].map(numbers).tolist(),
                        columns=list(grid_cols), index=sweep.index)
    frame = pd.concat([sweep.drop(columns=["grid"]), grid], axis=1)
    frame["analysis"] = frame["analysis"].map({"auto": "Auto-RQA", "cross": "Cross-RQA"})
    frame["measure"] = pd.Categorical(frame["metric"].map(RQA_LABEL),
                                      categories=[RQA_LABEL[m] for m in RQA_MEASURES],
                                      ordered=True)
    frame["contrast"] = pd.Categorical(frame["contrast"].map(CONTRAST_LABEL),
                                       categories=[CONTRAST_LABEL[c] for c in CONTRASTS],
                                       ordered=True)
    frame = frame.sort_values(["analysis", "measure", "contrast", *grid_cols])
    frame = with_dp(frame, ["est"])
    dp = {c: ("num2" if c == "radius" else "int") for c in grid_cols}
    return Table(
        number=number, title=title, frame=frame,
        columns={"analysis": "Analysis", "measure": "Measure", "contrast": "Contrast",
                 **grid_cols, "est": "<i>b</i>", "p": "<i>p</i>"},
        formatters={"measure": "html", "est": by_row(signed=True), "p": "p", **dp},
        note=note,
        stub_groups=["analysis", "measure", "contrast"],
        align={"analysis": "left", "measure": "left", "contrast": "left"})


def theiler_table(check: pd.DataFrame, number: str, title: str, note: str) -> Table:
    """Recurrence and determinism under the Theiler-window / l_min alternatives."""
    settings = {
        "tw2_minl4": "Theiler 2, l<sub>min</sub> = 4 (committed)",
        "tw20_minl4": "Theiler 20, l<sub>min</sub> = 4",
        "tw60_minl4": "Theiler 60, l<sub>min</sub> = 4",
        "tw2_minl2": "Theiler 2, l<sub>min</sub> = 2",
    }
    check = check.copy()
    check["condition"] = pd.Categorical(check["condition"], categories=COND, ordered=True)
    long = stack(check, settings, ["rec", "det"], "setting")
    return _noted(descriptive_table(
        long, ["rec", "det"], {"rec": "%REC", "det": "%DET"},
        ["setting"], {"setting": "Detection setting"}, number, title), note)


def _noted(table: Table, note: str) -> Table:
    table.note = note
    return table


# ----------------------------------------------------------------------
def build(window: pd.DataFrame, effects: pd.DataFrame, radius: pd.DataFrame | None,
          tau: pd.DataFrame | None, theiler: pd.DataFrame | None) -> list:
    """The tables, and nothing else -- no prose sections.

    Everything a note would have said is either in the manuscript's methods or
    recoverable from the code; the notes that remain say only what a reader cannot
    infer from a title, which is where a number came from.
    """
    signal_labels = {s: SIGNAL_LABEL[s] for s in SIGNALS}
    pair_labels = {p: PAIR_LABEL[p] for p, _ in CFG.CROSS_PAIRS}

    fx = split_metric(effects)
    linear_fx = fx[fx["measure"].isin(LINEAR_MEASURES) & fx["lmin"].isna()]
    auto_fx = fx[fx["measure"].isin(RQA_MEASURES) & fx["lmin"].isna()]
    cross_fx = fx[fx["lmin"].notna()].copy()
    cross_fx["lmin"] = cross_fx["lmin"].astype(int)

    window_desc = "Descriptives are over analysis windows, the unit the models are fit on."
    blocks = [
        composition_table(window),
        _noted(descriptive_table(
            stack(window, signal_labels, LINEAR_MEASURES, "signal"),
            LINEAR_MEASURES, LINEAR_LABEL, ["signal"], {"signal": "Signal"},
            "2", "Linear kinematic measures by signal and cognitive load"), window_desc),
        _noted(descriptive_table(
            stack(window, signal_labels, RQA_CORE, "signal"),
            RQA_CORE, RQA_LABEL, ["signal"], {"signal": "Signal"},
            "3", "Auto-recurrence measures by signal and cognitive load"), window_desc),
        _noted(descriptive_table(
            pd.concat([stack(window, {f"crqa_l{n}_{p}": lbl for p, lbl in pair_labels.items()},
                             RQA_CORE, "pairing").assign(lmin=n) for n in CROSS_MINL],
                      ignore_index=True),
            RQA_CORE, RQA_LABEL, ["pairing", "lmin"],
            {"pairing": "Pairing", "lmin": "<i>l</i><sub>min</sub>"},
            "4", "Cross-recurrence measures by gaze-head pairing and cognitive load"),
            "Pairings are axis-matched: horizontal pupil displacement against horizontal "
            "head translation, vertical against vertical, magnitude against magnitude. "
            "<i>l</i><sub>min</sub> = 2 is the committed setting; 4 is the comparison "
            "value. " + window_desc),
        contrast_table(
            order_by(linear_fx, signal_labels, LINEAR_LABEL, LINEAR_MEASURES),
            "5", "Cognitive-load contrasts on the linear kinematic measures",
            {"signal": "Signal", "measure": "Measure"}),
        contrast_table(
            order_by(auto_fx, signal_labels, RQA_LABEL, RQA_MEASURES),
            "6", "Cognitive-load contrasts on the auto-recurrence measures",
            {"signal": "Signal", "measure": "Measure"}),
        contrast_table(
            order_by(cross_fx, pair_labels, RQA_LABEL, RQA_MEASURES),
            "7", "Cognitive-load contrasts on the cross-recurrence measures",
            {"signal": "Pairing", "lmin": "<i>l</i><sub>min</sub>", "measure": "Measure"}),
    ]

    n = 8
    if radius is not None:
        blocks.append(sensitivity_table(
            radius, "radius", {"radius": "Radius"}, str(n),
            "Cognitive-load contrasts across the recurrence-radius grid",
            "Pupil-displacement magnitude, refit at each radius. The committed radii "
            "are 0.20 for auto-recurrence and 0.30 for cross-recurrence; a sweep holds "
            "one radius across both."))
        n += 1
    if tau is not None:
        blocks.append(sensitivity_table(
            tau, "embedding", {"tau": "<i>τ</i>", "m": "<i>m</i>"}, str(n),
            "Cognitive-load contrasts across the embedding grid",
            "Pupil-displacement magnitude, refit at each (<i>τ</i>, <i>m</i>). The "
            "committed embedding is <i>τ</i> = 20, <i>m</i> = 4."))
        n += 1
    if theiler is not None:
        blocks.append(theiler_table(
            theiler, str(n),
            "Pupil-magnitude recurrence under alternative detection settings",
            "The Theiler window excludes the diagonal band of temporally adjacent "
            "points; <i>l</i><sub>min</sub> sets the shortest run of recurrent points "
            "counted as a diagonal line. " + window_desc))
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=NB / "matb_case1_results_v2.csv",
                    help="window-level results table")
    ap.add_argument("--effects", type=Path, default=NB / "matb_case1_effects_v2.csv",
                    help="saved condition contrasts (ignored with --refit)")
    ap.add_argument("--refit", action="store_true",
                    help="recompute the contrasts from --results instead of reading them")
    ap.add_argument("--radius-sensitivity", type=Path,
                    default=NB / "matb_case1_sensitivity_radius.csv")
    ap.add_argument("--tau-sensitivity", type=Path,
                    default=NB / "matb_case1_sensitivity_tau.csv")
    ap.add_argument("--theiler", type=Path, default=NB / "matb_case1_theiler_check.csv")
    ap.add_argument("--out", type=Path, default=Path("case1_results.html"))
    ap.add_argument("--fragment", action="store_true",
                    help="emit a <style>+markup fragment instead of a full HTML document")
    ap.add_argument("--no-toc", action="store_true", help="omit the index of tables")
    args = ap.parse_args()

    window = load_results(args.results)
    effects = refit_effects(window) if args.refit else pd.read_csv(args.effects)
    optional = [pd.read_csv(p) if p and p.exists() else None
                for p in (args.radius_sensitivity, args.tau_sensitivity, args.theiler)]

    blocks = build(window, effects, *optional)
    html_out = render_report(
        "Case 1 (MATB): full results",
        blocks,
        standalone=not args.fragment,
        toc=not args.no_toc,
    )
    args.out.write_text(html_out, encoding="utf-8")
    print(f"wrote {args.out}  ({len(html_out):,} bytes, {len(blocks)} blocks)")


if __name__ == "__main__":
    main()
