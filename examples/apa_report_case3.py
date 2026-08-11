"""Build the Case 3 (Mirror Game) full-results report as a standalone HTML page.

The third of the trio with ``examples/apa_report_case1.py`` and
``examples/apa_report_case2.py``, and the same shape: aggregate to the level you
intend to describe, build a list of :class:`~pose_dynamics.reporting.Table`, hand it
to :func:`~pose_dynamics.reporting.render_report`. Only the labels and the table
definitions below are Case 3-specific.

Case 3 is the case study whose full results have never been published elsewhere, so
the manuscript's supplementary tables *are* this page: the per-keypoint breakdowns,
the surrogate-pair baseline, and the parameter sweeps are all quoted in the text but
only summarised there.

Inputs
------
The artifacts the Case 3 notebook writes, all of which ship with the repository. The
raw ZED recordings are not needed:

``--results``
    one row per dyad-trial: subset and per-role kinematics, keypoint-averaged
    cross-recurrence at a fixed radius (``crossfx_*``) and at a matched 2.5%
    recurrence rate (``cross_*``), and the two MdCRQA variants (``mdfx_*``, ``md_*``).
``--keypoint-kinematics`` / ``--surrogate`` / ``--auto-rqa``
    the per-landmark tables: kinematics per person-trial-keypoint, the
    real-versus-surrogate-pair cross-recurrence, and each participant's own
    auto-recurrence.
``--embedding-estimates`` / ``--crqa-sweep`` / ``--md-sweep`` / ``--embedding-sweep``
    the parameter evidence and the three robustness grids.

Unlike Case 1, the models here are refit rather than read from a saved coefficient
table, because the notebook writes no such table; every model is the one the
corresponding notebook cell fits, so the coefficients reproduce the manuscript's.

Usage::

    python examples/apa_report_case3.py --out case3_results.html
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pose_dynamics.case_studies.mirror_game import KINEMATIC_FEATURES, movement_long
from pose_dynamics.case_studies.mirror_game import config as CFG
from pose_dynamics.reporting import Table, describe_by, fmt_num, fmt_signed, render_report

NB = Path(__file__).resolve().parent.parent / "notebooks"

# ----------------------------------------------------------------------
# Display labels. The frames carry pipeline names; a reader wants the measure.
# ----------------------------------------------------------------------
COND = list(CFG.CONDITION_ORDER)                   # b2b, uni, f2f
COND_LABEL = dict(CFG.CONDITION_LABELS)
CONTRASTS = ["uni", "f2f"]                         # against back-to-back
CONTRAST_LABEL = {"uni": "Unidirectional − BTB", "f2f": "Face-to-face − BTB"}

ROLES = ["leader", "follower"]
ROLE_LABEL = {"leader": "Leader", "follower": "Follower"}

KEYPOINTS = ["head", "l_wrist", "r_wrist", "l_ankle", "r_ankle"]
KEYPOINT_LABEL = {"head": "Head", "l_wrist": "Left wrist", "r_wrist": "Right wrist",
                  "l_ankle": "Left ankle", "r_ankle": "Right ankle"}

KIN_LABEL = {
    "disp_mean": "Displacement, <i>M</i>", "disp_rms": "Displacement, RMS",
    "vel_mean": "Velocity, <i>M</i>", "vel_rms": "Velocity, RMS",
    "accel_mean": "Acceleration, <i>M</i>", "accel_rms": "Acceleration, RMS",
}

#: The recurrence measures reported for every recurrence analysis. The results table
#: also carries ``lmax``/``mean_line`` aliases of ``maxl_found``/``mean_line_length``
#: and the trend/complexity extras; the aliases would duplicate rows and are dropped.
RQA_MEASURES = ["perc_recur", "perc_determ", "laminarity", "mean_line_length",
                "std_line_length", "maxl_found", "entropy", "trapping_time",
                "vmax", "divergence", "complexity",
                "trend_lower_diag", "trend_upper_diag"]
RQA_LABEL = {
    "perc_recur": "%REC", "perc_determ": "%DET", "laminarity": "LAM",
    "mean_line_length": "L<sub>mean</sub>", "std_line_length": "L<sub>SD</sub>",
    "maxl_found": "L<sub>max</sub>", "entropy": "Entropy",
    "trapping_time": "Trapping time", "vmax": "V<sub>max</sub>",
    "divergence": "Divergence", "complexity": "Complexity",
    "trend_lower_diag": "Trend (lower diagonal)",
    "trend_upper_diag": "Trend (upper diagonal)",
    "radius": "Radius achieved",
}

#: The two ways a recurrence threshold can be set, and the prefix each carries in the
#: results table. Under the fixed rule %REC is an outcome; under the matched rule
#: %REC is pinned at 2.5% and the radius needed to reach it is the outcome instead.
CROSS_MODES = {"crossfx": "Fixed radius", "cross": "Matched 2.5% recurrence"}
MD_MODES = {"mdfx": "Fixed radius", "md": "Matched 2.5% recurrence"}


# ----------------------------------------------------------------------
# Number formatting
# ----------------------------------------------------------------------
#: Case 3's measures span six orders of magnitude -- a mean frame-to-frame
#: displacement near 0.005 and a maximum line length near 100 appear in the same
#: kind of table -- so no single column-wide precision is defensible. APA's
#: fixed-decimal convention applies *within* a measure, which is what this encodes:
#: each row picks the decimals giving its own values three significant figures.
SIG_FIGS = 3
DP_MIN, DP_MAX = 2, 8


def _sig_dp(values, sig: int = SIG_FIGS, lo: int = DP_MIN, hi: int = DP_MAX) -> int:
    """Decimals giving ``sig`` significant figures for the largest value in a row."""
    finite = [abs(float(v)) for v in values
              if isinstance(v, (int, float, np.floating, np.integer))
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
# Models
#
# Two specifications, chosen by what the measure is a property of. A kinematic
# measure describes one person, so the model carries role and a person effect
# nested in pair. Cross-recurrence describes the dyad -- there is no leader value
# and follower value to distinguish -- so it carries a pair intercept only.
# ----------------------------------------------------------------------
def _fit(formula: str, frame: pd.DataFrame, groups: str, vc: dict | None = None):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # An empty ``vc_formula`` is not the same as none at all: statsmodels then
        # builds a variance-component design it cannot name, and reading the
        # coefficients back raises.
        kwargs = {"vc_formula": vc} if vc else {}
        return smf.mixedlm(formula, frame, groups=frame[groups], **kwargs).fit()


def _terms(fit, keys: dict[str, str], **extra) -> list[dict]:
    """One row per named model term, carrying the coefficient and its test."""
    return [{**extra, "contrast": label, "beta": fit.params[term], "SE": fit.bse[term],
             "z": fit.tvalues[term], "p": fit.pvalues[term]}
            for term, label in keys.items() if term in fit.params]


def _condition_terms() -> dict[str, str]:
    return {f"C(condition, Treatment('b2b'))[T.{lvl}]": lvl for lvl in CONTRASTS}


def dyad_effects(frame: pd.DataFrame, metrics: list[str], **extra) -> pd.DataFrame:
    """Condition effects on a dyad-level measure: condition + trial order, pair intercept."""
    rows = []
    for metric in metrics:
        sub = frame[["pair", "trial", "condition", metric]].dropna()
        if sub[metric].nunique() < 2:
            continue
        fit = _fit(f"{metric} ~ C(condition, Treatment('b2b')) + trial", sub, "pair")
        rows += _terms(fit, _condition_terms(), metric=metric, n=len(sub), **extra)
    return pd.DataFrame(rows)


def person_effects(frame: pd.DataFrame, metrics: list[str], **extra) -> pd.DataFrame:
    """Condition effects on a person-level measure: + role, and a person effect in pair."""
    rows = []
    frame = frame.copy()
    frame["person_id"] = frame["pair"].astype(str) + "_" + frame["person"].astype(str)
    for metric in metrics:
        sub = frame[["pair", "person_id", "trial", "condition", "role", metric]].dropna()
        if sub[metric].nunique() < 2:
            continue
        fit = _fit(f"{metric} ~ C(condition, Treatment('b2b')) "
                   f"+ C(role, Treatment('leader')) + trial",
                   sub, "pair", {"person": "0 + C(person_id)"})
        terms = {**_condition_terms(),
                 "C(role, Treatment('leader'))[T.follower]": "role"}
        rows += _terms(fit, terms, metric=metric, n=len(sub), **extra)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Reshaping
# ----------------------------------------------------------------------
def stack(frame: pd.DataFrame, groups: dict[str, str], measures: list[str],
          group_name: str, keep: list[str] | None = None) -> pd.DataFrame:
    """Long frame with one block of rows per column-prefix group.

    ``groups`` maps a column-name prefix to its display label; the result is what
    :func:`~pose_dynamics.reporting.describe_by` wants.
    """
    keep = keep or ["condition"]
    blocks = []
    for prefix, label in groups.items():
        cols = {f"{prefix}_{m}": m for m in measures if f"{prefix}_{m}" in frame.columns}
        if not cols:
            continue
        block = frame[[*keep, *cols]].rename(columns=cols)
        block.insert(0, group_name, label)
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True)
    out[group_name] = pd.Categorical(out[group_name],
                                     categories=list(dict.fromkeys(groups.values())),
                                     ordered=True)
    return out


def label_effects(effects: pd.DataFrame, measures: list[str],
                  labels: dict[str, str]) -> pd.DataFrame:
    """Label and order an effects frame's measure and contrast columns."""
    out = effects.copy()
    out["measure"] = pd.Categorical(out["metric"].map(labels),
                                    categories=[labels[m] for m in measures], ordered=True)
    order = {**CONTRAST_LABEL, "role": "Follower − leader"}
    out["contrast"] = pd.Categorical(out["contrast"].map(order),
                                     categories=list(dict.fromkeys(order.values())),
                                     ordered=True)
    return out.dropna(subset=["measure"])


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def descriptive_table(long: pd.DataFrame, measures: list[str], labels: dict[str, str],
                      rows: list[str], row_headers: dict[str, str],
                      number: str, title: str, note: str = "") -> Table:
    frame, cols, _ = describe_by(long, metrics=measures, rows=rows,
                                 columns="condition", column_order=COND)
    value_cols = cols[len(rows) + 1:]
    frame = with_dp(frame, [c for c in value_cols if c.endswith(" M")])
    frame["measure"] = frame["metric"].map(labels)
    columns = {**row_headers, "measure": "Measure"}
    # The stub labels are written here, not read from data, so they may carry markup.
    fmts: dict[str, object] = {"measure": "html", **{r: "html" for r in rows}}
    for c in value_cols:
        columns[c] = "<i>M</i>" if c.endswith(" M") else "<i>SD</i>"
        fmts[c] = by_row()
    spanners = [("", len(rows) + 1)] + [(COND_LABEL[c], 2) for c in COND]
    return Table(number=number, title=title, frame=frame, columns=columns,
                 formatters=fmts, spanners=spanners, stub_groups=rows, note=note,
                 align={**{r: "left" for r in rows}, "measure": "left"})


def effects_table(effects: pd.DataFrame, number: str, title: str,
                  stub: dict[str, str], note: str = "") -> Table:
    frame = with_dp(effects, ["beta"])
    for col in stub:
        frame[col] = frame[col].astype(str)
    frame["contrast"] = frame["contrast"].astype(str)
    return Table(
        number=number, title=title, frame=frame,
        columns={**stub, "contrast": "Contrast", "beta": "<i>b</i>", "SE": "<i>SE</i>",
                 "z": "<i>z</i>", "p": "<i>p</i>", "n": "<i>N</i>"},
        formatters={**{c: "html" for c in stub},
                    "beta": by_row(signed=True), "SE": by_row(extra=1),
                    "z": "num2", "p": "p", "n": "int"},
        note=note,
        stub_groups=list(stub),
        align={**{c: "left" for c in stub}, "contrast": "left"})


def composition_table(results: pd.DataFrame, number: str) -> Table:
    rows = []
    for cond in COND:
        d = results[results["condition"] == cond]
        rows.append({
            "condition": COND_LABEL[cond],
            "dyads": d["pair"].nunique(),
            "trials": len(d),
            "person_trials": 2 * len(d),
            "per_dyad": len(d) / max(d["pair"].nunique(), 1),
        })
    return Table(
        number=number,
        title="Sample composition by visual-coupling condition",
        frame=pd.DataFrame(rows),
        columns={"condition": "Condition", "dyads": "Dyads", "trials": "Dyad-trials",
                 "person_trials": "Person-trials", "per_dyad": "Trials per dyad"},
        formatters={"dyads": "int", "trials": "int", "person_trials": "int",
                    "per_dyad": "num2"},
        note="Each dyad contributed two blocks of six trials, one with each partner "
             "leading. Every trial lasted 30 s.",
        align={"condition": "left"})


def pct_change_table(long: pd.DataFrame, number: str, title: str) -> Table:
    """Movement in each coupled condition as a percentage of that role's own baseline.

    The manuscript's role asymmetry is a claim about relative change, and the raw
    means it is read off (Table 2) do not show it directly: leaders and followers
    move different amounts to begin with.
    """
    means = long.groupby(["role", "condition"], observed=True)[KINEMATIC_FEATURES].mean()
    rows = []
    for role in ROLES:
        base = means.loc[(role, "b2b")]
        for feature in KINEMATIC_FEATURES:
            row = {"role": ROLE_LABEL[role], "measure": KIN_LABEL[feature],
                   "b2b": base[feature]}
            for cond in CONTRASTS:
                value = means.loc[(role, cond), feature]
                row[f"{cond}_pct"] = (value / base[feature] - 1) * 100
            rows.append(row)
    frame = with_dp(pd.DataFrame(rows), ["b2b"])
    return Table(
        number=number, title=title, frame=frame,
        columns={"role": "Role", "measure": "Measure", "b2b": "Back-to-back <i>M</i>",
                 "uni_pct": "Unidirectional", "f2f_pct": "Face-to-face"},
        formatters={"role": "html", "measure": "html", "b2b": by_row(),
                    "uni_pct": "signed", "f2f_pct": "signed"},
        spanners=[("", 3), ("% change from back-to-back", 2)],
        note="Condition means of the five-keypoint subset, computed per person-trial. "
             "Percentages are relative to that role's own back-to-back baseline.",
        stub_groups=["role"],
        align={"role": "left", "measure": "left"})


def role_interaction_table(long: pd.DataFrame, number: str, title: str) -> Table:
    """Condition x role model: does the manipulation change the leader's behaviour too?"""
    terms = {"C(condition, Treatment('b2b'))[T.uni]": "Unidirectional − BTB",
             "C(condition, Treatment('b2b'))[T.f2f]": "Face-to-face − BTB",
             "C(role, Treatment('leader'))[T.follower]": "Follower − leader",
             "C(condition, Treatment('b2b'))[T.uni]:"
             "C(role, Treatment('leader'))[T.follower]": "Unidirectional × follower",
             "C(condition, Treatment('b2b'))[T.f2f]:"
             "C(role, Treatment('leader'))[T.follower]": "Face-to-face × follower"}
    rows = []
    for metric in KINEMATIC_FEATURES:
        fit = _fit(f"{metric} ~ C(condition, Treatment('b2b')) "
                   f"* C(role, Treatment('leader'))", long, "pair")
        rows += _terms(fit, terms, metric=metric, n=len(long))
    frame = pd.DataFrame(rows)
    frame["measure"] = pd.Categorical(frame["metric"].map(KIN_LABEL),
                                      categories=[KIN_LABEL[m] for m in KINEMATIC_FEATURES],
                                      ordered=True)
    frame["contrast"] = pd.Categorical(frame["contrast"],
                                       categories=list(terms.values()), ordered=True)
    frame = frame.sort_values(["measure", "contrast"])
    return effects_table(
        frame, number, title, {"measure": "Measure"},
        note="Random intercept for pair. The interaction rows ask whether the "
             "condition effect differs by role; the condition rows are then the "
             "leader's effect.")


def surrogate_interaction_table(surrogate: pd.DataFrame, metrics: list[str],
                                number: str, title: str) -> Table:
    """Condition x pairing: is the coordination effect larger in real than fake pairs?

    The condition effect on its own cannot separate coordination from the fact that
    people move more when they can see each other, since a fixed radius responds to
    the movement itself. Two people who never interacted, doing the same task under
    the same instruction, carry the second without the first.
    """
    terms = {"C(condition, Treatment('b2b'))[T.uni]": "Unidirectional − BTB",
             "C(condition, Treatment('b2b'))[T.f2f]": "Face-to-face − BTB",
             "C(kind, Treatment('surrogate'))[T.real]": "Real − surrogate",
             "C(condition, Treatment('b2b'))[T.uni]:"
             "C(kind, Treatment('surrogate'))[T.real]": "Unidirectional × real",
             "C(condition, Treatment('b2b'))[T.f2f]:"
             "C(kind, Treatment('surrogate'))[T.real]": "Face-to-face × real"}
    agg = (surrogate.groupby(["pair", "trial", "condition", "kind"], observed=True)[metrics]
           .mean().reset_index())
    rows = []
    for metric in metrics:
        fit = _fit(f"{metric} ~ C(condition, Treatment('b2b')) "
                   f"* C(kind, Treatment('surrogate'))", agg, "pair")
        rows += _terms(fit, terms, metric=metric, n=len(agg))
    frame = pd.DataFrame(rows)
    frame["measure"] = pd.Categorical(frame["metric"].map(RQA_LABEL),
                                      categories=[RQA_LABEL[m] for m in metrics],
                                      ordered=True)
    frame["contrast"] = pd.Categorical(frame["contrast"],
                                       categories=list(terms.values()), ordered=True)
    frame = frame.sort_values(["measure", "contrast"])
    return effects_table(
        frame, number, title, {"measure": "Measure"},
        note="Metrics averaged across the five keypoints per trial, then modelled with "
             "a random intercept for pair. Surrogate pairing is the reference, so the "
             "interaction rows are the test: they ask how much of the condition effect "
             "belongs to the interaction rather than to how each person moved.")


def sweep_table(sweep: pd.DataFrame, metrics: list[str], grid: str, grid_header: str,
                number: str, title: str, note: str, grid_format: object = "num2",
                sort_key=None) -> Table:
    """Condition effects refit at every setting of a parameter grid.

    A sweep carrying a ``keypoint`` column is averaged across keypoints per trial
    first, so the grid is compared against the same keypoint-averaged measure the
    primary analysis reports rather than against a differently-scoped one.
    """
    keys = ["pair", "trial", "condition"]
    rows = []
    for value, block in sweep.groupby(grid, observed=True):
        frame = (block.groupby(keys, observed=True)[metrics].mean().reset_index()
                 if "keypoint" in block.columns else block)
        rows.append(dyad_effects(frame, metrics).assign(**{grid: value}))
    frame = label_effects(pd.concat(rows, ignore_index=True), metrics, RQA_LABEL)
    frame["_grid"] = frame[grid].map(sort_key) if sort_key else frame[grid]
    frame = with_dp(frame.sort_values(["measure", "contrast", "_grid"]), ["beta"])
    return Table(
        number=number, title=title, frame=frame,
        columns={"measure": "Measure", "contrast": "Contrast", grid: grid_header,
                 "beta": "<i>b</i>", "SE": "<i>SE</i>", "p": "<i>p</i>"},
        formatters={"measure": "html", "contrast": "text", grid: grid_format,
                    "beta": by_row(signed=True), "SE": by_row(extra=1), "p": "p"},
        note=note,
        stub_groups=["measure", "contrast"],
        align={"measure": "left", "contrast": "left", grid: "right"})


# ----------------------------------------------------------------------
def build(results: pd.DataFrame, kp_kin: pd.DataFrame | None,
          surrogate: pd.DataFrame | None, auto: pd.DataFrame | None,
          embedding: pd.DataFrame | None, crqa_sweep: pd.DataFrame | None,
          md_sweep: pd.DataFrame | None, emb_sweep: pd.DataFrame | None) -> list:
    """The tables, and nothing else -- no prose sections.

    Notes carry only what a reader cannot infer from a title: which rows a model was
    fit on, and which of two threshold rules produced a column.
    """
    long = movement_long(results)
    kp_labels = {k: KEYPOINT_LABEL[k] for k in KEYPOINTS}
    kp_order = list(kp_labels.values())
    core = ["perc_recur", "perc_determ", "maxl_found", "mean_line_length",
            "std_line_length", "trapping_time"]
    counter = iter(str(i) for i in range(1, 100))

    def as_keypoint(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(keypoint=pd.Categorical(frame["keypoint"].map(KEYPOINT_LABEL),
                                                    categories=kp_order, ordered=True))

    def by_mode(modes: dict[str, str]) -> pd.DataFrame:
        """Condition effects for one recurrence analysis under both threshold rules."""
        per_mode = [
            dyad_effects(results, [f"{prefix}_{m}" for m in RQA_MEASURES], mode=label)
            .assign(metric=lambda f, p=prefix: f["metric"].str.removeprefix(f"{p}_"))
            for prefix, label in modes.items()]
        return (label_effects(pd.concat(per_mode, ignore_index=True),
                              RQA_MEASURES, RQA_LABEL)
                .sort_values(["mode", "measure", "contrast"]))

    person_model = ("Fixed effects of condition, role, and trial order; random "
                    "intercepts for pair and for individual nested within pair. "
                    "Back-to-back is the reference condition.")
    dyad_model = ("Cross-recurrence is a property of the dyad rather than of either "
                  "participant, so these models carry a random intercept for pair and "
                  "no person-level effect. Condition and trial order are fixed effects, "
                  "with back-to-back as the reference condition.")

    blocks = [
        composition_table(results, next(counter)),
        descriptive_table(
            long.assign(role=pd.Categorical(long["role"].map(ROLE_LABEL),
                                            categories=[ROLE_LABEL[r] for r in ROLES],
                                            ordered=True)),
            KINEMATIC_FEATURES, KIN_LABEL, ["role"], {"role": "Role"},
            next(counter), "Movement magnitude by visual-coupling condition and role",
            "One row per person-trial, averaged across the five-keypoint subset "
            "(head, left/right wrists, left/right ankles)."),
        pct_change_table(long, next(counter),
                         "Movement magnitude relative to each role's back-to-back baseline"),
        effects_table(
            label_effects(person_effects(long, KINEMATIC_FEATURES),
                          KINEMATIC_FEATURES, KIN_LABEL).sort_values(["measure", "contrast"]),
            next(counter), "Condition effects on the subset-level kinematic measures",
            {"measure": "Measure"}, note=person_model),
        role_interaction_table(long, next(counter),
                               "Condition × role effects on the kinematic measures"),
    ]

    if kp_kin is not None:
        kp_fx = pd.concat([person_effects(kp_kin[kp_kin["keypoint"] == k],
                                          KINEMATIC_FEATURES, keypoint=KEYPOINT_LABEL[k])
                           for k in KEYPOINTS], ignore_index=True)
        kp_fx["keypoint"] = pd.Categorical(kp_fx["keypoint"], categories=kp_order,
                                           ordered=True)
        blocks += [
            descriptive_table(
                as_keypoint(kp_kin), KINEMATIC_FEATURES, KIN_LABEL,
                ["keypoint"], {"keypoint": "Keypoint"}, next(counter),
                "Movement magnitude by keypoint and visual-coupling condition",
                "One row per person-trial-keypoint, both roles pooled."),
            effects_table(
                label_effects(kp_fx, KINEMATIC_FEATURES, KIN_LABEL)
                .sort_values(["keypoint", "measure", "contrast"]),
                next(counter), "Condition effects on the kinematic measures, by keypoint",
                {"keypoint": "Keypoint", "measure": "Measure"},
                note="Fit separately within each keypoint. " + person_model),
        ]

    blocks += [
        descriptive_table(
            stack(results, CROSS_MODES, [*RQA_MEASURES, "radius"], "mode"),
            [*RQA_MEASURES, "radius"], RQA_LABEL, ["mode"], {"mode": "Threshold rule"},
            next(counter), "Keypoint-averaged cross-recurrence measures by condition",
            "Metrics computed per keypoint and averaged across the five before "
            "aggregation. Under the fixed rule (radius 0.30) %REC is an outcome; under "
            "the matched rule %REC is pinned at 2.5% and the radius needed to reach it "
            "is the outcome instead."),
        effects_table(
            by_mode(CROSS_MODES), next(counter),
            "Condition effects on the keypoint-averaged cross-recurrence measures",
            {"mode": "Threshold rule", "measure": "Measure"}, note=dyad_model),
        descriptive_table(
            stack(results, MD_MODES, [*RQA_MEASURES, "radius"], "mode"),
            [*RQA_MEASURES, "radius"], RQA_LABEL, ["mode"], {"mode": "Threshold rule"},
            next(counter), "Multidimensional cross-recurrence measures by condition",
            "Each participant's five keypoint-magnitude signals treated as one "
            "five-dimensional system. The fixed radius is 0.59 rather than 0.30 because "
            "delay-embedding a five-dimensional signal gives a twenty-dimensional state "
            "space, in which pairwise distances are larger; thresholds are not "
            "comparable across analyses of different dimensionality."),
        effects_table(
            by_mode(MD_MODES), next(counter),
            "Condition effects on the multidimensional cross-recurrence measures",
            {"mode": "Threshold rule", "measure": "Measure"}, note=dyad_model),
    ]

    if surrogate is not None:
        kinds = {"real": "Real pairs", "surrogate": "Surrogate pairs"}
        kp_cross = pd.concat(
            [dyad_effects(surrogate[(surrogate["kind"] == kind)
                                    & (surrogate["keypoint"] == keypoint)],
                          core, pairing=label, keypoint=KEYPOINT_LABEL[keypoint])
             for kind, label in kinds.items() for keypoint in KEYPOINTS],
            ignore_index=True)
        kp_cross["pairing"] = pd.Categorical(kp_cross["pairing"],
                                             categories=list(kinds.values()), ordered=True)
        kp_cross["keypoint"] = pd.Categorical(kp_cross["keypoint"], categories=kp_order,
                                              ordered=True)
        blocks += [
            descriptive_table(
                as_keypoint(surrogate[surrogate["kind"] == "real"]),
                core, RQA_LABEL, ["keypoint"], {"keypoint": "Keypoint"}, next(counter),
                "Cross-recurrence measures by keypoint and condition",
                "Real pairs at the fixed radius of 0.30, before the average across "
                "keypoints that the keypoint-averaged tables report."),
            effects_table(
                label_effects(kp_cross, core, RQA_LABEL)
                .sort_values(["pairing", "keypoint", "measure", "contrast"]),
                next(counter),
                "Condition effects on the cross-recurrence measures, by keypoint and pairing",
                {"pairing": "Pairing", "keypoint": "Keypoint", "measure": "Measure"},
                note="Surrogate pairs re-pair each trial's leader with a different "
                     "dyad's follower from the same condition, five per trial: two "
                     "people who never interacted, doing the same task under the same "
                     "instruction. " + dyad_model),
            surrogate_interaction_table(
                surrogate, core, next(counter),
                "Condition × pairing effects on the cross-recurrence measures"),
        ]

    if auto is not None:
        auto_fx = pd.concat([person_effects(auto[auto["keypoint"] == k], core,
                                            keypoint=KEYPOINT_LABEL[k])
                             for k in KEYPOINTS], ignore_index=True)
        auto_fx["keypoint"] = pd.Categorical(auto_fx["keypoint"], categories=kp_order,
                                             ordered=True)
        blocks.append(effects_table(
            label_effects(auto_fx, core, RQA_LABEL)
            .sort_values(["keypoint", "measure", "contrast"]),
            next(counter),
            "Condition effects on each participant's own auto-recurrence, by keypoint",
            {"keypoint": "Keypoint", "measure": "Measure"},
            note="Auto-recurrence involves one participant and no partner, so a "
                 "condition effect here is a change in how that person moved rather "
                 "than in how the two coordinated. " + person_model))

    if embedding is not None:
        emb_labels = {"ami_first_min": "AMI, first minimum",
                      "ami_1e": "AMI, 1/<i>e</i> crossing", "fnn_dim": "FNN dimension"}
        blocks.append(descriptive_table(
            as_keypoint(embedding), list(emb_labels), emb_labels,
            ["keypoint"], {"keypoint": "Keypoint"}, next(counter),
            "Embedding parameter estimates by keypoint and condition",
            "Estimated per person-trial-keypoint and pooled to commit a single "
            "<i>τ</i> = 20, <i>m</i> = 4 across all trials. Delays are in frames at "
            "30 Hz."))

    if crqa_sweep is not None:
        blocks.append(sweep_table(
            crqa_sweep, core, "radius", "Radius", next(counter),
            "Condition effects on the keypoint-averaged cross-recurrence measures "
            "across the radius grid",
            "Refit at each radius after averaging across the five keypoints. The "
            "committed radius is 0.30. " + dyad_model))
    if md_sweep is not None:
        blocks.append(sweep_table(
            md_sweep, core, "radius", "Radius", next(counter),
            "Condition effects on the multidimensional cross-recurrence measures "
            "across the radius grid",
            "Refit at each radius. The committed radius is 0.59. " + dyad_model))
    if emb_sweep is not None:
        blocks.append(sweep_table(
            emb_sweep.assign(grid=emb_sweep["tau"].astype(str) + ", "
                             + emb_sweep["m"].astype(str)),
            core, "grid", "<i>τ</i>, <i>m</i>", next(counter),
            "Condition effects on the keypoint-averaged cross-recurrence measures "
            "across the embedding grid",
            "Refit at each (<i>τ</i>, <i>m</i>) after averaging across the five "
            "keypoints. The committed embedding is <i>τ</i> = 20, <i>m</i> = 4. "
            + dyad_model,
            grid_format="text",
            sort_key=lambda s: tuple(int(x) for x in s.split(", "))))
    return blocks


# ----------------------------------------------------------------------
def load(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    if "condition" in frame:
        frame["condition"] = pd.Categorical(frame["condition"], categories=COND, ordered=True)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=NB / "mirror_case3_results.csv",
                    help="dyad-trial results table")
    ap.add_argument("--keypoint-kinematics", type=Path,
                    default=NB / "mirror_case3_keypoint_kinematics.csv")
    ap.add_argument("--surrogate", type=Path, default=NB / "mirror_case3_surrogate.csv")
    ap.add_argument("--auto-rqa", type=Path, default=NB / "mirror_case3_auto_rqa.csv")
    ap.add_argument("--embedding-estimates", type=Path,
                    default=NB / "mirror_case3_embedding_estimates.csv")
    ap.add_argument("--crqa-sweep", type=Path,
                    default=NB / "mirror_case3_crqa_keypoint_sweep.csv")
    ap.add_argument("--md-sweep", type=Path, default=NB / "mirror_case3_md_radius_sweep.csv")
    ap.add_argument("--embedding-sweep", type=Path,
                    default=NB / "mirror_case3_embedding_sweep.csv")
    ap.add_argument("--out", type=Path, default=Path("case3_results.html"))
    ap.add_argument("--fragment", action="store_true",
                    help="emit a <style>+markup fragment instead of a full HTML document")
    ap.add_argument("--no-toc", action="store_true", help="omit the index of tables")
    args = ap.parse_args()

    results = load(args.results)
    if results is None:
        raise SystemExit(f"no results table at {args.results}")
    optional = [load(p) for p in (args.keypoint_kinematics, args.surrogate, args.auto_rqa,
                                  args.embedding_estimates, args.crqa_sweep,
                                  args.md_sweep, args.embedding_sweep)]

    blocks = build(results, *optional)
    html_out = render_report(
        "Case 3 (Mirror Game): full results",
        blocks,
        standalone=not args.fragment,
        toc=not args.no_toc,
    )
    args.out.write_text(html_out, encoding="utf-8")
    print(f"wrote {args.out}  ({len(html_out):,} bytes, {len(blocks)} blocks)")


if __name__ == "__main__":
    main()
